/* -*- c++ -*- */

#include <stdio.h>

#include "device.h"

#define _NUM_SIMPLE_TIMER 40
#define _NUM_SIMPLE_COUNTER 30
#include <unistd.h>
#include <string.h>
#include <sched.h>
#define _MIN(A,B) (A<B)?A:B
#define _MAX(A,B) (A>B)?A:B
#define _SIZE_FCI_BATCHES 6

/* ---------------------------------------------------------------------- */

void Device::init_jk_ao2mo(int ncore, int nmo)
{
  double t0 = omp_get_wtime();

  // host initializes on each device
  
  for(int id=0; id<num_devices; ++id) {
    pm->dev_set_device(id);
    
    my_device_data * dd = &(device_data[id]);
    
    int size_j_pc = ncore*nmo;
    int size_k_pc = ncore*nmo;

    grow_array(dd->d_j_pc, size_j_pc, dd->size_j_pc, "j_pc", FLERR);
    grow_array(dd->d_k_pc, size_k_pc, dd->size_k_pc, "k_pc", FLERR);

    dd->active = 0;
  }
  
  int _size_buf_j_pc = num_devices*nmo*ncore;
  
  grow_array_host(buf_j_pc, _size_buf_j_pc, size_buf_j_pc, "h:buf_j_pc");
  
  int _size_buf_k_pc = num_devices*nmo*ncore;

  grow_array_host(buf_k_pc, _size_buf_k_pc, size_buf_k_pc, "h:buf_k_pc");
  
  double t1 = omp_get_wtime();
  t_array[8] += t1 - t0;
  // counts in pull ppaa
}

/* ---------------------------------------------------------------------- */

void Device::init_ppaa_papa_ao2mo( int nmo, int ncas)
{
  double t0 = omp_get_wtime();

  // initializing only cpu side, gpu ppaa will be a buffer array (dd->d_buf3) 

  int _size_buf_ppaa = num_devices*nmo*nmo*ncas*ncas;
  grow_array_host(buf_ppaa, _size_buf_ppaa, size_buf_ppaa, "h:buf_ppaa");

  int _size_buf_papa = num_devices*nmo*ncas*nmo*ncas;
  grow_array_host(buf_papa, _size_buf_papa, size_buf_papa, "h:buf_papa");

  double t1 = omp_get_wtime();
  t_array[8] += t1 - t0;
  // counts in pull ppaa_papa
}

/* ---------------------------------------------------------------------- */

void Device::init_eri_h2eff(int nmo, int ncas)
{
  double t0 = omp_get_wtime();
  
  // host initializes on each device 

  int ncas_pair = ncas*(ncas+1)/2;
  int size_eri_h2eff = nmo*ncas*ncas_pair;

  for(int id=0; id<num_devices; ++id) {
    pm->dev_set_device(id);

    my_device_data * dd = &(device_data[id]);

    dd->active = 0;

    grow_array(dd->d_eri_h2eff, size_eri_h2eff, dd->size_eri_h2eff, "eri_h2eff", FLERR);
  }
  
  int _size_buf_eri_h2eff = num_devices * size_eri_h2eff;

  grow_array_host(buf_eri_h2eff, _size_buf_eri_h2eff, size_buf_eri_h2eff, "h:buf_eri_h2eff");
  
  double t1 = omp_get_wtime();
  t_array[8] += t1 - t0;
  // counts in pull ppaa
}

/* ---------------------------------------------------------------------- */

void Device::extract_mo_cas(int ncas, int ncore, int nao)
{
  double t0 = omp_get_wtime();
  
  const int _size_mo_cas = ncas*nao; 
  for(int id=0; id<num_devices; ++id) {
    pm->dev_set_device(id);
    
    my_device_data * dd = &(device_data[id]);

    grow_array(dd->d_mo_cas, _size_mo_cas, dd->size_mo_cas, "mo_cas", FLERR);

#if 0 
    dim3 block_size(1,1,1);
    dim3 grid_size(_TILE(ncas, block_size.x), _TILE(nao, block_size.y));
    get_mo_cas<<<grid_size, block_size, 0, dd->stream>>>(dd->d_mo_coeff, dd->d_mo_cas, ncas, ncore, nao);
#else
    get_mo_cas(dd->d_mo_coeff,dd->d_mo_cas, ncas, ncore, nao);
#endif
  }
  
  double t1 = omp_get_wtime();
  t_array[7] += t1 - t0;
}

/* ---------------------------------------------------------------------- */

#if defined(_ENABLE_P2P)
void Device::pull_jk_ao2mo_v4(py::array_t<double> _j_pc, py::array_t<double> _k_pc, int nmo, int ncore)
{
  double t0 = omp_get_wtime();

  py::buffer_info info_j_pc = _j_pc.request(); //2D array (nmo*ncore)
  double * j_pc = static_cast<double*>(info_j_pc.ptr);
  
  py::buffer_info info_k_pc = _k_pc.request(); //2D array (nmo*ncore)
  double * k_pc = static_cast<double*>(info_k_pc.ptr);
  
  int N = nmo * ncore;

  std::vector<double *> pc_vec(num_devices);
  std::vector<double *> buf_vec(num_devices);
  std::vector<int> active(num_devices);
  
  for(int i=0; i<num_devices; ++i) {
    my_device_data * dd = &(device_data[i]);
    pc_vec[i] = dd->d_j_pc;
    buf_vec[i] = dd->d_buf1;
    active[i] = dd->active;
  }

  mgpu_reduce(pc_vec, buf_j_pc, N, true, buf_vec, active);

#pragma omp parallel for
  for(int i=0; i<nmo*ncore; ++i) j_pc[i] = buf_j_pc[i];

  // Pulling k_pc from all devices

  for(int i=0; i<num_devices; ++i) {
    my_device_data * dd = &(device_data[i]);
    pc_vec[i] = dd->d_k_pc;
    buf_vec[i] = dd->d_buf1;
  }
  
  mgpu_reduce(pc_vec, buf_k_pc, N, true, buf_vec, active);

#pragma omp parallel for
  for(int i=0; i<nmo*ncore; ++i) k_pc[i] = buf_k_pc[i];
  
  double t1 = omp_get_wtime();
  t_array[10] += t1 - t0;
  // counts in pull ppaa
}

#else

void Device::pull_jk_ao2mo_v4(py::array_t<double> _j_pc, py::array_t<double> _k_pc, int nmo, int ncore)
{
  double t0 = omp_get_wtime();

  py::buffer_info info_j_pc = _j_pc.request(); //2D array (nmo*ncore)
  double * j_pc = static_cast<double*>(info_j_pc.ptr);
  double * tmp;
  
  py::buffer_info info_k_pc = _k_pc.request(); //2D array (nmo*ncore)
  double * k_pc = static_cast<double*>(info_k_pc.ptr);
  
  int size = nmo*ncore;//*sizeof(double);

  printf("nmo= %i  ncore= %i\n",nmo, ncore);
  
  // Pulling j_pc from all devices
  
  for (int i=0; i<num_devices; ++i){
    pm->dev_set_device(i);
    my_device_data * dd = &(device_data[i]);
    
    tmp = &(buf_j_pc[i*nmo*ncore]);
    
    if(dd->active) pm->dev_pull_async(dd->d_j_pc, tmp, size*sizeof(double));
  }
  
  // Adding j_pc from all devices

  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);

    my_device_data * dd = &(device_data[i]);
    
    pm->dev_stream_wait();

    if(i > 0 && dd->active) {
      
      tmp = &(buf_j_pc[i * nmo* ncore]);
//#pragma omp parallel for
      for(int j=0; j<ncore*nmo; ++j) buf_j_pc[j] += tmp[j];
    }
  }
#ifdef _DEBUG_DEVICE
  for (int i=0; i<num_devices;++i){
      for (int j=0; j<nmo;++j){
          for (int k=0; k<ncore;++k){
              printf("%f\t",buf_j_pc[i*nmo*ncore +j*ncore+k]);
          } printf("\n");
      } printf("\n");
  } 
#endif
  //copy buf_j_pc[first nmo*ncore] to j_pc
  std::memcpy(j_pc,buf_j_pc,nmo*ncore*sizeof(double));

  // Pulling k_pc from all devices
  
  for (int i=0; i<num_devices; ++i){
    pm->dev_set_device(i);
    
    my_device_data * dd = &(device_data[i]);

    tmp = &(buf_k_pc[i*nmo*ncore]);
    
    if(dd->active) pm->dev_pull_async(dd->d_k_pc, tmp, size*sizeof(double));
  }
  
  // Adding k_pc from all devices
  
  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);
    
    my_device_data * dd = &(device_data[i]);
    
    pm->dev_stream_wait();

    if(i > 0 && dd->active) {
      
      tmp = &(buf_k_pc[i * nmo* ncore]);
//#pragma omp parallel for
      for(int j=0; j<ncore*nmo; ++j) buf_k_pc[j] += tmp[j];
    }
  }
    
  //copy buf_k_pc[first nmo*ncore] to k_pc
  std::memcpy(k_pc,buf_k_pc,nmo*ncore*sizeof(double));
  double t1 = omp_get_wtime();
  t_array[10] += t1 - t0;
  // counts in pull ppaa
}
#endif

/* ---------------------------------------------------------------------- */

#if defined(_ENABLE_P2P)
void Device::pull_ppaa_papa_ao2mo_v4(py::array_t<double> _ppaa, py::array_t<double> _papa, int nmo, int ncas)
{
  double t0 = omp_get_wtime();

  py::buffer_info info_ppaa = _ppaa.request(); //2D array (nmo*ncore)
  py::buffer_info info_papa = _papa.request(); //2D array (nmo*ncore)
  double * ppaa = static_cast<double*>(info_ppaa.ptr);
  double * papa = static_cast<double*>(info_papa.ptr);

  int N = nmo*nmo*ncas*ncas;
  
  // Pulling ppaa from all devices

  //  printf("nmo= %i  ncas= %i  N= %i\n",nmo, ncas, N);

  std::vector<double *> p_vec(num_devices);
  std::vector<double *> buf_vec(num_devices);
  std::vector<int> active(num_devices);
  
  for(int i=0; i<num_devices; ++i) {
    my_device_data * dd = &(device_data[i]);
    p_vec[i] = dd->d_ppaa; // pointing at d_buf3
    buf_vec[i] = dd->d_buf2;
    active[i] = dd->active;
  }

  mgpu_reduce(p_vec, buf_ppaa, N, true, buf_vec, active);

#pragma omp parallel for
  for(int i=0; i<N; ++i) ppaa[i] = buf_ppaa[i];

  // Pulling papa from all devices
  
  for(int i=0; i<num_devices; ++i) {
    my_device_data * dd = &(device_data[i]);
    p_vec[i] = dd->d_papa; // pointing at d_buf3
  }

  mgpu_reduce(p_vec, buf_papa, N, true, buf_vec, active);

#pragma omp parallel for
  for(int i=0; i<N; ++i) papa[i] = buf_papa[i];

  double t1 = omp_get_wtime();
  t_array[10] += t1 - t0;
  count_array[6] += 1; //doing this in ppaa pull, not in any inits or computes
}

#else
void Device::pull_ppaa_papa_ao2mo_v4(py::array_t<double> _ppaa, py::array_t<double> _papa, int nmo, int ncas)
{
  double t0 = omp_get_wtime();

  py::buffer_info info_ppaa = _ppaa.request(); //2D array (nmo*ncore)
  py::buffer_info info_papa = _papa.request(); //2D array (nmo*ncore)
  double * ppaa = static_cast<double*>(info_ppaa.ptr);
  double * papa = static_cast<double*>(info_papa.ptr);
  double * tmp;
  const int _size_ppaa = nmo*nmo*ncas*ncas;
  const int _size_papa = nmo*nmo*ncas*ncas;
  // Pulling ppaa from all devices
  
  for (int i=0; i<num_devices; ++i){
    pm->dev_set_device(i);

    my_device_data * dd = &(device_data[i]);

    tmp = &(buf_ppaa[i*_size_ppaa]);
    
    if (dd->active) pm->dev_pull_async(dd->d_ppaa, tmp, _size_ppaa*sizeof(double));
  }
  
  // Adding ppaa from all devices
  
  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);

    my_device_data * dd = &(device_data[i]);
    
    pm->dev_stream_wait();

    if(i > 0 && dd->active) {
      
      tmp = &(buf_ppaa[i * _size_ppaa]);
//#pragma omp parallel for
      for(int j=0; j<_size_ppaa; ++j) buf_ppaa[j] += tmp[j];
    }
  }
  //copy buf_ppaa[first nmo*nmo*ncas*ncas] to ppaa
  std::memcpy(ppaa,buf_ppaa,_size_ppaa*sizeof(double));

  // Pulling papa from all devices
  for (int i=0; i<num_devices; ++i){
    pm->dev_set_device(i);

    my_device_data * dd = &(device_data[i]);

    tmp = &(buf_papa[i*_size_papa]);
    
    if (dd->d_papa) pm->dev_pull_async(dd->d_papa, tmp, _size_papa*sizeof(double));
  }
  
  // Adding papa from all devices
  
  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);

    my_device_data * dd = &(device_data[i]);
    
    pm->dev_stream_wait();

    if(i > 0 && dd->active) {
      
      tmp = &(buf_papa[i * _size_papa]);
//#pragma omp parallel for
      for(int j=0; j<_size_papa; ++j) buf_papa[j] += tmp[j];
    }
  }
  //copy buf_papa[first nmo*nmo*ncas*ncas] to papa
  std::memcpy(papa,buf_papa,_size_papa*sizeof(double));
  double t1 = omp_get_wtime();
  t_array[10] += t1 - t0;
  count_array[6] += 1; //doing this in ppaa pull, not in any inits or computes
}
#endif

/* ---------------------------------------------------------------------- */

void Device::df_ao2mo_v4 (int blksize, int nmo, int nao, int ncore, int ncas, int naux, 
				  int count, size_t addr_dfobj)
{
  double t0 = omp_get_wtime();
  
  pm->dev_profile_start("AO2MO v4");

  const int device_id = count % num_devices;

  pm->dev_set_device(device_id);

  my_device_data * dd = &(device_data[device_id]);

  dd->active = 1;

  //  py::buffer_info info_eri1 = _eri1.request(); // 2D array (naux, nao_pair) nao_pair= nao*(nao+1)/2
  const int nao_pair = nao*(nao+1)/2;
  //  double * eri = static_cast<double*>(info_eri1.ptr);
  
  int _size_eri = naux * nao_pair;
  int _size_eri_unpacked = naux * nao * nao; 
  int _size_ppaa = nmo * nmo * ncas * ncas;

#ifdef _DEBUG_DEVICE
#if defined (_GPU_CUDA)
  size_t freeMem;size_t totalMem;
  freeMem=0;totalMem=0;
  cudaMemGetInfo(&freeMem, &totalMem);
  printf("Starting ao2mo Free memory %lu bytes, total memory %lu bytes\n",freeMem,totalMem);
#endif
#endif

  int max_size_buf = 2 * _size_ppaa;
  if(_size_eri_unpacked > max_size_buf) max_size_buf = _size_eri_unpacked;
  
  grow_array(dd->d_buf1, max_size_buf, dd->size_buf1, "buf1", FLERR);
  grow_array(dd->d_buf2, max_size_buf, dd->size_buf2, "buf2", FLERR);
  grow_array(dd->d_buf3, max_size_buf, dd->size_buf3, "buf3", FLERR);
  
  // I want to fit both ppaa and papa inside buf3 to remove it from cpu side
  // my guess is blksize*nao_s*nao_s > 2 * nmo_f * nmo_f * ncas_f * ncas_f (dd->size_eri_unpacked is for the entire system. Usually nao_s > sqrt(2)*nao_f, blksize = 240, ncas_f must be less than 15)
  double * d_buf = dd->d_buf1; 
  double * d_eri_unpacked = dd->d_buf2; 
  
  double * d_eri = nullptr;
  
  if(use_eri_cache) {
    //    d_eri = dd_fetch_eri(dd, eri, naux, nao_pair, addr_dfobj, count);
    d_eri = dd_fetch_eri(dd, nullptr, naux, nao_pair, addr_dfobj, count);
  } else {
    grow_array(dd->d_eri1, _size_eri, dd->size_eri1, "eri1", FLERR);
    d_eri = dd->d_eri1;
  }
  
  int * my_d_tril_map_ptr = dd_fetch_pumap(dd, nao, _PUMAP_2D_UNPACK);

  getjk_unpack_buf2(d_eri_unpacked, d_eri, my_d_tril_map_ptr, naux, nao, nao_pair);
  
  //bufpp = mo.T @ eri @ mo
  //buf = np.einsum('ijk,kl->ijl',eri_unpacked,mo_coeff),i=naux,j=nao,l=nao
  
  const double alpha = 1.0;
  const double beta = 0.0;
  const int nao2 = nao * nao;
  const int zero = 0;
  
  ml->set_handle();
  ml->gemm_batch((char *) "N", (char *) "N", &nao, &nao, &nao,
		 &alpha, d_eri_unpacked, &nao, &nao2, dd->d_mo_coeff, &nao, &zero, &beta, d_buf, &nao, &nao2, &naux);
  
  //bufpp = np.einsum('jk,ikl->ijl',mo_coeff.T,buf),i=naux,j=nao,l=nao
  
  double * d_bufpp = dd->d_buf2;

  ml->gemm_batch((char *) "T", (char *) "N", &nao, &nao, &nao,
		 &alpha, dd->d_mo_coeff, &nao, &zero, d_buf, &nao, &nao2, &beta, d_bufpp, &nao, &nao2, &naux);

  int _size_bufpa = naux*nmo*ncas;
  grow_array(dd->d_bufpa, _size_bufpa, dd->size_bufpa, "bufpa", FLERR);
  
  double * d_bufpa = dd->d_bufpa;

  get_bufpa(d_bufpp, d_bufpa, naux, nmo, ncore, ncas);

  // making papa on device, so no longer need to pull
  //double * bufpa = &(pin_bufpa[count*blksize*nmo*ncas]);
  //pm->dev_pull_async(d_bufpa, bufpa, naux*nmo*ncas*sizeof(double));

  double * d_fxpp = dd->d_buf1;
  
  // fxpp[str(k)] =bufpp.transpose(1,2,0);

  transpose_120(d_bufpp, d_fxpp, naux, nmo, nmo);

// calculate j_pc
  
  // k_cp += numpy.einsum('kij,kij->ij', bufpp[:,:ncore], bufpp[:,:ncore])

  int one = 1;
  int nmo_ncore = nmo * ncore;
  double beta_ = (count < num_devices) ? 0.0 : 1.0;
  
  ml->gemm_batch((char *) "N", (char *) "T", &one, &one, &naux,
		 &alpha, d_fxpp, &one, &naux, d_fxpp, &one, &naux, &beta_, dd->d_k_pc, &one, &one, &nmo_ncore);

  //bufd work

  int _size_bufd = naux*nmo;
  grow_array(dd->d_bufd, _size_bufd, dd->size_bufd, "bufd", FLERR);
  
  double * d_bufd = dd->d_bufd;

  get_bufd(d_bufpp, d_bufd, naux, nmo);
  
// calculate j_pc
  
  // self.j_pc += numpy.einsum('ki,kj->ij', bufd, bufd[:,:ncore])

  ml->gemm((char *) "N", (char *) "T", &ncore, &nmo, &naux,
	   &alpha, d_bufd, &nmo, d_bufd, &nmo, &beta_, dd->d_j_pc, &ncore);

  int _size_bufaa = naux*ncas*ncas;
  grow_array(dd->d_bufaa, _size_bufaa, dd->size_bufaa, "bufaa", FLERR);

  double * d_bufaa = dd->d_bufaa;

  get_bufaa(d_bufpp, d_bufaa, naux, nmo, ncore, ncas);

  const int ncas2 = ncas*ncas;
  const int nmo_ncas = nmo*ncas;

  // calculate ppaa
  dd->d_ppaa = dd->d_buf3;
  ml->gemm ((char *) "N", (char *) "N", &ncas2, &nao2, &naux,  
                   &alpha,  d_bufaa, &ncas2, d_fxpp, &naux, &beta_, dd->d_ppaa, &ncas2);                  
  
  // calculate papa
  //dd->d_papa = dd->d_buf3 + _size_ppaa*sizeof(double);
  dd->d_papa = dd->d_buf3 + _size_ppaa;
  ml->gemm ((char *) "N", (char *) "T", &nmo_ncas, &nmo_ncas, &naux, 
                  &alpha, d_bufpa, &nmo_ncas, d_bufpa, &nmo_ncas, &beta_, dd->d_papa, &nmo_ncas); 
#ifdef _DEBUG_DEVICE
#if defined (_GPU_CUDA)
  printf("LIBGPU :: Leaving Device::df_ao2mo_pass1_fdrv()\n"); 
  cudaMemGetInfo(&freeMem, &totalMem);
  printf("Ending ao2mo fdrv Free memory %lu bytes, total memory %lu bytes\n",freeMem,totalMem);
#endif
#endif
  
  pm->dev_profile_stop();
  
  double t1 = omp_get_wtime();
  t_array[9] += t1 - t0;
  // counts in pull ppaa
}

/* ---------------------------------------------------------------------- */
