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

void Device::update_h2eff_sub(int ncore, int ncas, int nocc, int nmo,
                              py::array_t<double> _umat, py::array_t<double> _h2eff_sub)
{
  double t0 = omp_get_wtime();

#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device :: Starting update_h2eff_sub function\n");
#endif

  pm->dev_profile_start("Setup initial h2eff_sub");
  
  py::buffer_info info_umat = _umat.request(); // 2d array nmo*nmo
  py::buffer_info info_h2eff_sub = _h2eff_sub.request();// 2d array (nmo * ncas) x (ncas*(ncas+1)/2)

  const int device_id = 0; //count % num_devices;

  pm->dev_set_device(device_id);

  my_device_data * dd = &(device_data[device_id]);

  const int ncas_pair = ncas * (ncas+1)/2;
  
  double * umat = static_cast<double*>(info_umat.ptr);
  double * h2eff_sub = static_cast<double*>(info_h2eff_sub.ptr);

#ifdef _DEBUG_DEVICE
#if defined (_GPU_CUDA)
  size_t freeMem;size_t totalMem;
  freeMem=0;totalMem=0;
  cudaMemGetInfo(&freeMem, &totalMem);
  printf("Starting h2eff_update Free memory %lu bytes, total memory %lu bytes\n",freeMem,totalMem);
#endif
#endif
  
  int _size_h2eff_unpacked = nmo*ncas*ncas*ncas;
  int _size_h2eff_packed = nmo*ncas*ncas_pair;

  grow_array(dd->jk.d_buf1, _size_h2eff_unpacked, dd->jk.size_buf1, "buf1", FLERR);
  //  grow_array(dd->jk.d_buf2, _size_h2eff_unpacked, dd->jk.size_buf2, "buf2", FLERR);
  //  grow_array(dd->jk.d_buf3, _size_h2eff_unpacked, dd->jk.size_buf3, "buf3", FLERR);
  
  double * d_h2eff_unpacked = dd->jk.d_buf1;

  grow_array(dd->d_ucas, ncas*ncas, dd->size_ucas, "ucas", FLERR);

  grow_array(dd->d_umat, nmo*nmo, dd->size_umat, "umat", FLERR);
  
  pm->dev_push_async(dd->d_umat, umat, nmo*nmo*sizeof(double));

#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- Setup update function\n");
#endif
  
  pm->dev_profile_next("extraction");
  
  //ucas = umat[ncore:nocc, ncore:nocc]

  extract_submatrix(dd->d_umat, dd->d_ucas, ncas, ncore, nmo);
  
  //h2eff_sub = h2eff_sub.reshape (nmo*ncas, ncas*(ncas+1)//2)
  //h2eff_sub = lib.numpy_helper.unpack_tril (h2eff_sub)
  //h2eff_sub = h2eff_sub.reshape (nmo, ncas, ncas, ncas)

  grow_array(dd->d_h2eff, _size_h2eff_packed, dd->size_h2eff, "h2eff", FLERR);
  
  double * d_h2eff_sub = dd->d_h2eff;
  
  pm->dev_push_async(d_h2eff_sub, h2eff_sub, _size_h2eff_packed * sizeof(double));

  pm->dev_profile_next("map creation and pushed");
  
  int * d_my_unpack_map_ptr = dd_fetch_pumap(dd, ncas, _PUMAP_H2EFF_UNPACK);

  pm->dev_profile_next("unpacking");

#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- created and pushed unpacking map\n");
#endif

  unpack_h2eff_2d(d_h2eff_sub, d_h2eff_unpacked, d_my_unpack_map_ptr, nmo, ncas, ncas_pair);
  
  pm->dev_profile_next("2 dgemms");
  
#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- unpacked h2eff_sub \n");
#endif
  
  //1. h2eff_sub = np.tensordot (ucas, h2eff_sub, axes=((0),(1))) # bpaa
  //2. h2eff_sub = np.tensordot (umat, h2eff_sub, axes=((0),(1))) # qbaa
  //3. h2eff_sub = np.tensordot (h2eff_sub, ucas, axes=((2),(0))) # qbab
  //4. h2eff_sub = np.tensordot (h2eff_sub, ucas, axes=((2),(0))) # qbbb
  // doing 3,4,tranpose, 1,2, tranpose
  
  const double alpha=1.0;
  const double beta=0.0;
  
  //h2eff_step1=([pi]jk,jJ->[pi]kJ)

  double * d_h2eff_step1 = dd->jk.d_buf2;

  int zero = 0;
  int ncas2 = ncas * ncas;
  int ncas_nmo = ncas * nmo;
  
  ml->set_handle();
  ml->gemm_batch((char *) "N", (char *) "N", &ncas, &ncas, &ncas,
		 &alpha, dd->d_ucas, &ncas, &zero, d_h2eff_unpacked, &ncas, &ncas2, &beta, d_h2eff_step1, &ncas, &ncas2, &ncas_nmo);
  
  //h2eff_step2=([pi]kJ,kK->[pi]JK
  
  double * d_h2eff_step2 = dd->jk.d_buf1;

  ml->gemm_batch((char *) "N", (char *) "T", &ncas, &ncas, &ncas,
		 &alpha, d_h2eff_step1, &ncas, &ncas2, dd->d_ucas, &ncas, &zero, &beta, d_h2eff_step2, &ncas, &ncas2, &ncas_nmo);
  
  pm->dev_profile_next("transpose");
  
#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- Finished first 2 cublasDgemmStridedBatched Functions \n");
#endif
  
  //h2eff_tranposed=(piJK->JKpi)
  
  double * d_h2eff_transposed = dd->jk.d_buf2;

  transpose_2310(d_h2eff_step2, d_h2eff_transposed, nmo, ncas);
  
  pm->dev_profile_next("last 2 dgemm");
  
#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- Finished transposing\n");
#endif
  
  double * d_h2eff_step3 = dd->jk.d_buf1;

  //h2eff_sub=np.einsum('iI,JKip->JKIp',ucas,h2eff_sub) h2eff=ncas,ncas,ncas,nmo; ucas=ncas,ncas

  ml->gemm_batch((char *) "N", (char *) "T", &nmo, &ncas, &ncas,
		 &alpha, d_h2eff_transposed, &nmo, &ncas_nmo, dd->d_ucas, &ncas, &zero, &beta, d_h2eff_step3, &nmo, &ncas_nmo, &ncas2);
  
  //h2eff_step4=([JK]Ip,pP->[JK]IP)

  double * d_h2eff_step4 = dd->jk.d_buf2;

  ml->gemm_batch((char *) "N", (char *) "N", &nmo, &ncas, &nmo,
		 &alpha, dd->d_umat, &nmo, &zero, d_h2eff_step3, &nmo, &ncas_nmo, &beta, d_h2eff_step4, &nmo, &ncas_nmo, &ncas2);
  
  pm->dev_profile_next("2nd transpose");

#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- Finished last 2 cublasDgemmStridedBatched Functions \n");
#endif

  double * d_h2eff_transpose2 = dd->jk.d_buf1;
  
  //h2eff_tranposed=(JKIP->PIJK) 3201

  transpose_3210(d_h2eff_step4, d_h2eff_transpose2, nmo, ncas);
  
#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- finished transposing back\n");
#endif
  
  //ix_i, ix_j = np.tril_indices (ncas)
  //h2eff_sub = h2eff_sub.reshape (nmo, ncas, ncas*ncas)
  //h2eff_sub = h2eff_sub[:,:,(ix_i*ncas)+ix_j]
  //h2eff_sub = h2eff_sub.reshape (nmo, -1)

  pm->dev_profile_next("second map and packing");
  
  int * d_my_pack_map_ptr = dd_fetch_pumap(dd, ncas, _PUMAP_H2EFF_PACK);

  pack_h2eff_2d(d_h2eff_transpose2, d_h2eff_sub, d_my_pack_map_ptr, nmo, ncas, ncas_pair);
  
#ifdef _DEBUG_H2EFF
  printf("LIBGPU :: Inside Device :: -- Freed map\n");
#endif
  
  pm->dev_pull_async(d_h2eff_sub, h2eff_sub, _size_h2eff_packed*sizeof(double));

  pm->dev_stream_wait(); // is this required or can we delay waiting?
  
  pm->dev_profile_stop();
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device :: Leaving update function\n");
#if defined (_GPU_CUDA)
  cudaMemGetInfo(&freeMem, &totalMem);
  
  printf("Ending h2eff_sub_update Free memory %lu bytes, total memory %lu bytes\n",freeMem,totalMem);
#endif
#endif
  
  double t1 = omp_get_wtime();
  t_array[5] += t1 - t0;
  count_array[3] += 1;
}

/* ---------------------------------------------------------------------- */
void Device::get_h2eff_df_v2(py::array_t<double> _cderi, 
                                int nao, int nmo, int ncas, int naux, int ncore, 
                                py::array_t<double> _eri, int count, size_t addr_dfobj) 
{
  double t0 = omp_get_wtime();

#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Inside Device::get_h2eff_df_v2()\n");
  printf("LIBGPU:: dfobj= %p count= %i combined= %lu %p update_dfobj= %i\n",(void*)(addr_dfobj), count, addr_dfobj+count, (void*)(addr_dfobj+count),update_dfobj);
#endif 
  pm->dev_profile_start("h2eff df setup");
  
  py::buffer_info info_eri = _eri.request(); //2D array nao * ncas * ncas_pair
  
  const int device_id = count % num_devices;
  
  pm->dev_set_device(device_id);
  
  my_device_data * dd = &(device_data[device_id]);

  dd->active = 1;
  //printf("nao: %i, nmo: %i, ncas: %i, naux: %i count %i\n",nao, nmo, ncas, naux, count);


  const int nao_pair = nao * (nao+1)/2;
  const int ncas_pair = ncas * (ncas+1)/2;
  const int _size_eri_h2eff = nmo*ncas*ncas_pair;
  const int _size_eri = naux*nao_pair;
  const int _size_mo_cas = nao*ncas;

  const int _size_eri_unpacked = naux * nao * nao;
  const int bump_buvp = naux * ncas * (ncas + nao);
  
  // buf2 will hold vuwm
  
  const int size_vuwm = ncas * ncas * ncas * nao;

  // buf1 will hold 1) cderi_unpacked 2) both bumP & buvP 3) vuwM

  const int size_cderi_unpacked = naux * nao * nao;
  
  const int _size_bPmu = naux*ncas*nao;
  const int _size_bPvu = naux*ncas*ncas;
  
  const int size_bumP_buvP = _size_bPmu + _size_bPvu;
  const int size_vuwM = nmo * ncas * ncas_pair;
  
  // int max_size_buf = (_size_eri_unpacked > _size_eri_h2eff) ? _size_eri_unpacked : _size_eri_h2eff;
  // if(size_vuwm > max_size_buf) max_size_buf = size_vuwm;
  // if(size_cderi_unpacked > max_size_buf) max_size_buf = size_cderi_unpacked;
  // if(size_bumP_buvP > max_size_buf) max_size_buf = size_bumP_buvP;
  // if(size_vuwM > max_size_buf) max_size_buf = size_vuwM;

  // if(device_id == 0)
  // printf("get_h2eff_df_v2 :: device_id= %i  naux= %i nmo= %i ncas= %i ncas_pair= %i nao= %i nao_pair= %i  _size_eri_unpacked= %i  _size_eri_h2eff= %i  size_vuwm= %i  size_cderi_unpacked= %i  size_bumP_buvP= %i\n",device_id,naux,nmo,ncas,ncas_pair,nao,nao_pair,_size_eri_unpacked, _size_eri_h2eff, size_vuwm, size_cderi_unpacked, size_bumP_buvP);

  int max_size_buf = size_cderi_unpacked;
  if(size_bumP_buvP > max_size_buf) max_size_buf = size_bumP_buvP;
  if(size_vuwM > max_size_buf) max_size_buf = size_vuwM; 
  if(size_vuwm > max_size_buf) max_size_buf = size_vuwm; 
  //printf("cderi_up: %i bump_buvP: %i vuWM:%i vuwm:%i size_buf%i\n",size_cderi_unpacked,size_bumP_buvP, size_vuwM, size_vuwm, max_size_buf);
  
  grow_array(dd->jk.d_buf1, max_size_buf, dd->jk.size_buf1, "buf1", FLERR); // holds cderi_unpacked and bumP+buvP and vuwM

  max_size_buf = size_bumP_buvP;
  if(size_vuwm > max_size_buf) max_size_buf = size_vuwm;
  
  grow_array(dd->jk.d_buf2, max_size_buf, dd->jk.size_buf2, "buf2", FLERR); // holds bPmu+bPvu and vuwm

  max_size_buf = _size_eri_h2eff;
  if(size_vuwM > max_size_buf) max_size_buf = size_vuwM;
  
  grow_array(dd->jk.d_buf3, max_size_buf, dd->jk.size_buf3, "buf3", FLERR); // holds eri_h2eff
  
  double * eri = static_cast<double*>(info_eri.ptr);
  double * d_mo_coeff = dd->d_mo_coeff;
  double * d_mo_cas = dd->d_mo_cas; 
 
  #if 0
    double * h_result = (double *)pm->dev_malloc_host(nao*ncas*sizeof(double)); 
    pm->dev_pull_async(d_mo_cas, h_result, nao*ncas*sizeof(double));
    pm->dev_barrier();
    printf("printing from mo_cas");
    for (int i=0; i<3; ++i){
      for (int j=0; j<3; ++j){
         printf("%f\t",h_result[i*nao + j]*1e7);}printf("\n");}printf("\n");
    pm->dev_free_host(h_result);

  #endif
  
  py::buffer_info info_cderi = _cderi.request(); // 2D array blksize * nao_pair
  double * cderi = static_cast<double*>(info_cderi.ptr);

  double * d_cderi = nullptr;
  
  if(use_eri_cache) {
    d_cderi = dd_fetch_eri(dd, cderi, naux, nao_pair, addr_dfobj, count);
  } else {
    grow_array(dd->jk.d_eri1, _size_eri, dd->jk.size_eri1, "eri1", FLERR);
    d_cderi = dd->jk.d_eri1;

    pm->dev_push_async(d_cderi, cderi, _size_eri * sizeof(double));
  }

  double * d_cderi_unpacked = dd->jk.d_buf1;

  int * d_my_unpack_map_ptr = dd_fetch_pumap(dd, nao, _PUMAP_2D_UNPACK);

  // CHRIS :: Start chunking w/r naux
  
  getjk_unpack_buf2(d_cderi_unpacked, d_cderi, d_my_unpack_map_ptr, naux, nao, nao_pair);
  
  #if 0 
    double * h_result = (double *)pm->dev_malloc_host(size_cderi_unpacked*sizeof(double)); 
    pm->dev_pull_async(d_cderi_unpacked, h_result, size_cderi_unpacked*sizeof(double));
    pm->dev_barrier();
    printf("printing from cderi unpacked results");
    for (int i=0; i<3; ++i){
      for (int j=0; j<3; ++j){
        for (int k=0; k<3; ++k){
         printf("%f\t",h_result[i*nao*nao + j*nao + k]*1e7);}printf("\n");}printf("\n");};
    pm->dev_free_host(h_result);
  #endif 


  //bPmu = np.einsum('Pmn,nu->Pmu',cderi,mo_cas)
  
  const double alpha = 1.0;
  const double beta = 0.0;
  int zero = 0;
  int nao2 = nao * nao;
  int ncas_nao = ncas * nao;
  int ncas2 = ncas * ncas;

  double * d_bPmu = dd->jk.d_buf2;
  
  ml->set_handle();
  ml->gemm_batch((char *) "N", (char *) "N", &nao, &ncas, &nao,
		 &alpha, d_cderi_unpacked, &nao, &nao2, d_mo_cas, &nao, &zero, &beta, d_bPmu, &nao, &ncas_nao, &naux);
  
  //bPvu = np.einsum('mv,Pmu->Pvu',mo_cas.conjugate(),bPmu)

  double * d_bPvu= dd->jk.d_buf2 + naux*ncas*nao;

  ml->set_handle();
  ml->gemm_batch((char *) "T", (char *) "N", &ncas, &ncas, &nao,
		 &alpha, d_mo_cas, &nao, &zero, d_bPmu, &nao, &ncas_nao, &beta, d_bPvu, &ncas, &ncas2, &naux);
  
  //eri = np.einsum('Pmw,Pvu->mwvu', bPmu, bPvu)

  //transpose bPmu

  double * d_bumP = dd->jk.d_buf1;

  transpose_120(d_bPmu, d_bumP, naux, ncas, nao, 1); // this call distributes work items differently 

  double * d_buvP = dd->jk.d_buf1 + naux*ncas*nao;

  // printf("size_buf1= %i  size_bumP= %i  size_buvP= %i  sum= %i\n",
  // 	 dd->size_buf, naux*ncas*nao, naux*ncas*ncas, naux*ncas*nao + naux*ncas*ncas);
  
  //transpose bPvu

  transpose_210(d_bPvu, d_buvP, naux, ncas, ncas);

  // printf("size_buf2= %i  _size_mwvu= %i\n",dd->size_buf, size_vuwm);
  
  double * d_vuwm = dd->jk.d_buf2; 

  ml->set_handle();
  ml->gemm((char *) "T", (char *) "N", &ncas_nao, &ncas2, &naux,
	   &alpha, d_bumP, &naux, d_buvP, &naux, &beta, d_vuwm, &ncas_nao);

  // CHRIS :: Stop chunking w/r naux
  
  double * d_vuwM = dd->jk.d_buf1;

  ml->set_handle();
  ml->gemm_batch((char *) "T", (char *) "T", &ncas, &nao, &nao,
		 &alpha, d_vuwm, &nao, &ncas_nao, d_mo_coeff, &nao, &zero, &beta, d_vuwM, &ncas, &ncas_nao, &ncas2);

  int * my_d_tril_map_ptr = dd_fetch_pumap(dd, ncas, _PUMAP_2D_UNPACK);
  
  if (count < num_devices) {
    pack_d_vuwM(d_vuwM, dd->h2eff.d_eri_h2eff, my_d_tril_map_ptr, nmo, ncas, ncas_pair);
  } else {
    pack_d_vuwM(d_vuwM, dd->jk.d_buf3, my_d_tril_map_ptr, nmo, ncas, ncas_pair);
    vecadd(dd->jk.d_buf3, dd->h2eff.d_eri_h2eff, _size_eri_h2eff);
  }

  #if 0
    double * h_result = (double *)pm->dev_malloc_host(size_vuwM*sizeof(double)); 
    if (count < num_devices) {
       pm->dev_pull_async(dd->h2eff.d_eri_h2eff, h_result, size_vuwM*sizeof(double));}
    else{
       pm->dev_pull_async(dd->jk.d_buf3, h_result, size_vuwM*sizeof(double));}
    pm->dev_barrier();
    printf("printing from h2eff results");
    for (int i=0; i<3; ++i){
      for (int j=0; j<3; ++j){
         printf("%f\t",h_result[i*ncas*ncas_pair + j]*1e5);}printf("\n");}printf("\n");
    pm->dev_free_host(h_result);
  #endif 

  pm->dev_profile_stop();
  
  double t1 = omp_get_wtime();
  t_array[6] += t1 - t0;//TODO: add the array size
  count_array[4] += 1; // see v1 comment
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Leaving Device::get_h2eff_df_v2()\n");
#endif  
}

/* ---------------------------------------------------------------------- */

#if defined(_ENABLE_P2P)
void Device::pull_eri_h2eff(py::array_t<double> _eri, int nmo, int ncas)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Inside Device::pull_eri_h2eff()\n");
#endif
  
  py::buffer_info info_eri = _eri.request(); //2D array (nmo * (ncas*ncas_pair))
  double * eri = static_cast<double*>(info_eri.ptr);

  const int ncas_pair = ncas*(ncas+1)/2;
  
  const int N = nmo*ncas * ncas_pair;

  std::vector<double *> e_vec(num_devices);
  std::vector<double *> buf_vec(num_devices);
  std::vector<int> active(num_devices);
  
  for(int i=0; i<num_devices; ++i) {
    my_device_data * dd = &(device_data[i]);
    e_vec[i] = dd->h2eff.d_eri_h2eff;
    buf_vec[i] = dd->jk.d_buf3;
    active[i] = dd->active;
  }

  mgpu_reduce(e_vec, buf_eri_h2eff, N, true, buf_vec, active);

#pragma omp parallel for
  for(int i=0; i<N; ++i) eri[i] = buf_eri_h2eff[i];
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Leaving Device::get_h2eff_df_v2()\n");
#endif
}
#else
void Device::pull_eri_h2eff(py::array_t<double> _eri, int nmo, int ncas)
{
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: -- Inside Device::pull_eri_h2eff()\n");
#endif
  
  py::buffer_info info_eri = _eri.request(); //2D array (nmo * (ncas*ncas_pair))
  double * eri = static_cast<double*>(info_eri.ptr);
  double * tmp;

  const int ncas_pair = ncas*(ncas+1)/2;
  const int size_eri_h2eff = nmo*ncas*ncas_pair;
  
  // Pulling eri from all devices
  
  for (int i=0; i<num_devices; ++i){
    pm->dev_set_device(i);

    my_device_data * dd = &(device_data[i]);

    tmp = &(buf_eri_h2eff[i*size_eri_h2eff]);
    
    if(dd->active) pm->dev_pull_async(dd->h2eff.d_eri_h2eff, tmp, size_eri_h2eff*sizeof(double));
  }
  
  // Adding eri from all devices
  
  for(int i=0; i<num_devices; ++i) {
    pm->dev_set_device(i);

    my_device_data * dd = &(device_data[i]);

    pm->dev_stream_wait();

    if(i > 0 && dd->active) {

      tmp = &(buf_eri_h2eff[i * size_eri_h2eff]);
//#pragma omp parallel for
      for(int j=0; j< size_eri_h2eff; ++j) buf_eri_h2eff[j] += tmp[j];
    }
  }
  
  std::memcpy(eri, buf_eri_h2eff, size_eri_h2eff*sizeof(double));
  
#ifdef _DEBUG_DEVICE
  printf("LIBGPU :: Leaving Device::get_h2eff_df_v2()\n");
#endif
}
#endif
