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

void Device::init_mo_grid(int ngrid, int nmo)
{
  printf("starting init mo_grid\n");
  double t0 = omp_get_wtime();
  
  for(int id=0; id<num_devices; ++id) {
    pm->dev_set_device(id);

    my_device_data * dd = &(device_data[id]);

    int size_mo_grid = ngrid*nmo;

    grow_array(dd->pdft.d_mo_grid, size_mo_grid, dd->pdft.size_mo_grid, "mo_grid", FLERR);

    dd->active = 0;
  }
  
  double t1 = omp_get_wtime();
  
  //TODO:t_array[] += t1 - t0;
  // counts in pull Pi
}

/* ---------------------------------------------------------------------- */

void Device::push_ao_grid(py::array_t<double> _ao, int ngrid, int nao, int count)
{
  printf("starting push_mo_grid\n");
  double t0 = omp_get_wtime();
  
  py::buffer_info info_ao = _ao.request(); // 2D array (ngrid, nao)
  double * ao = static_cast<double*>(info_ao.ptr);
  
  int id = count%num_devices;

  pm->dev_set_device(id);

  my_device_data * dd = &(device_data[id]);

  int size_ao_grid = ngrid*nao;

  grow_array(dd->pdft.d_ao_grid, size_ao_grid, dd->pdft.size_ao_grid, "ao_grid", FLERR);
  
  pm->dev_push_async(dd->pdft.d_ao_grid, ao, size_ao_grid*sizeof(double));
  
  double t1 = omp_get_wtime();
  
  //TODO:t_array[] += t1 - t0;
  // counts in pull Pi
}

/* ---------------------------------------------------------------------- */

void Device::compute_mo_grid(int ngrid, int nao, int nmo)
{
  printf("starting compute\n");
  double t0 = omp_get_wtime();
  const int device_id =0;// count % num_devices;
  pm->dev_set_device(device_id);
  my_device_data * dd = &(device_data[device_id]);
  
  dd->active = 1;

  const double alpha = 1.0;
  const double beta = 0.0;
  #if 0
  double * h_mo_coeff = (double *)pm->dev_malloc_host(nao*nmo*sizeof(double));
  pm->dev_pull_async(dd->d_mo_coeff, h_mo_coeff, nao*nmo*sizeof(double));
  pm->dev_stream_wait();
  for (int i =0;i<nao;++i){for (int j=0;j<nmo;++j){printf("%f\t",h_mo_coeff[i*nmo+j]);}printf("\n");}
  #endif
  ml->set_handle();
  ml->gemm((char *) "N", (char *) "N", 
             &nmo, &ngrid, &nao,
             &alpha, 
             dd->d_mo_coeff, &nmo, 
             dd->pdft.d_ao_grid, &nao, 
             &beta, 
             dd->pdft.d_mo_grid, &nmo
             );
  double t1 = omp_get_wtime();  
  //TODO:t_array[] += t1 - t0;
  // counts in pull Pi
}

/* ---------------------------------------------------------------------- */

void Device::pull_mo_grid(py::array_t<double>_mo, int ngrid, int nmo)
{
double t0 = omp_get_wtime();

py::buffer_info info_mo = _mo.request(); // 2D array (ngrid, nao)
double * mo = static_cast<double*>(info_mo.ptr);

for(int id=0; id<1; ++id) {
  pm->dev_set_device(id);
  my_device_data * dd = &(device_data[id]);
  int size_mo_grid = ngrid*nmo;
  
  if(dd->active) {pm->dev_pull_async(dd->pdft.d_mo_grid, mo, size_mo_grid*sizeof(double));
    
  pm->dev_stream_wait();}
  pm->dev_barrier(); 
  
}
double t1 = omp_get_wtime();
}

/* ---------------------------------------------------------------------- */
void Device::push_cascm2 (py::array_t<double> _cascm2, int ncas) 
{
  double t0 = omp_get_wtime();
   
  py::buffer_info info_cascm2 = _cascm2.request(); // 4D array (ncas, ncas, ncas, ncas)
  double * cascm2 = static_cast<double*>(info_cascm2.ptr);

  for(int id=0; id<1; ++id) {
    pm->dev_set_device(id);
    my_device_data * dd = &(device_data[id]);
    
    int size_cascm2 = ncas*ncas*ncas*ncas;

    grow_array(dd->pdft.d_cascm2, size_cascm2, dd->pdft.size_cascm2, "cascm2", FLERR);

    pm->dev_push_async(dd->pdft.d_cascm2, cascm2, size_cascm2*sizeof(double));
  }
  
  double t1 = omp_get_wtime();
  
  //TODO:t_array[] += t1 - t0;
  // counts in pull Pi
}

/* ---------------------------------------------------------------------- */


void Device::init_Pi(int ngrid)
{
  double t0 = omp_get_wtime();
  
  for(int id=0; id<1; ++id) {
    pm->dev_set_device(id);

    my_device_data * dd = &(device_data[id]);

    int size_Pi = ngrid;

    grow_array(dd->pdft.d_Pi, size_Pi, dd->pdft.size_Pi, "Pi", FLERR);
  }
  
  double t1 = omp_get_wtime();
  
  //TODO:t_array[] += t1 - t0;
  // counts in pull Pi
}

/* ---------------------------------------------------------------------- */
void Device::compute_rho_to_Pi(py::array_t<double> _rho, int ngrid, int count)
{
  double t0 = omp_get_wtime();
  const int device_id = count % num_devices;
  pm->dev_set_device(device_id);
  my_device_data * dd = &(device_data[device_id]);
  
  py::buffer_info info_rho = _rho.request(); // 1D array (ngrid)
  double * cascm2 = static_cast<double*>(info_rho.ptr);
  grow_array(dd->jk.d_rho, ngrid, dd->jk.size_rho, "rho", FLERR);
  pm->dev_push_async(dd->jk.d_rho, rho, ngrid*sizeof(double));
  get_rho_to_Pi(dd->jk.d_rho, dd->pdft.d_Pi, ngrid);
}
/* ---------------------------------------------------------------------- */

void Device::compute_Pi (int ngrid, int ncas, int nao, int count) 
{
  double t0 = omp_get_wtime();
  const int device_id = count % num_devices;
  pm->dev_set_device(device_id);
  my_device_data * dd = &(device_data[device_id]);
  const double alpha = 1.0;
  const double beta = 0.0;
  const int one = 1; 
  ml->set_handle();
  
  int _size_buf_pdft = ngrid*ncas*ncas;

  int _size_orig = dd->pdft.size_buf_pdft; // because grow_array() updates dd->pdft.size_buf_pdft on first call

  grow_array(dd->pdft.d_buf_pdft1, _size_buf_pdft, dd->pdft.size_buf_pdft, "buf_pdft1", FLERR);
  grow_array(dd->pdft.d_buf_pdft2, _size_buf_pdft, _size_orig,        "buf_pdft2", FLERR);

  int ncas2 = ncas*ncas;
  //make mo_grid to ngrid*ncas*ncas (ai,aj->aij)
  double * d_mo_grid = dd->pdft.d_buf_pdft1;  //mo grid is only ngrid*ncas, using buf_pdft1 because efficient to not allot more
  ml->set_handle();
  ml->gemm((char *) "N", (char *) "N", 
             &ncas, &ngrid, &nao,
             &alpha, 
             dd->d_mo_coeff, &ncas, 
             dd->pdft.d_ao_grid, &nao, 
             &beta, 
             d_mo_grid, &ncas
             );

  // do buf1 = aij, ijkl->akl, mo, cascm2
  double * d_gridkern = dd->pdft.d_buf_pdft2; //trying to make it close to pyscf-forge mcpdft
  #if 0
  ml->gemm ((char *) "N", (char *) "N",
             &ngrid, &ncas2, &ncas2, 
             &alpha,
             dd->pdft.d_buf_pdft1, &ngrid,
             dd->pdft.d_cascm2, &ncas2,
             &beta, 
             dd->pdft.d_buf_pdft2, &ngrid);
             
  #else
  make_buf_pdft(d_gridkern, dd->pdft.d_buf_pdft1, dd->pdft.d_cascm2, ngrid, ncas);
  #endif
  // do Pi = (akl,akl->a, buf1, mo)/2
  #if 0
  const double half=0.5;
  ml->gemm_batch ((char *) "N",(char *) "T", 
             &one, &one, &ncas2,
             &half, 
             dd->pdft.d_buf_pdft1, &ncas2, &ncas2, 
             dd->pdft.d_buf_pdft2, &ncas2, &ncas2, 
             &beta, 
             dd->pdft.d_Pi, &one, &one, 
             &ngrid);
  #else
  make_Pi_final(d_gridkern, dd->pdft.d_buf_pdft1, dd->pdft.d_Pi, ngrid, ncas);
  #endif
             
}

/* ---------------------------------------------------------------------- */

void Device::pull_Pi (py::array_t<double> _Pi, int ngrid, int count)
{
  double t0 = omp_get_wtime();

  py::buffer_info info_Pi = _Pi.request(); //1D array (ngrid)
  double * Pi = static_cast<double*>(info_Pi.ptr);

  int device_id = count%num_devices;

  pm->dev_set_device(device_id);
  my_device_data * dd = &(device_data[device_id]);

  if (dd->pdft.d_Pi) pm->dev_pull_async(dd->pdft.d_Pi, Pi, ngrid*sizeof(double));
  
}
