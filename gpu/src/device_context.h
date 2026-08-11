/* -*- c++ -*- */

#ifndef DEVICE_CONTEXT_H
#define DEVICE_CONTEXT_H

#include <vector>

#include "pm/pm.h"
#include "mathlib/mathlib.h"

struct DeviceJkData {
  int size_rho, size_vj, size_vk, size_buf1, size_buf2, size_buf3;
  int size_dms, size_dmtril, size_eri1;
  double *d_rho, *d_vj, *d_buf1, *d_buf2, *d_buf3, *d_vkk;
  double *d_dms, *d_dmtril, *d_eri1;
};

struct DeviceAo2moData {
  int size_j_pc, size_k_cp, size_k_pc, size_bufd, size_bufpa, size_bufaa;
  double *d_j_pc, *d_k_pc, *d_bufd, *d_bufpa, *d_bufaa;
  double *d_ppaa, *d_papa;
};

struct DeviceH2effData {
  int size_eri_unpacked, size_eri_h2eff;
  double *d_eri_h2eff;
};

struct DeviceImphamData {
};

struct DevicePdftData {
  int size_mo_grid, size_ao_grid, size_cascm2, size_Pi, size_buf_pdft;
  double *d_ao_grid, *d_cascm2, *d_mo_grid, *d_Pi, *d_buf_pdft1, *d_buf_pdft2;
};

struct DeviceFciData {
  int size_clinka, size_clinkb, size_cibra, size_ciket;
  int size_tdm1, size_tdm2, size_tdm2_p, size_pdm1, size_pdm2;
  int *d_clinka, *d_clinkb;
  double *d_cibra, *d_ciket;
  double *d_tdm1, *d_tdm2, *d_tdm2_p;
  double *d_tdm1h, *d_tdm3ha, *d_tdm3hb;
  double *d_pdm1, *d_pdm2;
  std::vector<int> type_pumap, size_pumap;
  std::vector<int *> pumap, d_pumap;
  int *d_pumap_ptr; // no explicit allocation
  int *pumap_ptr; // no explicit allocation
};

struct DeviceLassiData {
};

struct my_device_data {
  int device_id;
  int active; // was device used in calculation and has result to be accumulated?

  // Shared
  int size_mo_coeff;
  int size_mo_cas;
  int size_ucas;
  int size_umat;
  int size_h2eff;
  double * d_mo_coeff;
  double * d_mo_cas;
  double * d_ucas;
  double * d_umat;
  double * d_h2eff;

  // Per-domain data
  DeviceJkData jk;
  DeviceAo2moData ao2mo;
  DeviceH2effData h2eff;
  DeviceImphamData impham;
  DevicePdftData pdft;
  DeviceFciData fci;
  DeviceLassiData lassi;

  // we keep the following for now, but we don't explicitly use them anymore
  // besides, pm.h should defined a queue_t and mathlib.h a handle_t...

#if defined (_USE_GPU)
#if defined _GPU_CUBLAS
  cublasHandle_t handle;
  cudaStream_t stream;
#elif defined _GPU_HIPBLAS
  hipblasHandle_t handle;
  hipStream_t stream;
#elif defined _GPU_MKL
  int * handle;
  sycl::queue * stream;
#endif
#else
  int * handle;
  int * stream;
#endif
};

#endif
