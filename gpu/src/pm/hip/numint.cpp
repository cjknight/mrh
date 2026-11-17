/* -*- c++ -*- */

#if defined(_GPU_HIP)

#include "../../device/device_numint.h"

#include <stdio.h>

#define _NUMINT_BLOCK_SIZE 256
#define _TILE(A,B) (A + B - 1) / B

/* ---------------------------------------------------------------------- */

// one thread per grid point; c[k] is column-major (ngrid x nocc)
__global__ void _numint_contract_rho(double * rho, const double * c, int ngrid, int nocc, int nrow)
{
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if(g >= ngrid) return;

  const size_t stride = (size_t) ngrid * nocc; // between AO components
  const double * c0 = c + g;                    // c0[o*ngrid]: grid index fastest

  double r0 = 0.0;
  for(int o=0; o<nocc; ++o) r0 += c0[(size_t) o * ngrid] * c0[(size_t) o * ngrid];
  rho[g] = r0;

  if(nrow < 4) return;

  double tau = 0.0;
  for(int d=1; d<4; ++d) {
    const double * cd = c0 + d * stride;
    double rd = 0.0, td = 0.0;
    for(int o=0; o<nocc; ++o) {
      const double a = c0[(size_t) o * ngrid];
      const double e = cd[(size_t) o * ngrid];
      rd += a * e;
      td += e * e;
    }
    rho[(size_t) d * ngrid + g] = 2.0 * rd;
    tau += td;
  }

  if(nrow == 5) rho[(size_t) 4 * ngrid + g] = 0.5 * tau;
}

/* ---------------------------------------------------------------------- */

// grid-stride loop over the ngrid*nao elements of out
__global__ void _numint_scale_ao(double * out, const double * ao, const double * wv, int ngrid, int nao, int ncomp)
{
  const size_t n = (size_t) ngrid * nao;
  const size_t step = (size_t) blockDim.x * gridDim.x;

  for(size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += step) {
    const size_t g = idx % ngrid; // idx = i*ngrid + g
    double val = ao[idx] * wv[g];
    for(int k=1; k<ncomp; ++k) val += ao[k * n + idx] * wv[k * (size_t) ngrid + g];
    out[idx] = val;
  }
}

/* ---------------------------------------------------------------------- */

void DeviceNumint::numint_contract_rho(double * rho, const double * c, int ngrid, int nocc, int nrow)
{
  dim3 block_size(_NUMINT_BLOCK_SIZE, 1, 1);
  dim3 grid_size(_TILE(ngrid, block_size.x), 1, 1);

  hipStream_t s = *(ctx.pm->dev_get_queue());

  _numint_contract_rho<<<grid_size, block_size, 0, s>>>(rho, c, ngrid, nocc, nrow);

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- numint::contract_rho :: ngrid= %i  nocc= %i  nrow= %i\n", ngrid, nocc, nrow);
  _HIP_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceNumint::numint_scale_ao(double * out, const double * ao, const double * wv, int ngrid, int nao, int ncomp)
{
  const size_t n = (size_t) ngrid * nao;
  size_t nblocks = (n + _NUMINT_BLOCK_SIZE - 1) / _NUMINT_BLOCK_SIZE;
  if(nblocks > 1048576) nblocks = 1048576; // grid-stride loop covers the rest

  dim3 block_size(_NUMINT_BLOCK_SIZE, 1, 1);
  dim3 grid_size((unsigned int) nblocks, 1, 1);

  hipStream_t s = *(ctx.pm->dev_get_queue());

  _numint_scale_ao<<<grid_size, block_size, 0, s>>>(out, ao, wv, ngrid, nao, ncomp);

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- numint::scale_ao :: ngrid= %i  nao= %i  ncomp= %i\n", ngrid, nao, ncomp);
  _HIP_CHECK_ERRORS();
#endif
}

/* ---------------------------------------------------------------------- */

// grid-stride loop over the ngrid*nao elements of out
__global__ void _numint_scale_ao_terms(double * out, DeviceNumint::Terms t, int ngrid, int nao)
{
  const size_t n = (size_t) ngrid * nao;
  const size_t step = (size_t) blockDim.x * gridDim.x;

  for(size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += step) {
    const size_t g = idx % ngrid; // idx = i*ngrid + g
    double val = t.ao[0][idx] * t.wv[0][g];
    for(int k=1; k<t.n; ++k) val += t.ao[k][idx] * t.wv[k][g];
    out[idx] = val;
  }
}

/* ---------------------------------------------------------------------- */

void DeviceNumint::numint_scale_ao_terms(double * out, Terms t, int ngrid, int nao)
{
  const size_t n = (size_t) ngrid * nao;
  size_t nblocks = (n + _NUMINT_BLOCK_SIZE - 1) / _NUMINT_BLOCK_SIZE;
  if(nblocks > 1048576) nblocks = 1048576; // grid-stride loop covers the rest

  dim3 block_size(_NUMINT_BLOCK_SIZE, 1, 1);
  dim3 grid_size((unsigned int) nblocks, 1, 1);

  hipStream_t s = *(ctx.pm->dev_get_queue());

  _numint_scale_ao_terms<<<grid_size, block_size, 0, s>>>(out, t, ngrid, nao);

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- numint::scale_ao_terms :: ngrid= %i  nao= %i  n= %i\n", ngrid, nao, t.n);
  _HIP_CHECK_ERRORS();
#endif
}

#endif
