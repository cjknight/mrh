/* -*- c++ -*- */

#if defined(_GPU_SYCL) || defined(_GPU_SYCL_CUDA)

#include "../../device/device_numint.h"

#include <stdio.h>

#define _NUMINT_BLOCK_SIZE 256
#define _TILE(A,B) (A + B - 1) / B

/* ---------------------------------------------------------------------- */

void DeviceNumint::numint_contract_rho(double * rho, const double * c, int ngrid, int nocc, int nrow)
{
  sycl::queue * s = ctx.pm->dev_get_queue();

  const size_t local = _NUMINT_BLOCK_SIZE;
  const size_t global = (size_t) _TILE(ngrid, _NUMINT_BLOCK_SIZE) * local;

  // one work-item per grid point; c[k] is column-major (ngrid x nocc)
  s->parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
		  [=](sycl::nd_item<1> item) {
    const int g = (int) item.get_global_id(0);
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
  });

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- numint::contract_rho :: ngrid= %i  nocc= %i  nrow= %i\n", ngrid, nocc, nrow);
  ctx.pm->dev_check_errors();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceNumint::numint_scale_ao(double * out, const double * ao, const double * wv, int ngrid, int nao, int ncomp)
{
  sycl::queue * s = ctx.pm->dev_get_queue();

  const size_t n = (size_t) ngrid * nao;
  const size_t local = _NUMINT_BLOCK_SIZE;
  size_t ngroups = (n + local - 1) / local;
  if(ngroups > 1048576) ngroups = 1048576; // grid-stride loop covers the rest
  const size_t global = ngroups * local;

  s->parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
		  [=](sycl::nd_item<1> item) {
    for(size_t idx = item.get_global_id(0); idx < n; idx += global) {
      const size_t g = idx % ngrid; // idx = i*ngrid + g
      double val = ao[idx] * wv[g];
      for(int k=1; k<ncomp; ++k) val += ao[k * n + idx] * wv[k * (size_t) ngrid + g];
      out[idx] = val;
    }
  });

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- numint::scale_ao :: ngrid= %i  nao= %i  ncomp= %i\n", ngrid, nao, ncomp);
  ctx.pm->dev_check_errors();
#endif
}

#endif
