/* -*- c++ -*- */

#if defined(_GPU_SYCL) || defined(_GPU_SYCL_CUDA)

#include "../../device/device_dfgrad.h"

#include <stdio.h>

#define _DFGRAD_BLOCK_SIZE 256
#define _DFGRAD_MAX_GROUPS 1048576

/* ---------------------------------------------------------------------- */

static size_t _dfgrad_global(size_t n)
{
  size_t ngroups = (n + _DFGRAD_BLOCK_SIZE - 1) / _DFGRAD_BLOCK_SIZE;
  if(ngroups > _DFGRAD_MAX_GROUPS) ngroups = _DFGRAD_MAX_GROUPS; // grid-stride loop covers the rest
  if(ngroups == 0) ngroups = 1;
  return ngroups * _DFGRAD_BLOCK_SIZE;
}

void DeviceDFGrad::dfgrad_permute(double * out, const double * in, int nb, int naux, int nocc)
{
  sycl::queue * s = ctx.pm->dev_get_queue();

  const size_t n = (size_t) nb * nocc * naux;
  const size_t global = _dfgrad_global(n);

  s->parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(_DFGRAD_BLOCK_SIZE)),
		  [=](sycl::nd_item<1> item) {
    for(size_t idx = item.get_global_id(0); idx < n; idx += global) {
      const size_t b = idx % nb;
      const size_t t = idx / nb;
      const size_t o = t % nocc;
      const size_t P = t / nocc;
      out[idx] = in[b + nb * (P + (size_t) naux * o)];
    }
  });

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- dfgrad::permute :: nb= %i  naux= %i  nocc= %i\n", nb, naux, nocc);
  ctx.pm->dev_check_errors();
#endif
}

/* ---------------------------------------------------------------------- */

void DeviceDFGrad::dfgrad_pack_tril(double * out, const double * S, int nao, int nP)
{
  sycl::queue * s = ctx.pm->dev_get_queue();

  const size_t n = (size_t) nP * nao * nao;
  const size_t npair = (size_t) nao * (nao + 1) / 2;
  const size_t global = _dfgrad_global(n);

  s->parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(_DFGRAD_BLOCK_SIZE)),
		  [=](sycl::nd_item<1> item) {
    for(size_t idx = item.get_global_id(0); idx < n; idx += global) {
      const size_t j = idx % nao;
      const size_t t = idx / nao;
      const size_t i = t % nao;
      const size_t P = t / nao;
      if(j <= i) {
        const double s = S[idx]; // S_P[j + nao*i]
        out[i * (i + 1) / 2 + j + npair * P] = (j == i) ? 0.5 * s : s;
      }
    }
  });

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- dfgrad::pack_tril :: nao= %i  nP= %i\n", nao, nP);
  ctx.pm->dev_check_errors();
#endif
}

#endif
