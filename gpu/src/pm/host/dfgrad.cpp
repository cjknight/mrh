/* -*- c++ -*- */

#if defined(_USE_CPU)

#include "../../device/device_dfgrad.h"

#include <stdio.h>

/* ---------------------------------------------------------------------- */

void DeviceDFGrad::dfgrad_pack_tril(double * out, const double * S, int nao, int nP)
{
  const size_t n = (size_t) nP * nao * nao;
  const size_t npair = (size_t) nao * (nao + 1) / 2;
#pragma omp parallel for schedule(static)
  for(size_t idx=0; idx<n; ++idx) {
    const size_t j = idx % nao;
    const size_t t = idx / nao;
    const size_t i = t % nao;
    const size_t P = t / nao;
    if(j <= i) {
      const double v = S[idx]; // S_P[j + nao*i]
      out[i * (i + 1) / 2 + j + npair * P] = (j == i) ? 0.5 * v : v;
    }
  }
}

#endif
