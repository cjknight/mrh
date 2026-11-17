/* -*- c++ -*- */

#if defined(_GPU_CUDA)

#include "../../device/device_dfgrad.h"

#include <stdio.h>

#define _DFGRAD_BLOCK_SIZE 256
#define _DFGRAD_MAX_BLOCKS 1048576

/* ---------------------------------------------------------------------- */

__global__ void _dfgrad_pack_tril(double * out, const double * S, int nao, int nP)
{
  const size_t n = (size_t) nP * nao * nao;
  const size_t npair = (size_t) nao * (nao + 1) / 2;
  const size_t step = (size_t) blockDim.x * gridDim.x;
  for(size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x; idx < n; idx += step) {
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

/* ---------------------------------------------------------------------- */

static unsigned int _dfgrad_nblocks(size_t n)
{
  size_t nblocks = (n + _DFGRAD_BLOCK_SIZE - 1) / _DFGRAD_BLOCK_SIZE;
  if(nblocks > _DFGRAD_MAX_BLOCKS) nblocks = _DFGRAD_MAX_BLOCKS; // grid-stride loop covers the rest
  if(nblocks == 0) nblocks = 1;
  return (unsigned int) nblocks;
}

void DeviceDFGrad::dfgrad_pack_tril(double * out, const double * S, int nao, int nP)
{
  dim3 block_size(_DFGRAD_BLOCK_SIZE, 1, 1);
  dim3 grid_size(_dfgrad_nblocks((size_t) nP * nao * nao), 1, 1);

  cudaStream_t s = *(ctx.pm->dev_get_queue());

  _dfgrad_pack_tril<<<grid_size, block_size, 0, s>>>(out, S, nao, nP);

#ifdef _DEBUG_DEVICE
  printf("LIBGPU ::  -- dfgrad::pack_tril :: nao= %i  nP= %i\n", nao, nP);
  ctx.pm->dev_check_errors();
#endif
}

#endif
