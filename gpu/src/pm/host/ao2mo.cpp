/* -*- c++ -*- */

#if defined(_USE_CPU)

#include "../../device.h"

#include <stdio.h>

#define _RHO_BLOCK_SIZE 64
#define _DOT_BLOCK_SIZE 32
#define _CUDA_MAX_GRID_DIM_YZ 65535

/* ---------------------------------------------------------------------- */

/* ---------------------------------------------------------------------- */

void Device::get_bufpa(const double* bufpp, double* bufpa, int naux, int nmo, int ncore, int ncas)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<naux; ++i)
    for(int j=0; j<nmo; ++j)
      for(int k=0; k<ncas; ++k) {
        int inputIndex = (i*nmo + j)*nmo + k+ncore;
        int outputIndex = (i*nmo + j)*ncas + k;
        bufpa[outputIndex] = bufpp[inputIndex];
      }
}

/* ---------------------------------------------------------------------- */

void Device::get_bufaa(const double* bufpp, double* bufaa, int naux, int nmo, int ncore, int ncas)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<naux; ++i)
    for(int j=0; j<ncas; ++j)
      for(int k=0; k<ncas; ++k) {
        int inputIndex = (i*nmo + (j+ncore))*nmo + k+ncore;
        int outputIndex = (i*ncas + j)*ncas + k;
        bufaa[outputIndex] = bufpp[inputIndex];
      }
}

/* ---------------------------------------------------------------------- */

void Device::transpose_120(double * in, double * out, int naux, int nao, int ncas, int order)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<naux; ++i)
    for(int j=0; j<ncas; ++j)
      for(int k=0; k<nao; ++k) {
        int inputIndex = i*nao*ncas + j*nao + k;
        int outputIndex = j*nao*naux + k*naux + i;
        out[outputIndex] = in[inputIndex];
      }
}

/* ---------------------------------------------------------------------- */

void Device::get_bufd(const double* bufpp, double* bufd, int naux, int nmo)
{
#pragma omp parallel for collapse(2) schedule(static)
  for(int i=0; i<naux; ++i)
    for(int j=0; j<nmo; ++j)
      bufd[i*nmo + j] = bufpp[(i*nmo + j)*nmo + j];
}

/* ---------------------------------------------------------------------- */

void Device::transpose_210(double * in, double * out, int naux, int nao, int ncas)
{
#pragma omp parallel for collapse(3) schedule(static)
  for(int i=0; i<naux; ++i)
    for(int j=0; j<ncas; ++j)
      for(int k=0; k<nao; ++k) {
        int inputIndex = i*nao*ncas + j*nao + k;
        int outputIndex = k*ncas*naux + j*naux + i;
        out[outputIndex] = in[inputIndex];
      }
}


#endif
