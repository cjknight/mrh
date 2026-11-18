/* -*- c++ -*- */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <limits.h>

#include "device.h"

/* ---------------------------------------------------------------------- */

DeviceDFBuild::DeviceDFBuild(DeviceContext & _ctx)
  : ctx(_ctx), naoaux(0), naux(0), npair(0), metric_cd(0), h_buf(nullptr), size_h_buf(0)
{
  DFBuildBuffers empty;
  memset(&empty, 0, sizeof(DFBuildBuffers));
  bufs.assign(ctx.num_devices, empty);
}

/* ---------------------------------------------------------------------- */

void DeviceDFBuild::free_buffers(DFBuildBuffers & b)
{
  double ** ptrs[] = {&b.d_low, &b.d_j, &b.d_y};
  for(double ** p : ptrs) {
    if(*p) ctx.pm->dev_free(*p, "dfbuild");
    *p = nullptr;
  }
  memset(&b, 0, sizeof(DFBuildBuffers));
}

DeviceDFBuild::~DeviceDFBuild()
{
  for(int id=0; id<ctx.num_devices; ++id) {
    ctx.pm->dev_set_device(id);
    free_buffers(bufs[id]);
  }
  if(h_buf) ctx.pm->dev_free_host(h_buf);
}

/* ---------------------------------------------------------------------- */

// release device and pinned memory after the build (the blocks are large)
void DeviceDFBuild::free_dfbuild()
{
  for(int id=0; id<ctx.num_devices; ++id) {
    ctx.pm->dev_set_device(id);
    ctx.pm->dev_stream_wait();
    free_buffers(bufs[id]);
  }
  if(h_buf) ctx.pm->dev_free_host(h_buf);
  h_buf = nullptr;
  size_h_buf = 0;
}

/* ---------------------------------------------------------------------- */

DeviceDFBuild::DFBuildBuffers * DeviceDFBuild::select_device(int block)
{
  const int id = block % ctx.num_devices;
  ctx.pm->dev_set_device(id);
  return &(bufs[id]);
}

/* ---------------------------------------------------------------------- */

static double * as_ptr(py::array_t<double> & a) { return static_cast<double*>(a.request().ptr); }

static void check_size(py::array_t<double> & a, size_t n, const char * what)
{
  if((size_t) a.request().size != n) {
    printf("LIBGPU :: dfbuild %s has %zu elements; expected %zu\n", what, (size_t) a.request().size, n);
    exit(1);
  }
}

// device buffers are sized in int elements (grow_array)
static void check_int(size_t n, const char * what)
{
  if(n > (size_t) INT_MAX) {
    printf("LIBGPU :: dfbuild %s needs %zu elements; more than INT_MAX: use smaller blocks\n", what, n);
    exit(1);
  }
}

/* ---------------------------------------------------------------------- */

// low: CD: the Cholesky factor L (naoaux x naoaux, host C order); ED: pyscf's projection
// (naux x naoaux, host C order). Pushed to every device.
void DeviceDFBuild::init_dfbuild(py::array_t<double> _low, int _naoaux, int _naux, int _npair, int _metric_cd)
{
  double t0 = omp_get_wtime();

  naoaux = _naoaux;
  naux = _naux;
  npair = _npair;
  metric_cd = _metric_cd;

  if(metric_cd && naux != naoaux) {
    printf("LIBGPU :: init_dfbuild Cholesky metric with naux= %i != naoaux= %i\n", naux, naoaux);
    exit(1);
  }
  const size_t nlow = (size_t) naux * naoaux;
  check_size(_low, nlow, "low");
  check_int(nlow, "low");
  double * low = as_ptr(_low);

  for(int id=0; id<ctx.num_devices; ++id) {
    ctx.pm->dev_set_device(id);
    DFBuildBuffers & b = bufs[id];
    grow_array(ctx.pm, b.d_low, (int) nlow, b.size_low, "dfbuild_low", FLERR);
    ctx.pm->dev_push_async(b.d_low, low, nlow * sizeof(double));
  }
  for(int id=0; id<ctx.num_devices; ++id) {
    ctx.pm->dev_set_device(id);
    ctx.pm->dev_stream_wait(); // low is a Python temporary
  }

  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
}

/* ---------------------------------------------------------------------- */

// one block of pairs [p0, p0+nrow): ints is the Fortran (nrow, naoaux) libcint output (flattened);
// cderi is the host tensor (naux, npair), C order: rows P get cderi[P, p0:p0+nrow]
void DeviceDFBuild::compute_dfbuild(py::array_t<double> _ints, py::array_t<double> _cderi, int p0, int nrow, int block)
{
  double t0 = omp_get_wtime();

  const size_t nj = (size_t) nrow * naoaux;
  const size_t ny = (size_t) nrow * naux;
  check_size(_ints, nj, "ints");
  check_size(_cderi, (size_t) naux * npair, "cderi");
  check_int(nj, "ints block");
  check_int(ny, "cderi block");
  if(p0 < 0 || nrow < 0 || (size_t) p0 + nrow > (size_t) npair) {
    printf("LIBGPU :: compute_dfbuild pairs [%i, %i) outside [0, %i)\n", p0, p0 + nrow, npair);
    exit(1);
  }
  double * h_ints = as_ptr(_ints);
  double * cderi = as_ptr(_cderi);

  // Release the GIL for the push, the device work and the copy-back: DF.build computes the next
  // integral block in a prefetch thread (lib.map_with_prefetch), which needs the GIL to start its
  // libcint call. All Python objects are touched above; the arrays stay alive as arguments.
  py::gil_scoped_release nogil;

  DFBuildBuffers * b = select_device(block);

  grow_array(ctx.pm, b->d_j, (int) nj, b->size_j, "dfbuild_j", FLERR);
  ctx.pm->dev_push_async(b->d_j, h_ints, nj * sizeof(double));

  const double one = 1.0;
  const double zero = 0.0;
  double * d_y = b->d_j;
  if(metric_cd) {
    // Y L^T = J^T: L^T is the column-major view of the host (C order) L, upper triangular
    ctx.ml->set_handle();
    ctx.ml->trsm((char *) "R", (char *) "U", (char *) "N", (char *) "N", &nrow, &naux,
		 &one, b->d_low, &naoaux, b->d_j, &nrow);
  } else {
    // Y = J^T low^T: low^T is the column-major view of the host (C order) low; k = naoaux
    grow_array(ctx.pm, b->d_y, (int) ny, b->size_y, "dfbuild_y", FLERR);
    ctx.ml->set_handle();
    ctx.ml->gemm((char *) "N", (char *) "N", &nrow, &naux, &naoaux,
		 &one, b->d_j, &nrow, b->d_low, &naoaux,
		 &zero, b->d_y, &nrow);
    d_y = b->d_y;
  }

  if(ny > size_h_buf) {
    if(h_buf) ctx.pm->dev_free_host(h_buf);
    h_buf = (double *) ctx.pm->dev_malloc_host(ny * sizeof(double));
    size_h_buf = ny;
  }
  ctx.pm->dev_pull_async(d_y, h_buf, ny * sizeof(double));
  ctx.pm->dev_stream_wait(); // also the push of the Python temporary ints

  // Y column-major (nrow x naux): column P -> cderi[P, p0:p0+nrow]
#pragma omp parallel for
  for(int P=0; P<naux; ++P)
    memcpy(cderi + (size_t) P * npair + p0, h_buf + (size_t) P * nrow, (size_t) nrow * sizeof(double));

  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
}

/* ---------------------------------------------------------------------- */
/* Device facade forwarders                                               */
/* ---------------------------------------------------------------------- */

void Device::init_dfbuild(py::array_t<double> low, int naoaux, int naux, int npair, int metric_cd)
{ _dfbuild->init_dfbuild(low, naoaux, naux, npair, metric_cd); }

void Device::compute_dfbuild(py::array_t<double> ints, py::array_t<double> cderi, int p0, int nrow, int block)
{ _dfbuild->compute_dfbuild(ints, cderi, p0, nrow, block); }

void Device::free_dfbuild()
{ _dfbuild->free_dfbuild(); }
