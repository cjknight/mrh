/* -*- c++ -*- */

#include <stdio.h>

#include "device.h"

/* ---------------------------------------------------------------------- */

DeviceNumint::DeviceNumint(DeviceContext & _ctx)
  : ctx(_ctx), nao(0), nocc(0), nrow(0), ncomp(0), h_vmat(nullptr), size_h_vmat(0)
{
  NumintBuffers empty = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         0, 0, 0, 0, 0, 0, 0, 0, 0};
  bufs.assign(ctx.num_devices, empty);
}

/* ---------------------------------------------------------------------- */

DeviceNumint::~DeviceNumint()
{
  for(int id=0; id<ctx.num_devices; ++id) {
    ctx.pm->dev_set_device(id);
    NumintBuffers & b = bufs[id];
    if(b.d_cocc) ctx.pm->dev_free(b.d_cocc, "numint_cocc");
    if(b.d_ao) ctx.pm->dev_free(b.d_ao, "numint_ao");
    if(b.d_c) ctx.pm->dev_free(b.d_c, "numint_c");
    if(b.d_rho) ctx.pm->dev_free(b.d_rho, "numint_rho");
    if(b.d_wv) ctx.pm->dev_free(b.d_wv, "numint_wv");
    if(b.d_aow) ctx.pm->dev_free(b.d_aow, "numint_aow");
    if(b.d_vmat) ctx.pm->dev_free(b.d_vmat, "numint_vmat");
  }
  if(h_vmat) ctx.pm->dev_free_host(h_vmat);
}

/* ---------------------------------------------------------------------- */

void DeviceNumint::init_numint(py::array_t<double> _cocc, int _nao, int _nocc, int _nrow)
{
  double t0 = omp_get_wtime();

  if(_nrow != 1 && _nrow != 4 && _nrow != 5) {
    printf("LIBGPU :: init_numint() nrow= %i not supported (1 = LDA, 4 = GGA, 5 = meta-GGA)\n", _nrow);
    exit(1);
  }

  nao = _nao;
  nocc = _nocc;
  nrow = _nrow;
  ncomp = (nrow == 1) ? 1 : 4;

  py::buffer_info info_cocc = _cocc.request(); // 2D array (nao, nocc), C-contiguous
  double * cocc = static_cast<double*>(info_cocc.ptr);

  for(int id=0; id<ctx.num_devices; ++id) {
    ctx.pm->dev_set_device(id);
    NumintBuffers & b = bufs[id];

    grow_array(ctx.pm, b.d_cocc, nao*nocc, b.size_cocc, "numint_cocc", FLERR);
    grow_array(ctx.pm, b.d_vmat, nao*nao, b.size_vmat, "numint_vmat", FLERR);

    ctx.pm->dev_push_async(b.d_cocc, cocc, nao*nocc*sizeof(double));
    ctx.pm->dev_stream_wait(); // host array is a Python temporary

    b.active = 0;
    b.vmat_init = 0;
  }

  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
}

/* ---------------------------------------------------------------------- */

DeviceNumint::NumintBuffers * DeviceNumint::select_device(int block)
{
  const int id = block % ctx.num_devices;
  ctx.pm->dev_set_device(id);
  return &(bufs[id]);
}

/* ---------------------------------------------------------------------- */

void DeviceNumint::compute_numint_rho(py::array_t<double> _ao, py::array_t<double> _rho, int ngrid, int block)
{
  double t0 = omp_get_wtime();

  py::buffer_info info_ao = _ao.request();   // (ncomp, nao, ngrid), C-contiguous
  py::buffer_info info_rho = _rho.request(); // (nrow, ngrid), C-contiguous
  double * ao = static_cast<double*>(info_ao.ptr);
  double * rho = static_cast<double*>(info_rho.ptr);

  if((size_t) info_ao.size != (size_t) ncomp * ngrid * nao) {
    printf("LIBGPU :: compute_numint_rho() ao has %zu elements; expected ncomp*ngrid*nao= %i*%i*%i\n",
	   (size_t) info_ao.size, ncomp, ngrid, nao);
    exit(1);
  }

  NumintBuffers * b = select_device(block);

  grow_array(ctx.pm, b->d_ao, ncomp*ngrid*nao, b->size_ao, "numint_ao", FLERR);
  grow_array(ctx.pm, b->d_c, ncomp*ngrid*nocc, b->size_c, "numint_c", FLERR);
  grow_array(ctx.pm, b->d_rho, nrow*ngrid, b->size_rho, "numint_rho", FLERR);

  // the AO block is a Python temporary: wait for the copy before returning
  ctx.pm->dev_push_async(b->d_ao, ao, (size_t) ncomp*ngrid*nao*sizeof(double));
  ctx.pm->dev_stream_wait();

  // c[k](g,o) = sum_i ao[k](g,i) cocc[i,o]: column-major (ngrid x nao) . (nao x nocc), where the
  // C-contiguous cocc (nao, nocc) is the column-major (nocc x nao) matrix cocc^T
  {
    const double alpha = 1.0;
    const double beta = 0.0;

    for(int k=0; k<ncomp; ++k) {
      ctx.ml->set_handle();
      ctx.ml->gemm((char *) "N", (char *) "T", &ngrid, &nocc, &nao,
		   &alpha, b->d_ao + (size_t) k * nao * ngrid, &ngrid, b->d_cocc, &nocc,
		   &beta, b->d_c + (size_t) k * ngrid * nocc, &ngrid);
    }
  }

  numint_contract_rho(b->d_rho, b->d_c, ngrid, nocc, nrow);

  ctx.pm->dev_pull_async(b->d_rho, rho, nrow*ngrid*sizeof(double));
  ctx.pm->dev_stream_wait();

  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
}

/* ---------------------------------------------------------------------- */

void DeviceNumint::compute_numint_vmat(py::array_t<double> _wv, int ngrid, int block)
{
  double t0 = omp_get_wtime();

  py::buffer_info info_wv = _wv.request(); // (nrow, ngrid), C-contiguous
  double * wv = static_cast<double*>(info_wv.ptr);

  NumintBuffers * b = select_device(block);

  // the AO block of this device must be the one pushed by compute_numint_rho() for this block

  grow_array(ctx.pm, b->d_wv, nrow*ngrid, b->size_wv, "numint_wv", FLERR);
  grow_array(ctx.pm, b->d_aow, ngrid*nao, b->size_aow, "numint_aow", FLERR);

  ctx.pm->dev_push_async(b->d_wv, wv, nrow*ngrid*sizeof(double));
  ctx.pm->dev_stream_wait(); // wv is a Python temporary

  b->active = 1;

  // aow = sum_{k<ncomp} ao[k] * wv[k];  vmat (+)= ao0^T . aow, as pyscf (column-major ngrid x nao
  // operands; the row-major host array holds its transpose, which does not matter since the caller
  // forms vmat + vmat^T).

  numint_scale_ao(b->d_aow, b->d_ao, b->d_wv, ngrid, nao, ncomp);

  {
    const double alpha = 1.0;
    const double beta = b->vmat_init ? 1.0 : 0.0;

    ctx.ml->set_handle();
    ctx.ml->gemm((char *) "T", (char *) "N", &nao, &nao, &ngrid,
		 &alpha, b->d_ao, &ngrid, b->d_aow, &ngrid,
		 &beta, b->d_vmat, &nao);
  }

  b->vmat_init = 1;

  // meta-GGA tau term: v1 = sum_d (ao[d] * wv[4])^T ao[d] is symmetric, so adding v1/2 to vmat
  // gives vmat + vmat^T = (pyscf vmat + vmat^T) + v1. wv[4] is already halved by the caller.

  if(nrow == 5) {
    const double alpha = 0.5;
    const double beta = 1.0;
    const size_t offset_wv4 = (size_t) 4 * ngrid;

    for(int d=1; d<4; ++d) {
      double * d_aod = b->d_ao + (size_t) d * ngrid * nao;

      numint_scale_ao(b->d_aow, d_aod, b->d_wv + offset_wv4, ngrid, nao, 1);

      ctx.ml->set_handle();
      ctx.ml->gemm((char *) "T", (char *) "N", &nao, &nao, &ngrid,
		   &alpha, d_aod, &ngrid, b->d_aow, &ngrid,
		   &beta, b->d_vmat, &nao);
    }
  }

  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
}

/* ---------------------------------------------------------------------- */

void DeviceNumint::pull_numint_vmat(py::array_t<double> _vmat)
{
  double t0 = omp_get_wtime();

  py::buffer_info info_vmat = _vmat.request(); // (nao, nao); accumulated into
  double * vmat = static_cast<double*>(info_vmat.ptr);

  const int N = nao * nao;

  grow_array_host(ctx.pm, h_vmat, N, size_h_vmat, "h:numint_vmat");

  for(int id=0; id<ctx.num_devices; ++id) {
    NumintBuffers & b = bufs[id];
    if(!b.active) continue;

    ctx.pm->dev_set_device(id);
    ctx.pm->dev_pull_async(b.d_vmat, h_vmat, N*sizeof(double));
    ctx.pm->dev_stream_wait();

#pragma omp parallel for
    for(int i=0; i<N; ++i) vmat[i] += h_vmat[i];

    b.active = 0;
    b.vmat_init = 0;
  }

  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
}

/* ---------------------------------------------------------------------- */
/* Device facade forwarders                                               */
/* ---------------------------------------------------------------------- */

void Device::init_numint(py::array_t<double> _cocc, int nao, int nocc, int nrow)
{ _numint->init_numint(_cocc, nao, nocc, nrow); }

void Device::compute_numint_rho(py::array_t<double> _ao, py::array_t<double> _rho, int ngrid, int block)
{ _numint->compute_numint_rho(_ao, _rho, ngrid, block); }

void Device::compute_numint_vmat(py::array_t<double> _wv, int ngrid, int block)
{ _numint->compute_numint_vmat(_wv, ngrid, block); }

void Device::pull_numint_vmat(py::array_t<double> _vmat)
{ _numint->pull_numint_vmat(_vmat); }
