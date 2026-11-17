/* -*- c++ -*- */

#include <stdio.h>

#include "device.h"

/* ---------------------------------------------------------------------- */

DeviceNumint::DeviceNumint(DeviceContext & _ctx)
  : ctx(_ctx), nao(0), nocc(0), nrow(0), grad(0), ncomp(0), ncomp_ao(0), nvmat(1),
    h_vmat(nullptr), size_h_vmat(0)
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

void DeviceNumint::init_numint(py::array_t<double> _cocc, int _nao, int _nocc, int _nrow, int _grad)
{
  double t0 = omp_get_wtime();

  if(_nrow != 1 && _nrow != 4 && _nrow != 5) {
    printf("LIBGPU :: init_numint() nrow= %i not supported (1 = LDA, 4 = GGA, 5 = meta-GGA)\n", _nrow);
    exit(1);
  }

  nao = _nao;
  nocc = _nocc;
  nrow = _nrow;
  grad = _grad ? 1 : 0;
  ncomp = (nrow == 1) ? 1 : 4;
  ncomp_ao = grad ? ((nrow == 1) ? 4 : 10) : ncomp;
  nvmat = grad ? 3 : 1;

  py::buffer_info info_cocc = _cocc.request(); // 2D array (nao, nocc), C-contiguous
  double * cocc = static_cast<double*>(info_cocc.ptr);

  for(int id=0; id<ctx.num_devices; ++id) {
    ctx.pm->dev_set_device(id);
    NumintBuffers & b = bufs[id];

    grow_array(ctx.pm, b.d_cocc, nao*nocc, b.size_cocc, "numint_cocc", FLERR);
    grow_array(ctx.pm, b.d_vmat, nvmat*nao*nao, b.size_vmat, "numint_vmat", FLERR);

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

  py::buffer_info info_ao = _ao.request();   // (ncomp_ao, nao, ngrid), C-contiguous
  py::buffer_info info_rho = _rho.request(); // (nrow, ngrid), C-contiguous
  double * ao = static_cast<double*>(info_ao.ptr);
  double * rho = static_cast<double*>(info_rho.ptr);

  if((size_t) info_ao.size != (size_t) ncomp_ao * ngrid * nao) {
    printf("LIBGPU :: compute_numint_rho() ao has %zu elements; expected ncomp*ngrid*nao= %i*%i*%i\n",
	   (size_t) info_ao.size, ncomp_ao, ngrid, nao);
    exit(1);
  }

  NumintBuffers * b = select_device(block);

  grow_array(ctx.pm, b->d_ao, ncomp_ao*ngrid*nao, b->size_ao, "numint_ao", FLERR);
  grow_array(ctx.pm, b->d_c, ncomp*ngrid*nocc, b->size_c, "numint_c", FLERR);
  grow_array(ctx.pm, b->d_rho, nrow*ngrid, b->size_rho, "numint_rho", FLERR);

  // the AO block is a Python temporary: wait for the copy before returning
  ctx.pm->dev_push_async(b->d_ao, ao, (size_t) ncomp_ao*ngrid*nao*sizeof(double));
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

  if(grad) {
    vmat_grad(b, ngrid);

    double t1 = omp_get_wtime();
    LIBGPU_PROFILE(ctx, t1 - t0);
    return;
  }

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

// Gradient contributions of one grid block (pyscf.grad.rks.get_vxc before the final minus sign).
// pyscf's _dot_ao_ao(ao1, ao2)[i,j] = sum_g ao1[g,i] ao2[g,j]; with column-major (ngrid x nao)
// operands, gemm("T","N") of (ao2, ao1) gives that matrix in the row-major layout of the host.

void DeviceNumint::vmat_grad(NumintBuffers * b, int ngrid)
{
  const size_t nn = (size_t) nao * nao;
  const size_t na = (size_t) ngrid * nao;
  const double * ao = b->d_ao;
  double * aow = b->d_aow;

  // Hessian components: XX, XY, XZ = 4, 5, 6; YX, YY, YZ = 5, 7, 8; ZX, ZY, ZZ = 6, 8, 9
  const int H[3][3] = {{4, 5, 6}, {5, 7, 8}, {6, 8, 9}};

  auto ao_k = [&](int k) { return ao + k * na; };
  auto wv_k = [&](int k) { return (const double *) b->d_wv + (size_t) k * ngrid; };

  // vmat[x] += ao1^T . ao2; the first product into each vmat[x] of an evaluation overwrites it
  int first = !b->vmat_init;
  auto dot = [&](const double * ao1, const double * ao2, int x) {
    const double alpha = 1.0;
    const double beta = first ? 0.0 : 1.0;
    ctx.ml->set_handle();
    ctx.ml->gemm((char *) "T", (char *) "N", &nao, &nao, &ngrid,
		 &alpha, ao2, &ngrid, ao1, &ngrid,
		 &beta, b->d_vmat + x * nn, &nao);
  };

  // _d1_dot_(vmat, ao[1:4], sum_k ao[k] * wv[k]): k < 1 for LDA, k < 4 for GGA/meta-GGA
  numint_scale_ao(aow, ao, b->d_wv, ngrid, nao, ncomp);
  for(int x=0; x<3; ++x) dot(ao_k(1+x), aow, x);
  first = 0;
  b->vmat_init = 1;

  if(nrow == 1) return;

  // _d1_dot_(vmat, _make_dR_dao_w(ao, wv), ao[0])
  for(int x=0; x<3; ++x) {
    Terms t = {{ao_k(1+x), ao_k(H[x][0]), ao_k(H[x][1]), ao_k(H[x][2])},
	       {wv_k(0), wv_k(1), wv_k(2), wv_k(3)}, 4};
    numint_scale_ao_terms(aow, t, ngrid, nao);
    dot(aow, ao_k(0), x);
  }

  if(nrow < 5) return;

  // _tau_grad_dot_: for d, vmat[x] += ao[H[x][d]]^T . (ao[1+d] * wv[4])
  for(int d=0; d<3; ++d) {
    Terms t = {{ao_k(1+d), nullptr, nullptr, nullptr}, {wv_k(4), nullptr, nullptr, nullptr}, 1};
    numint_scale_ao_terms(aow, t, ngrid, nao);
    for(int x=0; x<3; ++x) dot(ao_k(H[x][d]), aow, x);
  }
}

/* ---------------------------------------------------------------------- */

void DeviceNumint::pull_numint_vmat(py::array_t<double> _vmat)
{
  double t0 = omp_get_wtime();

  py::buffer_info info_vmat = _vmat.request(); // (nao, nao) or (3, nao, nao); accumulated into
  double * vmat = static_cast<double*>(info_vmat.ptr);

  const int N = nvmat * nao * nao;

  if((size_t) info_vmat.size != (size_t) N) {
    printf("LIBGPU :: pull_numint_vmat() vmat has %zu elements; expected %i\n", (size_t) info_vmat.size, N);
    exit(1);
  }

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

void Device::init_numint(py::array_t<double> _cocc, int nao, int nocc, int nrow, int grad)
{ _numint->init_numint(_cocc, nao, nocc, nrow, grad); }

void Device::compute_numint_rho(py::array_t<double> _ao, py::array_t<double> _rho, int ngrid, int block)
{ _numint->compute_numint_rho(_ao, _rho, ngrid, block); }

void Device::compute_numint_vmat(py::array_t<double> _wv, int ngrid, int block)
{ _numint->compute_numint_vmat(_wv, ngrid, block); }

void Device::pull_numint_vmat(py::array_t<double> _vmat)
{ _numint->pull_numint_vmat(_vmat); }
