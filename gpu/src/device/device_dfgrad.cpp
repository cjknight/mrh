/* -*- c++ -*- */

#include <stdio.h>
#include <string.h>

#include "device.h"

/* ---------------------------------------------------------------------- */

DeviceDFGrad::DeviceDFGrad(DeviceContext & _ctx)
  : ctx(_ctx), nao(0), naux(0), nocc(0), npair(0), with_j(0), aux_response(0), metric_cd(0),
    h_buf(nullptr), size_h_buf(0)
{
  DFGradBuffers empty;
  memset(&empty, 0, sizeof(DFGradBuffers));
  bufs.assign(ctx.num_devices, empty);
}

/* ---------------------------------------------------------------------- */

void DeviceDFGrad::free_buffers(DFGradBuffers & b)
{
  double ** ptrs[] = {&b.d_orbolF, &b.d_orbolC, &b.d_orborF, &b.d_dm, &b.d_dm_tril, &b.d_jinv, &b.d_rhoj,
		      &b.d_rhoj_raw, &b.d_rhok_oo, &b.d_vj, &b.d_vk, &b.d_drhoj, &b.d_vkaux,
		      &b.d_int3c, &b.d_rhok, &b.d_y, &b.d_x, &b.d_t, &b.d_s, &b.d_sp};
  for(double ** p : ptrs) {
    if(*p) ctx.pm->dev_free(*p, "dfgrad");
    *p = nullptr;
  }
  const int active = b.active;
  memset(&b, 0, sizeof(DFGradBuffers));
  b.active = active;
}

DeviceDFGrad::~DeviceDFGrad()
{
  for(int id=0; id<ctx.num_devices; ++id) {
    ctx.pm->dev_set_device(id);
    free_buffers(bufs[id]);
  }
  if(h_buf) ctx.pm->dev_free_host(h_buf);
}

/* ---------------------------------------------------------------------- */

// release device memory between gradient evaluations (the blocks are large)
void DeviceDFGrad::free_dfgrad()
{
  for(int id=0; id<ctx.num_devices; ++id) {
    ctx.pm->dev_set_device(id);
    ctx.pm->dev_stream_wait();
    free_buffers(bufs[id]);
    bufs[id].active = 0;
  }
}

/* ---------------------------------------------------------------------- */

DeviceDFGrad::DFGradBuffers * DeviceDFGrad::select_device(int block)
{
  const int id = block % ctx.num_devices;
  ctx.pm->dev_set_device(id);
  bufs[id].active = 1;
  return &(bufs[id]);
}

/* ---------------------------------------------------------------------- */

static double * as_ptr(py::array_t<double> & a) { return static_cast<double*>(a.request().ptr); }

static void check_size(py::array_t<double> & a, size_t n, const char * what)
{
  if((size_t) a.request().size != n) {
    printf("LIBGPU :: dfgrad %s has %zu elements; expected %zu\n", what, (size_t) a.request().size, n);
    exit(1);
  }
}

/* ---------------------------------------------------------------------- */

void DeviceDFGrad::init_dfgrad(py::array_t<double> _orbol, py::array_t<double> _orbor, py::array_t<double> _dm,
			       py::array_t<double> _dm_tril, py::array_t<double> _metric,
			       int _nao, int _naux, int _nocc, int _with_j, int _aux_response, int _metric_cd)
{
  double t0 = omp_get_wtime();

  nao = _nao;
  naux = _naux;
  nocc = _nocc;
  npair = nao * (nao + 1) / 2;
  with_j = _with_j;
  aux_response = _aux_response;
  metric_cd = _metric_cd;

  // orbol, orbor: C-contiguous (nao, nocc); dm: C-contiguous (nao, nao) = orbor . orbol^T, passed
  // transposed so that d_dm is its column-major copy; dm_tril: (npair); metric (naux, naux): the
  // Cholesky factor L passed transposed (column-major lower triangle on the device) if metric_cd,
  // else pyscf's dense pseudo-inverse j2c (symmetric)
  check_size(_orbol, (size_t) nao*nocc, "orbol");
  check_size(_orbor, (size_t) nao*nocc, "orbor");
  check_size(_dm, (size_t) nao*nao, "dm");
  check_size(_metric, (size_t) naux*naux, "metric");
  if(with_j && aux_response) check_size(_dm_tril, (size_t) npair, "dm_tril");

  double * orbol = as_ptr(_orbol);
  double * orbor = as_ptr(_orbor);
  double * dm = as_ptr(_dm);
  double * metric = as_ptr(_metric);

  // column-major copies of orbol and orbor (nao x nocc)
  std::vector<double> orbolF((size_t) nao*nocc), orborF((size_t) nao*nocc);
#pragma omp parallel for
  for(int i=0; i<nao; ++i)
    for(int o=0; o<nocc; ++o) {
      orbolF[i + (size_t) nao*o] = orbol[(size_t) i*nocc + o];
      orborF[i + (size_t) nao*o] = orbor[(size_t) i*nocc + o];
    }

  const int zero = 0;

  for(int id=0; id<ctx.num_devices; ++id) {
    ctx.pm->dev_set_device(id);
    DFGradBuffers & b = bufs[id];
    b.active = 0;

    grow_array(ctx.pm, b.d_orbolF, nao*nocc, b.size_orbolF, "dfgrad_orbolF", FLERR);
    grow_array(ctx.pm, b.d_orbolC, nao*nocc, b.size_orbolC, "dfgrad_orbolC", FLERR);
    grow_array(ctx.pm, b.d_orborF, nao*nocc, b.size_orborF, "dfgrad_orborF", FLERR);
    grow_array(ctx.pm, b.d_jinv, naux*naux, b.size_jinv, "dfgrad_jinv", FLERR);
    grow_array(ctx.pm, b.d_vk, 3*nao*nao, b.size_vk, "dfgrad_vk", FLERR);

    ctx.pm->dev_push_async(b.d_orbolF, orbolF.data(), (size_t) nao*nocc*sizeof(double));
    ctx.pm->dev_push_async(b.d_orbolC, orbol, (size_t) nao*nocc*sizeof(double));
    ctx.pm->dev_push_async(b.d_orborF, orborF.data(), (size_t) nao*nocc*sizeof(double));
    ctx.pm->dev_push_async(b.d_jinv, metric, (size_t) naux*naux*sizeof(double));

    size_t bytes = (size_t) 3*nao*nao*sizeof(double);
    ctx.ml->memset(b.d_vk, &zero, &bytes);

    if(with_j) {
      grow_array(ctx.pm, b.d_dm, nao*nao, b.size_dm, "dfgrad_dm", FLERR);
      grow_array(ctx.pm, b.d_rhoj_raw, naux, b.size_rhoj_raw, "dfgrad_rhoj_raw", FLERR);
      grow_array(ctx.pm, b.d_rhoj, naux, b.size_rhoj, "dfgrad_rhoj", FLERR);
      grow_array(ctx.pm, b.d_vj, 3*nao*nao, b.size_vj, "dfgrad_vj", FLERR);
      ctx.pm->dev_push_async(b.d_dm, dm, (size_t) nao*nao*sizeof(double));
      bytes = (size_t) naux*sizeof(double);
      ctx.ml->memset(b.d_rhoj_raw, &zero, &bytes);
      bytes = (size_t) 3*nao*nao*sizeof(double);
      ctx.ml->memset(b.d_vj, &zero, &bytes);
    }

    if(aux_response) {
      grow_array(ctx.pm, b.d_rhok_oo, naux*nocc*nocc, b.size_rhok_oo, "dfgrad_rhok_oo", FLERR);
      grow_array(ctx.pm, b.d_vkaux, 3*naux, b.size_vkaux, "dfgrad_vkaux", FLERR);
      bytes = (size_t) naux*nocc*nocc*sizeof(double);
      ctx.ml->memset(b.d_rhok_oo, &zero, &bytes);
      bytes = (size_t) 3*naux*sizeof(double);
      ctx.ml->memset(b.d_vkaux, &zero, &bytes);
      if(with_j) {
	grow_array(ctx.pm, b.d_dm_tril, npair, b.size_dm_tril, "dfgrad_dm_tril", FLERR);
	grow_array(ctx.pm, b.d_drhoj, 3*naux, b.size_drhoj, "dfgrad_drhoj", FLERR);
	ctx.pm->dev_push_async(b.d_dm_tril, as_ptr(_dm_tril), (size_t) npair*sizeof(double));
	bytes = (size_t) 3*naux*sizeof(double);
	ctx.ml->memset(b.d_drhoj, &zero, &bytes);
      }
    }

    ctx.pm->dev_stream_wait(); // host arrays are Python temporaries
  }

  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
}

/* ---------------------------------------------------------------------- */

// A: one AO block [p0, p0+nb) of (P|mn); int3c is (nao, nb, naux) Fortran order; rhok is the
// host output (naux, nocc, nb) C-contiguous.
void DeviceDFGrad::compute_dfgrad_rhok(py::array_t<double> _int3c, py::array_t<double> _rhok, int p0, int nb, int block)
{
  double t0 = omp_get_wtime();

  check_size(_int3c, (size_t) nao*nb*naux, "int3c (A)");
  check_size(_rhok, (size_t) naux*nocc*nb, "rhok (A)");
  double * h_int3c = as_ptr(_int3c);
  double * h_rhok = as_ptr(_rhok);

  // Release the GIL for the push and the device work: get_jk computes the next integral block
  // in a prefetch thread (lib.map_with_prefetch), which needs the GIL to start its libcint call.
  // All Python objects are touched above; the arrays stay alive as arguments of this call.
  py::gil_scoped_release nogil;

  DFGradBuffers * b = select_device(block);

  grow_array(ctx.pm, b->d_int3c, nao*nb*naux, b->size_int3c, "dfgrad_int3c", FLERR);
  grow_array(ctx.pm, b->d_y, nb*naux*nocc, b->size_y, "dfgrad_y", FLERR);
  if(!metric_cd) grow_array(ctx.pm, b->d_x, nb*naux*nocc, b->size_x, "dfgrad_x", FLERR);

  ctx.pm->dev_push_async(b->d_int3c, h_int3c, (size_t) nao*nb*naux*sizeof(double));
  ctx.pm->dev_stream_wait(); // Python temporary

  const double one = 1.0;
  const double zero = 0.0;
  const int inc = 1;

  // rhoj_raw[P] += sum_{i,b} A[(i,b),P] Dm[i, p0+b]: A as column-major (nao*nb x naux)
  if(with_j) {
    const int m = nao * nb;
    ctx.ml->set_handle();
    ctx.ml->gemv((char *) "T", &m, &naux, &one, b->d_int3c, &m, b->d_dm + (size_t) p0*nao, &inc,
		 &one, b->d_rhoj_raw, &inc);
  }

  // Y_P (nb x nocc) = A_P^T orbor, A_P = (P|i b) column-major (nao x nb): Y layout [P][o][b], i.e. the
  // column-major (nb*nocc x naux) matrix Y[(b,o), P]
  const int m = nb * nocc;
  {
    const int strideA = nao * nb;
    const int stride0 = 0;
    ctx.ml->set_handle();
    ctx.ml->gemm_batch((char *) "T", (char *) "N", &nb, &nocc, &nao,
		       &one, b->d_int3c, &nao, &strideA, b->d_orborF, &nao, &stride0,
		       &zero, b->d_y, &nb, &m, &naux);
  }

  // rhok = Y J^-1 over P (pyscf: J^-1 v with v = Y^T)
  double * d_rhok = b->d_y;
  if(metric_cd) {
    // J = L L^T: Y <- Y L^-T, then Y <- Y L^-1
    ctx.ml->set_handle();
    ctx.ml->trsm((char *) "R", (char *) "L", (char *) "T", (char *) "N", &m, &naux,
		 &one, b->d_jinv, &naux, b->d_y, &m);
    ctx.ml->trsm((char *) "R", (char *) "L", (char *) "N", (char *) "N", &m, &naux,
		 &one, b->d_jinv, &naux, b->d_y, &m);
  } else {
    // the host j2c (C order) is j2c^T column-major: X = Y j2c^T, X^T = j2c Y^T
    ctx.ml->set_handle();
    ctx.ml->gemm((char *) "N", (char *) "N", &m, &naux, &naux,
		 &one, b->d_y, &m, b->d_jinv, &naux,
		 &zero, b->d_x, &m);
    d_rhok = b->d_x;
  }

  // rhok_oo[P][o][j] += sum_b orbol[p0+b, j] rhok[P,o,b]: column-major (nocc x nocc*naux)
  if(aux_response) {
    const int n = nocc * naux;
    ctx.ml->set_handle();
    ctx.ml->gemm((char *) "N", (char *) "N", &nocc, &n, &nb,
		 &one, b->d_orbolC + (size_t) p0*nocc, &nocc, d_rhok, &nb,
		 &one, b->d_rhok_oo, &nocc);
  }

  ctx.pm->dev_pull_async(d_rhok, h_rhok, (size_t) naux*nocc*nb*sizeof(double));
  ctx.pm->dev_stream_wait();

  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
}

/* ---------------------------------------------------------------------- */

// add device d_src (n doubles) of device id into host out
void DeviceDFGrad::pull_sum(double * out, double * d_src, size_t n, int id)
{
  if(n > size_h_buf) {
    if(h_buf) ctx.pm->dev_free_host(h_buf);
    h_buf = (double *) ctx.pm->dev_malloc_host(n * sizeof(double));
    size_h_buf = n;
  }

  ctx.pm->dev_set_device(id);
  ctx.pm->dev_pull_async(d_src, h_buf, n * sizeof(double));
  ctx.pm->dev_stream_wait();

#pragma omp parallel for
  for(size_t i=0; i<n; ++i) out[i] += h_buf[i];
}

/* ---------------------------------------------------------------------- */

// sums of the per-device rhoj_raw (naux) and rhok_oo (naux, nocc, nocc), accumulated into the arguments
void DeviceDFGrad::pull_dfgrad_rho(py::array_t<double> _rhoj, py::array_t<double> _rhok_oo)
{
  double t0 = omp_get_wtime();

  if(with_j) check_size(_rhoj, (size_t) naux, "rhoj");
  if(aux_response) check_size(_rhok_oo, (size_t) naux*nocc*nocc, "rhok_oo");

  for(int id=0; id<ctx.num_devices; ++id) {
    if(!bufs[id].active) continue;
    if(with_j) pull_sum(as_ptr(_rhoj), bufs[id].d_rhoj_raw, naux, id);
    if(aux_response) pull_sum(as_ptr(_rhok_oo), bufs[id].d_rhok_oo, (size_t) naux*nocc*nocc, id);
  }

  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
}

/* ---------------------------------------------------------------------- */

void DeviceDFGrad::set_dfgrad_rhoj(py::array_t<double> _rhoj)
{
  if(!with_j) return;
  check_size(_rhoj, (size_t) naux, "rhoj");

  for(int id=0; id<ctx.num_devices; ++id) {
    ctx.pm->dev_set_device(id);
    ctx.pm->dev_push_async(bufs[id].d_rhoj, as_ptr(_rhoj), (size_t) naux*sizeof(double));
    ctx.pm->dev_stream_wait();
  }
}

/* ---------------------------------------------------------------------- */

// B: aux block [P0, P0+nP) of (d_x i j|P), memory [x][P][j][i]; rhok is (nP, nocc, nao) C-contiguous
void DeviceDFGrad::compute_dfgrad_ip1(py::array_t<double> _int3c, py::array_t<double> _rhok, int P0, int nP, int block)
{
  double t0 = omp_get_wtime();

  const size_t nn = (size_t) nao * nao;
  check_size(_int3c, 3 * nn * nP, "int3c (ip1)");
  check_size(_rhok, (size_t) nP*nocc*nao, "rhok (ip1)");
  double * h_int3c = as_ptr(_int3c);
  double * h_rhok = as_ptr(_rhok);

  // Release the GIL for the push and the device work: get_jk computes the next integral block
  // in a prefetch thread (lib.map_with_prefetch), which needs the GIL to start its libcint call.
  // All Python objects are touched above; the arrays stay alive as arguments of this call.
  py::gil_scoped_release nogil;

  DFGradBuffers * b = select_device(block);

  grow_array(ctx.pm, b->d_int3c, 3*nao*nao*nP, b->size_int3c, "dfgrad_int3c", FLERR);
  grow_array(ctx.pm, b->d_rhok, nP*nocc*nao, b->size_rhok, "dfgrad_rhok", FLERR);
  grow_array(ctx.pm, b->d_t, nao*nocc*nP, b->size_t, "dfgrad_t", FLERR);

  ctx.pm->dev_push_async(b->d_int3c, h_int3c, 3 * nn * nP * sizeof(double));
  ctx.pm->dev_push_async(b->d_rhok, h_rhok, (size_t) nP*nocc*nao*sizeof(double));
  ctx.pm->dev_stream_wait(); // Python temporaries

  const double one = 1.0;
  const double zero = 0.0;
  const int inc = 1;
  const int nao2 = nao * nao;
  const int stride0 = 0;
  const int strideT = nao * nocc;
  const int ncol = nocc * nP;

  for(int x=0; x<3; ++x) {
    const double * I = b->d_int3c + x * nn * nP;

    // vj[x](i,j) += sum_P I[(i,j),P] rhoj[P]
    if(with_j) {
      ctx.ml->set_handle();
      ctx.ml->gemv((char *) "N", &nao2, &nP, &one, I, &nao2, b->d_rhoj + P0, &inc,
		   &one, b->d_vj + x * nn, &inc);
    }

    // T(i,(o,P)) = sum_j I_P(i,j) orbol(j,o)
    ctx.ml->set_handle();
    ctx.ml->gemm_batch((char *) "N", (char *) "N", &nao, &nocc, &nao,
		       &one, I, &nao, &nao2, b->d_orbolF, &nao, &stride0,
		       &zero, b->d_t, &nao, &strideT, &nP);

    // vk[x](i,k) += sum_(o,P) T(i,(o,P)) rhok(k,(o,P))
    ctx.ml->set_handle();
    ctx.ml->gemm((char *) "N", (char *) "T", &nao, &nao, &ncol,
		 &one, b->d_t, &nao, b->d_rhok, &nao,
		 &one, b->d_vk + x * nn, &nao);
  }

  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
}

/* ---------------------------------------------------------------------- */

// C: aux block [P0, P0+nP) of (i j|d_x P), s2 packed, memory [x][P][pair]; rhok_oo is (nP, nocc, nocc)
void DeviceDFGrad::compute_dfgrad_ip2(py::array_t<double> _int3c, py::array_t<double> _rhok_oo, int P0, int nP, int block)
{
  double t0 = omp_get_wtime();

  const size_t nn = (size_t) nao * nao;
  check_size(_int3c, (size_t) 3 * npair * nP, "int3c (ip2)");
  check_size(_rhok_oo, (size_t) nP*nocc*nocc, "rhok_oo (ip2)");
  double * h_int3c = as_ptr(_int3c);
  double * h_rhok_oo = as_ptr(_rhok_oo);

  // Release the GIL for the push and the device work: get_jk computes the next integral block
  // in a prefetch thread (lib.map_with_prefetch), which needs the GIL to start its libcint call.
  // All Python objects are touched above; the arrays stay alive as arguments of this call.
  py::gil_scoped_release nogil;

  DFGradBuffers * b = select_device(block);

  grow_array(ctx.pm, b->d_int3c, 3*npair*nP, b->size_int3c, "dfgrad_int3c", FLERR);
  grow_array(ctx.pm, b->d_rhok, nP*nocc*nocc, b->size_rhok, "dfgrad_rhok", FLERR);
  grow_array(ctx.pm, b->d_t, nao*nocc*nP, b->size_t, "dfgrad_t", FLERR);
  grow_array(ctx.pm, b->d_s, nao*nao*nP, b->size_s, "dfgrad_s", FLERR);
  grow_array(ctx.pm, b->d_sp, npair*nP, b->size_sp, "dfgrad_sp", FLERR);

  ctx.pm->dev_push_async(b->d_int3c, h_int3c, (size_t) 3*npair*nP*sizeof(double));
  ctx.pm->dev_push_async(b->d_rhok, h_rhok_oo, (size_t) nP*nocc*nocc*sizeof(double));
  ctx.pm->dev_stream_wait(); // Python temporaries

  const double one = 1.0;
  const double zero = 0.0;
  const int inc = 1;
  const int ione = 1;
  const int stride0 = 0;
  const int strideZ = nao * nocc;
  const int strideS = nao * nao;
  const int ncol = nocc * nP;

  // drhoj[x,P] = sum_pair I2[x,P,pair] dm_tril[pair]
  if(with_j)
    for(int x=0; x<3; ++x) {
      ctx.ml->set_handle();
      ctx.ml->gemv((char *) "T", &npair, &nP, &one, b->d_int3c + (size_t) x*npair*nP, &npair,
		   b->d_dm_tril, &inc, &zero, b->d_drhoj + x*naux + P0, &inc);
    }

  // M_P (nocc x nocc, column-major) = rhok_oo[P]^T;  Z = orbor . [M_P]  (nao x nocc*nP)
  ctx.ml->set_handle();
  ctx.ml->gemm((char *) "N", (char *) "N", &nao, &ncol, &nocc,
	       &one, b->d_orborF, &nao, b->d_rhok, &nocc,
	       &zero, b->d_t, &nao);

  // S_P = Z_P orbol^T + orbol Z_P^T
  ctx.ml->set_handle();
  ctx.ml->gemm_batch((char *) "N", (char *) "T", &nao, &nao, &nocc,
		     &one, b->d_t, &nao, &strideZ, b->d_orbolF, &nao, &stride0,
		     &zero, b->d_s, &nao, &strideS, &nP);
  ctx.ml->set_handle();
  ctx.ml->gemm_batch((char *) "N", (char *) "T", &nao, &nao, &nocc,
		     &one, b->d_orbolF, &nao, &stride0, b->d_t, &nao, &strideZ,
		     &one, b->d_s, &nao, &strideS, &nP);

  dfgrad_pack_tril(b->d_sp, b->d_s, nao, nP);

  // vkaux[x,P] = sum_pair I2[x,P,pair] Sp[P,pair]
  for(int x=0; x<3; ++x) {
    ctx.ml->set_handle();
    ctx.ml->gemm_batch((char *) "T", (char *) "N", &ione, &ione, &npair,
		       &one, b->d_int3c + (size_t) x*npair*nP, &npair, &npair, b->d_sp, &npair, &npair,
		       &zero, b->d_vkaux + x*naux + P0, &ione, &ione, &nP);
  }

  (void) nn;

  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
}

/* ---------------------------------------------------------------------- */

// sums of the per-device vj, vk (3, nao, nao; transposed, see header), drhoj and vkaux (3, naux),
// accumulated into the arguments
void DeviceDFGrad::pull_dfgrad(py::array_t<double> _vj, py::array_t<double> _vk,
			       py::array_t<double> _drhoj, py::array_t<double> _vkaux)
{
  double t0 = omp_get_wtime();

  const size_t n3 = (size_t) 3 * nao * nao;
  check_size(_vk, n3, "vk");
  if(with_j) check_size(_vj, n3, "vj");
  if(aux_response) check_size(_vkaux, (size_t) 3*naux, "vkaux");
  if(aux_response && with_j) check_size(_drhoj, (size_t) 3*naux, "drhoj");

  for(int id=0; id<ctx.num_devices; ++id) {
    if(!bufs[id].active) continue;
    pull_sum(as_ptr(_vk), bufs[id].d_vk, n3, id);
    if(with_j) pull_sum(as_ptr(_vj), bufs[id].d_vj, n3, id);
    if(aux_response) pull_sum(as_ptr(_vkaux), bufs[id].d_vkaux, (size_t) 3*naux, id);
    if(aux_response && with_j) pull_sum(as_ptr(_drhoj), bufs[id].d_drhoj, (size_t) 3*naux, id);
  }

  double t1 = omp_get_wtime();
  LIBGPU_PROFILE(ctx, t1 - t0);
}

/* ---------------------------------------------------------------------- */
/* Device facade forwarders                                               */
/* ---------------------------------------------------------------------- */

void Device::init_dfgrad(py::array_t<double> orbol, py::array_t<double> orbor, py::array_t<double> dm,
			 py::array_t<double> dm_tril, py::array_t<double> metric,
			 int nao, int naux, int nocc, int with_j, int aux_response, int metric_cd)
{ _dfgrad->init_dfgrad(orbol, orbor, dm, dm_tril, metric, nao, naux, nocc, with_j, aux_response, metric_cd); }

void Device::compute_dfgrad_rhok(py::array_t<double> int3c, py::array_t<double> rhok, int p0, int nb, int block)
{ _dfgrad->compute_dfgrad_rhok(int3c, rhok, p0, nb, block); }

void Device::pull_dfgrad_rho(py::array_t<double> rhoj, py::array_t<double> rhok_oo)
{ _dfgrad->pull_dfgrad_rho(rhoj, rhok_oo); }

void Device::set_dfgrad_rhoj(py::array_t<double> rhoj)
{ _dfgrad->set_dfgrad_rhoj(rhoj); }

void Device::compute_dfgrad_ip1(py::array_t<double> int3c, py::array_t<double> rhok, int P0, int nP, int block)
{ _dfgrad->compute_dfgrad_ip1(int3c, rhok, P0, nP, block); }

void Device::compute_dfgrad_ip2(py::array_t<double> int3c, py::array_t<double> rhok_oo, int P0, int nP, int block)
{ _dfgrad->compute_dfgrad_ip2(int3c, rhok_oo, P0, nP, block); }

void Device::pull_dfgrad(py::array_t<double> vj, py::array_t<double> vk, py::array_t<double> drhoj, py::array_t<double> vkaux)
{ _dfgrad->pull_dfgrad(vj, vk, drhoj, vkaux); }

void Device::free_dfgrad()
{ _dfgrad->free_dfgrad(); }
