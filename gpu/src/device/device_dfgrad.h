/* -*- c++ -*- */

#ifndef DEVICE_DFGRAD_H
#define DEVICE_DFGRAD_H

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "../pm/pm.h"
#include "../mathlib/mathlib.h"
#include "device_context.h"

using namespace PM_NS;
using namespace MATHLIB_NS;

// Density-fitted J/K nuclear-gradient contractions (pyscf.df.grad.rhf.get_jk) for one density
// matrix dm = orbor . orbol^T (nao x nocc factors). The 3-center integrals are computed on the
// host (libcint) and pushed block by block in their native memory order; blocks are assigned
// to devices round-robin (block % num_devices) and each device accumulates its own partial sums,
// which the pull_* calls add up.
//
//   A  compute_dfgrad_rhok (AO blocks [p0, p0+nb) of (P|mn), memory [P][b][i], i fastest):
//        rhoj_raw[P] += sum_ib (P|i b) Dm[i, p0+b]                               (if with_j)
//        Y[P][o][b]   = sum_i (P|i b) orbor[i,o]; rhok = J^-1 Y over P, as pyscf's metric solver:
//                       two triangular solves with the Cholesky factor L (metric_cd), else a
//                       GEMM with the dense pseudo-inverse pyscf uses when the Cholesky fails
//                       (an explicit inverse is not accurate enough: the jkfit metrics of large
//                       molecules have cond ~1e11)
//        rhok block   -> host (rhok[P, o, p0+b])
//        rhok_oo[P][o][j] += sum_b rhok[P,o,p0+b] orbol[p0+b, j]                  (if aux_response)
//   B  compute_dfgrad_ip1 (aux blocks [P0, P0+nP) of (d_x i j|P), memory [x][P][j][i]):
//        vj[x](i,j) += sum_P rhoj[P] I_x[i,j,P]                                  (if with_j)
//        T[x](i,(o,P)) = sum_j I_x[i,j,P] orbol[j,o];  vk[x](i,k) += sum_{o,P} T(i,(o,P)) rhok[P,o,k]
//   C  compute_dfgrad_ip2 (aux blocks of (i j|d_x P), s2 packed pairs, memory [x][P][pair]):
//        drhoj[x,P] = sum_pair I2[x,P,pair] dm_tril[pair]                          (if with_j)
//        vkaux[x,P] = sum_mn I2[x,P,mn] (orbor H_P^T orbol^T)[m,n], H_P = rhok_oo[P]
//
// vj, vk are accumulated column-major: the host arrays hold vj[x]^T, vk[x]^T.
class DeviceDFGrad {

public:

  DeviceDFGrad(DeviceContext & ctx);
  ~DeviceDFGrad();

  // orchestration (invoked through the Device facade)
  void init_dfgrad(py::array_t<double> _orbol, py::array_t<double> _orbor, py::array_t<double> _dm,
		   py::array_t<double> _dm_tril, py::array_t<double> _metric,
		   int nao, int naux, int nocc, int with_j, int aux_response, int metric_cd);
  void compute_dfgrad_rhok(py::array_t<double> _int3c, py::array_t<double> _rhok, int p0, int nb, int block);
  void pull_dfgrad_rho(py::array_t<double> _rhoj, py::array_t<double> _rhok_oo);
  void set_dfgrad_rhoj(py::array_t<double> _rhoj);
  void compute_dfgrad_ip1(py::array_t<double> _int3c, py::array_t<double> _rhok, int P0, int nP, int block);
  void compute_dfgrad_ip2(py::array_t<double> _int3c, py::array_t<double> _rhok_oo, int P0, int nP, int block);
  void pull_dfgrad(py::array_t<double> _vj, py::array_t<double> _vk,
		   py::array_t<double> _drhoj, py::array_t<double> _vkaux);
  void free_dfgrad();

  // kernel launchers (implemented in pm/<platform>/dfgrad.cpp)

  // out[i*(i+1)/2 + j + npair*P] = S_P[j + nao*i] (j < i), 0.5 * S_P[i + nao*i] (j == i), for symmetric
  // S_P = S + P*nao*nao; npair = nao*(nao+1)/2
  void dfgrad_pack_tril(double * out, const double * S, int nao, int nP);

private:

  struct DFGradBuffers {
    // per evaluation
    double * d_orbolF, * d_orbolC, * d_orborF, * d_dm, * d_dm_tril, * d_jinv, * d_rhoj;
    double * d_rhoj_raw, * d_rhok_oo, * d_vj, * d_vk, * d_drhoj, * d_vkaux;
    // per block work
    double * d_int3c, * d_rhok, * d_y, * d_x, * d_t, * d_s, * d_sp;
    int size_orbolF, size_orbolC, size_orborF, size_dm, size_dm_tril, size_jinv, size_rhoj;
    int size_rhoj_raw, size_rhok_oo, size_vj, size_vk, size_drhoj, size_vkaux;
    int size_int3c, size_rhok, size_y, size_x, size_t, size_s, size_sp;
    int active; // device holds contributions for this evaluation
  };

  DFGradBuffers * select_device(int block);
  void free_buffers(DFGradBuffers & b);
  void pull_sum(double * out, double * d_src, size_t n, int id);

  DeviceContext & ctx;

  std::vector<DFGradBuffers> bufs;

  int nao, naux, nocc, npair, with_j, aux_response, metric_cd;

  double * h_buf; // pinned staging for the per-device pulls
  size_t size_h_buf;
};

#endif
