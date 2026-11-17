/* -*- c++ -*- */

#ifndef DEVICE_NUMINT_H
#define DEVICE_NUMINT_H

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "../pm/pm.h"
#include "../mathlib/mathlib.h"
#include "device_context.h"

using namespace PM_NS;
using namespace MATHLIB_NS;

// DFT XC numerical integration (pyscf.dft.numint.nr_rks) for one density matrix given
// by its occupied orbitals cocc = C[:,occ>0] * sqrt(occ) (nao x nocc).
//
// AO values are passed in pyscf's eval_ao memory order, grid index fastest: a C-contiguous
// (ncomp, nao, ngrid) array (= eval_ao(...).transpose(0,2,1)), so each component is a
// column-major (ngrid x nao) matrix with leading dimension ngrid.
//
// Per grid block (assigned to device block % num_devices), the Python side drives:
//   compute_numint_rho  : push AO values; c[k] = ao[k] . cocc (column-major ngrid x nocc);
//                         rho[nrow, ngrid] = (rho, d/dx, d/dy, d/dz, tau) back to host
//   (host: libxc on rho -> wv = weight * vxc, with wv[0] and wv[4] halved as in nr_rks)
//   compute_numint_vmat : vmat += (sum_c ao[c] * wv[c])^T ao[0]   [+ 1/2 sum_d (ao[d] * wv[4])^T ao[d]]
// then pull_numint_vmat sums the per-device vmat; the caller forms vmat + vmat^T.
//
// nrow selects the functional type: 1 = LDA (ncomp = 1 AO component), 4 = GGA,
// 5 = meta-GGA without laplacian (ncomp = 4: value + gradient).
//
// grad = 1 (pyscf.grad.rks.get_vxc): the AO block also carries the derivative needed for the
// nuclear gradient (ncomp_ao = 4 for LDA, 10 = value, gradient, Hessian xx,xy,xz,yy,yz,zz for
// GGA/meta-GGA; rho still uses the first ncomp_rho), and compute_numint_vmat accumulates
// vmat[3, nao, nao] as _d1_dot_ / _gga_grad_sum_ / _tau_grad_dot_ (no symmetrization; the caller
// negates). wv is weight * vxc with wv[0] (GGA, meta-GGA) and wv[4] halved, as get_vxc.
//
// Per-device buffers are owned here (not in my_device_data); the kernel launchers
// are implemented per platform in pm/<platform>/numint.cpp.
class DeviceNumint {

public:

  DeviceNumint(DeviceContext & ctx);
  ~DeviceNumint();

  // orchestration (invoked through the Device facade)
  void init_numint(py::array_t<double> _cocc, int nao, int nocc, int nrow, int grad);
  void compute_numint_rho(py::array_t<double> _ao, py::array_t<double> _rho, int ngrid, int block);
  void compute_numint_vmat(py::array_t<double> _wv, int ngrid, int block);
  void pull_numint_vmat(py::array_t<double> _vmat);

  // kernel launchers (implemented in pm/<platform>/numint.cpp)

  // rho[r*ngrid+g] from c[k] (column-major ngrid x nocc, component stride ngrid*nocc):
  // r=0 sum_o c0^2; r=1..3 2 sum_o c0 c_r; r=4 (nrow == 5) 1/2 sum_r sum_o c_r^2
  void numint_contract_rho(double * rho, const double * c, int ngrid, int nocc, int nrow);

  // out[i*ngrid+g] = sum_{k<ncomp} ao[k*nao*ngrid + i*ngrid + g] * wv[k*ngrid + g]
  void numint_scale_ao(double * out, const double * ao, const double * wv, int ngrid, int nao, int ncomp);

  // out[i*ngrid+g] = sum_{t<n} ao_t[i*ngrid + g] * wv_t[g]   (component pointers, n <= 4)
  struct Terms {
    const double * ao[4];
    const double * wv[4];
    int n;
  };
  void numint_scale_ao_terms(double * out, Terms t, int ngrid, int nao);

private:

  struct NumintBuffers {
    double * d_cocc, * d_ao, * d_c, * d_rho, * d_wv, * d_aow, * d_vmat;
    int size_cocc, size_ao, size_c, size_rho, size_wv, size_aow, size_vmat;
    int active;    // device holds a vmat contribution for this evaluation
    int vmat_init; // vmat has been written (first gemm overwrites, later ones accumulate)
  };

  NumintBuffers * select_device(int block);
  void vmat_grad(NumintBuffers * b, int ngrid); // grad = 1 contributions of one block

  DeviceContext & ctx;

  std::vector<NumintBuffers> bufs;

  int nao, nocc, nrow;
  int grad;      // 0: nr_rks vmat[nao, nao]; 1: gradient vmat[3, nao, nao]
  int ncomp;     // AO components used for rho (1 or 4)
  int ncomp_ao;  // AO components pushed per block
  int nvmat;     // 1 or 3

  double * h_vmat; // pinned staging for the per-device vmat pull
  int size_h_vmat;
};

#endif
