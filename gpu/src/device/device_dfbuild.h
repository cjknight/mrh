/* -*- c++ -*- */

#ifndef DEVICE_DFBUILD_H
#define DEVICE_DFBUILD_H

#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#include "../pm/pm.h"
#include "../mathlib/mathlib.h"
#include "device_context.h"

using namespace PM_NS;
using namespace MATHLIB_NS;

// Metric solve of the in-memory DF tensor build (pyscf.df.incore.cholesky_eri, as DF.build calls it
// when the tensor fits in max_memory). The 3-center integrals are computed on the host (libcint),
// one block of s2-packed AO pairs [p0, p0+nrow) at a time; each block is pushed to a device
// (block % num_devices), solved there and written back into the host tensor, which stays pyscf's
// ordinary _cderi. The device ERI cache is not touched: the first get_jk fills it from _cderi.
//
//   ints   block of (pair|P), memory as libcint returns it: Fortran (nrow, naoaux), i.e. the
//          column-major (nrow x naoaux) matrix J^T
//   CD     metric j2c = L L^T; low is L (host C order, so the column-major L^T on the device):
//          Y = J^T L^-T, one trsm in place
//   ED     low (naux x naoaux, host C order: column-major low^T), pyscf's eigenvalue projection:
//          Y = J^T low^T, one gemm (k = naoaux)
//   Y      column-major (nrow x naux) = row-major (naux x nrow): row P is cderi[P, p0:p0+nrow]
class DeviceDFBuild {

public:

  DeviceDFBuild(DeviceContext & ctx);
  ~DeviceDFBuild();

  // orchestration (invoked through the Device facade)
  void init_dfbuild(py::array_t<double> _low, int naoaux, int naux, int npair, int metric_cd);
  void compute_dfbuild(py::array_t<double> _ints, py::array_t<double> _cderi, int p0, int nrow, int block);
  void free_dfbuild();

private:

  struct DFBuildBuffers {
    double * d_low, * d_j, * d_y;
    int size_low, size_j, size_y;
  };

  DFBuildBuffers * select_device(int block);
  void free_buffers(DFBuildBuffers & b);

  DeviceContext & ctx;

  std::vector<DFBuildBuffers> bufs;

  int naoaux, naux, npair, metric_cd;

  double * h_buf; // pinned staging for the pull of Y
  size_t size_h_buf;
};

#endif
