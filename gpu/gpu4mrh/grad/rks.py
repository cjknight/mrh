'''GPU version of pyscf.grad.rks.get_vxc (XC part of the RKS nuclear gradient, no grid
response), used by both the plain and the density-fitted RKS gradients.

Same scheme as gpu4mrh.dft.numint.nr_rks: CPU eval_ao (with second derivatives for GGA and
meta-GGA) and libxc, the density and the _d1_dot_ / _gga_grad_sum_ / _tau_grad_dot_ products
on the devices. Falls back to the CPU get_vxc under the same conditions as nr_rks.
'''
import numpy
from pyscf.lib import logger
from pyscf.grad import rks as rks_grad

from gpu4mrh.dft import numint as gpu_numint

_cpu_get_vxc = rks_grad.get_vxc

def get_vxc (ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
             max_memory=2000, verbose=None):
    gpu, cocc = gpu_numint.gpu_cocc (ni, mol, xc_code, dms)
    if cocc is None:
        return _cpu_get_vxc (ni, mol, grids, xc_code, dms, relativity=relativity, hermi=hermi,
                             max_memory=max_memory, verbose=verbose)

    t0 = (logger.process_clock (), logger.perf_counter ())
    xctype = ni._xc_type (xc_code)
    nao = mol.nao

    def block_weights (rho, weight):
        rho_xc = rho[0] if xctype == 'LDA' else rho
        vxc = ni.eval_xc_eff (xc_code, rho_xc, deriv=1, xctype=xctype, spin=0)[1]
        wv = weight * vxc.reshape (len (rho), -1)
        if xctype != 'LDA': wv[0] *= .5
        if xctype == 'MGGA': wv[4] *= .5 # for the factor 1/2 in tau
        return wv

    vmat = gpu_numint.xc_block_loop (gpu, ni, mol, grids, xc_code, cocc,
                                     numpy.zeros ((3, nao, nao)), True, block_weights)
    logger.timer (mol, 'gpu4mrh grad get_vxc', *t0)
    # - sign because nabla_X = -nabla_x
    return None, -vmat
