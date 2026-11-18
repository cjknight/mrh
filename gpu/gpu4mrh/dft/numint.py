'''GPU version of pyscf.dft.numint.nr_rks (XC energy and potential matrix, RKS).

The AO values are evaluated on the CPU (pyscf eval_ao, which applies the grids' screening
mask) and pushed per grid block; libxc also stays on the CPU. On the devices, per block:
c = ao . C_occ, the density (and gradient, tau) on the grid, the scaled AO sum and the XC
matrix GEMMs. Grid blocks are assigned to devices round-robin.

Handled: one density matrix tagged with mo_coeff/mo_occ (the SCF density; pyscf then uses
eval_rho2), LDA/GGA/meta-GGA without laplacian. Everything else falls back to the CPU
nr_rks: untagged or several density matrices, negative occupations, complex, xctype HF/NLC,
and laplacian meta-GGAs (which the CPU path rejects with its own error).
'''
import numpy
from pyscf import lib
from pyscf.lib import logger
from pyscf.dft import numint
from pyscf.dft.gen_grid import BLKSIZE

from gpu4mrh.lib.utils import resolve_use_gpu
from mrh.my_pyscf.gpu import libgpu

# Device memory cap per grid block for the AO values plus one work array, in MB:
# (ncomp + 1) * ngrid * nao * 8 bytes, ncomp = AO components (1 or 4; gradient 4 or 10).
max_block_mb = 4000

# True: evaluate the AO values of every grid block into one host buffer (the device copy is
# complete before the next block is evaluated) instead of a fresh array per block. Off: on Aurora,
# with ~150 GB of huge-page memory resident (in-memory DF tensors), the reused buffer was slower
# (SCF XC 5.5 s vs 3.0 s per cycle, anouar); with nothing resident both take the same time.
reuse_buffers = False

_NROW = {'LDA': 1, 'GGA': 4, 'MGGA': 5}
_INT_MAX = 2**31 - 1

_cpu_nr_rks = numint.nr_rks

def _occupied_orbitals (dms):
    '''cocc = C[:,occ>0] * sqrt(occ) for a single real density matrix tagged with its
    orbitals (as pyscf eval_rho2), else None.'''
    mo_coeff = getattr (dms, 'mo_coeff', None)
    mo_occ = getattr (dms, 'mo_occ', None)
    if mo_coeff is None or mo_occ is None: return None
    if numpy.asarray (dms).ndim != 2: return None
    mo_coeff = numpy.asarray (mo_coeff)
    mo_occ = numpy.asarray (mo_occ)
    if mo_coeff.ndim != 2 or mo_occ.ndim != 1: return None
    if numpy.iscomplexobj (mo_coeff) or numpy.iscomplexobj (dms): return None
    if numpy.any (mo_occ < -numint.OCCDROP): return None
    pos = mo_occ > numint.OCCDROP
    return numpy.ascontiguousarray (mo_coeff[:,pos] * numpy.sqrt (mo_occ[pos]))

def _grid_blocks (ngrids, nao, ncomp, ndev):
    '''(p0, p1) grid blocks: multiples of BLKSIZE (to index grids.non0tab), at most
    max_block_mb of device memory each, and a multiple of ndev blocks to balance devices.'''
    cap = int (max_block_mb * 1e6 / ((ncomp + 1) * nao * 8))
    cap = min (cap, _INT_MAX // (ncomp * nao))
    cap = max (BLKSIZE, cap // BLKSIZE * BLKSIZE)
    nblk = -(-ngrids // cap)
    nblk = -(-nblk // ndev) * ndev
    blksize = -(-ngrids // nblk)
    blksize = -(-blksize // BLKSIZE) * BLKSIZE
    return [(p0, min (p0 + blksize, ngrids)) for p0 in range (0, ngrids, blksize)]

def _laplacian_mgga (xc_code):
    return any (x in xc_code.upper () for x in ('CC06', 'CS', 'BR89', 'MK00'))

def gpu_cocc (ni, mol, xc_code, dms):
    '''(gpu, cocc) if this XC evaluation can run on the device, else (None, None).'''
    gpu = resolve_use_gpu (mol)
    xctype = ni._xc_type (xc_code)
    if gpu is None or xctype not in _NROW or (xctype == 'MGGA' and _laplacian_mgga (xc_code)):
        return None, None
    cocc = _occupied_orbitals (dms)
    if cocc is None: return None, None
    return gpu, cocc

def xc_block_loop (gpu, ni, mol, grids, xc_code, cocc, vmat, grad, block_weights):
    '''Device XC integration over the grid blocks (see module docstring). block_weights
    (rho[nrow, ng], weight[ng]) -> wv[nrow, ng] does the host part of each block (libxc);
    the per-device matrices are summed into vmat ((nao, nao), or (3, nao, nao) for grad).'''
    if grids.coords is None:
        grids.build (with_non0tab=True)
    non0tab = grids.non0tab if mol is grids.mol else None
    xctype = ni._xc_type (xc_code)
    nao = mol.nao
    nocc = cocc.shape[1]
    nrow = _NROW[xctype]
    ao_deriv = (0 if xctype == 'LDA' else 1) + int (bool (grad))
    ncomp = (1, 4, 10)[ao_deriv]
    ngrids = grids.weights.size
    if nocc == 0: return vmat

    libgpu.init_numint (gpu, cocc, nao, nocc, nrow, int (bool (grad)))
    ndev = libgpu.get_num_devices (gpu)
    blocks = _grid_blocks (ngrids, nao, ncomp, ndev)
    aobuf = None
    if reuse_buffers:
        aobuf = numpy.empty (ncomp * nao * max (p1 - p0 for p0, p1 in blocks))
    for block, (p0, p1) in enumerate (blocks):
        ng = p1 - p0
        mask = None if non0tab is None else non0tab[p0//BLKSIZE:]
        ao = ni.eval_ao (mol, grids.coords[p0:p1], deriv=ao_deriv, non0tab=mask,
                         cutoff=grids.cutoff, out=aobuf)
        # eval_ao's memory order is (ncomp, nao, ngrid): pass that buffer, no copy
        ao = numpy.ascontiguousarray (ao.reshape (ncomp, ng, nao).transpose (0,2,1))
        rho = numpy.empty ((nrow, ng))
        libgpu.compute_numint_rho (gpu, ao, rho, ng, block)
        ao = None
        wv = numpy.ascontiguousarray (block_weights (rho, grids.weights[p0:p1]))
        libgpu.compute_numint_vmat (gpu, wv, ng, block)
    libgpu.pull_numint_vmat (gpu, vmat)
    return vmat

def nr_rks (ni, mol, grids, xc_code, dms, relativity=0, hermi=1,
            max_memory=2000, verbose=None):
    gpu, cocc = gpu_cocc (ni, mol, xc_code, dms)
    if cocc is None:
        return _cpu_nr_rks (ni, mol, grids, xc_code, dms, relativity=relativity, hermi=hermi,
                            max_memory=max_memory, verbose=verbose)

    t0 = (logger.process_clock (), logger.perf_counter ())
    xctype = ni._xc_type (xc_code)
    nrow = _NROW[xctype]
    nao = mol.nao
    acc = [0.0, 0.0] # nelec, excsum

    def block_weights (rho, weight):
        rho_xc = rho[0] if xctype == 'LDA' else rho
        exc, vxc = ni.eval_xc_eff (xc_code, rho_xc, deriv=1, xctype=xctype, spin=0)[:2]
        den = rho[0] * weight
        acc[0] += den.sum ()
        acc[1] += numpy.dot (den, exc)
        wv = weight * vxc.reshape (nrow, -1)
        wv[0] *= .5 # vmat + vmat.T at the end
        if nrow == 5: wv[4] *= .5 # 1/2 in tau
        return wv

    vmat = xc_block_loop (gpu, ni, mol, grids, xc_code, cocc, numpy.zeros ((nao, nao)), False,
                          block_weights)
    nelec, excsum = acc
    vmat = vmat + vmat.T
    logger.timer (mol, 'gpu4mrh nr_rks', *t0)

    if isinstance (dms, numpy.ndarray) and vmat.dtype != dms.dtype:
        vmat = numpy.asarray (vmat, dtype=dms.dtype)
    return nelec, excsum, vmat
