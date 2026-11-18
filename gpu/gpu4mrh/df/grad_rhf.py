'''GPU version of pyscf.df.grad.rhf.get_jk (density-fitted J/K nuclear-gradient contractions).

The 3-center integrals stay on the CPU (libcint), computed one block ahead of the device work
(lib.map_with_prefetch) and pushed in their native memory order; the contractions run on the
devices (DeviceDFGrad), blocks assigned round-robin:

  A  AO blocks of (P|mn): rhoj (unsolved), rhok = J^-1 (P|mn) orbor and rhok_oo; rhok is
     kept in host memory (pyscf writes it to an HDF5 temporary file)
  B  aux blocks of (d_x m n|P): vj, vk
  C  aux blocks of (m n|d_x P) (auxbasis response): drhoj and vkaux
The metric solve follows pyscf's _gen_metric_solver: two triangular solves with the Cholesky
factor (CD), or, if the Cholesky fails or decompose_j2c='ED', a GEMM with the same dense
pseudo-inverse pyscf uses. (An explicit inverse in place of the triangular solves is not
accurate enough: the aug-jkfit metric of a 66-atom molecule has cond ~1e11, and J^-1 applied as
a GEMM loses ~3 digits there.) The (d_x P|Q) terms and the per-atom sums stay on the host as in
pyscf.

Handled: one real density matrix, with_k (any with_j), either auxbasis_response. Falls back to
the CPU get_jk otherwise (J only, several density matrices, complex, only_dfj).
'''
import os
import ctypes
import numpy
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.ao2mo.outcore import balance_partition
from pyscf.df.grad import rhf as df_rhf_grad

from gpu4mrh.lib.utils import resolve_use_gpu
from mrh.my_pyscf.gpu import libgpu

# Device memory cap per integral block (integrals plus work arrays), in MB
max_block_mb = 4000

_INT_MAX = 2**31 - 1

_cpu_get_jk = df_rhf_grad.get_jk

# GPU4MRH_DFGRAD_CHECK=1: debug check of the aux-response K terms against host references (rhok_oo
# from the host rhok; vkaux per aux block as pyscf); prints the deviations, slow
_check = os.environ.get ('GPU4MRH_DFGRAD_CHECK', '0') not in ('', '0')

def _vkaux_ref (int3c, orbol, orbor, rhok_oo_blk, nao):
    '''pyscf's (ij|d_x P) rhok_oo contraction for one aux block; int3c (3, npair, nP) as returned
    by the s2ij int3c2e_ip2 wrapper.'''
    from pyscf.ao2mo import _ao2mo
    nP, nocc = rhok_oo_blk.shape[0], orbol.shape[1]
    orbol = numpy.asfortranarray (orbol) # as pyscf's _decompose_rdm1*: AO2MOmmm_bra reads it so
    buf = numpy.empty ((3, nP, nocc, nao))
    _ao2mo.libao2mo.AO2MOnr_e2_drv (_ao2mo.libao2mo.AO2MOtranse2_nr_s2,
        _ao2mo.libao2mo.AO2MOmmm_bra_nr_s2, buf.ctypes.data_as (ctypes.c_void_p),
        int3c.ctypes.data_as (ctypes.c_void_p), orbol.ctypes.data_as (ctypes.c_void_p),
        ctypes.c_int (3*nP), ctypes.c_int (nao), (ctypes.c_int*4)(0, nocc, 0, nao),
        lib.c_null_ptr (), ctypes.c_int (0))
    int3c_ij = lib.dot (buf.reshape (-1, nao), orbor).reshape (3, nP, nocc, nocc)
    return lib.einsum ('xpij,pij->xp', int3c_ij, rhok_oo_blk)

def _ranges (loc, nbytes, nelem, cap_bytes):
    '''balance_partition shell ranges of at most cap_bytes per block, given the device bytes and
    the elements of the largest (int-indexed) array per basis function of the block.'''
    blk = int (min (cap_bytes // nbytes, _INT_MAX // nelem))
    blk = max (blk, int (max (loc[1:] - loc[:-1])))
    return [(shl0, shl1) for shl0, shl1, _ in balance_partition (loc, blk)]

def _metric (auxmol, decompose_j2c, lindep):
    '''As pyscf.df.grad.rhf._gen_metric_solver: (True, L^T, solver) with L the lower Cholesky
    factor ((naux, naux), C order, i.e. L column-major), else (False, j2c, solver) with the dense
    pseudo-inverse j2c.'''
    int2c = auxmol.intor ('int2c2e', aosym='s1')
    if decompose_j2c.upper () == 'CD':
        try:
            low = scipy.linalg.cho_factor (int2c, lower=True)
            def solver (v):
                return scipy.linalg.cho_solve (low, v, overwrite_b=True)
            return True, numpy.ascontiguousarray (low[0].T), solver
        except (numpy.linalg.LinAlgError, scipy.linalg.LinAlgError):
            pass
    w, v = scipy.linalg.eigh (int2c)
    mask = w > lindep
    v1 = v[:,mask]
    j2c = lib.dot (v1/w[mask], v1.conj ().T)
    def solver (v):
        return lib.dot (j2c, v)
    return False, numpy.ascontiguousarray (j2c), solver

def get_jk (mf_grad, mol=None, dm=None, hermi=0, with_j=True, with_k=True,
            decompose_j2c='CD', lindep=df_rhf_grad.LINEAR_DEP_THRESHOLD):
    if mol is None: mol = mf_grad.mol
    gpu = resolve_use_gpu (mol)
    use_gpu = (gpu is not None and with_k and not getattr (mf_grad.base, 'only_dfj', False))
    if use_gpu:
        if dm is None: dm = mf_grad.base.make_rdm1 ()
        use_gpu = numpy.asarray (dm).ndim == 2 and not numpy.iscomplexobj (dm)
    if not use_gpu:
        return _cpu_get_jk (mf_grad, mol=mol, dm=dm, hermi=hermi, with_j=with_j, with_k=with_k,
                            decompose_j2c=decompose_j2c, lindep=lindep)

    t0 = (logger.process_clock (), logger.perf_counter ())
    auxmol = df_rhf_grad._make_auxmol (mf_grad.base.with_df, mol)
    nbas, nao, naux = mol.nbas, mol.nao, auxmol.nao
    ao_loc, aux_loc = mol.ao_loc, auxmol.ao_loc
    npair = nao * (nao + 1) // 2
    aux_response = bool (mf_grad.auxbasis_response)
    dms = numpy.asarray (dm)

    if hermi == 1:
        orbol, orbor = df_rhf_grad._decompose_rdm1 (mf_grad, mol, dm)
    else:
        orbol, orbor = df_rhf_grad._decompose_rdm1_svd (mf_grad, mol, dm)
    orbol = numpy.ascontiguousarray (orbol[0])
    orbor = numpy.ascontiguousarray (orbor[0])
    nocc = orbol.shape[1]

    metric_cd, metric, solve_j2c = _metric (auxmol, decompose_j2c, lindep)

    # rhoj: (P|mn) (orbor orbol^T)_mn, as pyscf; passed transposed (column-major on the device)
    dm_cm = numpy.ascontiguousarray (orbol @ orbor.T)
    dm_tril = numpy.zeros (1)
    if with_j and aux_response:
        idx = numpy.arange (nao)
        dm_tril = lib.pack_tril (dms + dms.T)
        dm_tril[idx*(idx+1)//2+idx] *= .5

    libgpu.init_dfgrad (gpu, orbol, orbor, dm_cm, dm_tril, metric, nao, naux, nocc,
                        int (with_j), int (aux_response), int (metric_cd))
    metric = dm_cm = None
    cap = max_block_mb * 1e6
    t1 = logger.timer_debug1 (mf_grad, 'gpu4mrh df grad init', *t0)

    # A: (P|mn) by AO blocks -> rhoj (unsolved), rhok (host), rhok_oo
    get_int3c_s1 = df_rhf_grad._int3c_wrapper (mol, auxmol, 'int3c2e', 's1')
    ranges = _ranges (ao_loc, 8 * (nao*naux + 2*naux*nocc), max (nao, nocc) * naux, cap)
    def intor_a (rng):
        shl0, shl1 = rng
        return rng, get_int3c_s1 ((0, nbas, shl0, shl1, 0, auxmol.nbas))
    rhok = numpy.empty ((naux, nocc, nao))
    for block, ((shl0, shl1), int3c) in enumerate (lib.map_with_prefetch (intor_a, ranges)):
        p0, p1 = ao_loc[shl0], ao_loc[shl1]
        buf = numpy.empty ((naux, nocc, p1-p0))
        libgpu.compute_dfgrad_rhok (gpu, int3c.reshape (-1, order='F'), buf, p0, p1-p0, block)
        int3c = None
        rhok[:,:,p0:p1] = buf
    buf = None
    rhoj = numpy.zeros (naux)
    rhok_oo = numpy.zeros ((naux, nocc, nocc)) if aux_response else numpy.zeros (1)
    libgpu.pull_dfgrad_rho (gpu, rhoj, rhok_oo)
    if _check and aux_response:
        ref = lib.dot (rhok.reshape (-1, nao), orbol).reshape (naux, nocc, nocc)
        err = abs (rhok_oo - ref).max (axis=(1,2))
        logger.note (mf_grad, 'GPU4MRH_DFGRAD_CHECK rhok_oo: max|dev-host| %.3e  max|host| %.3e  bad P %d / %d',
                     err.max (), abs (ref).max (), (err > 1e-8).sum (), naux)
        ref = err = None
    if with_j:
        rhoj = solve_j2c (rhoj[:,None])[:,0]
        libgpu.set_dfgrad_rhoj (gpu, numpy.ascontiguousarray (rhoj))
    t1 = logger.timer_debug1 (mf_grad, 'gpu4mrh df grad rhoj, rhok', *t1)

    # B: (d_x m n|P) by aux blocks -> vj, vk
    get_int3c_ip1 = df_rhf_grad._int3c_wrapper (mol, auxmol, 'int3c2e_ip1', 's1')
    ranges = _ranges (aux_loc, 8 * (3*nao*nao + 2*nocc*nao), 3*nao*nao, cap)
    def intor_b (rng):
        shl0, shl1 = rng
        return rng, get_int3c_ip1 ((0, nbas, 0, nbas, shl0, shl1))
    for block, ((shl0, shl1), int3c) in enumerate (lib.map_with_prefetch (intor_b, ranges)):
        p0, p1 = aux_loc[shl0], aux_loc[shl1]
        int3c = int3c.transpose (0,3,2,1) # memory order [x][P][j][i]
        assert int3c.flags.c_contiguous
        libgpu.compute_dfgrad_ip1 (gpu, int3c.reshape (-1), rhok[p0:p1], p0, p1-p0, block)
        int3c = None
    rhok = None
    t1 = logger.timer_debug1 (mf_grad, 'gpu4mrh df grad vj, vk', *t1)

    # C: (m n|d_x P) by aux blocks -> drhoj, vkaux
    vkaux_ref = numpy.zeros ((3, naux)) if (_check and aux_response) else None
    blocks_c = []
    if aux_response:
        get_int3c_ip2 = df_rhf_grad._int3c_wrapper (mol, auxmol, 'int3c2e_ip2', 's2ij')
        ranges = _ranges (aux_loc, 8 * (4*npair + nao*nao + nocc*nao), 3*npair + nao*nao, cap)
        def intor_c (rng):
            shl0, shl1 = rng
            return rng, get_int3c_ip2 ((0, nbas, 0, nbas, shl0, shl1))
        for block, ((shl0, shl1), int3c) in enumerate (lib.map_with_prefetch (intor_c, ranges)):
            p0, p1 = aux_loc[shl0], aux_loc[shl1]
            blocks_c.append ((p0, p1))
            if vkaux_ref is not None:
                vkaux_ref[:,p0:p1] = _vkaux_ref (int3c, orbol, orbor, rhok_oo[p0:p1], nao)
            int3c = int3c.transpose (0,2,1) # memory order [x][P][pair]
            assert int3c.flags.c_contiguous
            libgpu.compute_dfgrad_ip2 (gpu, int3c.reshape (-1), rhok_oo[p0:p1], p0, p1-p0, block)
            int3c = None
        t1 = logger.timer_debug1 (mf_grad, 'gpu4mrh df grad aux', *t1)

    vj = numpy.zeros ((3, nao, nao)) if with_j else numpy.zeros (1)
    vk = numpy.zeros ((3, nao, nao))
    drhoj = numpy.zeros ((3, naux)) if (with_j and aux_response) else numpy.zeros (1)
    vkaux = numpy.zeros ((3, naux)) if aux_response else numpy.zeros (1)
    libgpu.pull_dfgrad (gpu, vj, vk, drhoj, vkaux)
    libgpu.free_dfgrad (gpu)
    if vkaux_ref is not None:
        err = abs (vkaux - vkaux_ref).max (axis=0)
        logger.note (mf_grad, 'GPU4MRH_DFGRAD_CHECK vkaux (ij|d_x P) part: max|dev-host| %.3e  max|host| %.3e  bad P %d / %d',
                     err.max (), abs (vkaux_ref).max (), (err > 1e-8).sum (), naux)
        for block, (p0, p1) in enumerate (blocks_c):
            e = err[p0:p1].max ()
            if e > 1e-8:
                logger.note (mf_grad, '    block %d (device %d) P [%d, %d) nP %d: max err %.3e',
                             block, block % libgpu.get_num_devices (gpu), p0, p1, p1-p0, e)
        err = vkaux_ref = None
    vk = vk.transpose (0,2,1) # column-major on the device
    if with_j: vj = vj.transpose (0,2,1)

    if not aux_response:
        logger.timer (mf_grad, 'gpu4mrh df grad vj and vk', *t0)
        return (-vj if with_j else None), -vk

    # (d/dX P|Q) and per-atom sums, as pyscf
    int2c_e1 = auxmol.intor ('int2c2e_ip1')
    tmp = lib.einsum ('pij,qji->pq', rhok_oo, rhok_oo)
    vkaux -= lib.einsum ('xpq,pq->xp', int2c_e1, tmp)
    tmp = rhok_oo = None
    auxslices = auxmol.aoslice_by_atom ()[:,2:]
    def per_atom (v):
        v = numpy.array ([-v[:,p0:p1].sum (axis=1) for p0, p1 in auxslices])
        return v.reshape (1, 1, -1, 3)
    vk = lib.tag_array (-vk, aux=per_atom (vkaux))
    if with_j:
        vjaux = drhoj * rhoj
        vjaux -= lib.einsum ('xpq,p,q->xp', int2c_e1, rhoj, rhoj)
        vj = lib.tag_array (-vj, aux=per_atom (vjaux))
    logger.timer (mf_grad, 'gpu4mrh df grad vj and vk', *t0)
    if with_j: return vj, vk
    else: return None, vk
