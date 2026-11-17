'''GPU version of pyscf.df.grad.rhf.get_jk (density-fitted J/K nuclear-gradient contractions).

The 3-center integrals stay on the CPU (libcint), computed one block ahead of the device work
(lib.map_with_prefetch) and pushed in their native memory order; the contractions run on the
devices (DeviceDFGrad), blocks assigned round-robin:

  A  AO blocks of (P|mn): rhoj (unsolved), rhok = J^-1 (P|mn) orbor and rhok_oo; rhok is
     kept in host memory (pyscf writes it to an HDF5 temporary file)
  B  aux blocks of (d_x m n|P): vj, vk
  C  aux blocks of (m n|d_x P) (auxbasis response): drhoj and vkaux
The metric solve is a GEMM with the dense J^-1 from pyscf's own metric solver (cho_solve with
the identity for CD; the pseudo-inverse pyscf already uses for ED). The (d_x P|Q) terms and the
per-atom sums stay on the host as in pyscf.

Handled: one real density matrix, with_k (any with_j), either auxbasis_response. Falls back to
the CPU get_jk otherwise (J only, several density matrices, complex, only_dfj).
'''
import numpy
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

def _ranges (loc, nbytes, nelem, cap_bytes):
    '''balance_partition shell ranges of at most cap_bytes per block, given the device bytes and
    the elements of the largest (int-indexed) array per basis function of the block.'''
    blk = int (min (cap_bytes // nbytes, _INT_MAX // nelem))
    blk = max (blk, int (max (loc[1:] - loc[:-1])))
    return [(shl0, shl1) for shl0, shl1, _ in balance_partition (loc, blk)]

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

    # J^-1 from pyscf's metric solver; the device applies it as a GEMM
    int2c = auxmol.intor ('int2c2e', aosym='s1')
    solve_j2c = df_rhf_grad._gen_metric_solver (int2c, decompose_j2c, lindep)
    int2c = None
    jinv = numpy.ascontiguousarray (solve_j2c (numpy.eye (naux)))

    # rhoj: (P|mn) (orbor orbol^T)_mn, as pyscf; passed transposed (column-major on the device)
    dm_cm = numpy.ascontiguousarray (orbol @ orbor.T)
    dm_tril = numpy.zeros (1)
    if with_j and aux_response:
        idx = numpy.arange (nao)
        dm_tril = lib.pack_tril (dms + dms.T)
        dm_tril[idx*(idx+1)//2+idx] *= .5

    libgpu.init_dfgrad (gpu, orbol, orbor, dm_cm, dm_tril, jinv, nao, naux, nocc,
                        int (with_j), int (aux_response))
    jinv = dm_cm = None
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
    if aux_response:
        get_int3c_ip2 = df_rhf_grad._int3c_wrapper (mol, auxmol, 'int3c2e_ip2', 's2ij')
        ranges = _ranges (aux_loc, 8 * (4*npair + nao*nao + nocc*nao), 3*npair + nao*nao, cap)
        def intor_c (rng):
            shl0, shl1 = rng
            return rng, get_int3c_ip2 ((0, nbas, 0, nbas, shl0, shl1))
        for block, ((shl0, shl1), int3c) in enumerate (lib.map_with_prefetch (intor_c, ranges)):
            p0, p1 = aux_loc[shl0], aux_loc[shl1]
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
