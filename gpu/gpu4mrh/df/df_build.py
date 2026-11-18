'''DF.build with the metric solve on the device.

Replaces pyscf's in-memory build (DF.build -> pyscf.df.incore.cholesky_eri, taken when the tensor
fits in max_memory). The result is the same: an ordinary host ``_cderi`` (naux, nao_pair) array,
which every PySCF code path can use, and which the first get_jk copies into the device ERI cache.
Only the work changes:

  * libcint computes the 3-center integrals on the host, one block of s2-packed AO pairs at a
    time, in a prefetch thread (lib.map_with_prefetch), as pyscf's on-disk build does;
  * each block goes to a device (block % num_devices), which applies the metric (Cholesky:
    triangular solve; eigenvalue decomposition: the projection GEMM), and comes back into its
    columns of ``_cderi``.

pyscf's own in-memory loop runs these steps serially with single-threaded copies, so it is slower
than writing the tensor to disk (anouar, 66 atoms: 309 s in memory vs 142 s on disk; this build
23 s for the full tensor). Builds that do not fit in max_memory, or with a file requested in
``_cderi_to_save``, are left to pyscf unchanged (on disk). The memory test is pyscf's, except that
a metric with linear dependencies is counted with the functions its projection keeps (pyscf
counts all of them: anouar's omega tensor, 1957 of 5092).

The metric is factored on the host exactly as pyscf does (scipy Cholesky, else the eigenvalue
projection with pyscf's linear-dependency threshold), so the tensor matches pyscf's to rounding.
'''
import numpy
import scipy.linalg
from pyscf import gto, lib
from pyscf.lib import logger
from pyscf.df import addons, incore
from pyscf.df.outcore import _guess_shell_ranges

from mrh.my_pyscf.gpu import libgpu
from gpu4mrh.df.grad_rhf import prefetch_blocks

use_device = True # False: DF.build is pyscf's
block_mb = 2000.  # target size of one host block of integrals (MB)
blocks_per_device = 2 # at least this many blocks per device when the blocks are small
host_local = True # host tensor on the threads' local memory node (see cholesky_eri)

_INT_MAX = 2**31 - 1 # device buffers are sized in int elements

def _block_rows (npair, naoaux, naux, ndev, max_memory):
    '''Number of AO pairs per block: about block_mb of integrals, at least blocks_per_device
    blocks per device, within max_memory (two integral blocks in flight plus the staging of one
    result, besides the tensor itself) and within the device buffers' int sizes.'''
    rows = int (block_mb * 1e6 / 8 / naoaux)
    rows = min (rows, -(-npair // (blocks_per_device * ndev)))
    words = max_memory * 1e6 / 8 - naux * npair
    rows = min (rows, int (.9 * words / (2 * naoaux + naux)))
    rows = min (rows, _INT_MAX // max (naoaux, naux))
    return max (rows, 1)

def metric (mol, auxmol, int2c='int2c2e', decompose_j2c='cd', lindep=incore.LINEAR_DEP_THR):
    '''The metric factor as pyscf.df.incore.cholesky_eri: (low, decompose_j2c), low the Cholesky
    factor L (naoaux, naoaux), or if that fails the eigenvalue projection (naux, naoaux) that drops
    the linearly dependent functions.'''
    j2c = auxmol.intor (int2c, hermi=1)
    if decompose_j2c == 'eig':
        return incore._eig_decompose (mol, j2c, lindep), 'eig'
    try:
        return scipy.linalg.cholesky (j2c, lower=True), 'cd'
    except scipy.linalg.LinAlgError:
        return incore._eig_decompose (mol, j2c, lindep), 'eig'

def cholesky_eri (mol, auxmol, gpu, int3c='int3c2e', int2c='int2c2e', max_memory=2000,
                  decompose_j2c='cd', lindep=incore.LINEAR_DEP_THR, verbose=0, low=None):
    '''As pyscf.df.incore.cholesky_eri (aosym 's2ij', comp 1) with the metric applied on the device.
    low, decompose_j2c: the metric factor from metric(), if already computed.

    Returns:
        2D array of (naux, nao*(nao+1)/2) in C-contiguous
    '''
    t0 = (logger.process_clock (), logger.perf_counter ())
    log = logger.new_logger (mol, verbose)

    if low is None:
        low, decompose_j2c = metric (mol, auxmol, int2c, decompose_j2c, lindep)
    naux, naoaux = low.shape
    log.debug ('size of aux basis %d', naux)
    t1 = log.timer_debug1 ('gpu4mrh df build 2c2e', *t0)

    int3c = gto.moleintor.ascint3 (mol._add_suffix (int3c))
    atm, bas, env = gto.mole.conc_env (mol._atm, mol._bas, mol._env,
                                       auxmol._atm, auxmol._bas, auxmol._env)
    ao_loc = gto.moleintor.make_loc (bas, int3c)
    nao = int (ao_loc[mol.nbas])
    npair = nao * (nao+1) // 2
    assert int (ao_loc[-1]) - nao == naoaux

    ndev = libgpu.get_num_devices (gpu)
    rows = _block_rows (npair, naoaux, naux, ndev, max_memory)
    shranges = _guess_shell_ranges (mol, rows, 's2ij')
    log.debug ('gpu4mrh df build: %d blocks of up to %d AO pairs (%.1f MB) on %d devices',
               len (shranges), max (x[2] for x in shranges),
               max (x[2] for x in shranges) * naoaux * 8 / 1e6, ndev)
    cintopt = gto.moleintor.make_cintopt (atm, bas, env, int3c)

    cderi = numpy.empty ((naux, npair))
    if host_local:
        # Before the first touch: the host tensor is cold (the working copy is the device ERI
        # cache), so keep it on the threads' local memory node. Under numactl --preferred=<HBM>
        # it would otherwise fill the preferred fast memory first (Aurora: 61 of 64 GB), and every
        # later hot array would stall on failed huge-page compaction there before falling back.
        log.debug ('gpu4mrh df build: host tensor on memory node %d (-1: default placement)',
                   libgpu.host_mem_local (cderi))
    def intor (sh_range, out):
        bstart, bend, nrow = sh_range
        shls_slice = (bstart, bend, 0, mol.nbas, mol.nbas, mol.nbas+auxmol.nbas)
        return sh_range, gto.moleintor.getints3c (int3c, atm, bas, env, shls_slice, 1, 's2ij',
                                                  ao_loc, cintopt, out=out)
    size = max (x[2] for x in shranges) * naoaux

    libgpu.init_dfbuild (gpu, numpy.ascontiguousarray (low), naoaux, naux, npair,
                         int (decompose_j2c == 'cd'))
    low = None
    p1 = 0
    try:
        for block, ((bstart, bend, nrow), ints) in enumerate (prefetch_blocks (intor, shranges, size)):
            p0 = int (ao_loc[bstart]) * (int (ao_loc[bstart]) + 1) // 2
            assert p0 == p1 and ints.shape == (nrow, naoaux) and ints.flags.f_contiguous
            p1 = p0 + nrow
            libgpu.compute_dfbuild (gpu, ints.reshape (-1, order='F'), cderi, p0, nrow, block)
            ints = None
            log.debug1 ('gpu4mrh df build [%d/%d], AO [%d:%d], nrow = %d',
                        block+1, len (shranges), bstart, bend, nrow)
    finally:
        libgpu.free_dfbuild (gpu)
    assert p1 == npair
    log.timer ('gpu4mrh df build cholesky_eri', *t0)
    return cderi

def build (dfobj, gpu, cpu_build):
    '''DF.build for a DF object on the device: the in-memory build with the metric solve on the
    device; pyscf's build (cpu_build) for everything else.'''
    if not use_device: return cpu_build (dfobj)
    if dfobj._cderi is not None and dfobj.auxmol is None: return cpu_build (dfobj) # given tensor
    if isinstance (dfobj._cderi_to_save, str): return cpu_build (dfobj) # file requested
    mol = dfobj.mol
    auxmol = addons.make_auxmol (mol, dfobj.auxbasis)
    if mol.cart != auxmol.cart: return cpu_build (dfobj) # pyscf raises
    nao, naux = mol.nao_nr (), auxmol.nao_nr ()
    nao_pair = nao * (nao+1) // 2
    max_memory = dfobj.max_memory - lib.current_memory ()[0]
    low = decompose_j2c = None
    if not nao_pair * naux * 8 / 1e6 < .9 * max_memory:
        # pyscf's test counts all auxiliary functions; the eigenvalue projection of a singular
        # metric (e.g. omega-attenuated) keeps fewer, so its tensor may fit after all
        low, decompose_j2c = metric (mol, auxmol, mol._add_suffix ('int2c2e'))
        if not nao_pair * low.shape[0] * 8 / 1e6 < .9 * max_memory:
            return cpu_build (dfobj) # pyscf: on disk

    t0 = (logger.process_clock (), logger.perf_counter ())
    log = logger.Logger (dfobj.stdout, dfobj.verbose)
    dfobj.check_sanity ()
    dfobj.dump_flags ()
    dfobj.auxmol = auxmol
    dfobj._cderi = cholesky_eri (mol, auxmol, gpu, int3c=mol._add_suffix ('int3c2e'),
                                 int2c=mol._add_suffix ('int2c2e'), max_memory=max_memory,
                                 decompose_j2c=decompose_j2c or 'cd', verbose=log, low=low)
    log.timer ('gpu4mrh df build', *t0)
    return dfobj
