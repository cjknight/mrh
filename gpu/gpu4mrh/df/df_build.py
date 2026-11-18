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
than writing the tensor to disk (anouar, 66 atoms: 309 s in memory vs 142 s on disk). Builds that
pyscf would write to disk (tensor too large for max_memory, or a file requested in
``_cderi_to_save``) are left to pyscf unchanged.

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

use_device = True # False: DF.build is pyscf's
block_mb = 2000.  # target size of one host block of integrals (MB)
blocks_per_device = 2 # at least this many blocks per device when the blocks are small

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

def cholesky_eri (mol, auxmol, gpu, int3c='int3c2e', int2c='int2c2e', max_memory=2000,
                  decompose_j2c='cd', lindep=incore.LINEAR_DEP_THR, verbose=0):
    '''As pyscf.df.incore.cholesky_eri (aosym 's2ij', comp 1) with the metric applied on the device.

    Returns:
        2D array of (naux, nao*(nao+1)/2) in C-contiguous
    '''
    t0 = (logger.process_clock (), logger.perf_counter ())
    log = logger.new_logger (mol, verbose)

    j2c = auxmol.intor (int2c, hermi=1)
    if decompose_j2c == 'eig':
        low = incore._eig_decompose (mol, j2c, lindep)
    else:
        try:
            low = scipy.linalg.cholesky (j2c, lower=True)
            decompose_j2c = 'cd'
        except scipy.linalg.LinAlgError:
            low = incore._eig_decompose (mol, j2c, lindep)
            decompose_j2c = 'eig'
    j2c = None
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
    def intor (sh_range):
        bstart, bend, nrow = sh_range
        shls_slice = (bstart, bend, 0, mol.nbas, mol.nbas, mol.nbas+auxmol.nbas)
        return sh_range, gto.moleintor.getints3c (int3c, atm, bas, env, shls_slice, 1, 's2ij',
                                                  ao_loc, cintopt)

    libgpu.init_dfbuild (gpu, numpy.ascontiguousarray (low), naoaux, naux, npair,
                         int (decompose_j2c == 'cd'))
    low = None
    p1 = 0
    try:
        for block, ((bstart, bend, nrow), ints) in enumerate (lib.map_with_prefetch (intor, shranges)):
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
    if not nao_pair * naux * 8 / 1e6 < .9 * max_memory: return cpu_build (dfobj) # pyscf: on disk

    t0 = (logger.process_clock (), logger.perf_counter ())
    log = logger.Logger (dfobj.stdout, dfobj.verbose)
    dfobj.check_sanity ()
    dfobj.dump_flags ()
    dfobj.auxmol = auxmol
    dfobj._cderi = cholesky_eri (mol, auxmol, gpu, int3c=mol._add_suffix ('int3c2e'),
                                 int2c=mol._add_suffix ('int2c2e'), max_memory=max_memory,
                                 verbose=log)
    log.timer ('gpu4mrh df build', *t0)
    return dfobj
