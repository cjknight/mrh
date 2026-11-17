import unittest
from unittest import mock
from collections import Counter
import numpy
from pyscf import gto, dft, lib
from pyscf.df.df import DF

# gpu4mrh's DF get_jk reads the host DF blocks (DF.loop) only when the device cache cannot supply all of
# them, so each DF object is read once per SCF instead of once per get_jk. Range-separated hybrid: the
# full-range and the omega DF objects are both cached. Small blockdim so there are several blocks.

ATOM = "O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59"
BLOCKDIM = 20
OMEGA = 0.33

def _make (use_gpu=None):
    kwargs = {} if use_gpu is None else {'use_gpu': use_gpu}
    mol = gto.M(atom=ATOM, basis='cc-pvdz', verbose=0, **kwargs)
    mf = dft.RKS(mol).density_fit()
    mf.with_df.blockdim = BLOCKDIM
    mf.xc = 'cam-b3lyp'
    mf.grids.level = 2
    mf.conv_tol = 1e-11
    return mf

def setUpModule():
    global e_ref, dm_ref, vj_ref, vk_ref, vklr_ref
    assert getattr(lib.param, 'use_gpu', None) is None
    mf = _make()
    e_ref = mf.kernel()
    dm_ref = mf.make_rdm1()
    vj_ref = mf.with_df.get_jk(dm_ref, with_k=False)[0]
    vk_ref = mf.with_df.get_jk(dm_ref, with_j=False)[1]
    vklr_ref = mf.with_df.get_jk(dm_ref, with_j=False, omega=OMEGA)[1]

def tearDownModule():
    global e_ref, dm_ref, vj_ref, vk_ref, vklr_ref
    del e_ref, dm_ref, vj_ref, vk_ref, vklr_ref

class KnownValues (unittest.TestCase):

    @classmethod
    def setUpClass (cls):
        global libgpu, df_jk
        from mrh.my_pyscf.gpu import libgpu
        from gpu4mrh import patch_pyscf
        from gpu4mrh.df import df_jk
        cls.gpu = libgpu.init()

    @classmethod
    def tearDownClass (cls):
        libgpu.destroy_device(cls.gpu)

    def _scf_counting_loops (self, skip):
        calls = Counter()
        cpu_loop = DF.loop
        def loop (dfobj, blksize=None):
            calls[id(dfobj)] += 1
            return cpu_loop(dfobj, blksize)
        mf = _make(self.gpu)
        with mock.patch.object(DF, 'loop', loop), mock.patch.object(df_jk, 'skip_cached_loop', skip):
            e = mf.kernel()
        return mf, e, calls

    def test_scf_reads_host_blocks_once (self):
        mf, e, calls = self._scf_counting_loops(True)
        self.assertAlmostEqual(e, e_ref, 9)
        self.assertEqual(len(calls), 2) # full-range and omega DF objects
        self.assertEqual(set(calls.values()), {1}, calls)

    def test_skip_matches_streaming (self):
        mf, e_stream, calls = self._scf_counting_loops(False)
        self.assertGreater(min(calls.values()), 1)
        mf, e_skip, calls = self._scf_counting_loops(True)
        self.assertAlmostEqual(e_skip, e_stream, 12)

    def test_j_only_from_cache (self):
        mf = _make(self.gpu)
        mf.kernel()
        vj = mf.with_df.get_jk(dm_ref, with_k=False)[0]
        self.assertAlmostEqual(abs(vj - vj_ref).max(), 0, 9)

    def test_k_only (self):
        # with_j=False: libgpu skips J; vj is 0 as in pyscf.df.df_jk.get_jk. Twice: host blocks, then cache.
        mf = _make(self.gpu)
        for it in range(2):
            vj, vk = mf.with_df.get_jk(dm_ref, with_j=False)
            vjlr, vklr = mf.with_df.get_jk(dm_ref, with_j=False, omega=OMEGA)
            with self.subTest(it=it):
                self.assertIs(vj, 0)
                self.assertIs(vjlr, 0)
                self.assertAlmostEqual(abs(vk - vk_ref).max(), 0, 9)
                self.assertAlmostEqual(abs(vklr - vklr_ref).max(), 0, 9)

    def test_other_partition_not_trusted (self):
        mf = _make(self.gpu)
        mf.kernel()
        dfobj, nao = mf.with_df, mf.mol.nao
        naux = df_jk._cached_block_naux(self.gpu, dfobj, BLOCKDIM, nao)
        self.assertIsNotNone(naux)
        self.assertEqual(sum(naux), dfobj.get_naoaux())
        self.assertIsNone(df_jk._cached_block_naux(self.gpu, dfobj, BLOCKDIM+1, nao))

if __name__ == "__main__":
    print("Full Tests for gpu4mrh DF get_jk skipping the host DF loop on device-cache hits")
    unittest.main()
