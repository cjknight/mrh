import os
import tempfile
import unittest
from unittest import mock
import numpy
from pyscf import gto, dft, df, lib
from pyscf.df import incore, addons

# gpu4mrh's DF.build (gpu4mrh.df.df_build: in-memory DF tensor, metric solve on the device) against
# pyscf's: Cholesky and eigenvalue-projection metrics, many blocks, the range-separated (omega)
# tensor, the DF-RKS energy, builds pyscf writes to disk (left to pyscf), and a geometry change.

ATOM = "O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59"
ATOM2 = "O 0 0 0.02; H 0 0.78 0.58; H 0 -0.75 0.61"
BASIS, AUXBASIS = 'cc-pvtz', 'cc-pvtz-jkfit'
BLOCK_MB = 0.5 # tiny blocks: many blocks per device

def _mol (atom=ATOM, use_gpu=None):
    kwargs = {} if use_gpu is None else {'use_gpu': use_gpu}
    return gto.M(atom=atom, basis=BASIS, verbose=0, **kwargs)

def _mf (mol):
    mf = dft.RKS(mol).density_fit(auxbasis=AUXBASIS)
    mf.xc = 'wb97m-v'
    mf.nlc = False # no VV10 (and no pyscf-dispersion)
    mf.grids.level = 2
    mf.conv_tol = 1e-11
    return mf

def setUpModule():
    global e_ref
    assert getattr(lib.param, 'use_gpu', None) is None
    e_ref = _mf(_mol()).kernel()

def tearDownModule():
    global e_ref
    del e_ref

class KnownValues (unittest.TestCase):

    @classmethod
    def setUpClass (cls):
        global libgpu, df_build
        from mrh.my_pyscf.gpu import libgpu
        from gpu4mrh import patch_pyscf
        from gpu4mrh.df import df_build
        cls.gpu = libgpu.init()

    @classmethod
    def tearDownClass (cls):
        libgpu.destroy_device(cls.gpu)

    def _calls (self):
        return mock.patch.object(libgpu, 'compute_dfbuild', wraps=libgpu.compute_dfbuild)

    def test_cholesky_eri_matches_incore (self):
        mol = _mol()
        auxmol = addons.make_auxmol(mol, AUXBASIS)
        with mock.patch.object(df_build, 'block_mb', BLOCK_MB), self._calls() as calls:
            for dec in ('cd', 'eig'):
                with self.subTest(decompose_j2c=dec):
                    ref = incore.cholesky_eri(mol, auxmol=auxmol, decompose_j2c=dec)
                    n0 = calls.call_count
                    out = df_build.cholesky_eri(mol, auxmol, self.gpu, decompose_j2c=dec, max_memory=4000)
                    self.assertGreaterEqual(calls.call_count - n0, 2 * libgpu.get_num_devices(self.gpu))
                    self.assertEqual(out.shape, ref.shape)
                    self.assertTrue(out.flags.c_contiguous)
                    self.assertAlmostEqual(abs(out - ref).max(), 0, 12)

    def test_df_build_matches_cpu (self):
        d0 = df.DF(_mol(), auxbasis=AUXBASIS).build()
        d1 = df.DF(_mol(use_gpu=self.gpu), auxbasis=AUXBASIS)
        with mock.patch.object(df_build, 'block_mb', BLOCK_MB), self._calls() as calls:
            d1.build()
            self.assertGreater(calls.call_count, 0)
            self.assertIsInstance(d1._cderi, numpy.ndarray)
            self.assertAlmostEqual(abs(d1._cderi - d0._cderi).max(), 0, 12)
            # range-separated tensor (eigenvalue projection: the omega metric is singular here)
            n0 = calls.call_count
            with d1.range_coulomb(0.3) as r1, d0.range_coulomb(0.3) as r0:
                r1.build()
                r0.build()
                self.assertGreater(calls.call_count, n0)
                self.assertEqual(r1._cderi.shape, r0._cderi.shape)
                self.assertAlmostEqual(abs(r1._cderi - r0._cderi).max(), 0, 12)

    def test_projected_metric_fits (self):
        # omega tensor: 139 auxiliary functions, 83 kept by the eigenvalue projection. pyscf's memory
        # test counts 139 (1.90 MB, on disk at max_memory 1.6 MB); in memory it is 83 (1.14 MB).
        d0 = df.DF(_mol(), auxbasis=AUXBASIS)
        d1 = df.DF(_mol(use_gpu=self.gpu), auxbasis=AUXBASIS)
        with self._calls() as calls, d1.range_coulomb(0.3) as r1, d0.range_coulomb(0.3) as r0:
            r0.build()
            r1.max_memory = 1.6
            with mock.patch.object(lib, 'current_memory', return_value=(0., 0.)):
                r1.build()
            self.assertGreater(calls.call_count, 0)
            self.assertIsInstance(r1._cderi, numpy.ndarray)
            self.assertEqual(r1._cderi.shape, (83, 1711))
            self.assertAlmostEqual(abs(r1._cderi - r0._cderi).max(), 0, 12)

    def test_rks_energy (self):
        mf = _mf(_mol(use_gpu=self.gpu))
        with mock.patch.object(df_build, 'block_mb', BLOCK_MB), self._calls() as calls:
            e = mf.kernel()
        self.assertGreaterEqual(calls.call_count, 2) # full and omega tensors
        self.assertAlmostEqual(e, e_ref, 9)

    def test_disk_builds_stay_cpu (self):
        with self._calls() as calls:
            with tempfile.TemporaryDirectory() as tmp:
                d = df.DF(_mol(use_gpu=self.gpu), auxbasis=AUXBASIS)
                d._cderi_to_save = os.path.join(tmp, 'cderi.h5')
                d.build()
                self.assertIsInstance(d._cderi, str)
            d = df.DF(_mol(use_gpu=self.gpu), auxbasis=AUXBASIS)
            d.max_memory = lib.current_memory()[0] + 1 # tensor does not fit: pyscf writes a file
            d.build()
            self.assertNotIsInstance(d._cderi, numpy.ndarray)
        self.assertEqual(calls.call_count, 0)

    def test_geometry_change (self):
        d = df.DF(_mol(use_gpu=self.gpu), auxbasis=AUXBASIS).build()
        mol2 = _mol(ATOM2, use_gpu=self.gpu)
        with self._calls() as calls:
            d.reset(mol2)
            d.build()
        self.assertGreater(calls.call_count, 0)
        ref = df.DF(_mol(ATOM2), auxbasis=AUXBASIS).build()._cderi
        self.assertAlmostEqual(abs(d._cderi - ref).max(), 0, 12)

if __name__ == "__main__":
    print("Full Tests for gpu4mrh DF tensor build (df.df.DF.build)")
    unittest.main()
