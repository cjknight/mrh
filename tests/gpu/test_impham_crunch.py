'''
Tests for the GPU path of the impurity Hamiltonian in asynchronous LASSCF
(ImpuritySCF._update_impham_1_ in my_pyscf/mcscf/lasscf_async/crunch.py).

Unlike test_impham_cderi.py, which drives the libgpu kernels through its own copy of the
Python loop, these tests go through crunch.py itself, so they also cover
  - that the transformed integrals are actually installed for the impurity solver: in
    with_df._cderi for the density-fitted fragment impurities, and in _cderi/_eri for the
    conventional-ERI pair impurities built in combine.py (an earlier version of this path
    sent the pair impurities through the 4-center kernel, whose output buffer was never sized)
  - the fallback to the CPU path when the device DF-block cache cannot be used
'''
import unittest
import numpy as np
from pyscf import gto, scf, lib
from pyscf.df import DF
from pyscf.mcscf import avas
from mrh.tests.gpu.geometry_generator import generator
from mrh.my_pyscf.gpu import libgpu

NFRAGS = 2

def setUpModule():
    global gpu, mf, mo_guess, frags, crunch
    from gpu4mrh import patch_pyscf
    from mrh.my_pyscf.mcscf.lasscf_async import crunch
    gpu = libgpu.init ()
    lib.param.use_gpu = gpu # so that the SCF get_jk caches the full-system DF blocks on device
    mol = gto.M (atom=generator (NFRAGS), basis='6-31g', verbose=0, output='/dev/null')
    mf = scf.RHF (mol).density_fit ().run ()
    ncas, nelecas, mo_guess = avas.kernel (mf, ['C 2pz'], verbose=0)
    frags = [list (range (1+4*i, 3+4*i)) for i in range (NFRAGS)]

def tearDownModule():
    global gpu, mf, mo_guess, frags, crunch
    libgpu.destroy_device (gpu)
    lib.param.use_gpu = None
    del gpu, mf, mo_guess, frags, crunch

def _run_async (use_gpu):
    from mrh.my_pyscf.mcscf.lasscf_async import LASSCF
    las = LASSCF (mf, [2]*NFRAGS, [2]*NFRAGS, use_gpu=use_gpu)
    las.kernel (las.set_fragments_(frags, mo_guess))
    return las

class KnownValues (unittest.TestCase):

    def test_installed_cderi (self):
        '''Every impurity update on the GPU path takes the GPU route (no silent fallback) and
        installs the same integrals as the CPU transform: in with_df._cderi for the DF fragment
        impurities, and in _cderi/_eri for the conventional-ERI pair impurities of combine.py.'''
        stats = {'gpu': 0, 'fallback': 0, 'checked': 0, 'checked_df': 0, 'checked_eri': 0,
                 'maxdiff': 0.0}
        get_gpu = crunch.ImpuritySCF._get_imporb_cderi_gpu
        update1 = crunch.ImpuritySCF._update_impham_1_
        def counting_get_gpu (self, *args, **kwargs):
            cderi = get_gpu (self, *args, **kwargs)
            stats['fallback' if cderi is None else 'gpu'] += 1
            return cderi
        def checking_update1 (self, *args, **kwargs):
            out = update1 (self, *args, **kwargs)
            full_df = self.mol._las._scf.with_df
            ref = self._get_imporb_cderi_cpu (full_df, self.mol.get_imporb_coeff ())
            if getattr (self, 'with_df', None) is not None: # DF fragment impurity
                diff = np.amax (np.abs (np.asarray (self.with_df._cderi) - ref))
                stats['checked_df'] += 1
            else: # conventional-ERI pair impurity (combine.py)
                diff = max (np.amax (np.abs (self._cderi - ref)),
                            np.amax (np.abs (self._eri - ref.T @ ref)))
                stats['checked_eri'] += 1
            stats['checked'] += 1
            stats['maxdiff'] = max (stats['maxdiff'], diff)
            return out
        crunch.ImpuritySCF._get_imporb_cderi_gpu = counting_get_gpu
        crunch.ImpuritySCF._update_impham_1_ = checking_update1
        try:
            las = _run_async (gpu)
        finally:
            crunch.ImpuritySCF._get_imporb_cderi_gpu = get_gpu
            crunch.ImpuritySCF._update_impham_1_ = update1
        self.assertTrue (las.converged)
        with self.subTest ('GPU path taken'):
            self.assertGreater (stats['gpu'], 0)
            self.assertEqual (stats['fallback'], 0)
        with self.subTest ('installed cderi == CPU transform'):
            self.assertEqual (stats['checked'], stats['gpu'])
            # both impurity kinds must be exercised for this test to mean anything
            self.assertGreater (stats['checked_df'], 0)
            self.assertGreater (stats['checked_eri'], 0)
            self.assertLess (stats['maxdiff'], 1e-10)

    def test_energy_vs_cpu (self):
        las_cpu = _run_async (None)
        las_gpu = _run_async (gpu)
        self.assertTrue (las_cpu.converged)
        self.assertTrue (las_gpu.converged)
        self.assertAlmostEqual (las_gpu.e_tot, las_cpu.e_tot, 8)

    def test_direct (self):
        '''GPU transform of the cached full-system blocks into an arbitrary (non-orthogonal,
        non-square) orbital basis matches the CPU transform.'''
        c = np.random.default_rng (0).random ((mf.mol.nao, 7))
        ref = crunch.ImpuritySCF._get_imporb_cderi_cpu (None, mf.with_df, c)
        cderi = crunch.ImpuritySCF._get_imporb_cderi_gpu (None, gpu, mf.with_df, c)
        self.assertIsNotNone (cderi)
        self.assertEqual (cderi.shape, ref.shape)
        self.assertLess (np.amax (np.abs (cderi - ref)), 1e-10)

    def test_fallback_uncached (self):
        '''A DF object whose blocks were never cached on the device -> CPU fallback (None),
        rather than reading unpopulated device memory.'''
        with_df = DF (mf.mol, auxbasis=mf.with_df.auxbasis).build ()
        c = np.random.default_rng (1).random ((mf.mol.nao, 5))
        self.assertIsNone (crunch.ImpuritySCF._get_imporb_cderi_gpu (None, gpu, with_df, c))

    def test_fallback_update_pending (self):
        '''A pending device-cache update -> CPU fallback (None). Otherwise the device would
        try to refresh the cached blocks from a null host pointer.'''
        c = np.random.default_rng (2).random ((mf.mol.nao, 5))
        libgpu.set_update_dfobj_(gpu, 1)
        try:
            cderi = crunch.ImpuritySCF._get_imporb_cderi_gpu (None, gpu, mf.with_df, c)
        finally:
            libgpu.set_update_dfobj_(gpu, 0)
        self.assertIsNone (cderi)

if __name__ == "__main__":
    print ("Tests for the GPU impurity Hamiltonian in asynchronous LASSCF")
    unittest.main ()
