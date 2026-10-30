import gc
import unittest
import numpy
from pyscf import gto, dft, lib

# A geometry step through the GPU DF J/K: SCF at G0, then the gradient scanner at G1 (what geomeTRIC does),
# against a fresh CPU calculation at G1. libgpu caches the 3-center integrals by id(dfobj); the scanner keeps the
# DF object, so without a reload exact exchange keeps the G0 integrals (|dE| ~ 2.5 Ha after one step).

G0 = "O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59"
G1 = "O 0 0 0.05; H 0 0.80 0.55; H 0 -0.72 0.63"
CASES = [("pbe", 0), ("pbe0", 0), ("wb97m-v", 0), ("pbe0", 1)]   # (xc, spin); spin 1 = ROKS cation

def _make (atom, xc, spin, basis='cc-pvdz', use_gpu=None):
    # the device handle travels on the Mole (never lib.param.use_gpu); the CPU references are built
    # before gpu4mrh patches gto.M, so the keyword is only passed when there is a handle
    kwargs = {} if use_gpu is None else {'use_gpu': use_gpu}
    mol = gto.M(atom=atom, basis=basis, charge=spin, spin=spin, verbose=0, **kwargs)
    mf = (dft.RKS(mol) if spin == 0 else dft.ROKS(mol)).density_fit()
    mf.xc = xc
    mf.conv_tol = 1e-11
    mf.grids.level = 2
    mf.nlcgrids.level = 1
    return mf

def _reference (xc, spin):
    mf = _make(G1, xc, spin)
    e = mf.kernel()
    return e, mf.nuc_grad_method().kernel()

def _scanned (xc, spin, use_gpu):
    mf = _make(G0, xc, spin, use_gpu=use_gpu)
    mf.kernel()
    scan = mf.nuc_grad_method().as_scanner()
    mol1 = mf.mol.set_geom_(gto.M(atom=G1, unit="Angstrom").atom_coords(), unit="Bohr", inplace=False)
    return scan(mol1)

def _cached_blocks (gpu, dfobj_addr):
    # number of device ERI blocks cached under this DF object address, -1 if none
    from mrh.my_pyscf.gpu import libgpu
    arg = numpy.array([-1, -1, -1, -1], dtype=numpy.int32)
    libgpu.get_dfobj_status(gpu, dfobj_addr, arg)
    return int(arg[2])

def setUpModule():
    global refs, ref_tz
    assert getattr(lib.param, 'use_gpu', None) is None
    refs = {c: _reference(*c) for c in CASES}
    ref_tz = _make(G1, 'pbe0', 0, basis='cc-pvtz').kernel()

def tearDownModule():
    global refs, ref_tz
    del refs, ref_tz

class KnownValues (unittest.TestCase):

    @classmethod
    def setUpClass (cls):
        from mrh.my_pyscf.gpu import libgpu
        from gpu4mrh import patch_pyscf
        cls.libgpu = libgpu
        cls.gpu = libgpu.init()

    @classmethod
    def tearDownClass (cls):
        cls.libgpu.destroy_device(cls.gpu)

    def test_scanner_step (self):
        for c in CASES:
            e_ref, g_ref = refs[c]
            e, g = _scanned(*c, self.gpu)
            with self.subTest (xc=c[0], spin=c[1]):
                self.assertAlmostEqual (e, e_ref, 9)
                self.assertLess (numpy.abs(g - g_ref).max(), 1e-6)

    def test_same_dfobj_new_basis (self):
        # cc-pVDZ -> cc-pVTZ on the same DF object (same id): the cached blocks change shape
        mf = _make(G0, 'pbe0', 0, use_gpu=self.gpu)
        mf.kernel()
        mf_tz = _make(G1, 'pbe0', 0, basis='cc-pvtz', use_gpu=self.gpu)
        mf_tz.with_df = mf.with_df
        mf_tz.with_df.reset(mf_tz.mol)
        self.assertAlmostEqual (mf_tz.kernel(), ref_tz, 9)

    def test_freed_dfobj_evicted (self):
        # the cache is keyed by id(dfobj): a DF object that is garbage collected must not leave blocks that a
        # later DF object at the same address would read (evictions are applied on the next get_jk)
        mf = _make(G0, 'pbe0', 0, use_gpu=self.gpu)
        mf.kernel()
        addr = id(mf.with_df)
        self.assertEqual (_cached_blocks(self.gpu, addr), 1)
        del mf
        gc.collect()
        mf1 = _make(G1, 'pbe0', 0, use_gpu=self.gpu)
        self.assertAlmostEqual (mf1.kernel(), refs[('pbe0', 0)][0], 9)
        if id(mf1.with_df) != addr:
            self.assertEqual (_cached_blocks(self.gpu, addr), -1)

    def test_rsh_scanner_evicts (self):
        # DF.reset replaces the range-separated sub-DF objects on every geometry step; their blocks must not pile up
        mf = _make(G0, 'wb97m-v', 0, use_gpu=self.gpu)
        mf.kernel()
        old = [id(x) for x in mf.with_df._rsh_df.values()]
        self.assertTrue (old)
        scan = mf.nuc_grad_method().as_scanner()
        mol1 = mf.mol.set_geom_(gto.M(atom=G1, unit="Angstrom").atom_coords(), unit="Bohr", inplace=False)
        e, g = scan(mol1)
        self.assertAlmostEqual (e, refs[('wb97m-v', 0)][0], 9)
        new = [id(x) for x in scan.base.with_df._rsh_df.values()]
        for addr in set(old) - set(new):
            self.assertEqual (_cached_blocks(self.gpu, addr), -1)

if __name__ == "__main__":
    print("Tests for GPU DF J/K after a geometry change")
    unittest.main()

