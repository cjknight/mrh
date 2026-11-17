import unittest
from unittest import mock
import numpy
from pyscf import gto, dft, lib

# gpu4mrh's NumInt.nr_rks (XC energy and matrix on the device) against the CPU nr_rks, for LDA,
# GGA and meta-GGA (incl. range-separated hybrids), with many grid blocks; a full DF-RKS SCF; and
# the CPU fallback for density matrices without orbitals.

ATOM = "O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59"
XCS = ['lda,vwn', 'pbe', 'pbe0', 'scan', 'm06-l', 'wb97m-v', 'wb97m-d3bj']
BLOCK_MB = 2.0 # tiny device blocks: many grid blocks per call

def _mol (use_gpu=None):
    kwargs = {} if use_gpu is None else {'use_gpu': use_gpu}
    return gto.M(atom=ATOM, basis='cc-pvtz', verbose=0, **kwargs)

def _mf (mol, xc):
    mf = dft.RKS(mol).density_fit()
    mf.xc = xc
    mf.grids.level = 2
    mf.conv_tol = 1e-11
    return mf

def setUpModule():
    global refs, e_scf_ref
    assert getattr(lib.param, 'use_gpu', None) is None
    mol = _mol()
    refs = {}
    for xc in XCS:
        mf = _mf(mol, xc)
        mf.kernel()
        dm = mf.make_rdm1()
        refs[xc] = (dm, mf._numint.nr_rks(mol, mf.grids, xc, dm))
    e_scf_ref = _mf(mol, 'wb97m-d3bj').kernel()

def tearDownModule():
    global refs, e_scf_ref
    del refs, e_scf_ref

class KnownValues (unittest.TestCase):

    @classmethod
    def setUpClass (cls):
        global libgpu, gnum
        from mrh.my_pyscf.gpu import libgpu
        from gpu4mrh import patch_pyscf
        from gpu4mrh.dft import numint as gnum
        cls.gpu = libgpu.init()

    @classmethod
    def tearDownClass (cls):
        libgpu.destroy_device(cls.gpu)

    def test_nr_rks_matches_cpu (self):
        mol = _mol(self.gpu)
        with mock.patch.object(gnum, 'max_block_mb', BLOCK_MB):
            for xc in XCS:
                dm, (n_ref, e_ref, v_ref) = refs[xc]
                mf = _mf(mol, xc)
                mf.grids.build(with_non0tab=True)
                ncomp = 1 if mf._numint._xc_type(xc) == 'LDA' else 4
                self.assertGreater(len(gnum._grid_blocks(mf.grids.weights.size, mol.nao, ncomp, 1)), 5)
                n, e, v = mf._numint.nr_rks(mol, mf.grids, xc, dm)
                with self.subTest(xc=xc):
                    self.assertAlmostEqual(n, n_ref, 10)
                    self.assertAlmostEqual(e, e_ref, 10)
                    self.assertAlmostEqual(abs(v - v_ref).max(), 0, 10)

    def test_scf_energy (self):
        # DF J/K and XC both on the device: wB97M-D3BJ (range-separated meta-GGA hybrid)
        e = _mf(_mol(self.gpu), 'wb97m-d3bj').kernel()
        self.assertAlmostEqual(e, e_scf_ref, 9)

    def test_untagged_dm_uses_cpu (self):
        dm, (n_ref, e_ref, v_ref) = refs['pbe']
        mol = _mol(self.gpu)
        mf = _mf(mol, 'pbe')
        mf.grids.build(with_non0tab=True)
        with mock.patch.object(libgpu, 'compute_numint_rho', side_effect=AssertionError('GPU path used')):
            n, e, v = mf._numint.nr_rks(mol, mf.grids, 'pbe', numpy.asarray(dm))
        self.assertAlmostEqual(e, e_ref, 10)
        self.assertAlmostEqual(abs(v - v_ref).max(), 0, 10)

if __name__ == "__main__":
    print("Full Tests for gpu4mrh DFT numint (nr_rks)")
    unittest.main()
