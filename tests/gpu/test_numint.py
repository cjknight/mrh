import unittest
from unittest import mock
import numpy
from pyscf import gto, dft, lib
from pyscf.grad import rks as rks_grad

# gpu4mrh's NumInt.nr_rks (XC energy and matrix on the device) and grad.rks.get_vxc (XC part of
# the nuclear gradient) against the CPU versions, for LDA, GGA and meta-GGA (incl. range-separated
# hybrids), with many grid blocks; a full DF-RKS SCF and gradient; and the CPU fallback for density
# matrices without orbitals.

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
    global refs, grad_refs, e_scf_ref, de_ref
    assert getattr(lib.param, 'use_gpu', None) is None
    mol = _mol()
    refs = {}
    grad_refs = {}
    for xc in XCS:
        mf = _mf(mol, xc)
        mf.kernel()
        dm = mf.make_rdm1()
        refs[xc] = (dm, mf._numint.nr_rks(mol, mf.grids, xc, dm))
        grad_refs[xc] = rks_grad.get_vxc(mf._numint, mol, mf.grids, xc, dm)[1]
    mf = _mf(mol, 'wb97m-d3bj')
    e_scf_ref = mf.kernel()
    de_ref = mf.nuc_grad_method().kernel()

def tearDownModule():
    global refs, grad_refs, e_scf_ref, de_ref
    del refs, grad_refs, e_scf_ref, de_ref

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

    def test_get_vxc_grad_matches_cpu (self):
        mol = _mol(self.gpu)
        from gpu4mrh.grad import rks as grks
        self.assertIs(rks_grad.get_vxc, grks.get_vxc)
        with mock.patch.object(gnum, 'max_block_mb', BLOCK_MB):
            for xc in XCS:
                dm, _ = refs[xc]
                mf = _mf(mol, xc)
                mf.grids.build(with_non0tab=True)
                ncomp = 4 if mf._numint._xc_type(xc) == 'LDA' else 10
                self.assertGreater(len(gnum._grid_blocks(mf.grids.weights.size, mol.nao, ncomp, 1)), 5)
                exc, v = rks_grad.get_vxc(mf._numint, mol, mf.grids, xc, dm)
                with self.subTest(xc=xc):
                    self.assertIsNone(exc)
                    self.assertEqual(v.shape, (3, mol.nao, mol.nao))
                    self.assertAlmostEqual(abs(v - grad_refs[xc]).max(), 0, 10)

    def test_scf_energy_and_gradient (self):
        # DF J/K and XC both on the device: wB97M-D3BJ (range-separated meta-GGA hybrid)
        mf = _mf(_mol(self.gpu), 'wb97m-d3bj')
        e = mf.kernel()
        self.assertAlmostEqual(e, e_scf_ref, 9)
        de = mf.nuc_grad_method().kernel()
        self.assertAlmostEqual(abs(de - de_ref).max(), 0, 8)

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
    print("Full Tests for gpu4mrh DFT numint (nr_rks, grad get_vxc)")
    unittest.main()
