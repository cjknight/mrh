import unittest
from unittest import mock
import numpy
from pyscf import gto, dft, lib

# gpu4mrh's pyscf.df.grad.rhf.get_jk (DF J/K nuclear-gradient contractions on the device) against
# the CPU version: J+K, K only, hermi 0/1, range-separated (omega), with and without the auxiliary
# basis response, with many integral blocks; full DF-RKS nuclear gradients of hybrids.
#
# Tolerances: the device applies the dense metric inverse J^-1 (pyscf: Cholesky solve), which for
# the jkfit metric (cond ~1e6) moves single vj/vk elements by ~1e-10. The omega-attenuated metric is
# numerically singular, so pyscf's own rhoj there is only defined to ~1e-8 (summation order); the
# traced gradients agree to ~1e-12.

ATOM = "O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59"
BLOCK_MB = 3.0 # tiny device blocks: many integral blocks per phase
CASES = [dict(), dict(hermi=1), dict(with_j=False), dict(omega=0.3), dict(omega=0.3, with_j=False)]
XCS = ['pbe0', 'wb97m-v']

def _mol (use_gpu=None):
    kwargs = {} if use_gpu is None else {'use_gpu': use_gpu}
    return gto.M(atom=ATOM, basis='cc-pvtz', verbose=0, **kwargs)

def _mf (mol, xc):
    mf = dft.RKS(mol).density_fit()
    mf.xc = xc
    mf.grids.level = 2
    mf.conv_tol = 1e-11
    # no VV10: 'wb97m-v' here is wb97m-d3bj without the D3 term, so pyscf-dispersion is not needed
    mf.nlc = False
    return mf

def _get_jk (mf, mol, dm, kw, aux):
    g = mf.nuc_grad_method()
    g.auxbasis_response = aux
    return g.get_jk(mol, dm, **kw)

def setUpModule():
    global mf_ref, dm_ref, jk_refs, grad_refs
    assert getattr(lib.param, 'use_gpu', None) is None
    mol = _mol()
    mf_ref = _mf(mol, 'wb97m-v')
    mf_ref.kernel()
    dm_ref = mf_ref.make_rdm1()
    jk_refs = [(kw, aux, _get_jk(mf_ref, mol, dm_ref, kw, aux)) for kw in CASES for aux in (True, False)]
    grad_refs = {}
    for xc in XCS:
        mf = _mf(_mol(), xc)
        mf.kernel()
        grad_refs[xc] = (mf, mf.nuc_grad_method().kernel())

def tearDownModule():
    global mf_ref, dm_ref, jk_refs, grad_refs
    del mf_ref, dm_ref, jk_refs, grad_refs

class KnownValues (unittest.TestCase):

    @classmethod
    def setUpClass (cls):
        global libgpu, gg
        from mrh.my_pyscf.gpu import libgpu
        from gpu4mrh import patch_pyscf
        from gpu4mrh.df import grad_rhf as gg
        cls.gpu = libgpu.init()

    @classmethod
    def tearDownClass (cls):
        libgpu.destroy_device(cls.gpu)

    def test_get_jk_matches_cpu (self):
        mol = _mol(self.gpu)
        mf = _mf(mol, 'wb97m-v')
        mf.__dict__.update({k: v for k, v in mf_ref.__dict__.items() if k in ('mo_coeff', 'mo_occ', 'mo_energy', 'e_tot', 'converged')})
        with mock.patch.object(gg, 'max_block_mb', BLOCK_MB), \
             mock.patch.object(libgpu, 'compute_dfgrad_ip1', wraps=libgpu.compute_dfgrad_ip1) as ip1:
            for kw, aux, (vj0, vk0) in jk_refs:
                vj1, vk1 = _get_jk(mf, mol, dm_ref, kw, aux)
                tol_j = 7 if 'omega' in kw else 9
                with self.subTest(kw=kw, aux=aux):
                    self.assertAlmostEqual(abs(vk1 - vk0).max(), 0, 9)
                    if aux: self.assertAlmostEqual(abs(vk1.aux - vk0.aux).max(), 0, 9)
                    if vj0 is None:
                        self.assertIsNone(vj1)
                    else:
                        self.assertAlmostEqual(abs(vj1 - vj0).max(), 0, tol_j)
                        if aux: self.assertAlmostEqual(abs(vj1.aux - vj0.aux).max(), 0, tol_j)
        self.assertGreaterEqual(ip1.call_count, 5 * len(jk_refs))

    def test_nuc_grad_matches_cpu (self):
        # gradient of the CPU-converged SCF, with the molecule switched to the device: DF J/K and
        # XC gradient terms on the device, nothing else changes
        with mock.patch.object(gg, 'max_block_mb', BLOCK_MB):
            for xc in XCS:
                mf, de_ref = grad_refs[xc]
                with mock.patch.object(mf.mol, 'use_gpu', self.gpu, create=True), \
                     mock.patch.object(libgpu, 'compute_dfgrad_ip1', wraps=libgpu.compute_dfgrad_ip1) as ip1:
                    de = mf.nuc_grad_method().kernel()
                with self.subTest(xc=xc):
                    self.assertGreater(ip1.call_count, 0)
                    self.assertAlmostEqual(abs(de - de_ref).max(), 0, 11)

    def test_multiple_dms_use_cpu (self):
        mol = _mol(self.gpu)
        mf = _mf(mol, 'wb97m-v')
        g = mf_ref.nuc_grad_method()
        g.mol = mol
        dms = numpy.stack([dm_ref, dm_ref])
        with mock.patch.object(libgpu, 'init_dfgrad', side_effect=AssertionError('GPU path used')):
            vj, vk = g.get_jk(mol, dms)
        self.assertEqual(vk.shape, (2, 3, mol.nao, mol.nao))

if __name__ == "__main__":
    print("Full Tests for gpu4mrh DF J/K nuclear gradient (df.grad.rhf.get_jk)")
    unittest.main()
