import unittest
from unittest import mock
import numpy
from pyscf import gto, dft, lib

# gpu4mrh's pyscf.df.grad.rhf.get_jk (DF J/K nuclear-gradient contractions on the device) against
# the CPU version: J+K, K only, hermi 0/1, range-separated (omega), with and without the auxiliary
# basis response, with many integral blocks; full DF-RKS nuclear gradients of hybrids.
#
# Tolerances: the omega-attenuated metric is numerically singular (pyscf uses its eigenvalue
# pseudo-inverse), so anything through it is only defined to summation order: pyscf's own CPU
# get_jk(omega=0.3) changes by up to ~3e-9 in vk (and ~1e-8 in vj) and its wb97m-v gradient by up to
# ~5e-10 between OMP_NUM_THREADS = 1, 2, 4, 8. The omega cases and range-separated gradients are
# checked at that level, everything else tightly. BUTANE (aug-cc-pvtz-jkfit, cond ~8e8) checks the Cholesky path: a
# dense J^-1 in place of the triangular solves loses ~2e-6 there (cond ~1e11 for the 66-atom
# production case, where it broke the gradient).

ATOM = "O 0 0 0; H 0 0.76 0.59; H 0 -0.76 0.59"
BLOCK_MB = 3.0 # tiny device blocks: many integral blocks per phase
CASES = [dict(), dict(hermi=1), dict(with_j=False), dict(omega=0.3), dict(omega=0.3, with_j=False)]
XCS = ['pbe0', 'wb97m-v']
BUTANE = ("C 0 0 0; C 1.53 0 0; C 2.04 1.44 0; C 3.57 1.44 0; H -0.39 1.02 0; H -0.39 -0.51 0.88; "
          "H -0.39 -0.51 -0.88; H 1.92 -0.51 0.88; H 1.92 -0.51 -0.88; H 1.65 1.95 0.88; H 1.65 1.95 -0.88; "
          "H 3.96 0.42 0; H 3.96 1.95 0.88; H 3.96 1.95 -0.88")

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

def _butane (use_gpu=None):
    kwargs = {} if use_gpu is None else {'use_gpu': use_gpu}
    mol = gto.M(atom=BUTANE, basis='cc-pvdz', verbose=0, **kwargs)
    mf = dft.RKS(mol).density_fit(auxbasis='aug-cc-pvtz-jkfit')
    mf.xc = 'pbe0'
    mf.grids.level = 1
    mf.conv_tol = 1e-10
    return mf

def setUpModule():
    global mf_ref, dm_ref, jk_refs, grad_refs, butane_ref
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
    mf = _butane()
    mf.kernel()
    g = mf.nuc_grad_method()
    butane_ref = (mf, g.get_jk(mf.mol, mf.make_rdm1()), g.kernel())

def tearDownModule():
    global mf_ref, dm_ref, jk_refs, grad_refs, butane_ref
    del mf_ref, dm_ref, jk_refs, grad_refs, butane_ref

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
                tol_k = 8 if 'omega' in kw else 9
                with self.subTest(kw=kw, aux=aux):
                    self.assertAlmostEqual(abs(vk1 - vk0).max(), 0, tol_k)
                    if aux: self.assertAlmostEqual(abs(vk1.aux - vk0.aux).max(), 0, tol_k)
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
                tol = 9 if dft.libxc.rsh_coeff(xc)[0] != 0 else 11 # range-separated: omega metric
                with self.subTest(xc=xc):
                    self.assertGreater(ip1.call_count, 0)
                    self.assertAlmostEqual(abs(de - de_ref).max(), 0, tol)

    def test_ill_conditioned_metric (self):
        mf, (vj0, vk0), de0 = butane_ref
        with mock.patch.object(gg, 'max_block_mb', BLOCK_MB), \
             mock.patch.object(mf.mol, 'use_gpu', self.gpu, create=True), \
             mock.patch.object(libgpu, 'init_dfgrad', wraps=libgpu.init_dfgrad) as init:
            g = mf.nuc_grad_method()
            vj1, vk1 = g.get_jk(mf.mol, mf.make_rdm1())
            de1 = g.kernel()
        self.assertEqual(init.call_args_list[0].args[-1], 1) # Cholesky path
        self.assertAlmostEqual(abs(vk1 - vk0).max(), 0, 9)
        self.assertAlmostEqual(abs(vj1 - vj0).max(), 0, 9)
        self.assertAlmostEqual(abs(vk1.aux - vk0.aux).max(), 0, 9)
        self.assertAlmostEqual(abs(vj1.aux - vj0.aux).max(), 0, 9)
        self.assertAlmostEqual(abs(de1 - de0).max(), 0, 10)

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
