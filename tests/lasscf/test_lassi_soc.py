import unittest
import numpy as np
from pyscf import gto, scf, lib, mcscf
from c2h6n4_struct import structure as struct
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.mcscf.soc_int import compute_hso, amfi_dm
from mrh.my_pyscf.mcscf.lassi_op_o0 import si_soc
from mrh.my_pyscf.mcscf.lasscf_o0 import LASSCF
from mrh.my_pyscf.mcscf.lassi import make_stdm12s, roots_make_rdm12s

def setUpModule():
    global mol1, mf1, mol2, mf2, las2
    mol1 = gto.M (atom="""
        O  0.000000  0.000000  0.000000
        H  0.758602  0.000000  0.504284
        H  -0.758602  0.000000  0.504284
    """, basis='631g',symmetry=True,
    output='test_lassi_soc1.log',
    verbose=lib.logger.DEBUG)
    mf1 = scf.RHF (mol1).run ()
   
    # NOTE: Test systems don't have to be scientifically meaningful, but they do need to
    # be "mathematically" meaningful. I.E., you can't just test zero. You need a test case
    # where the effect of the thing you are trying to test is numerically large enough
    # to be reproduced on any computer. Calculations that don't converge can't be used
    # as test cases for this reason.
    mol2 = struct (2.0, 2.0, '6-31g', symmetry=False)
    mol2.output = 'test_lassi_soc2.log'
    mol2.verbose = lib.logger.DEBUG
    mol2.build ()
    mf2 = scf.RHF (mol2).run ()
    las2 = LASSCF (mf2, (4,4), (4,4), spin_sub=(1,1))
    las2.mo_coeff = las2.localize_init_guess ((list (range (3)), list (range (9,12))), mf2.mo_coeff)
    # NOTE: for 2-fragment and above, you will ALWAYS need to remember to do the line above.
    # If you skip it, you can expect your orbitals not to converge.
    las2.state_average_(weights=[1,0,0,0,0],
                            spins=[[0,0],[2,0],[-2,0],[0,2],[0,-2]],
                            smults=[[1,1],[3,1],[3,1],[1,3],[1,3]])
    # NOTE: Be careful about state selection. You have to select states that can actually be coupled
    # by a 1-body SOC operator. For instance, spins=[0,0] and spins=[2,2] would need at least a 2-body
    # operator to couple.
    las2.kernel ()

def tearDownModule():
    global mol1, mf1, mol2, mf2, las2
    mol1.stdout.close()
    mol2.stdout.close()
    del mol1, mf1, mol2, mf2, las2

class KnownValues (unittest.TestCase):

    # NOTE: In OpenMolcas, when using the ANO-RCC basis sets, the AMFI operator is switched from the Breit-Pauli
    # to the Douglass-Kroll Hamiltonian. There is no convenient way to switch this off; the only workaround
    # I've found is to "disguise" the basis set as something unrelated to ANO-RCC by copying and pasting it into
    # a separate file. Therefore, for now, we can only compare results from non-relativistic basis sets between
    # the two codes, until we implement Douglass-Kroll ourselves.

    def test_soc_int (self):
        # Obtained from OpenMolcas v22.02
        int_ref = 2*np.array ([0.0000000185242348, 0.0000393310222742, 0.0000393310222742, 0.0005295974407740]) 
        
        amfi_int = compute_hso (mol1, amfi_dm (mol1), amfi=True)
        amfi_int = amfi_int[2][amfi_int[2] > 0]
        amfi_int = np.sort (amfi_int.imag)
        self.assertAlmostEqual (lib.fp (amfi_int), lib.fp (int_ref), 8)

    def test_soc_1frag (self):
        # References obtained from OpenMolcas v22.10 (locally-modified to enable changing the speed of light,
        # see https://gitlab.com/MatthewRHermes/OpenMolcas/-/tree/amfi_speed_of_light)
        esf_ref = [0.0000000000,] + ([0.7194945289,]*3) + ([0.7485251565,]*3)
        eso_ref = [-0.0180900821,0.6646578117,0.6820416863,0.7194945289,0.7485251565,0.8033618737,0.8040680811]
        hso_ref = np.zeros ((7,7), dtype=np.complex128)
        hso_ref[1,0] =  0 - 10982.305j # T(+1)
        hso_ref[3,0] =  0 + 10982.305j # T(-1)
        hso_ref[4,2] =  10524.501 + 0j # T(+1)
        hso_ref[5,1] = -10524.501 + 0j # T(-1)
        hso_ref[5,3] =  10524.501 + 0j # T(+1)
        hso_ref[6,2] = -10524.501 + 0j # T(-1)
        hso_ref[5,0] =  0 - 18916.659j # T(0) < testing both this and T(+-1) is the reason I did 2 triplets
        
        las = LASSCF (mf1, (6,), (8,), spin_sub=(1,), wfnsym_sub=('A1',)).run (conv_tol_grad=1e-7)
        las.state_average_(weights=[1,0,0,0,0,0,0],
                           spins=[[0,],[2,],[0,],[-2,],[2,],[0,],[-2,],],
                           smults=[[1,],[3,],[3,],[3,],[3,],[3,],[3,],],
                           wfnsyms=([['A1',],]+([['B1',],]*3)+([['A2',],]*3)))
                           #wfnsyms=([['A1',],['B1',],['A2',],['B2',]]))
        las.lasci ()
        e0 = las.e_states[0]
        with self.subTest (deltaE='SF'):
            esf_test = las.e_states - e0
            self.assertAlmostEqual (lib.fp (esf_test), lib.fp (esf_ref), 6)
        with lib.light_speed (10):
            e_roots, si = las.lassi (opt=0, soc=True, break_symmetry=True)
        eso_test = e_roots - e0
        with self.subTest (deltaE='SO'):
            self.assertAlmostEqual (lib.fp (eso_test), lib.fp (eso_ref), 6)
        hso_test = (si * eso_test[None,:]) @ si.conj ().T
        from pyscf.data import nist
        au2cm = nist.HARTREE2J / nist.PLANCK / nist.LIGHT_SPEED_SI * 1e-2
        hso_test *= au2cm
        hso_test = np.around (hso_test, 8)
        for i, j in zip (*np.where (hso_ref)):
            with self.subTest (hso=(i,j)):
                # The spin-pure states have arbitrary sign, but they're all
                # real, so it's just +- 1. I want to do this instead of abs
                # because whether something's on the real or imaginary number
                # line should be consistent.
                try:
                    self.assertAlmostEqual (hso_test[i,j],hso_ref[i,j],1)
                except AssertionError:
                    self.assertAlmostEqual (hso_test[i,j],-hso_ref[i,j],1)


    def test_soc_2frag (self):
        ## stationary test for >1 frag calc
        esf_ref = np.array ([-296.6356767693,-296.6354236887,-296.6354236887,-296.6354236887,-296.6354236887])
        eso_ref = np.array ([-296.6357061838,-296.6356871348,-296.6356871348,-296.6351604534,-296.6351310388])
        with self.subTest (deltaE='SF'):
            self.assertAlmostEqual (lib.fp (las2.e_states), lib.fp (esf_ref), 8)
        # Light speed value chosen because it changes the ground state from a triplet to 
        # a contaminated quasi-singlet.
        with lib.light_speed (5):
            e_test, si_test = las2.lassi (opt=0, soc=True, break_symmetry=True)
        with self.subTest (deltaE='SO'):
            self.assertAlmostEqual (lib.fp (e_test), lib.fp (eso_ref), 8)

    def test_soc_stdm12s (self):
        pass
        #stdm1s_test, stdm2s_test = make_stdm12s (las2, soc=True, opt=0)    
        ## stationary test for roots_make_stdm12s
  
    def test_soc_rdm12s (self):
        pass
        #rdm1s_test, rdm2s_test = roots_make_rdm12s (las2, las2.ci, si_ref, soc=True, opt=0)
        ## stationary test for roots_make_rdm12s

if __name__ == "__main__":
    print("Full Tests for SOC")
    unittest.main()
