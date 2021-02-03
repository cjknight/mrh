import numpy as np
import scipy, time
from pyscf import fci, lib, mcscf
from pyscf.mcscf.mc1step import _fake_h_for_fast_casci
from mrh.my_pyscf.mcpdft import mcpdft
logger = lib.logger

def get_heff_cas (mc, mo_coeff, ci, link_index=None):
    ncore, ncas, nelec = mc.ncore, mc.ncas, mc.nelecas
    nocc = ncore + ncas
    mo_core = mo_coeff[:,:ncore]
    mo_cas = mo_coeff[:,ncore:nocc]

    veff1, veff2 = mc.get_pdft_veff (mo=mo_coeff, ci=ci, incl_coul=False, paaa_only=True)
    h1_ao = (mc.get_hcore () + veff1
           + mc._scf.get_j (dm=mc.make_rdm1 (mo_coeff=mo_coeff, ci=ci)))

    h0 = (mc._scf.energy_nuc () 
        + (h1_ao @ mo_core).ravel ().dot (mo_core.conj ().ravel ())
        + np.trace (veff2._vhf_c[:ncore,:ncore])/2)
    h1 = (mo_cas.conj ().T @ (h1_ao) @ mo_cas
        + veff2.vhf_c[ncore:nocc,ncore:nocc])
    h2 = np.zeros ([ncas,]*4) # Forward-compatibility for outcore pdft veff...
    for i in range (ncore, nocc): 
        h2[i-ncore] = veff2.ppaa[i][ncore:nocc].copy ()

    return h0, h1, h2

def _ci_min_epdft_fp (mc, mo_coeff, ci0, hcas=None, verbose=None):
    ''' Minimize the PDFT energy of a single state by repeated diagonalizations of the effective 
    PDFT Hamiltonian hpdft = Pcas (vnuc + dE/drdm1 op1 + dE/drdm2 op2) Pcas
    (as if that makes sense...) 

    Args:
        mc : mcscf object
        mo_coeff : ndarray of shape (nao,nmo)
        ci0 : ndarray of size (ndeta*ndetb)
            Initial guess CI vector; required!

    Kwargs:
        hcas : (float, [ncas,]*2 ndarray, [ncas,]*4 ndarray) or None
            The true Hamiltonian projected into the active space
        verbose : integer
            logger verbosity of function output; defaults to mc.verbose

    Returns:
        epdft : float
            Minimized MC-PDFT energy
        h0_pdft : float
            At convergence, the constant term of hpdft
            You might need this because ????
        ci1 : ndarray of size (ndeta*ndetb)
            Optimized CI vector
        ecas : float or None
            <ci1|hcas|ci1>
    '''
    t0 = (time.clock (), time.time ())
    ncas, nelecas = mc.ncas, mc.nelecas
    if verbose is None: verbose = mc.verbose
    log = logger.new_logger (mc, verbose)
    if hasattr (mc.fcisolver, 'gen_linkstr'):
        linkstrl = mc.fcisolver.gen_linkstr(ncas, nelecas, True)
    else:
        linkstrl = None 
    h0_pdft, h1_pdft, h2_pdft = get_heff_cas (mc, mo_coeff, ci0)
    max_memory = max(400, mc.max_memory-lib.current_memory()[0])

    epdft = 0
    chc_last = 0
    ecas = None
    ci1 = ci0.copy ()
    for it in range (mc.max_cycle_fp):
        h2eff = mc.fcisolver.absorb_h1e (h1_pdft, h2_pdft, ncas, nelecas, 0.5)
        hc = mc.fcisolver.contract_2e (h2eff, ci1, ncas, nelecas, link_index=linkstrl).ravel ()
        chc = ci1.conj ().ravel ().dot (hc)
        ci_grad = hc - (chc * ci1.ravel ())
        ci_grad_norm = ci_grad.dot (ci_grad)
        epdft_last = epdft
        with lib.temporary_env (mc, ci=ci1): # TODO: remove; fix mcpdft.kernel or write a better function
            epdft = mcpdft.kernel (mc, mc.otfnal, verbose=0)[0]

        dchc = chc + h0_pdft - chc_last # careful; don't mess up ci_grad
        depdft = epdft - epdft_last
        if hcas is None:
            log.info ('MC-PDFT CI fp iter %d EPDFT = %e, |grad| = %e, dEPDFT = %e, d<c.Hpdft.c> = %e', it, epdft, ci_grad_norm, depdft, dchc)
        else:
            h2eff = mc.fcisolver.absorb_h1e (hcas[1], hcas[2], ncas, nelecas, 0.5)
            hc = mc.fcisolver.contract_2e (h2eff, ci1, ncas, nelecas, link_index=linkstrl).ravel ()
            ecas = ci1.conj ().ravel ().dot (hc) + hcas[0]
            log.info ('MC-PDFT CI fp iter %d ECAS = %e, EPDFT = %e, |grad| = %e, dEPDFT = %e, d<c.Hpdft.c> = %e', it, ecas, epdft, ci_grad_norm, depdft, dchc)
         

        if ci_grad_norm < mc.conv_tol_ci_fp and np.abs (dchc) < 1e-8: break
       
        chc_last, ci1 = mc.fcisolver.kernel (h1_pdft, h2_pdft, ncas, nelecas,
                                               ci0=ci1, verbose=log,
                                               max_memory=max_memory,
                                               ecore=h0_pdft)
        h0_pdft, h1_pdft, h2_pdft = get_heff_cas (mc, mo_coeff, ci1)
        # putting this at the bottom to 1) get a good max_memory outside the loop with 2) as few integrations as possible

    log.timer ('MC-PDFT CI fp iteration', *t0)
    return epdft, h0_pdft, ci1, ecas
    
def mc1step_casci(mc, mo_coeff, ci0=None, eris=None, verbose=None, envs=None):
    ''' Wrapper for _ci_min_epdft_fp to mcscf.mc1step.casci function '''
    if ci0 is None: ci0 = mc.ci
    if verbose is None: verbose = mc.verbose
    t0 = (time.clock (), time.time ())
    ncas, nelecas = mc.ncas, mc.nelecas
    linkstrl = mc.fcisolver.gen_linkstr(ncas, nelecas, True)
    linkstr  = mc.fcisolver.gen_linkstr(ncas, nelecas, False)
    log = lib.logger.new_logger (mc, verbose)
    if eris is None:
        h0_cas, h1_cas = mcscf.casci.h1e_for_cas (mc, mo_coeff, mc.ncas, mc.ncore)
        h2_cas = mcscf.casci.CASCI.ao2mo (mc, mo_coeff)
    else:
        fcasci = _fake_h_for_fast_casci (mc, mo_coeff, eris)
        h1_cas, h0_cas = fcasci.get_h1eff ()
        h2_cas = fcasci.get_h2eff ()

    if ci0 is None: 
        # Use real Hamiltonian? Or use HF?
        hdiag = mc.fcisolver.make_hdiag (h1_cas, h2_cas, mc.ncas, mc.nelecas)
        ci0 = mc.fcisolver.get_init_guess (ncas, nelecas, 1, hdiag)[0]

    epdft, h0_pdft, ci1, ecas = _ci_min_epdft_fp (mc, mo_coeff, ci0, 
        hcas=(h0_cas,h1_cas,h2_cas), verbose=verbose)
    eci = epdft - h0_cas

    if envs is not None and log.verbose >= lib.logger.INFO:
        log.debug('CAS space CI energy = %.15g', eci)

        if getattr(mc.fcisolver, 'spin_square', None):
            ss = mc.fcisolver.spin_square(ci1, mc.ncas, mc.nelecas)
        else:
            ss = None

        if 'imicro' in envs:  # Within CASSCF iteration
            if ss is None:
                log.info('macro iter %d (%d JK  %d micro), '
                         'MC-PDFT E = %.15g  dE = %.8g  CASCI E = %.15g',
                         envs['imacro'], envs['njk'], envs['imicro'],
                         epdft, epdft-envs['elast'], ecas)
            else:
                log.info('macro iter %d (%d JK  %d micro), '
                         'MC-PDFT E = %.15g  dE = %.8g  S^2 = %.7f  CASCI E = %.15g',
                         envs['imacro'], envs['njk'], envs['imicro'],
                         epdft, epdft-envs['elast'], ss[0], ecas)
            if 'norm_gci' in envs:
                log.info('               |grad[o]|=%5.3g  '
                         '|grad[c]|= %s  |ddm|=%5.3g',
                         envs['norm_gorb0'],
                         envs['norm_gci'], envs['norm_ddm'])
            else:
                log.info('               |grad[o]|=%5.3g  |ddm|=%5.3g',
                         envs['norm_gorb0'], envs['norm_ddm'])
        else:  # Initialization step
            if ss is None:
                log.info('MC-PDFT E = %.15g  CASCI E = %.15g', epdft, ecas)
            else:
                log.info('MC-PDFT E = %.15g  S^2 = %.7f  CASCI E = %.15g', epdft, ss[0], ecas)

    return epdft, ecas, ci1

#def update_casdm(mc, mo, u, fcivec, e_cas, eris, envs={}):



