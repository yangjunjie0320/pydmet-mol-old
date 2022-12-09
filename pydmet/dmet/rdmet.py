import sys, os
from collections.abc import Iterable
from functools import reduce

import numpy
import scipy

import pyscf
from pyscf import gto, scf, cc
from pyscf import ao2mo, lo, tools
from pyscf.lib import logger
from pyscf.lib import chkfile

class EmbeddingProblem(object):
    pass

class EmbeddingResults(object):
    pass

class EmbeddingBasis(object):
    pass

class SolverMixin(object):
    verbose  = 5
    stdout   = sys.stdout
    tmp_dir  = None

    conv_tol = 1e-8
    conv_tol_grad = 1e-5
    max_cycle = 50

    def dump_res(self):
        raise NotImplementedError
    
    def load_res(self):
        raise NotImplementedError

    def kernel(self, emb_prob, load_dir=None, save_dir=None):
        raise NotImplementedError

    def solve_mf(self, emb_prob, load_dir=None, save_dir=None):
        norb   = emb_prob.neo
        nelecs = emb_prob.nelecs
        neleca, nelecb = nelecs
        assert neleca == nelecb
        dm0 = emb_prob.dm0

        m = gto.M()
        m.nelectron     = neleca + nelecb
        m.spin          = 0
        m.incore_anyway = True
        m.build()

        s1e = numpy.eye(norb)
        f1e = emb_prob.f1e
        h2e = emb_prob.h2e

        h = pyscf.scf.RHF(m)
        h.verbose   = self.verbose
        h.stdout    = self.stdout
        h.get_ovlp  = lambda *args: s1e
        h.get_hcore = lambda *args: f1e
        h._eri      = h2e
        h.conv_tol  = self.conv_tol
        h.max_cycle = self.max_cycle
        h.kernel(dm0=dm0)

        if h.converged:
            logger.info(self, 'RHF converged')
        else:
            logger.info(self, 'RHF not converged')
        
        return h

    def energy_elec(self, rdm1, rdm2, emb_prob=None):
        imp_eo_idx  = emb_prob.imp_eo_idx
        f1e = emb_prob.f1e
        h1e = emb_prob.h1e
        h2e = emb_prob.h2e
        
        ene_elec    = numpy.einsum('pq,  qp  ->', (f1e + h1e)[imp_eo_idx, :], rdm1[:, imp_eo_idx])       / 2.0
        ene_elec   += numpy.einsum('pqrs,pqrs->', h2e[imp_eo_idx, :, :, :],   rdm2[imp_eo_idx, :, :, :]) / 2.0
        return ene_elec

class RHF(SolverMixin):

    def kernel(self, emb_prob, load_dir=None, save_dir=None):
        h = self.solve_mf(emb_prob, load_dir, save_dir)

        mo_coeff = h.mo_coeff
        rdm1 = h.make_rdm1()
        rdm2 = h.make_rdm2()

        scf_res = {
            'mo_coeff': mo_coeff,
            'rdm1': rdm1,
            'rdm2': rdm2,
        }

        if save_dir is not None:
            assert os.path.isdir(save_dir)
            save_file = os.path.join(save_dir, 'rhf.h5')
            logger.info(self, 'Saving RHF results to %s', save_file)
            chkfile.save(save_file, 'scf', scf_res)

        emb_res = EmbeddingResults()
        emb_res.neo    = emb_prob.neo
        emb_res.nelecs = emb_prob.nelecs
        emb_res.rdm1    = rdm1
        emb_res.rdm1_ao = None
        emb_res.energy_elec = self.energy_elec(rdm1, rdm2, emb_prob)
        return emb_res

class RCCSD(SolverMixin):
    def kernel(self, emb_prob, load_dir, save_dir):
        norb   = emb_prob.neo
        nelecs = emb_prob.nelecs
        neleca, nelecb = nelecs
        assert neleca == nelecb

        m = gto.M()
        m.nelectron     = nelecs
        m.incore_anyway = True

        s1e = numpy.eye(norb)
        h1e = emb_prob.h1e
        h2e = emb_prob.h2e

        h = pyscf.scf.RHF(m)
        h.get_hcore = lambda *args: h1e
        h.get_ovlp  = lambda *args: s1e
        h._eri      = h2e
        h.conv_tol  = self.conv_tol
        h.max_cycle = self.max_cycle
        h.kernel()

        if h.converged:
            logger.info(self, 'RHF converged')
        else:
            logger.info(self, 'RHF not converged')

        mo_coeff = h.mo_coeff
        rdm1 = h.make_rdm1()
        rdm2 = h.make_rdm2()

        scf_res = {
            'mo_coeff': mo_coeff,
            'rdm1': rdm1,
            'rdm2': rdm2,
        }

        if save_dir is not None:
            assert os.path.isdir(save_dir)
            save_file = os.path.join(save_dir, 'rhf.h5')
            logger.info(self, 'Saving RHF results to %s', save_file)
            chkfile.save(save_file, 'scf', scf_res)

        cc = pyscf.cc.RCCSD(h)
        cc.conv_tol = self.conv_tol
        cc.max_cycle = self.max_cycle
        cc.kernel()

        rdm1 = cc.make_rdm1(ao_repr=True)
        rdm2 = cc.make_rdm2(ao_repr=True)

        if cc.converged:
            logger.info(self, 'RCCSD converged')
        else:
            logger.info(self, 'RCCSD not converged')

        cc_res = {
            't1': cc.t1,
            't2': cc.t2,
            'rdm1': rdm1,
            'rdm2': rdm2,
        }

        if save_dir is not None:
            assert os.path.isdir(save_dir)
            save_file = os.path.join(save_dir, 'rccsd.h5')
            logger.info(self, 'Saving RCCSD results to %s', save_file)
            chkfile.save(save_file, 'cc', cc_res)
        
        emb_res = EmbeddingResults()
        emb_res.rdm1 = rdm1
        emb_res.rdm2 = rdm2
        return emb_res

class DMETMixin(object):
    """
    The base class for DMET. Different DMET methods can be classified by different
    spin symmetry.
    """
    verbose = 4
    stdout  = sys.stdout

    conv_tol  = 1e-6
    max_cycle = 50

    save_dir = None
    load_dir = None

    def _method_name(self):
        method_name = []
        for c in self.__class__.__mro__:
            if issubclass(c, DMETMixin) and c is not DMETMixin:
                method_name.append(c.__name__)
        return '-'.join(method_name)

    def dump_flags(self):
        log = logger.new_logger(self, self.verbose)

        log.info("\n")
        log.info("******** %s ********", self.__class__)
        log.info('method = %s', self._method_name())

    def build_env(self):
        raise NotImplementedError

    def build_mf(self):
        raise NotImplementedError

    def build_emb_basis(self):
        raise NotImplementedError

    def build_emb_prob(self):
        raise NotImplementedError

    def get_emb_solver(self, ifrag):
        solver = self.solver
        assert solver is not None

        if isinstance(solver, Iterable):
            solver_list = list(solver)
            assert len(solver_list) == self.nfrag

            solver_obj = solver[ifrag]
        else:
            solver_obj = solver

        assert isinstance(solver_obj, SolverMixin)
        return solver_obj

    def transform_dm_ao_to_lo(self):
        raise NotImplementedError

    def transform_dm_lo_to_ao(self):
        raise NotImplementedError

    def transform_h_ao_to_lo(self):
        raise NotImplementedError

    def transform_h_ao_to_lo(self):
        raise NotImplementedError

    def kernel(self, dm0=None):
        self.dump_flags()
        log = logger.new_logger(self, self.verbose)

        m   = self.m
        mf  = self.build_mf(m, dm0=dm0)
        assert mf.converged
        coeff_ao_lo = self.coeff_ao_lo

        hcore_ao = mf.get_hcore()
        ovlp_ao  = mf.get_ovlp()
        eri_ao   = None
        if hasattr(mf, '_eri'):
            eri_ao = mf._eri

        self.build_env()
        nfrag = self.nfrag
        imp_lo_idx_list = self.imp_lo_idx_list
        env_lo_idx_list = self.env_lo_idx_list
        assert len(imp_lo_idx_list) == nfrag
        assert len(env_lo_idx_list) == nfrag

        dm_ll_ao_pre = mf.make_rdm1()
        dm_ll_lo_pre = self.transform_dm_ao_to_lo(dm_ll_ao_pre, coeff_ao_lo=coeff_ao_lo, ovlp_ao=ovlp_ao)
        dm_ll_ao_cur = None
        dm_ll_lo_cur = None

        ene_hl_cur = 0.0
        ene_hl_pre = None

        dm_hl_ao_cur = numpy.zeros_like(dm_ll_ao_pre)
        dm_hl_lo_cur = None

        for ifrag, imp_lo_idx, env_lo_idx in zip(range(nfrag), imp_lo_idx_list, env_lo_idx_list):
            save_dir = self.save_dir 
            if save_dir is not None:
                save_dir = os.path.join(save_dir, 'frag_%d' % ifrag)
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)

            load_dir = None

            emb_basis = self.build_emb_basis(
                imp_lo_idx, env_lo_idx, dm_ll_lo_pre,
                coeff_ao_lo=coeff_ao_lo, ovlp_ao=ovlp_ao
                )

            emb_solver = self.get_emb_solver(ifrag)
            emb_prob   = self.build_emb_prob(
                m, mf, emb_basis=emb_basis,
                hcore_ao=hcore_ao, eri_ao=eri_ao,
                dm_ll_lo=dm_ll_lo_pre, 
                dm_ll_ao=dm_ll_ao_pre
                )

            # Note that the embbedding results from the solver may
            # not be consistent with the DMET type. So we need to
            # transform the results to the DMET type.
            emb_res = emb_solver.kernel(
                emb_prob=emb_prob,
                save_dir=save_dir, 
                load_dir=load_dir,
            )

            ene_hl_cur   += emb_res.energy_elec
            dm_hl_ao_cur += self.get_rdm1_ao(emb_res, emb_basis)

        dm_hl_lo_cur = self.transform_dm_ao_to_lo(dm_hl_ao_cur, coeff_ao_lo=coeff_ao_lo, ovlp_ao=ovlp_ao)
        print("ene_hl_cur = %8.6f"%ene_hl_cur)

class RDMETWithHF(DMETMixin):
    '''The class for solving spin restricted DMET problem in molecular system
    and the supercell gamma point periodic system.
    '''

    def __init__(self, mol, coeff_ao_lo=None, imp_lo_idx_list=None, solver=None):
        self.m       = mol
        self.stdout  = mol.stdout
        self.verbose = mol.verbose

        assert coeff_ao_lo is not None
        self.coeff_ao_lo = coeff_ao_lo

        assert imp_lo_idx_list is not None
        self.imp_lo_idx_list = imp_lo_idx_list
        self.nfrag           = len(imp_lo_idx_list)
        self.env_lo_idx_list = None

        if solver is None:
            solver = RHF()
        else:
            if isinstance(solver, Iterable):
                solver = list(solver)
                for s in solver:
                    assert isinstance(s, SolverMixin)
                assert len(solver) == self.nfrag
            else:
                assert isinstance(solver, SolverMixin)
        self.solver = solver

    def transform_dm_ao_to_lo(self, dm_ao, ovlp_ao=None, coeff_ao_lo=None):
        dm_lo    = reduce(numpy.dot, (coeff_ao_lo.T, ovlp_ao, dm_ao, ovlp_ao, coeff_ao_lo))
        return dm_lo

    def transform_dm_lo_to_ao(self, dm_lo, ovlp_ao=None, coeff_ao_lo=None):
        dm_ao    = reduce(numpy.dot, (coeff_ao_lo, dm_lo, coeff_ao_lo.T))
        return dm_ao

    def transform_h_ao_to_lo(self, h_ao, ovlp_ao=None, coeff_ao_lo=None):
        h_lo    = reduce(numpy.dot, (coeff_ao_lo, h_ao, coeff_ao_lo.T))
        return h_lo

    def transform_h_lo_to_ao(self, h_lo, ovlp_ao=None, coeff_ao_lo=None):
        h_ao    = reduce(numpy.dot, (coeff_ao_lo.T, ovlp_ao, h_lo, ovlp_ao, coeff_ao_lo))
        return h_ao

    def build_env(self):
        if self.env_lo_idx_list is None:
            self.env_lo_idx_list = []
            for imp_lo_idx in self.imp_lo_idx_list:
                env_lo_idx = [i for i in range(self.m.nao) if i not in imp_lo_idx]
                self.env_lo_idx_list.append(env_lo_idx)

    def build_mf(self, m, dm0=None):
        log = logger.new_logger(self, self.verbose)
        assert isinstance(m, pyscf.gto.Mole)
        mf = pyscf.scf.hf.RHF(m)
        mf.conv_tol  = self.conv_tol
        mf.max_cycle = self.max_cycle
        mf.kernel(dm0=dm0)

        if mf.converged:
            log.info('HF converged')
        else:
            log.info('HF not converged')

        return mf

    def get_rdm1_ao(self, emb_res, emb_basis):
        neo  = emb_res.neo

        imp_eo_idx  = emb_basis.imp_eo_idx
        bath_eo_idx = emb_basis.bath_eo_idx
        coeff_ao_eo = emb_basis.coeff_ao_eo
        coeff_ao_eo_imp  = coeff_ao_eo[:, imp_eo_idx]
        coeff_ao_eo_bath = coeff_ao_eo[:, bath_eo_idx]

        rdm1_emb_eo = emb_res.rdm1
        rdm1_emb_eo_imp_imp   = rdm1_emb_eo[imp_eo_idx, :][:, imp_eo_idx]
        rdm1_emb_eo_imp_bath  = rdm1_emb_eo[imp_eo_idx, :][:, bath_eo_idx]
        rdm1_emb_eo_bath_imp  = rdm1_emb_eo[bath_eo_idx, :][:, imp_eo_idx]

        rdm1_ao  = reduce(numpy.dot, (coeff_ao_eo_imp, rdm1_emb_eo_imp_imp,  coeff_ao_eo_imp.T))
        rdm1_ao += reduce(numpy.dot, (coeff_ao_eo_imp, rdm1_emb_eo_imp_bath, coeff_ao_eo_bath.T)) * 0.5
        rdm1_ao += reduce(numpy.dot, (coeff_ao_eo_bath, rdm1_emb_eo_bath_imp, coeff_ao_eo_imp.T)) * 0.5

        return rdm1_ao

    def build_emb_basis(self, imp_lo_idx, env_lo_idx, dm_ll_lo, ovlp_ao=None, coeff_ao_lo=None):
        """Build the embedding basis for the given impurity and environment orbitals.
        
        Parameters:
            imp_lo_idx: list of int
                The indices of the impurity orbitals.
            env_lo_idx: list of int
                The indices of the environment orbitals.
            dm_ll_lo: 2D numpy array
                The density matrix in the local basis, from the low level (mean field) calculation.
            ovlp_ao: 2D numpy array
                The overlap matrix in the AO basis.
            coeff_ao_lo: 2D numpy array
                The transformation matrix from the AO basis to the local basis.
            
        Returns:
            coeff_ao_eo: 2D numpy array
                The transformation matrix from the AO basis to the embedding basis.
            coeff_lo_eo: 2D numpy array
                The transformation matrix from the local basis to the embedding basis.
        """
        nlo_imp = len(imp_lo_idx)
        nlo_env = len(env_lo_idx)

        coeff_ao_lo = self.coeff_ao_lo

        imp_imp_lo_ix = numpy.ix_(imp_lo_idx, imp_lo_idx)
        env_env_lo_ix = numpy.ix_(env_lo_idx, env_lo_idx)
        imp_env_lo_ix = numpy.ix_(imp_lo_idx, env_lo_idx)

        dm_imp_imp_lo = dm_ll_lo[imp_imp_lo_ix]
        dm_env_env_lo = dm_ll_lo[env_env_lo_ix]
        dm_imp_env_lo = dm_ll_lo[imp_env_lo_ix]

        assert dm_imp_imp_lo.shape == (nlo_imp, nlo_imp)
        assert dm_env_env_lo.shape == (nlo_env, nlo_env)
        assert dm_imp_env_lo.shape == (nlo_imp, nlo_env)

        u, s, vh = numpy.linalg.svd(dm_imp_env_lo, full_matrices=False)
        coeff_lo_eo_imp_imp  = numpy.eye(nlo_imp)
        coeff_lo_eo_env_bath = vh.T

        # The embedding basis is the union of the impurity and environment basis.
        coeff_ao_lo_imp = coeff_ao_lo[:, imp_lo_idx]
        coeff_ao_lo_env = coeff_ao_lo[:, env_lo_idx]
        coeff_ao_eo_imp  = numpy.dot(coeff_ao_lo_imp, coeff_lo_eo_imp_imp)
        coeff_ao_eo_bath = numpy.dot(coeff_ao_lo_env, coeff_lo_eo_env_bath)
        coeff_ao_eo = numpy.hstack((coeff_ao_eo_imp, coeff_ao_eo_bath))
        coeff_lo_eo = reduce(numpy.dot, (coeff_ao_lo.T, ovlp_ao, coeff_ao_eo))

        neo = coeff_ao_eo.shape[1]
        nlo_imp  = coeff_ao_eo_imp.shape[1]
        nlo_env  = coeff_ao_lo_env.shape[1]
        neo_imp  = coeff_ao_eo_imp.shape[1]
        neo_bath = coeff_ao_eo_bath.shape[1]
        assert coeff_ao_eo.shape == (nao, neo)
        assert coeff_lo_eo.shape == (nlo, neo)

        emb_basis = EmbeddingBasis()
        emb_basis.nlo = nlo
        emb_basis.nao = nao
        emb_basis.neo = neo
        emb_basis.neo_imp = neo_imp
        emb_basis.neo_bath = neo_bath
        emb_basis.imp_eo_idx  = range(0, neo_imp)
        emb_basis.bath_eo_idx = range(neo_imp, neo)
        emb_basis.coeff_ao_eo = coeff_ao_eo
        emb_basis.coeff_lo_eo = coeff_lo_eo

        return emb_basis

    def build_emb_prob(self, m, mf,   emb_basis=None,
                       hcore_ao=None, eri_ao=None, 
                       dm_ll_lo=None, dm_ll_ao=None):
        """Build the embedding problem for the given impurity and environment orbitals.
        
        Parameters:
            mf : pyscf.scf.hf.RHF
                The mean field object, will use the get_veff method to get the
                effective potential for the core part of the density matrix.
                If the eri_ao is not given, will use the mol object in the mf
                to perform the integral transformation.
            coeff_lo_eo : numpy.ndarray
                The coefficient for LO to EO transformation.
            coeff_ao_eo : numpy.ndarray
                The coefficient for AO to EO transformation.
            ovlp_ao : numpy.ndarray
                The overlap matrix in AO basis, if not given, will compute it
                with the mf object.
            hcore_ao : numpy.ndarray
                The core Hamiltonian in AO basis, if not given, will compute it
                with the mf object.
            dm_ll_lo : numpy.ndarray
                The density matrix in LO basis.
            dm_ll_ao : numpy.ndarray
                The density matrix in AO basis.
            """

        assert dm_ll_lo is not None
        assert dm_ll_ao is not None

        neo = emb_basis.neo
        neo_imp = emb_basis.neo_imp
        neo_bath = emb_basis.neo_bath
        imp_eo_idx = emb_basis.imp_eo_idx
        bath_eo_idx = emb_basis.bath_eo_idx

        coeff_ao_eo = emb_basis.coeff_ao_eo
        coeff_lo_eo = emb_basis.coeff_lo_eo

        # Get the valence part of the density matrix in LO basis
        dm_ll_eo     = reduce(numpy.dot, (coeff_lo_eo.T, dm_ll_lo, coeff_lo_eo))
        dm_ll_lo_val = reduce(numpy.dot, (coeff_lo_eo,   dm_ll_eo, coeff_lo_eo.T))
        dm_ll_lo_cor = dm_ll_lo - dm_ll_lo_val
        dm_ll_ao_cor = reduce(numpy.dot, (coeff_ao_lo, dm_ll_lo_cor, coeff_ao_lo.T))

        # Build the veff of core part
        veff_ao_cor  = mf.get_veff(dm=dm_ll_ao_cor)
        veff_eo_cor  = reduce(numpy.dot, (coeff_ao_eo.T, veff_ao_cor, coeff_ao_eo))
        h1e_eo       = reduce(numpy.dot, (coeff_ao_eo.T, hcore_ao, coeff_ao_eo))
        f1e_eo       = h1e_eo + veff_eo_cor

        eri_eo = None
        if isinstance(m, pyscf.gto.Mole) or isinstance(m, pyscf.gto.Mole):
            eri_eo = pyscf.ao2mo.kernel(m, coeff_ao_eo, compact=False)
        else:
            assert eri_ao is not None
            eri_eo = pyscf.ao2mo.kernel(eri_ao, coeff_ao_eo, compact=False)
        eri_eo_full = pyscf.ao2mo.restore(1, eri_eo, neo)

        nelec = numpy.einsum('ii->', dm_ll_eo)
        nelec = numpy.round(nelec)
        nelec = int(nelec)

        assert nelec % 2 == 0
        nelecs = (nelec // 2, nelec // 2)
        
        emb_prob = EmbeddingProblem()
        emb_prob.nao    = nao
        emb_prob.nlo    = nlo
        emb_prob.neo    = neo
        emb_prob.neo_imp  = neo_imp
        emb_prob.neo_bath = neo_bath
        emb_prob.imp_eo_idx = imp_eo_idx
        emb_prob.bath_eo_idx = bath_eo_idx
        emb_prob.nelecs = nelecs
        emb_prob.dm0    = dm_ll_eo
        emb_prob.h1e    = h1e_eo
        emb_prob.f1e    = f1e_eo
        emb_prob.h2e    = eri_eo_full

        return emb_prob


if __name__ == '__main__':
    import pyscf
    from pyscf import lo

    imp_atm_idx_list = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    nfrag = len(imp_atm_idx_list)

    mol = pyscf.gto.Mole()
    mol.build(
        atom = """
           O         0.4183272099    0.1671038379    0.1010361156
           H         0.8784893276   -0.0368266484    0.9330933285
           H        -0.3195928737    0.7774121014    0.3045311682
           O         3.0208058979    0.6163509592   -0.7203724735
           H         3.3050376617    1.4762564664   -1.0295977027
           H         2.0477791789    0.6319690134   -0.7090745711
           O         2.5143150551   -0.2441947452    1.8660305097
           H         2.8954132119   -1.0661605274    2.1741344071
           H         3.0247679096    0.0221180670    1.0833062723
        """,
        basis = 'sto3g',
        verbose=0
    )
    ao_labels = mol.ao_labels(fmt=False)

    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    ovlp = mf.get_ovlp()
    ovlp_sqrt = scipy.linalg.sqrtm(ovlp)

    coeff_ao_lo = lo.PM(mol, mf.mo_coeff).kernel()
    nao, nlo = coeff_ao_lo.shape

    imp_ao_idx_list  = [[] for _ in range(nfrag)]
    imp_lo_idx_list  = [[] for _ in range(nfrag)]
    w = numpy.einsum("mn,np->mp", ovlp_sqrt, coeff_ao_lo)

    for ao_idx in range(nao):
        ao_label = ao_labels[ao_idx]
        atm_idx  = ao_label[0]
        for ifrag, imp_atm_idx in enumerate(imp_atm_idx_list):
            if atm_idx in imp_atm_idx:
                imp_ao_idx_list[ifrag].append(ao_idx)
                break

    for imp_lo_idx in range(nlo):
        ww_frags = []
        for ifrag in range(nfrag):
            ww = w[imp_ao_idx_list[ifrag], imp_lo_idx]
            ww_frag = numpy.einsum("m,m->", ww, ww)
            ww_frags.append(ww_frag)

        frag_idx = numpy.argmax(ww_frags)
        imp_lo_idx_list[frag_idx].append(imp_lo_idx)
        assert ww_frags[numpy.argmax(ww_frags)] > 0.9

    for ifrag in range(nfrag):
        print("Fragment %d: " % ifrag)
        print("  AO indices: ", imp_ao_idx_list[ifrag])
        print("  LO indices: ", imp_lo_idx_list[ifrag])

    dmet_obj = RDMETWithHF(mol, coeff_ao_lo=coeff_ao_lo, imp_lo_idx_list=imp_lo_idx_list, solver=None)
    dmet_obj.verbose = 0
    dmet_obj.kernel(dm0 = mf.make_rdm1())
    print("E = %.12f" % mf.energy_elec()[0])
