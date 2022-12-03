import sys, os
from functools import reduce

import numpy
import scipy

import pyscf
from pyscf import gto, scf, cc
from pyscf import ao2mo, lo, tools
from pyscf.lib import logger

class EmbeddingResults(object):
    pass

class SolverMixin(object):
    tmp_dir  = None
    def kernel(self, emb_prob, load_dir, save_dir):
        raise NotImplementedError
            
class RCCSD(SolverMixin):
    def kernel(self, emb_prob, load_dir, save_dir):
        scf_res = None
        if load_dir is not None:
            scf_file = os.path.join(load_dir, 'rhf.chk')
            assert os.path.exists(scf_file)
            scf_res = pyscf.scf.chkfile.load(scf_file, 'scf')

            cc_file = os.path.join(load_dir, 'rccsd.chk')
            assert os.path.exists(cc_file)
            cc_res = pyscf.cc.chkfile.load(cc_file, 'cc')

        m = Mole()
        mf = scf.RHF(m)
        mf.load_scf(scf_res)
        mf.kernel()

        cc = RCCSD(mf)
        cc.__dict__.update(cc_res)
        cc.kernel()

        if save_dir is not None:
            pyscf.scf.chkfile.save(save_dir, 'scf', mf)
            pyscf.cc.chkfile.save(save_dir, 'cc', cc)

        res = EmbeddingResults()
        res.mf = mf
        res.dm_hl = cc.make_rdm1()
        res.dm_hh = cc.make_rdm2()
        



class DMETMFMixin(object):
    _ovlp  = None
    _hcore = None
    _eri   = None

class DMETMixin(object):
    """
    The base class for DMET. Different DMET methods can be classified by different
    spin symmetry.
    """
    verbose = 4
    stdout  = sys.stdout

    conv_tol  = 1e-6
    max_cycle = 50

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

    def build_mf(self, m, dm0=None):
        raise NotImplementedError

    def build_emb_basis(self, imp_idx, dm0=None):
        """
        Build the embedding basis for the impurity.
        """
        raise NotImplementedError

    def build_emb_prob(self, coeff_ao_eo):
        raise NotImplementedError

    def get_emb_solver(self, ifrag):
        solver = self.solver
        assert not isinstance(solver, str)
        assert solver is not None

        if hasattr(solver, '__getitem__'):
            s = solver[ifrag]
        else:
            s = solver

        assert isinstance(s, SolverMixin)
        return s

    def transform_dm_ao_to_lo(self, dm_ao):
        raise NotImplementedError

    def transform_dm_lo_to_ao(self, dm_lo):
        raise NotImplementedError

    def transform_h_ao_to_lo(self, dm_ao):
        raise NotImplementedError

    def transform_h_ao_to_lo(self, dm_lo):
        raise NotImplementedError

    def kernel(self, dm0=None):
        self.dump_flags()
        
        m   = self.m
        mf  = self.build_mf(m, dm0=dm0)
        assert mf.converged
        coeff_ao_lo = self.coeff_ao_lo

        dm_ll_ao_pre = mf.make_rdm1()
        dm_ll_lo_pre = self.transform_dm_ao_to_lo(dm_ll_ao_pre, coeff_ao_lo=coeff_ao_lo, mf=mf)
        dm_ll_ao_cur = None
        dm_ll_lo_cur = None
        
        emb_res_list = []

        for ifrag in range(self.nfrag):
            imp_idx     = self.imp_idx_list[ifrag]
            env_idx     = self.env_idx_list[ifrag]
            res         = self.build_emb_basis(imp_idx, env_idx, dm0)
            coeff_ao_eo = res[0]
            coeff_lo_eo = res[1]

            emb_solver = self.get_emb_solver(ifrag=ifrag, dmet_iter=0)
            emb_prob   = self.build_emb_prob(
                coeff_lo_eo=coeff_lo_eo, coeff_ao_eo=coeff_ao_eo
                )
            emb_res    = emb_solver.kernel(emb_prob)
            emb_res_list.append(emb_res)

class RHFMole(pyscf.scf.hf.RHF, DMETMFMixin):
    def get_ovlp(self, mol=None):
        ovlp = self._ovlp
        if ovlp is None:
            if mol is None:
                mol = self.mol
            ovlp = pyscf.scf.hf.get_ovlp(mol)
        return ovlp

    def get_hcore(self, mol=None):
        hcore = self._hcore
        if hcore is None:
            if mol is None:
                mol = self.mol
            hcore = pyscf.scf.hf.get_hcore(mol)
        return hcore

class RDMETWithHF(DMETMixin):
    '''The class for solving spin restricted DMET problem in molecular system
    and the supercell gamma point periodic system.
    '''
    def __init__(self, mol, coeff_ao_lo=None, imp_lo_list=None, solver="ccsd"):
        self.m       = mol
        self.stdout  = mol.stdout
        self.verbose = mol.verbose

        assert coeff_ao_lo is not None
        self.coeff_ao_lo = coeff_ao_lo

        assert imp_lo_list is not None
        self.imp_lo_list = imp_lo_list

        self.solver = solver

    def transform_dm_ao_to_lo(self, dm_ao, coeff_ao_lo=None, mf=None):
        assert isinstance(mf, RHFMole)
        ovlp_ao  = mf.get_ovlp()
        dm_lo    = reduce(numpy.dot, (coeff_ao_lo.T, ovlp_ao, dm_ao, ovlp_ao, coeff_ao_lo))
        return dm_lo

    def transform_dm_lo_to_ao(self, dm_lo, coeff_ao_lo=None, mf=None):
        assert isinstance(mf, RHFMole)
        dm_ao    = reduce(numpy.dot, (coeff_ao_lo, dm_lo, coeff_ao_lo.T))
        return dm_ao

    def transform_h_ao_to_lo(self, h_ao, coeff_ao_lo=None, mf=None):
        assert isinstance(mf, RHFMole)
        h_lo    = reduce(numpy.dot, (coeff_ao_lo, h_ao, coeff_ao_lo.T))
        return h_lo

    def transform_h_lo_to_ao(self, h_lo, coeff_ao_lo=None, mf=None):
        assert isinstance(mf, RHFMole)
        ovlp_ao = mf.get_ovlp()
        h_ao    = reduce(numpy.dot, (coeff_ao_lo, ovlp_ao, h_lo, ovlp_ao, coeff_ao_lo.T))
        return h_ao
        
    def kernel(self, dm0=None):
        coeff_ao_lo = self.coeff_ao_lo
        nao, nlo    = coeff_ao_lo.shape

        mf       = self.build_scf(dm0=dm0)
        dm_ll_ao = mf.make_rdm1()

        coeff_ao_lo = self.coeff_ao_lo
        ovlp_ao     = mf.get_ovlp()
        hcore_ao    = mf.get_hcore()
        fock_ao     = mf.get_fock()
        eri_ao      = mf._eri
        assert eri_ao is not None

        dm_ll_ao = mf.make_rdm1()
        dm_ll_lo = self.transform_dm_ao_to_lo(dm_ll_ao, ovlp_ao)
        
        dm_ll_lo_pre = dm_ll_lo
        dm_ll_lo_cur = None

        nlo_imp_tot = sum([len(frag.imp_lo_idx) * (len(frag.imp_lo_idx) + 1) // 2 for frag in self.frag_list])
        vcors_pre   = numpy.zeros((nlo_imp_tot, ))
        vcors_cur   = None

        dmet_ene_pre = None
        dmet_ene_cur = None

        dmet_iter = 0
        is_dmet_max_iter  = False
        is_dmet_converged = False

        while not is_dmet_converged and not is_dmet_max_iter:
            dmet_ene = 0.0
            dm_hl_ao = numpy.zeros_like(dm_ll_ao)

            for ifrag, frag in enumerate(self.frag_list):
                imp_lo_idx = frag.imp_lo_idx
                env_lo_idx = frag.env_lo_idx

                nlo_imp = len(imp_lo_idx)
                nlo_env = len(env_lo_idx)

                assert nlo_imp + nlo_env == nlo

                imp_ix     = numpy.ix_(imp_lo_idx, imp_lo_idx)
                env_ix     = numpy.ix_(env_lo_idx, env_lo_idx)
                imp_env_ix = numpy.ix_(imp_lo_idx, env_lo_idx)

                dm_imp_imp_lo = dm_ll_lo_pre[imp_ix]
                dm_env_env_lo = dm_ll_lo_pre[env_ix]
                dm_imp_env_lo = dm_ll_lo_pre[imp_env_ix]

                assert dm_imp_imp_lo.shape == (nlo_imp, nlo_imp)
                assert dm_env_env_lo.shape == (nlo_env, nlo_env)
                assert dm_imp_env_lo.shape == (nlo_imp, nlo_env)

                u, s, vh = numpy.linalg.svd(dm_imp_env_lo, full_matrices=False)
                coeff_lo_eo_imp = numpy.eye(nlo_imp)
                coeff_lo_eo_env = vh.T

                coeff_ao_eo_imp = numpy.dot(coeff_ao_lo[:, frag.imp_lo_idx], coeff_lo_eo_imp)
                coeff_ao_eo_env = numpy.dot(coeff_ao_lo[:, frag.env_lo_idx], coeff_lo_eo_env)
                coeff_ao_eo = numpy.hstack((coeff_ao_eo_imp, coeff_ao_eo_env))
                coeff_lo_eo = reduce(numpy.dot, (coeff_ao_lo.T, ovlp_ao, coeff_ao_eo))

                neo = coeff_ao_eo.shape[1]
                assert coeff_ao_eo.shape == (nao, neo)
                assert coeff_lo_eo.shape == (nlo, neo)

                dm_ll_eo     = reduce(numpy.dot, (coeff_lo_eo.T, dm_ll_lo_pre, coeff_lo_eo))
                dm_ll_lo_val = reduce(numpy.dot, (coeff_lo_eo,   dm_ll_eo, coeff_lo_eo.T))
                dm_ll_lo_cor = dm_ll_lo_pre - dm_ll_lo_val
                dm_ll_ao_cor = reduce(numpy.dot, (coeff_ao_lo, dm_ll_lo_cor, coeff_ao_lo.T))

                veff_ao_cor  = mf.get_veff(dm=dm_ll_ao_cor)

                veff_eo_cor  = reduce(numpy.dot, (coeff_ao_eo.T, veff_ao_cor, coeff_ao_eo))

                h1e_eo      = reduce(numpy.dot, (coeff_ao_eo.T, hcore_ao, coeff_ao_eo))
                f1e_eo      = h1e_eo + veff_eo_cor
                eri_eo      = pyscf.ao2mo.incore.full(eri_ao, coeff_ao_eo)
                eri_eo_full = pyscf.ao2mo.restore(1, eri_eo, neo)

                hl_solver, dm1_emb_eo, dm2_emb_eo  = build_ccsd_solver(f1e_eo, eri_eo, dm0=dm_ll_eo)

                dmet_ene += numpy.einsum('pq,qp->', h1e_eo[:nlo_imp, :], dm1_emb_eo[:, :nlo_imp]) / 2.0
                dmet_ene += numpy.einsum('pq,qp->', f1e_eo[:nlo_imp, :], dm1_emb_eo[:, :nlo_imp]) / 2.0
                dmet_ene += numpy.einsum('pqrs,pqrs->', eri_eo_full[:nlo_imp, :, :, :], dm2_emb_eo[:nlo_imp, :, :, :]) / 2.0
                
                dm1_imp_imp_eo = dm1_emb_eo[:nlo_imp, :nlo_imp]
                dm1_imp_env_eo = dm1_emb_eo[:nlo_imp, nlo_imp:]
                dm1_env_imp_eo = dm1_emb_eo[nlo_imp:, :nlo_imp]

                dm_hl_ao += reduce(numpy.dot, (coeff_ao_eo_imp, dm1_imp_imp_eo, coeff_ao_eo_imp.T))
                dm_hl_ao += reduce(numpy.dot, (coeff_ao_eo_imp, dm1_imp_env_eo, coeff_ao_eo_env.T)) * 0.5
                dm_hl_ao += reduce(numpy.dot, (coeff_ao_eo_env, dm1_env_imp_eo, coeff_ao_eo_imp.T)) * 0.5

            dmet_ene_cur = dmet_ene
            # print("SCF energy = %16.12f, DMET energy = %16.12f" % (mf.energy_elec()[0], dmet_ene))

            def get_dm_err(vcors):
                vcor_lo = numpy.zeros((nlo, nlo))
                assert vcors.shape == (nlo_imp_tot,)
                nlo_imp_tot_ = 0

                for ifrag, frag in enumerate(self.frag_list):
                    imp_lo_idx = frag.imp_lo_idx
                    nlo_imp    = len(imp_lo_idx)
                    imp_ix     = numpy.ix_(imp_lo_idx, imp_lo_idx)

                    vcor_tmp   = numpy.zeros((nlo_imp, nlo_imp))
                    vcor_tmp[numpy.tril_indices(nlo_imp)] = vcors[nlo_imp_tot_:nlo_imp_tot_+nlo_imp*(nlo_imp+1)//2]
                    vcor_lo[imp_ix] = vcor_tmp

                    nlo_imp_tot_ += nlo_imp*(nlo_imp+1)//2
                
                assert nlo_imp_tot_ == nlo_imp_tot
                assert vcors.shape  == (nlo_imp_tot,)
                vcor_ao = reduce(numpy.dot, (coeff_ao_lo, ovlp_ao, vcor_lo, ovlp_ao, coeff_ao_lo.T))

                ene_mo, coeff_lo_mo = mf.eig(fock_ao + vcor_ao, ovlp_ao)
                dm_fit_ao = mf.make_rdm1(coeff_lo_mo)

                dm_err = numpy.linalg.norm(dm_fit_ao - dm_hl_ao)
                return dm_err

            fitting_iter = 0
            def callback(vcors):
                nonlocal fitting_iter
                dm_err = get_dm_err(vcors)
                # print("fitting_iter = %6d, dm_err = %12.8f" % (fitting_iter, dm_err))
                fitting_iter += 1

            vcors_init = numpy.zeros((nlo_imp_tot,))
            res = scipy.optimize.minimize(
                get_dm_err, vcors_init, method='BFGS', 
                tol=1e-6, callback=callback, 
                options={'disp': False, 'maxiter': 1000}
            )

            vcors_cur = res.x
            vcor_lo   = numpy.zeros((nlo, nlo))

            nlo_imp_tot_ = 0
            for ifrag, frag in enumerate(self.frag_list):
                imp_lo_idx = frag.imp_lo_idx
                nlo_imp    = len(imp_lo_idx)
                imp_ix     = numpy.ix_(imp_lo_idx, imp_lo_idx)

                vcor_tmp   = numpy.zeros((nlo_imp, nlo_imp))
                vcor_tmp[numpy.tril_indices(nlo_imp)] = vcors_cur[nlo_imp_tot_:nlo_imp_tot_+nlo_imp*(nlo_imp+1)//2]
                vcor_lo[imp_ix] = vcor_tmp

                nlo_imp_tot_ += nlo_imp*(nlo_imp+1)//2

            assert nlo_imp_tot_ == nlo_imp_tot
           
            vcor_ao = reduce(numpy.dot, (coeff_ao_lo, ovlp_ao, vcor_lo, ovlp_ao, coeff_ao_lo.T))

            ene_mo, coeff_lo_mo = mf.eig(fock_ao + vcor_ao, ovlp_ao)
            dm_ll_ao     = mf.make_rdm1(coeff_lo_mo)
            dm_ll_lo_cur = reduce(numpy.dot, (coeff_ao_lo.T, ovlp_ao, dm_ll_ao, ovlp_ao, coeff_ao_lo))

            dm_err    = numpy.linalg.norm(dm_ll_lo_cur - dm_ll_lo_pre)
            vcors_err = numpy.linalg.norm(vcors_cur - vcors_pre)
            ene_err   = 1.0 if dmet_ene_pre is None else abs(dmet_ene_cur - dmet_ene_pre)
            dmet_ene_pre = dmet_ene_cur

            is_dmet_converged = (dm_err < self.dmet_tol) and (vcors_err < self.dmet_tol)
            is_dmet_converged = is_dmet_converged or (dmet_iter >= self.dmet_max_iter)
            is_dmet_max_iter  = (dmet_iter >= self.dmet_max_iter)

            print("DMET iter = %4d, DMET energy = %16.12f, DMET error = %6.4e, vcors_err = %6.4e, dm_err = %6.4e" % (dmet_iter, dmet_ene_cur, ene_err, vcors_err, dm_err))

            dm_ll_lo_pre = dm_ll_lo_cur
            vcors_pre    = vcors_cur

            dmet_iter += 1


if __name__ == '__main__':
    import pyscf
    from pyscf import lo

    imp_atm_idx_list = [[0, 1, 2], [3, 5, 6], [4, 7, 8]] # , [9, 12, 13], [10, 14, 15], [11, 16, 17]]
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
        verbose=4
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
    ww = numpy.einsum("mn,np->mp", ovlp_sqrt, coeff_ao_lo)

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
            ww_frag = numpy.einsum("m,m->", ww[imp_ao_idx_list[ifrag], imp_lo_idx], ww[imp_ao_idx_list[ifrag], imp_lo_idx])
            ww_frags.append(ww_frag)

        frag_idx = numpy.argmax(ww_frags)
        imp_lo_idx_list[frag_idx].append(imp_lo_idx)
        assert ww_frags[numpy.argmax(ww_frags)] > 0.9

    dmet = MoleculeDMET(mol, coeff_ao_lo, imp_lo_idx_list)
    dmet.kernel()

