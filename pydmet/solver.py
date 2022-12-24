import os
import sys
from collections.abc import Iterable
from functools import reduce

import numpy
import scipy

import pyscf
from pyscf import ao2mo, cc, gto
from pyscf import lo, scf, tools
from pyscf.scf import cphf   
from pyscf.lib import chkfile, logger
from pyscf.scf.hf import dot_eri_dm
from pyscf.scf import _response_functions  # noqa

class SolverMixin(object):
    verbose  = 0
    stdout   = sys.stdout
    tmp_dir  = None

    conv_tol = 1e-8
    conv_tol_grad = 1e-5
    max_cycle = 50

    def dump_flags(self):
        pass

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

        s1e  = numpy.eye(norb)
        f1e  = emb_prob.f1e
        mu1e = emb_prob.id_imp * emb_prob.mu
        h2e  = emb_prob.h2e
        imp_eo_idx = emb_prob.imp_eo_idx

        mf = pyscf.scf.RHF(m)
        mf.verbose   = self.verbose
        mf.stdout    = self.stdout
        mf.get_ovlp  = lambda *args: s1e
        mf.get_hcore = lambda *args: f1e + mu1e
        mf._eri      = h2e
        mf.conv_tol  = self.conv_tol
        mf.max_cycle = self.max_cycle
        mf.kernel(dm0=dm0)

        if mf.converged:
            logger.info(self, 'RHF converged')
        else:
            logger.warn(self, 'RHF not converged')

        mo_occ    = mf.mo_occ
        mo_energy = mf.mo_energy
        mo_coeff  = mf.mo_coeff

        mf_res = {
            'mo_coeff': mo_coeff,
            'rdm1': mf.make_rdm1(),
        }

        if save_dir is not None:
            assert os.path.isdir(save_dir)
            save_file = os.path.join(save_dir, 'mf.h5')
            logger.info(self, 'Saving SCF results to %s', save_file)
            chkfile.save(save_file, 'scf', mf_res)

        nvir = mo_occ[mo_occ == 0].size
        nocc = mo_occ[mo_occ >  0].size
        orbv = mo_coeff[:, mo_occ == 0]
        orbo = mo_coeff[:, mo_occ >  0]

        imp_1e_eo = emb_prob.id_imp
        mu_1e_oo = reduce(numpy.dot, (orbo.T, imp_1e_eo, orbo))
        mu_1e_vv = reduce(numpy.dot, (orbv.T, imp_1e_eo, orbv))
        mu_1e_ov = reduce(numpy.dot, (orbv.T, imp_1e_eo, orbo))
        mu_1e_vo = reduce(numpy.dot, (orbv.T, imp_1e_eo, orbo))

        # set singlet=None, generate function for CPHF type response kernel
        vresp = mf.gen_response(singlet=None, hermi=1)   
        def fvind(x):  # For singlet, closed shell ground state
            dm   = reduce(numpy.dot, (orbv, x.reshape(nvir, nocc)*2, orbo.T))
            v1ao = vresp(dm+dm.T)
            return reduce(numpy.dot, (orbv.T, v1ao, orbo)).ravel()

        z_vo  = cphf.solve(fvind, mo_energy, mo_occ, mu_1e_vo,
                           max_cycle=100, tol=1e-8, 
                           verbose=self.verbose)[0]
        z_vo  = z_vo.reshape(nvir, nocc)

        dn_dmu = 4.0 * numpy.einsum("ai,ai->", mu_1e_vo, z_vo)
        return mf, dn_dmu

    def energy_elec(self, rdm1, rdm2, emb_prob=None):
        imp_eo_idx  = emb_prob.imp_eo_idx
        f1e = emb_prob.f1e
        h1e = emb_prob.h1e
        h2e = emb_prob.h2e
        ene_elec    = numpy.einsum('pq,  qp  ->', (f1e + h1e)[imp_eo_idx, :], rdm1[:, imp_eo_idx])       / 2.0
        ene_elec   += numpy.einsum('pqrs,pqrs->', h2e[imp_eo_idx, :, :, :],   rdm2[imp_eo_idx, :, :, :]) / 2.0
        return ene_elec

    def finalize(self, rdm1, rdm2, emb_prob=None, dn_dmu_hf=None):
        from pydmet.embedding import EmbeddingResults
        emb_res = EmbeddingResults()
        emb_res.rdm1      = rdm1
        emb_res.rdm1_ao   = None
        emb_res.dn_dmu_hf = dn_dmu_hf
        emb_res.energy_elec = self.energy_elec(rdm1, rdm2, emb_prob)
        return emb_res

class RHF(SolverMixin):
    def kernel(self, emb_prob, load_dir=None, save_dir=None):
        mf, dn_dmu_hf = self.solve_mf(emb_prob, load_dir, save_dir)

        mo_coeff = mf.mo_coeff
        rdm1     = mf.make_rdm1()
        rdm2     = mf.make_rdm2()

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

        emb_res = self.finalize(rdm1, rdm2, emb_prob, dn_dmu_hf=dn_dmu_hf)
        # for the specific solver
        emb_res.mo_coeff  = mo_coeff # Save MO coefficients as results
        return emb_res

class RCCSD(SolverMixin):
    conv_tol_normt   = 1e-6
    is_approx_lambda = False

    def kernel(self, emb_prob, load_dir=None, save_dir=None):
        mf, dn_dmu = self.solve_mf(emb_prob, load_dir, save_dir)

        mo_coeff = mf.mo_coeff
        rdm1     = mf.make_rdm1()
        rdm2     = mf.make_rdm2()

        cc = pyscf.cc.RCCSD(mf)
        cc.verbose   = self.verbose
        cc.conv_tol  = self.conv_tol
        cc.conv_tol_normt = self.conv_tol_normt
        cc.max_cycle = self.max_cycle
        eris = cc.ao2mo(mo_coeff)
        tmp  = cc.kernel(eris=eris)
        t1   = tmp[1]
        t2   = tmp[2]

        if cc.converged:
            logger.info(self, 'RCCSD converged')
        else:
            logger.warn(self, 'RCCSD not converged')

        l1 = t1
        l2 = t2
        if not self.is_approx_lambda:
            l1, l2 = cc.solve_lambda(t1=t1, t2=t2, l1=l1, l2=l2, eris=eris)

        rdm1 = cc.make_rdm1(t1=t1, t2=t2, l1=l1, l2=l2, ao_repr=True)
        rdm2 = cc.make_rdm2(t1=t1, t2=t2, l1=l1, l2=l2, ao_repr=True)

        cc_res = {
            't1':  cc.t1,
            't2':  cc.t2,
            'rdm1': rdm1,
            'rdm2': rdm2,
        }

        if save_dir is not None:
            assert os.path.isdir(save_dir)
            save_file = os.path.join(save_dir, 'rccsd.h5')
            logger.info(self, 'Saving RCCSD results to %s', save_file)
            chkfile.save(save_file, 'cc', cc_res)
        
        emb_res = self.finalize(rdm1, rdm2, emb_prob)
        emb_res.t1 = t1 # Save t1 as results
        emb_res.t2 = t2 # Save t2 as results
        emb_res.l1 = l1
        emb_res.l2 = l2
        emb_res.mo_coeff = mo_coeff # Save MO coefficients as results
        emb_res.dn_dmu = dn_dmu
        return emb_res
