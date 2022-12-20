import os, sys 
from collections.abc import Iterable
from functools import reduce
from itertools import chain

import numpy
import scipy

import pyscf
from pyscf import ao2mo, cc, gto
from pyscf import lo, scf, tools
from pyscf.lib import chkfile, logger
from pyscf.scf.hf import dot_eri_dm

class DMET(object):
    """The base class for DMET. Different DMET methods can be classified by different
    spin symmetry.

    Attributes:
        verbose : 
            The verbosity level. 0 for silent, 4 for the maximum verbosity.
        stdout : 
            The stdout, default to sys.stdout.
        vir_lo_idx :
            The indices of the virtual orbitals to be excluded from the
            bath construction.
        cor_lo_idx :
            The indices of the core orbitals to be excluded from the
            bath construction.
        conv_tol :
            The convergence threshold for the DMET one-body density matrix,
            correlation potential and the energy.
        max_cycle :
            The maximum number of DMET iterations. If wish to to one-shot
            DMET, use or max_cycle=1.
        solver_conv_tol :
            The convergence threshold for the solver.
        solver_max_cycle :
            The maximum number of solver iterations.
        save_dir :
            The directory to save the results, default to None, which will 
            generate a temporary directory, and the results will be saved
            in the temporary directory.
        restart_dir :
            The directory to restart the DMET calculation, default to None.
            If the directory is not None, the calculation will try to load
            the results from the directory and restart the calculation, 
            if the loading is failed, the calculation will stop.

    Methods:
        kernel :
            The main function to run the DMET calculation. Shared by all the DMET
            child classes.
    """
    verbose = 0
    stdout  = sys.stdout

    min_weight = 0.4
    vir_lo_idx = None
    cor_lo_idx = None

    max_cycle     = 50
    conv_tol      = 1e-6
    conv_tol_grad = 1e-4

    solver_conv_tol  = 1e-6
    solver_max_cycle = 50

    save_dir = None
    load_dir = None
    restart_dir = None

    def __init__(self, mf, coeff_ao_lo=None, imp_lo_idx_list=None, 
                 solver=None, mu_fitting=None, vcor_fitting=None):
        if not mf.converged:
            sys.stderr.write('Warning: mean-field object is not converged.')

        self._m      = mf.mol
        self.stdout  = mf.stdout
        self.verbose = mf.verbose
        self._coeff_ao_lo = coeff_ao_lo

        self._nfrag = None
        self._imp_lo_idx_list = imp_lo_idx_list
        self._env_lo_idx_list = None

        self._solver = solver
        self._mu_fitting = mu_fitting
        self._vcor_fitting = vcor_fitting

        self._base   = mf
        self._hcore_ao = None
        self._ovlp_ao  = None
        self._fock_ao  = None

    def _method_name(self):
        method_name = []
        for c in self.__class__.__mro__:
            if issubclass(c, DMET) and c is not DMET:
                method_name.append(c.__name__)
        return '-'.join(method_name)

    def dump_flags(self):
        log = logger.new_logger(self, self.verbose)

        log.info("\n")
        log.info("******** %s ********", self.__class__)
        log.info('method = %s', self._method_name())

        solver = self.solver
        assert solver is not None
        self.solver.dump_flags()

        nfrag = self.nfrag
        self.dump_frag_info()

        # self.dump_mu_info()
        # self.dump_vcor_info()

    def dump_frag_info(self):
        log = logger.new_logger(self, self.verbose)

        coeff_ao_lo = self.coeff_ao_lo
        nao, nlo    = coeff_ao_lo.shape

        nfrag = self.nfrag
        imp_lo_idx_list = self.imp_lo_idx_list
        env_lo_idx_list = self.env_lo_idx_list

        lo_label_list = self.get_lo_label_list()

        log.info("\n")
        for ifrag in range(nfrag):
            log.info("******** Fragment-%d ********", ifrag)
            imp_lo_idx = imp_lo_idx_list[ifrag]
            env_lo_idx = env_lo_idx_list[ifrag]

            log.info("Impurity orbitals:")
            for p in imp_lo_idx:
                lo_label = lo_label_list[p]
                if lo_label is None:
                    log.info("%10s"%(f"LO-{p}"))
                else:
                    s = "%10s: %10s"%(f"LO-{p}", lo_label)
                    log.info(s)

            log.info("\nEnvironment orbitals:")
            for p in env_lo_idx:
                lo_label = lo_label_list[p]
                if lo_label is None:
                    log.info("%10s"%(f"LO-{p}"))
                else:
                    s = "%10s: %10s"%(f"LO-{p}", lo_label)
                    log.info(s)

            log.info("")

        if (0 if self.vir_lo_idx is None else len(self.vir_lo_idx)) > 0:
            log.info("Virtual orbitals excluded from the bath construction:")
            for p in self.vir_lo_idx:
                lo_label = lo_label_list[p]
                if lo_label is None:
                    log.info("   LO-%d", p)
                else:
                    log.info("   LO-%3d %10s", p, lo_label)
        else:
            log.info("No virtual orbitals excluded from the bath construction.")

        lo_idx_list  = list(chain.from_iterable(imp_lo_idx_list))
        lo_idx_list += [] if self.vir_lo_idx is None else self.vir_lo_idx
        lo_idx_list += [] if self.cor_lo_idx is None else self.cor_lo_idx
        lo_idx_list  = list(numpy.sort(lo_idx_list))

        for p in range(nlo):
            if p not in lo_idx_list: #TODO: and p not in self.vir_lo_idx:
                lo_label = lo_label_list[p]
                if lo_label is None:
                    log.warn(
                        "LO %d (%s) is not assigned to either any fragment or the core/virtual part.\n Please make sure this is what you want.", p
                        )
                else:
                    log.warn(
                        "LO %3d is not assigned to any either any fragment or the core/virtual part.\n Please make sure this is what you want.", p
                        )


    def dump_solver_info(self):
        log = logger.new_logger(self, self.verbose)

        log.info("\n")
        log.info("******** %s ********", self.__class__)
        log.info('method = %s', self._method_name())

        solver = self.solver

        log.info("\n******** Solver ********")
        if isinstance(solver, Iterable):
            solver_list = list(solver)
            log.info("List of solver objects are given.")
            assert len(solver_list) == self.nfrag

            for solver_obj in solver_list:
                assert isinstance(solver_obj, SolverMixin)
                # TODO: solver_obj.dump_flags()
        else:
            solver_obj = solver
            assert isinstance(solver_obj, SolverMixin)
            # TODO: solver_obj.dump_flags()

    def sanity_check(self):
        raise NotImplementedError

    def build(self):
        if self._fock_ao is None or self._hcore_ao is None or self._ovlp_ao is None:
            self.build_mf()

        assert self._base.converged
        assert self._base is not None
        assert self._fock_ao is not None
        assert self._hcore_ao is not None
        assert self._ovlp_ao is not None

        if self._nfrag is None:
            self.build_frag()

        assert self._coeff_ao_lo is not None

    def build_mf(self):
        raise NotImplementedError

    def build_frag(self):
        raise NotImplementedError

    def get_frag(self):
        raise NotImplementedError

    def make_emb_basis(self):
        raise NotImplementedError

    def make_emb_prob(self):
        raise NotImplementedError

    def make_emb_solver(self, ifrag):
        from pydmet.solver import SolverMixin
        solver = self.solver
        assert solver is not None

        if isinstance(solver, Iterable):
            nfrag = self.nfrag
            assert len(solver_list) == self.nfrag
            solver_obj = solver[ifrag]

        else:
            solver_obj = solver

        assert isinstance(solver_obj, SolverMixin)
        return solver_obj

    def get_veff_ao(self, dm_ao=None):
        assert self._base is not None
        veff_ao = self._base.get_veff(dm_ao=dm_ao)
        return veff_ao

    def get_mf_rdm1_ao(self, mo_energy=None, mo_coeff=None):
        if mo_energy is None or mo_coeff is None:
            assert self._base is not None
            assert self._base.converged
            mo_energy = self._base.mo_energy
            mo_coeff  = self._base.mo_coeff

        mo_occ = self._base.get_occ(mo_energy, mo_coeff)
        dm_ao  = self._base.make_rdm1(mo_coeff, mo_occ)
        return dm_ao

    def get_nelec(self, dm_lo=None, dm_ao=None, ovlp_ao=None):
        raise NotImplementedError

    def get_vcor(self):
        """Get the correlation potential from the vcor parameters (parameters in LO).
        Return the correlation potential in both AO and LO basis.
        """
        raise NotImplementedError

    def transform_dm_ao_to_lo(self):
        raise NotImplementedError

    def transform_dm_lo_to_ao(self):
        raise NotImplementedError

    def transform_h_ao_to_lo(self):
        raise NotImplementedError

    def transform_h_ao_to_lo(self):
        raise NotImplementedError

    def solve_mu_fitting(self, mu0=0.0, nelec_tot_target=None, dm_ll_lo=None, dm_ll_ao=None):
        """Solve the high-level problem with given chemical potential.
        
        Parameters:
            mu : float
                The chemical potential.
            dm_ll_lo : numpy.ndarray
                The low-level density matrix in local basis.
            dm_ll_ao : numpy.ndarray
                The low-level density matrix in AO basis.
                
        Returns:
            energy_elec : float
                The electronic energy.
            nelec_tot : float
                The total number of electrons.
            dm_hl_ao : numpy.ndarray
                The high-level density matrix in AO basis.
            dm_hl_lo : numpy.ndarray
                The high-level density matrix in local basis.
        """

        # The chemical potential shall be added to the diagonal terms of
        # in LO basis, here we use the AO basis equivalent.
        nfrag = self.nfrag
        coeff_ao_lo = self.coeff_ao_lo
        imp_lo_idx_list = self.imp_lo_idx_list
        env_lo_idx_list = self.env_lo_idx_list

        log = logger.new_logger(self, self.verbose)
        log.info("\n")
        log.info("******** Solve High-Level with Fitting Chemical Potential ********")

        res = self.solve_hl_with_mu(0.0, dm_ll_lo=dm_ll_lo, dm_ll_ao=dm_ll_ao)
        
        mu_list = []
        nelec_tot_list = []

        for mu in [-0.008, -0.004, 0.00, 0.004, 0.008]:
            log.info("mu = %f", mu)
            res = self.solve_hl_with_mu(mu, dm_ll_lo=dm_ll_lo, dm_ll_ao=dm_ll_ao)
            energy_elec, nelec_tot, dm_hl_ao, dm_hl_lo = res
            mu_list.append(mu)
            nelec_tot_list.append(nelec_tot)

        dn_dmu = (nelec_tot_list[4] - nelec_tot_list[0]) / (mu_list[4] - mu_list[0])
        print("dn_dmu = %12.8f" % dn_dmu)

        dn_dmu = (nelec_tot_list[3] - nelec_tot_list[1]) / (mu_list[3] - mu_list[1])
        print("dn_dmu = %12.8f" % dn_dmu)

        assert 1 == 2

        # energy_elec   = 0.0
        # dm_hl_ao_list = []

        # res_list = []
        
        # def nelec_err(mu):
        #     res = self.solve_hl_with_mu(mu, dm_ll_lo=dm_ll_lo, dm_ll_ao=dm_ll_ao)
            
        #     energy_elec, nelec_tot, dm_hl_ao, dm_hl_lo = res
        #     nelec_err = nelec_tot - nelec_tot_target

        #     res_list.append((mu, nelec_err, *res))
        #     return nelec_err

        # mu  = scipy.optimize.brentq(nelec_err, mu0 - 1.0, mu0 + 1.0)
        # for res in res_list:
        #     mu, nelec_err, energy_elec, nelec_tot, dm_hl_ao, dm_hl_lo = res
        #     print("mu = %12.6f, nelec_err = % 6.4e, energy_elec = % 12.6f, nelec_tot = % 12.6f" % (mu, nelec_err, energy_elec, nelec_tot))
        # res_list = None
        
        # return mu

    def solve_vcor_fitting(self):
        raise NotImplementedError

    def solve_hl_with_mu(self, mu=0.0, dm_ll_lo=None, dm_ll_ao=None):
        """Solve the high-level problem with given chemical potential.
        
        Parameters:
            mu : float
                The chemical potential.
            dm_ll_lo : numpy.ndarray
                The low-level density matrix in local basis.
            dm_ll_ao : numpy.ndarray
                The low-level density matrix in AO basis.
                
        Returns:
            energy_elec : float
                The electronic energy.
            nelec_tot : float
                The total number of electrons.
            dm_hl_ao : numpy.ndarray
                The high-level density matrix in AO basis.
            dm_hl_lo : numpy.ndarray
                The high-level density matrix in local basis.
        """

        # The chemical potential shall be added to the diagonal terms of
        # in LO basis, here we use the AO basis equivalent.

        solver = self.solver
        nfrag = self.nfrag
        imp_lo_idx_list = self.imp_lo_idx_list
        env_lo_idx_list = self.env_lo_idx_list

        log = logger.new_logger(self, self.verbose)
        log.info("\n")
        log.info("******** Solve High-Level Problems with mu = %6.4f ********", mu)

        energy_elec   = 0.0
        dn_dmu        = 0.0
        dm_hl_ao_list = []

        for ifrag in range(nfrag):
            imp_lo_idx = imp_lo_idx_list[ifrag]
            env_lo_idx = env_lo_idx_list[ifrag]
            
            save_dir = self.save_dir 
            if save_dir is not None:
                save_dir = os.path.join(save_dir, 'frag_%d' % ifrag)
                if not os.path.isdir(save_dir):
                    os.makedirs(save_dir)

            load_dir = None

            emb_basis = self.make_emb_basis(
                imp_lo_idx, env_lo_idx, 
                dm_ll_ao=dm_ll_ao,
                dm_ll_lo=dm_ll_lo,
                )

            emb_solver = self.make_emb_solver(ifrag)

            emb_prob   = self.make_emb_prob(
                mu=mu, emb_basis=emb_basis,
                dm_ll_ao=dm_ll_ao,
                dm_ll_lo=dm_ll_lo,
                )

            # Note that the embbedding results from the solver may
            # not be consistent with the DMET type. So we need to
            # transform the results to the DMET type.
            emb_res = emb_solver.kernel(
                emb_prob=emb_prob,
                save_dir=save_dir, 
                load_dir=load_dir,
            )

            energy_elec += emb_res.energy_elec
            dn_dmu      += emb_res.dn_dmu
            dm_hl_ao_list.append(self.get_emb_rdm1_ao(emb_res, emb_basis))

        print("energy_elec = % 12.6f, dn_dmu = % 12.6f" % (energy_elec, dn_dmu))
        dm_hl_ao  = self.combine_rdm1_ao(dm_hl_ao_list)
        dm_hl_lo  = self.transform_dm_ao_to_lo(dm_hl_ao)
        nelec_tot = self.get_nelec_tot(dm_ao=dm_hl_ao, dm_lo=dm_hl_lo)

        return energy_elec, nelec_tot, dm_hl_ao, dm_hl_lo

    def kernel(self, mu0=0.0, dm0=None):
        log = logger.new_logger(self, self.verbose)

        self.build()
        self.dump_flags()

        nfrag = self.nfrag
        imp_lo_idx_list = self.imp_lo_idx_list
        env_lo_idx_list = self.env_lo_idx_list
        
        dm_ll_ao   = self.get_mf_rdm1_ao()
        dm_ll_lo   = self.transform_dm_ao_to_lo(dm_ll_ao)
        nelec_tot0 = self.get_nelec_tot(dm_ao=dm_ll_ao, dm_lo=dm_ll_lo)

        log.info("nelec_tot0 = % 8.6f", nelec_tot0)

        mu_fitting   = self.mu_fitting
        vcor_fitting = self.vcor_fitting

        # Run the DMET high-level problem with given mu0.
        emb_prob_list = []
        emb_res_list  = []

        for ifrag in range(nfrag):
            imp_lo_idx = imp_lo_idx_list[ifrag]
            env_lo_idx = env_lo_idx_list[ifrag]

            emb_basis = self.make_emb_basis(
                imp_lo_idx, env_lo_idx, 
                dm_ll_ao=dm_ll_ao,
                dm_ll_lo=dm_ll_lo,
                )

            emb_solver = self.make_emb_solver(ifrag)

            emb_prob   = self.make_emb_prob(
                mu=mu0, emb_basis=emb_basis,
                dm_ll_ao=dm_ll_ao,
                dm_ll_lo=dm_ll_lo,
                )

            emb_res = emb_solver.kernel(
                emb_prob=emb_prob,
                save_dir=None,
                load_dir=None,
            )

            emb_prob_list.append(emb_prob)
            emb_res_list.append(emb_res)

        assert 1 == 2

        if mu_fitting is not None:
            mu_fitting.solve(
                mu0=mu0,
                emb_prob_list=emb_prob_list,
                emb_res_list=emb_res_list,
            )

        if vcor_fitting is not None:
            pass

        assert 1 == 2

class RHF(DMET):
    '''The class for solving spin restricted DMET problem in molecular system
    and the supercell gamma point periodic system.
    '''

    def build_mf(self):
        log = logger.new_logger(self, self.verbose)

        assert self._base is not None
        assert isinstance(self._base, pyscf.scf.hf.SCF)

        if isinstance(self._base, pyscf.dft.rks.KohnShamDFT):
            log.warn("The mean-field object is a Kohn-Sham DFT object.")
        
        assert isinstance(self._base, pyscf.scf.hf.RHF), "Shall be initialized with RHF object."
        assert self._base.converged, "The mean-field object is not converged."

        dm_ao = self._base.make_rdm1()

        if self._hcore_ao is None:
            self._hcore_ao = self._base.get_hcore()
        else:
            log.warn("_hcore_ao is set manually,\nplease make sure it is what you want.")

        if self._ovlp_ao is None:
            self._ovlp_ao = self._base.get_ovlp()
        else:
            log.warn("_ovlp_ao is set manually,\nplease make sure it is what you want.")

        if self._fock_ao is None:
            self._fock_ao = self._base.get_fock(
                h1e=self._hcore_ao, dm=dm_ao,
                )
        else:
            log.warn("_fock_ao is set manually,\nplease make sure it is what you want.")

        mo_energy, mo_coeff = self._base.eig(self._fock_ao, self._ovlp_ao)
        mo_occ = self._base.get_occ(mo_energy=mo_energy, mo_coeff=mo_coeff)
        norm_gorb = numpy.linalg.norm(self._base.get_grad(mo_coeff, mo_occ, self._fock_ao))
        norm_gorb = norm_gorb / numpy.sqrt(norm_gorb.size)

        log.info("norm_gorb = % 12.6e", norm_gorb)
        assert norm_gorb < self.conv_tol_grad, "The mean-field object is not converged."

    def transform_dm_ao_to_lo(self, dm_ao):
        ovlp_ao     = self.get_ovlp_ao()
        coeff_ao_lo = self.coeff_ao_lo

        if ovlp_ao is not None:
            dm_lo = reduce(numpy.dot, (coeff_ao_lo.T, ovlp_ao, dm_ao, ovlp_ao, coeff_ao_lo))
        else:
            dm_lo = reduce(numpy.dot, (coeff_ao_lo.T, dm_ao, coeff_ao_lo))

        return dm_lo

    def transform_dm_lo_to_ao(self, dm_lo):
        coeff_ao_lo = self.coeff_ao_lo
        dm_ao = reduce(numpy.dot, (coeff_ao_lo, dm_lo, coeff_ao_lo.T))
        return dm_ao

    def transform_h_ao_to_lo(self, h_ao):
        coeff_ao_lo = self.coeff_ao_lo
        h_lo    = reduce(numpy.dot, (coeff_ao_lo, h_ao, coeff_ao_lo.T))
        return h_lo

    def transform_h_lo_to_ao(self, h_lo):
        ovlp_ao     = self.get_ovlp_ao()
        coeff_ao_lo = self.coeff_ao_lo

        if ovlp_ao is not None:
            h_ao    = reduce(numpy.dot, (coeff_ao_lo.T, ovlp_ao, h_lo, ovlp_ao, coeff_ao_lo))
        else:
            h_ao    = reduce(numpy.dot, (coeff_ao_lo.T, h_lo, coeff_ao_lo))

        return h_ao

    def build_frag(self):
        """Build the environment orbital index list for each fragment.
        Exclude the core and virtual LOs.
        """
        log = logger.new_logger(self, self.verbose)
        coeff_ao_lo = self.coeff_ao_lo
        nao, nlo = coeff_ao_lo.shape

        imp_lo_idx_list = self.imp_lo_idx_list
        nfrag = len(imp_lo_idx_list)

        env_lo_idx_list = None
        if self._env_lo_idx_list is None:
            vir_lo_idx = self.vir_lo_idx
            if self.vir_lo_idx is None:
                vir_lo_idx = []

            cor_lo_idx = self.cor_lo_idx
            if self.cor_lo_idx is None:
                cor_lo_idx = []

            env_lo_idx_list = []
            for ifrag in range(nfrag):
                imp_lo_idx = imp_lo_idx_list[ifrag]
                env_lo_idx = []
                for p in range(nlo):
                    is_env = p not in imp_lo_idx
                    is_env = is_env and (p not in vir_lo_idx)
                    is_env = is_env and (p not in cor_lo_idx)
                    if is_env:
                        env_lo_idx.append(p)
                env_lo_idx_list.append(env_lo_idx)

        else:
            log.warn("_imp_lo_idx_list is already set, will not modify it.")
            env_lo_idx_list = list(self._env_lo_idx_list)

        assert env_lo_idx_list is not None
        assert len(imp_lo_idx_list) == nfrag
        assert len(env_lo_idx_list) == nfrag
        
        self._nfrag = nfrag
        self._imp_lo_idx_list = imp_lo_idx_list
        self._env_lo_idx_list = env_lo_idx_list

    @property
    def nfrag(self):
        assert self._nfrag is not None
        return self._nfrag

    @property
    def imp_lo_idx_list(self):
        assert self._imp_lo_idx_list is not None
        return self._imp_lo_idx_list

    @imp_lo_idx_list.setter
    def imp_lo_idx_list(self, imp_lo_idx_list):
        log = logger.new_logger(self, self.verbose)
        if self._imp_lo_idx_list is not None:
            log.warn('_imp_lo_idx_list is already set, overwrite it.')
        self._imp_lo_idx_list = imp_lo_idx_list

    @property
    def env_lo_idx_list(self):
        assert self._env_lo_idx_list is not None
        return self._env_lo_idx_list
    
    @env_lo_idx_list.setter
    def env_lo_idx_list(self, env_lo_idx_list):
        log = logger.new_logger(self, self.verbose)
        if self._env_lo_idx_list is not None:
            log.warn('_env_lo_idx_list is already set, overwrite it.')
        self._env_lo_idx_list = env_lo_idx_list

    @property
    def coeff_ao_lo(self):
        assert self._coeff_ao_lo is not None
        return self._coeff_ao_lo

    @coeff_ao_lo.setter
    def coeff_ao_lo(self, coeff_ao_lo):
        log = logger.new_logger(self, self.verbose)
        if self._coeff_ao_lo is not None:
            log.warn('_coeff_ao_lo is already set, overwrite it.')
        self._coeff_ao_lo = coeff_ao_lo

    @property
    def solver(self):
        assert self._solver is not None
        return self._solver

    @solver.setter
    def solver(self, solver):
        log = logger.new_logger(self, self.verbose)
        if self._solver is not None:
            log.warn('_solver is already set, overwrite it.')
        self._solver = solver

    @property
    def vcor_fitting(self):
        return self._vcor_fitting

    @vcor_fitting.setter
    def vcor_fitting(self, vcor_fitting):
        log = logger.new_logger(self, self.verbose)
        if self._vcor_fitting is not None:
            log.warn('_vcor_fitting is already set, overwrite it.')
        self._vcor_fitting = vcor_fitting

    @property
    def mu_fitting(self):
        return self._mu_fitting

    @mu_fitting.setter
    def mu_fitting(self, mu_fitting):
        log = logger.new_logger(self, self.verbose)
        if self._mu_fitting is not None:
            log.warn('_mu_fitting is already set, overwrite it.')
        self._mu_fitting = mu_fitting

    def get_eig(self, fock_ao=None):
        """Get the eigenvalues and eigenvectors of the Fock matrix.

        Parameters:
            fock_ao : the Fock matrix in AO basis
            ovlp_ao : the overlap matrix in AO basis
        
        Returns:
            mo_energy : the eigenvalues
            mo_coeff : the eigenvectors
        """
        ovlp_ao = self.get_ovlp_ao()
        nao     = fock_ao.shape[0]

        if ovlp_ao is None:
            ovlp_ao = numpy.eye(nao)

        return self._base.eig(fock_ao, ovlp_ao)

    def get_ovlp_ao(self):
        return self._ovlp_ao

    def get_hcore_ao(self):
        return self._hcore_ao

    def get_fock_ao(self, dm_ao=None):
        fock_ao = self._fock_ao
        if dm_ao is not None:
            fock_ao = self._base.get_fock(dm=dm_ao)
        return fock_ao

    def get_rdm1_ao(self, mo_energy=None, mo_coeff=None):
        mo_occ = self._base.get_occ(mo_energy, mo_coeff)
        return self._base.make_rdm1(mo_coeff, mo_occ)

    def get_veff_ao(self, dm_ao=None):
        return self._base.get_veff(dm=dm_ao)

    def get_eri(self, orbs=None, aosym=4, dataname="eri"):
        """Get the 2-electron integrals in the given basis.

        Args:
            coeffs : the coefficients of the given basis
        
        Returns:
            eri : the 2-electron integrals in the given basis
        """
        eri      = None
        eri_file = None # self.eri_file

        if orbs is None:
            eri = self._base._eri
        else:
            if isinstance(self._base.mol, pyscf.gto.Mole):
                eri = pyscf.ao2mo.kernel(
                    self._base.mol, orbs, aosym=aosym, 
                    dataname=dataname,
                    verbose=self.verbose
                    )

            else:
                assert self._base._eri is not None
                eri = pyscf.ao2mo.kernel(
                    self._base._eri, orbs, aosym=aosym, 
                    verbose=self.verbose
                    )
        
        assert eri is not None
        return eri

    def get_lo_label_list(self):
        """Generate the labels of the local orbitals.
        """
        from pydmet.tools.mol_lo_tools import lo_weight_on_ao
        
        log = logger.new_logger(self, self.verbose)
        log.info('Set the labels of the local orbitals')

        m = self._m
        if isinstance(m, pyscf.gto.Mole):
            ovlp_ao     = self.get_ovlp_ao()
            coeff_ao_lo = self.coeff_ao_lo

            nao, nlo = coeff_ao_lo.shape
            assert ovlp_ao.shape == (nao, nao)

            ao_label_list = m.ao_labels()
            lo_label_list = []
            
            w2_ao_lo = lo_weight_on_ao(m, coeff_ao_lo, ovlp_ao)

            for p in range(nlo):
                n = numpy.argmax(w2_ao_lo[:, p]) # n stands for the index of the AO

                if w2_ao_lo[n, p] > self.min_weight:
                    lo_label_list.append(ao_label_list[n])
                else:
                    log.warn('No label is assigned to the %d-th LO, weight: %6.4e'%(p, w2_ao_lo[n, p]))
                    lo_label_list.append(None)

        else:
            for p in range(nlo):
                lo_label_list.append(None)
                
        return lo_label_list

    def combine_emb_rdm1_ao(self, rdm1_ao_list):
        """Combine the 1-RDMs in the AO basis.

        Parameters:
            rdm1_ao_list : the list of 1-RDMs in the AO basis
        
        Returns:
            rdm1_ao : the combined 1-RDM in the AO basis
        """
        return numpy.einsum("fpq->pq", rdm1_ao_list)

    def get_emb_rdm1_ao(self, emb_res, emb_basis):
        """Get the 1-RDM from the solutions of the embedding problem.
        
        Parameters:
            emb_res : the embedding problem solution
            emb_basis : the embedding basis
        
        Returns:
            rdm1_emb_ao : the 1-RDM in AO basis from the solutions
            of the embedding problem.
        """

        imp_eo_idx  = emb_basis.imp_eo_idx
        bath_eo_idx = emb_basis.bath_eo_idx
        coeff_ao_eo = emb_basis.coeff_ao_eo
        coeff_ao_eo_imp  = coeff_ao_eo[:, imp_eo_idx]
        coeff_ao_eo_bath = coeff_ao_eo[:, bath_eo_idx]

        rdm1_emb_eo = emb_res.rdm1
        rdm1_emb_eo_imp_imp   = rdm1_emb_eo[imp_eo_idx, :][:, imp_eo_idx]
        rdm1_emb_eo_imp_bath  = rdm1_emb_eo[imp_eo_idx, :][:, bath_eo_idx]
        rdm1_emb_eo_bath_imp  = rdm1_emb_eo[bath_eo_idx, :][:, imp_eo_idx]
        rdm1_emb_eo_bath_bath = rdm1_emb_eo[bath_eo_idx, :][:, bath_eo_idx]

        rdm1_ao  = reduce(numpy.dot, (coeff_ao_eo_imp, rdm1_emb_eo_imp_imp,  coeff_ao_eo_imp.T))
        rdm1_ao += reduce(numpy.dot, (coeff_ao_eo_imp, rdm1_emb_eo_imp_bath, coeff_ao_eo_bath.T)) * 0.5
        rdm1_ao += reduce(numpy.dot, (coeff_ao_eo_bath, rdm1_emb_eo_bath_imp, coeff_ao_eo_imp.T)) * 0.5

        return rdm1_ao

    def get_nelec_tot(self, dm_lo=None, dm_ao=None):
        nelec = None
        
        if dm_lo is not None:
            nelec = numpy.einsum('ii', dm_lo)
        else:
            assert dm_ao is not None
            ovlp_ao = self.get_ovlp_ao()

            if ovlp_ao is not None:
                nelec = numpy.einsum('ij,ji', dm_ao, ovlp_ao)
            else:
                nelec = numpy.einsum('ii', dm_ao)

        assert nelec is not None
        return nelec.real

    def get_vcor(self, vcor_param_lo):
        """Get the correlation potential from the vcor parameters (parameters in LO).
        Return the correlation potential in both AO and LO basis.
        """
        coeff_ao_lo = self.coeff_ao_lo
        nao, nlo    = coeff_ao_lo.shape
        vcor_ao     = numpy.zeros((nao, nao))
        vcor_lo     = numpy.zeros((nlo, nlo))

        nfrag = self.nfrag
        imp_lo_idx_list = self.imp_lo_idx_list
        env_lo_idx_list = self.env_lo_idx_list

        param_count = 0
        for ifrag in range(nfrag):
            imp_lo_idx = imp_lo_idx_list[ifrag]
            env_lo_idx = env_lo_idx_list[ifrag]
            nimp = len(imp_lo_idx)
            diag_idx = numpy.diag_indices(nimp)
            tril_idx = numpy.tril_indices(nimp)
            nparam   = tril_idx[0].size

            vcor_frag_lo     = numpy.zeros((nimp, nimp))
            vcor_param_start = param_count
            vcor_param_end   = param_count + nparam
            vcor_param_frag  = vcor_param_lo[vcor_param_start:vcor_param_end]

            vcor_frag_lo[tril_idx] = vcor_param_frag
            vcor_frag_lo += vcor_frag_lo.T
            vcor_frag_lo[diag_idx] *= 0.5

            param_count  += nparam

            frag_ix = numpy.ix_(imp_lo_idx, imp_lo_idx)
            vcor_lo[frag_ix] = vcor_frag_lo

        vcor_ao = self.transform_h_lo_to_ao(vcor_lo)
        return vcor_ao, vcor_lo

    def make_emb_basis(self, imp_lo_idx, env_lo_idx, dm_ll_ao=None, dm_ll_lo=None):
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
        ovlp_ao     = self.get_ovlp_ao()
        nao, nlo    = coeff_ao_lo.shape

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

        from pydmet.embedding import EmbeddingBasis
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

    def make_emb_prob(self, mu=0.0, emb_basis=None, dm_ll_ao=None, dm_ll_lo=None):
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

        log = logger.Logger(self.stdout, self.verbose)

        log.info("Build the embedding problem.")

        assert dm_ll_lo is not None
        assert dm_ll_ao is not None

        neo      = emb_basis.neo
        neo_imp  = emb_basis.neo_imp
        neo_bath = emb_basis.neo_bath

        imp_eo_idx  = emb_basis.imp_eo_idx
        bath_eo_idx = emb_basis.bath_eo_idx

        coeff_ao_eo = emb_basis.coeff_ao_eo
        coeff_lo_eo = emb_basis.coeff_lo_eo

        hcore_ao = self.get_hcore_ao()
        fock_ao  = self.get_fock_ao()
        ovlp_ao = self.get_ovlp_ao()
        eri_eo = self.get_eri(orbs=coeff_ao_eo)
        eri_eo_full = pyscf.ao2mo.restore(1, eri_eo, neo)

        # Get the valence part of the density matrix in LO basis
        dm_ll_eo       = reduce(numpy.dot, (coeff_lo_eo.T, dm_ll_lo, coeff_lo_eo))
        j1e_eo, k1e_eo = dot_eri_dm(eri_eo_full, dm_ll_eo, hermi=1, with_j=True, with_k=True)

        # Clever way of building f1e_eo

        h1e_eo   = reduce(numpy.dot, (coeff_ao_eo.T, hcore_ao, coeff_ao_eo))
        f1e_eo   = reduce(numpy.dot, (coeff_ao_eo.T, fock_ao,  coeff_ao_eo))
        f1e_eo  -= (j1e_eo - k1e_eo * 0.5)
        
        id_imp  = numpy.zeros((neo, neo))
        id_imp[imp_eo_idx, imp_eo_idx] = 1.0

        nelec = numpy.einsum('ii->', dm_ll_eo)
        nelec = numpy.round(nelec)
        nelec = int(nelec)

        assert nelec % 2 == 0
        nelecs = (nelec // 2, nelec // 2)
        
        from pydmet.embedding import EmbeddingProblem
        emb_prob = EmbeddingProblem()
        emb_prob.neo    = neo
        emb_prob.neo_imp  = neo_imp
        emb_prob.neo_bath = neo_bath
        emb_prob.imp_eo_idx = imp_eo_idx
        emb_prob.bath_eo_idx = bath_eo_idx
        emb_prob.nelecs = nelecs
        emb_prob.dm0    = dm_ll_eo
        emb_prob.h1e    = h1e_eo
        emb_prob.f1e    = f1e_eo
        emb_prob.mu     = mu
        emb_prob.id_imp = id_imp
        emb_prob.h2e    = eri_eo_full

        return emb_prob

class MoleculeDMET(RHF):
    pass