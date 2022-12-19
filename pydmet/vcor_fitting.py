class VcorFittingMixin(object):
    vcor_fitting_max_cycle = 100
    vcor_fitting_tol       = 1e-6
    vcor_fitting_gtol      = 1e-3
    vcor_fitting_method    = "BFGS"
    vcor_fitting_jac       = None
    vcor_fitting_hess      = None
    vcor_fitting_hessp     = None
    vcor_fitting_bounds    = None
    vcor_fitting_callback  = None
    vcor_fitting_options   = None

    def gen_dm_err(self):
        pass

class LeastSquareFitting(VcorFittingMixin):

    def solve_vcor_fitting(self, vcor_params_lo_0, 
                                 fock_ao=None,  fock_lo=None, 
                                 dm_hl_ao=None, dm_hl_lo=None):

        """Fit the vcor to given high-level density matrix.

        Parameters:
            vcor_params_lo_0 : numpy.ndarray
                The initial guess of the vcor parameters.
            fock_ao : numpy.ndarray
                The Fock matrix in AO basis, if not given, will compute it
                with the mf object.
            fock_lo : numpy.ndarray
                The Fock matrix in LO basis, if not given, will compute it
                with the mf object.
            dm_hl_ao : numpy.ndarray
                The high-level density matrix in AO basis, if not given, will
                compute it with the mf object.
            dm_hl_lo : numpy.ndarray
                The high-level density matrix in LO basis, if not given, will
                compute it with the mf object.
        
        Returns:
            vcor_params : numpy.ndarray
                The fitted parameters of the vcor.
        """
        log = logger.new_logger(self, self.verbose)

        iter_fitting = 0
        res_list     = []
        
        def get_dm_err(vcor_params_lo):
            vcor_ao, vcor_lo = self.get_vcor(vcor_params_lo)
            
            f_ao = fock_ao + vcor_ao
            f_lo = fock_lo + vcor_lo

            mo_energy, mo_coeff = self.eig(f_ao)
            dm_ll_ao  = self.get_mf_rdm1_ao(mo_coeff, mo_energy)
            dm_ll_lo  = self.transform_dm_ao_to_lo(dm_ll_ao)
            dm_err    = self.get_dm_err_imp(
                dm_hl_ao=dm_hl_ao, dm_hl_lo=dm_hl_lo, 
                dm_ll_ao=dm_ll_ao, dm_ll_lo=dm_ll_lo
                )

            iter_fitting += 1
            res = {
                'iter': iter_fitting, 
                'dm_err': dm_err,
                'vcor_params_lo': vcor_params_lo,
                
            }
            res_list.append(res)

            log.info("iter fitting = %4d, dm_err = % 6.4e", iter_fitting, dm_err)
            return dm_err

        vcor_fitting_method  = self.vcor_fitting_method
        vcor_fitting_options = self.vcor_fitting_options

        if vcor_fitting_options is None:
            vcor_fitting_options = {
                'gtol':    self.vcor_fitting_gtol,
                'disp':    self.verbose >= 5,
                'maxiter': self.vcor_fitting_max_cycle,
            }

        log.info("Start vcor fitting process.")
        res = scipy.optimize.minimize(
            get_dm_err, vcor_params_lo_0, method=vcor_fitting_method,
            jac=self.vcor_fitting_jac, hess=self.vcor_fitting_hess,
            hessp=self.vcor_fitting_hessp, bounds=self.vcor_fitting_bounds,
            tol=self.vcor_fitting_tol,     options=vcor_fitting_options,
            callback=self.vcor_fitting_callback,
        )

        iter_fitting = None
        res_list     = None

        vcor_params_lo = res.x
        is_converged   = res.success
        if not is_converged:
            log.warn("vcor fitting is not converged.")
        
        return vcor_params_lo