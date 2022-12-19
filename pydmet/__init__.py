import pyscf

import pydmet
from pydmet import embedding
from pydmet import solver

def RHF(m, solver=None, imp_lo_idx_list=None, 
           coeff_ao_lo=None,
           is_mu_fitting : bool = True,
           is_vcor_fitting : bool = True):

    from pydmet.dmet import DMETwithRHF

    dmet_obj = DMETwithRHF(m)
    dmet_obj.solver = solver
    dmet_obj.imp_lo_idx_list = imp_lo_idx_list
    dmet_obj.coeff_ao_lo = coeff_ao_lo
    
#     if is_mu_fitting:
#         dmet_obj.mu_fitting  = embedding.MuFitting(dmet_obj)
       
#     if is_vcor_fitting:
#         dmet_obj.vcor_fitting = embedding.VcorFitting(dmet_obj)

    return dmet_obj