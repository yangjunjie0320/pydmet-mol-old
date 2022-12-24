import os, sys
import numpy, scipy

import pyscf
from pyscf import lo

import pydmet

from pydmet.tools import mol_lo_tools
from pydmet import solver

TOL = os.environ.get("TOL", 1e-8)

# Take care that the finite difference calculation of dn_dmu is super-sensitive
# to the convergence threshold of the HF solver, please make sure that the HF
# solver is converged to a very high accuracy.

def build_3h2o(basis="sto3g"):
    import pyscf
    from   pyscf import lo

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
        basis = basis,
        verbose=0
    )

    frag_atms_list   = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    mf = pyscf.scf.RHF(mol)
    mf.verbose = 0
    mf.max_cycle = 100
    mf.conv_tol  = TOL
    mf.conv_tol_grad = TOL
    mf.kernel()

    return mol, mf, frag_atms_list

def build_lo(mf, basis="sto3g"):
    coeff_ao_lo = None
    lo_path = "./lo-3h2o-%s.log" % basis
    if os.path.exists(lo_path):
        coeff_ao_lo = numpy.loadtxt(lo_path, delimiter=",")
    else:
        pm = lo.PM(mf.mol, mf.mo_coeff)
        pm.conv_tol = TOL
        coeff_ao_lo = pm.kernel()
        numpy.savetxt(lo_path, coeff_ao_lo, delimiter=",", fmt="% 20.16f")
    return coeff_ao_lo
    
def test_set_up(basis="sto3g"):
    mol, mf, imp_atms_list = build_3h2o(basis)
    ovlp_ao = mf.get_ovlp()
    coeff_ao_lo = build_lo(mf, basis)

    imp_lo_idx_list = mol_lo_tools.partition_lo_to_imps(
        imp_atms_list, mol=mol, coeff_ao_lo=coeff_ao_lo,
        min_weight=0.4
    )

    dmet_obj = pydmet.RHF(
        mf, None, imp_lo_idx_list,
        is_mu_fitting=True, 
        is_vcor_fitting=True
        )

    dmet_obj.build()

def test_dmet_rhf(basis="sto3g"):
    mol, mf, imp_atms_list = build_3h2o(basis)
    ovlp_ao = mf.get_ovlp()
    coeff_ao_lo = build_lo(mf, basis)

    hcore_ao    = mf.get_hcore()
    ovlp_ao     = mf.get_ovlp()
    dm_ao       = mf.make_rdm1()
    fock_ao     = mf.get_fock(h1e=hcore_ao, dm=dm_ao)
    ene_hf_ref  = mf.energy_elec()[0]
    rdm1_hf_ref = mf.make_rdm1()

    imp_lo_idx_list = mol_lo_tools.partition_lo_to_imps(
        imp_atms_list, mol=mol, coeff_ao_lo=coeff_ao_lo,
        min_weight=0.8
    )

    rhf_solver = solver.RHF()
    rhf_solver.max_cycle = 100
    rhf_solver.conv_tol      = TOL * TOL
    rhf_solver.conv_tol_grad = TOL * TOL
    rhf_solver.verbose = 0

    dmet_obj = pydmet.RHF(
        mf, solver=rhf_solver, 
        coeff_ao_lo=coeff_ao_lo,
        imp_lo_idx_list=imp_lo_idx_list,
        is_mu_fitting=False, 
        is_vcor_fitting=False
        )
    dmet_obj.verbose   = 0
    dmet_obj._hcore_ao = hcore_ao
    dmet_obj._ovlp_ao  = ovlp_ao
    dmet_obj._fock_ao  = fock_ao
    dmet_obj.build()
    dmet_obj.dump_flags()

    dm_ll_ao = dm_ao
    dm_ll_lo = dmet_obj.transform_dm_ao_to_lo(dm_ll_ao)

    nfrag = dmet_obj.nfrag
    imp_lo_idx_list = dmet_obj.imp_lo_idx_list
    env_lo_idx_list = dmet_obj.env_lo_idx_list
    
    energy_elec = 0.0
    dn_dmu_hf   = 0.0
    dm_hl_ao    = 0.0

    for ifrag in range(nfrag):
        imp_lo_idx = imp_lo_idx_list[ifrag]
        env_lo_idx = env_lo_idx_list[ifrag]

        emb_basis = dmet_obj.make_emb_basis(
            imp_lo_idx, env_lo_idx, 
            dm_ll_ao=dm_ll_ao,
            dm_ll_lo=dm_ll_lo,
            )

        emb_prob  = dmet_obj.make_emb_prob(
            mu=0.0, emb_basis=emb_basis,
            dm_ll_ao=dm_ll_ao,
            dm_ll_lo=dm_ll_lo,
            )

        emb_res = rhf_solver.kernel(
            emb_prob=emb_prob,
            save_dir=None,
            load_dir=None,
        )

        energy_elec += emb_res.energy_elec
        dn_dmu_hf   += emb_res.dn_dmu_hf
        dm_hl_ao    += dmet_obj.get_emb_rdm1_ao(emb_res, emb_basis)

def test_dmet_rhf_dn_dmu(basis="sto3g"):
    mol, mf, imp_atms_list = build_3h2o(basis)
    ovlp_ao = mf.get_ovlp()
    coeff_ao_lo = build_lo(mf, basis)

    hcore_ao    = mf.get_hcore()
    ovlp_ao     = mf.get_ovlp()
    dm_ao       = mf.make_rdm1()
    fock_ao     = mf.get_fock(h1e=hcore_ao, dm=dm_ao)
    ene_hf_ref  = mf.energy_elec()[0]
    rdm1_hf_ref = mf.make_rdm1()

    imp_lo_idx_list = mol_lo_tools.partition_lo_to_imps(
        imp_atms_list, mol=mol, coeff_ao_lo=coeff_ao_lo,
        min_weight=0.8
    )

    rhf_solver = solver.RHF()
    rhf_solver.max_cycle = 100
    rhf_solver.conv_tol      = TOL * TOL
    rhf_solver.conv_tol_grad = TOL * TOL
    rhf_solver.verbose = 0

    dmet_obj = pydmet.RHF(
        mf, solver=rhf_solver, 
        coeff_ao_lo=coeff_ao_lo,
        imp_lo_idx_list=imp_lo_idx_list,
        is_mu_fitting=False, 
        is_vcor_fitting=False
        )
    dmet_obj.verbose   = 0
    dmet_obj._hcore_ao = hcore_ao
    dmet_obj._ovlp_ao  = ovlp_ao
    dmet_obj._fock_ao  = fock_ao
    dmet_obj.build()
    dmet_obj.dump_flags()

    dm_ll_ao = dm_ao
    dm_ll_lo = dmet_obj.transform_dm_ao_to_lo(dm_ll_ao)

    nfrag = dmet_obj.nfrag
    imp_lo_idx_list = dmet_obj.imp_lo_idx_list
    env_lo_idx_list = dmet_obj.env_lo_idx_list


    mu_list  = [0.1, 0.0, -0.1]
    dmu_list = [8e-4, 4e-4, 2e-4, 1e-4]

    for mu in mu_list:
        energy_elec = 0.0
        dn_dmu_hf   = 0.0
        dm_hl_ao    = 0.0

        for ifrag in range(nfrag):
            imp_lo_idx = imp_lo_idx_list[ifrag]
            env_lo_idx = env_lo_idx_list[ifrag]

            emb_basis = dmet_obj.make_emb_basis(
                imp_lo_idx, env_lo_idx, 
                dm_ll_ao=dm_ll_ao,
                dm_ll_lo=dm_ll_lo,
                )

            emb_prob  = dmet_obj.make_emb_prob(
                mu=mu, emb_basis=emb_basis,
                dm_ll_ao=dm_ll_ao,
                dm_ll_lo=dm_ll_lo,
                )

            emb_res = rhf_solver.kernel(
                emb_prob=emb_prob,
                save_dir=None,
                load_dir=None,
            )

            energy_elec += emb_res.energy_elec
            dn_dmu_hf   += emb_res.dn_dmu_hf
            dm_hl_ao    += dmet_obj.get_emb_rdm1_ao(emb_res, emb_basis)

        for dmu in dmu_list:

            dm_hl_ao = 0.0
            
            for ifrag in range(nfrag):
                imp_lo_idx = imp_lo_idx_list[ifrag]
                env_lo_idx = env_lo_idx_list[ifrag]

                emb_basis = dmet_obj.make_emb_basis(
                    imp_lo_idx, env_lo_idx, 
                    dm_ll_ao=dm_ll_ao,
                    dm_ll_lo=dm_ll_lo,
                    )

                emb_prob  = dmet_obj.make_emb_prob(
                    mu=mu+dmu, emb_basis=emb_basis,
                    dm_ll_ao=dm_ll_ao,
                    dm_ll_lo=dm_ll_lo,
                    )

                emb_res = rhf_solver.kernel(
                    emb_prob=emb_prob,
                    save_dir=None,
                    load_dir=None,
                )

                dm_hl_ao += dmet_obj.get_emb_rdm1_ao(emb_res, emb_basis)

            nelec_tot_1 = numpy.einsum("ij,ji", dm_hl_ao, ovlp_ao)
        
            dm_hl_ao = 0.0
            
            for ifrag in range(nfrag):
                imp_lo_idx = imp_lo_idx_list[ifrag]
                env_lo_idx = env_lo_idx_list[ifrag]

                emb_basis = dmet_obj.make_emb_basis(
                    imp_lo_idx, env_lo_idx, 
                    dm_ll_ao=dm_ll_ao,
                    dm_ll_lo=dm_ll_lo,
                    )

                emb_prob  = dmet_obj.make_emb_prob(
                    mu=mu-dmu, emb_basis=emb_basis,
                    dm_ll_ao=dm_ll_ao,
                    dm_ll_lo=dm_ll_lo,
                    )

                emb_res = rhf_solver.kernel(
                    emb_prob=emb_prob,
                    save_dir=None,
                    load_dir=None,
                )

                dm_hl_ao    += dmet_obj.get_emb_rdm1_ao(emb_res, emb_basis)

            nelec_tot_2 = numpy.einsum("ij,ji", dm_hl_ao, ovlp_ao)

            dn_dmu_fd = (nelec_tot_1 - nelec_tot_2) / (2.0 * dmu)
            print("mu = % 6.4f, dmu = %6.4e, dn_dmu_hf = %8.6f, dn_dmu_fd = %8.6f, err = % 6.4e" % (mu, dmu, dn_dmu_hf, dn_dmu_fd, dn_dmu_hf-dn_dmu_fd))
        

if __name__ == '__main__':
    # test_set_up()
    # test_dmet_rhf()
    test_dmet_rhf_dn_dmu()