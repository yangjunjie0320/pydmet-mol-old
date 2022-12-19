import os, sys
import numpy, scipy

import pyscf
from pyscf.tools.dump_mat import dump_rec

import pydmet
from pydmet.tools import mol_lo_tools

def build_3h2o_lo(basis="sto3g"):
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
        basis = basis, verbose=0
    )

    mf = pyscf.scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()

    coeff_ao_lo = lo.PM(mol, mf.mo_coeff).kernel()

    return mol, coeff_ao_lo

def test_lo_weight_on_ao(basis="sto3g"):
    mol, coeff_ao_lo = build_3h2o_lo(basis)
    w2_ao_lo = mol_lo_tools.lo_weight_on_ao(mol, coeff_ao_lo)
    
    nao, nlo = coeff_ao_lo.shape
    assert w2_ao_lo.shape == (nao, nlo)

    for p in range(nlo):
        assert abs(w2_ao_lo[:, p].sum() - 1.0) < 1e-8

def test_lo_weight_on_atom(basis="sto3g"):
    mol, coeff_ao_lo = build_3h2o_lo(basis)
    w2_atm_lo = mol_lo_tools.lo_weight_on_atom(mol, coeff_ao_lo)
    
    natm, nlo = w2_atm_lo.shape
    assert w2_atm_lo.shape == (natm, nlo)

    for p in range(nlo):
        assert abs(w2_atm_lo[:, p].sum() - 1.0) < 1e-8

def test_lo_weight_on_frag(basis="sto3g"):
    mol, coeff_ao_lo = build_3h2o_lo(basis)
    w2_frag_lo = mol_lo_tools.lo_weight_on_frag(mol, coeff_ao_lo)
    
    nfrag, nlo = w2_frag_lo.shape
    assert w2_frag_lo.shape == (nfrag, nlo)

    for p in range(nlo):
        assert abs(w2_frag_lo[:, p].sum() - 1.0) < 1e-8

def test_partition_lo_to_atms(basis="sto3g"):
    mol, coeff_ao_lo = build_3h2o_lo(basis)

    ao_labels = mol.ao_labels()
    ovlp_ao   = mol.intor("int1e_ovlp")
    w2_ao_lo  = mol_lo_tools.lo_weight_on_ao(mol, coeff_ao_lo, ovlp_ao=ovlp_ao)
    w2_atm_lo = mol_lo_tools.lo_weight_on_atom(mol, coeff_ao_lo, ovlp_ao=ovlp_ao)
    lo_idx_on_atm_list = mol_lo_tools.partition_lo_to_atms(
        mol, coeff_ao_lo=coeff_ao_lo, w2_atm_lo=w2_atm_lo, min_weight=0.8
        )

    nao, nlo = coeff_ao_lo.shape
    assert len([lo_idx for lo_idx_on_atm in lo_idx_on_atm_list for lo_idx in lo_idx_on_atm]) == nlo

    for p in range(nlo):
        w2_ao_lo_p = w2_ao_lo[:, p]
        w2_atm_lo_p = w2_atm_lo[:, p]

        iatm = numpy.argmax(w2_atm_lo_p)
        assert p in lo_idx_on_atm_list[iatm]

        iao = numpy.argmax(w2_ao_lo_p)
        iatm_from_ao = int(ao_labels[iao][0])
        assert iatm == iatm_from_ao

def test_partition_lo_to_frags(basis="sto3g"):
    mol, coeff_ao_lo = build_3h2o_lo(basis)
    frag_atms_list   = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

    ao_labels  = mol.ao_labels()
    ovlp_ao    = mol.intor_symmetric("int1e_ovlp")
    w2_atm_lo  = mol_lo_tools.lo_weight_on_atom(mol, coeff_ao_lo, ovlp_ao=ovlp_ao)
    w2_frag_lo = mol_lo_tools.lo_weight_on_frag(
        frag_atms_list, mol=mol, ovlp_ao=ovlp_ao,
        coeff_ao_lo=coeff_ao_lo
        )
    lo_idx_on_frag_list = mol_lo_tools.partition_lo_to_frags(
        frag_atms_list, mol=mol, coeff_ao_lo=coeff_ao_lo, 
        w2_frag_lo=w2_frag_lo, min_weight=0.8
        )

    nao, nlo = coeff_ao_lo.shape
    assert len([lo_idx for lo_idx_on_frag in lo_idx_on_frag_list for lo_idx in lo_idx_on_frag]) == nlo

    for p in range(nlo):
        w2_atm_lo_p = w2_atm_lo[:, p]
        w2_frag_lo_p = w2_frag_lo[:, p]

        ifrag = numpy.argmax(w2_frag_lo_p)
        assert p in lo_idx_on_frag_list[ifrag]

        iatm = numpy.argmax(w2_atm_lo_p)
        ifrag_from_atm = None
        for ifrag, atm_list in enumerate(frag_atms_list):
            if iatm in atm_list:
                ifrag_from_atm = ifrag
                break
        assert ifrag == ifrag_from_atm

    

if __name__ == "__main__":
    test_lo_weight_on_ao(basis="sto3g")
    test_lo_weight_on_ao(basis="ccpvdz")
    test_lo_weight_on_atom(basis="sto3g")
    test_lo_weight_on_atom(basis="ccpvdz")
    test_partition_lo_to_atms(basis="sto3g")
    test_partition_lo_to_atms(basis="ccpvdz")
    test_partition_lo_to_frags(basis="sto3g")
    test_partition_lo_to_frags(basis="ccpvdz")