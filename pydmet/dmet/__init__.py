import pyscf
from pyscf.gto.mole import Mole
from pyscf import pbc
from pyscf.pbc.gto import Cell

import pydmet

def RHF(mol_or_mf=None, coeff_ao_lo=None, solver=None, imp_lo_idx_list=None):
    mol      = None
    mf       = None
    dmet_obj = None
        
    if isinstance(mol_or_mf, pyscf.scf.hf.SCF):
        mf  = mol_or_mf
        mol = mf.mol
    else:
        mol = mf_or_mol

    is_molecule = isinstance(mol, Mole) and not isinstance(mol, Cell)
    is_lattice  = isinstance(mol, Cell)
    is_model    = not is_molecule and not is_lattice

    if mf is None:
        if is_molecule:
            mf = pyscf.scf.RHF(mol)
            mf.kernel()

        elif is_lattice:
            raise NotImplementedError

        elif is_model:
            raise NotImplementedError

        else:
            raise RuntimeError("Unknown type: %s" % type(mol))

    dmet_obj = None

    if is_molecule:
        from pydmet.dmet.molecule import rhf
        from pydmet.dmet.molecule.rhf import MoleculeDMET
        dmet_obj = MoleculeDMET(mf)
    
    elif is_lattice:
        raise NotImplementedError

    elif is_model:
        raise NotImplementedError

    else:
        raise RuntimeError("Unknown type: %s" % type(mol))

    dmet_obj.solver = solver
    dmet_obj.imp_lo_idx_list = imp_lo_idx_list
    dmet_obj.coeff_ao_lo = coeff_ao_lo

    return dmet_obj

RDMET       = RHF
DMETwithRHF = RHF
RDMETwithHF = RHF