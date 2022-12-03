def build_fragment_list(mol, lo_labels):
    """
    Input:
        mol: pyscf.gto.Mole
        lo_labels: list of str
            labels of the localized orbitals

    Output:
        frag_list: list of Fragment instances
    """
    pass

class FragmentMixin(object):
    pass

class MoleculeFragmentWithSolver(FragmentMixin):
    pass

class LatticeFragmentWithSolver(FragmentMixin):
    pass
