import scipy, numpy
import pyscf

def lo_weight_on_ao(mol : pyscf.gto.Mole = None, coeff_ao_lo : numpy.ndarray = None, ovlp_ao : numpy.ndarray = None):
    """A function to calculate the weight of local orbitals on atomic orbitals.

    Parameters:
        mol : pyscf.gto.Mole
            The molecule object.
        coeff_ao_lo : numpy.ndarray
            The coefficient matrix to transform atomic orbitals 
            to local orbitals.
        ovlp_ao : numpy.ndarray
            The overlap matrix of atomic orbitals.

    Returns:
        w2_ao_lo : numpy.ndarray
            The weight of local orbitals on atomic orbitals.
    """

    if ovlp_ao is None:
        assert mol is not None
        ovlp_ao = mol.intor_symmetric('int1e_ovlp')
    assert ovlp_ao is not None

    nao = ovlp_ao.shape[0]
    nlo = coeff_ao_lo.shape[1]
    assert coeff_ao_lo.shape == (nao, nlo)
    assert ovlp_ao.shape == (nao, nao)

    ovlp_ao_sqrt = scipy.linalg.sqrtm(ovlp_ao)
    w_ao_lo      = numpy.dot(ovlp_ao_sqrt, coeff_ao_lo)
    w2_ao_lo     = numpy.einsum('np,np->np', w_ao_lo, w_ao_lo)

    return w2_ao_lo

def lo_weight_on_atom(mol : pyscf.gto.Mole, coeff_ao_lo : numpy.ndarray = None, ovlp_ao : numpy.ndarray = None):
    """A function to calculate the weight of local orbitals on atomic orbitals.

    Parameters:
        mol : pyscf.gto.Mole
            The molecule object.
        coeff_ao_lo : numpy.ndarray
            The coefficient matrix to transform atomic orbitals 
            to local orbitals.
        ovlp_ao : numpy.ndarray
            The overlap matrix of atomic orbitals.

    Returns:
        w2_atm_lo : numpy.ndarray
            The weight of local orbitals on each atom.
    """

    if ovlp_ao is None:
        ovlp_ao = mol.intor_symmetric('int1e_ovlp')
    assert ovlp_ao is not None

    natm = mol.natm
    nao = mol.nao
    nlo = coeff_ao_lo.shape[1]
    assert coeff_ao_lo.shape == (nao, nlo)
    assert ovlp_ao.shape == (nao, nao)

    w2_ao_lo  = lo_weight_on_ao(mol, coeff_ao_lo, ovlp_ao)
    w2_atm_lo = numpy.zeros((natm, nlo))
    ao_slice_by_atom = mol.aoslice_by_atom()

    for iatm in range(natm):
        ao_slice_iatm   = ao_slice_by_atom[iatm, 2:4]
        w2_atm_lo[iatm] = numpy.sum(w2_ao_lo[ao_slice_iatm[0]:ao_slice_iatm[1]], axis=0)

    return w2_atm_lo

def lo_weight_on_frag(frag_atms_list : list, mol : pyscf.gto.Mole,
                      coeff_ao_lo : numpy.ndarray = None, 
                      ovlp_ao : numpy.ndarray = None):
    """A function to calculate the weight of local orbitals on fragments.

    Parameters:
        frag_atms_list : list
            A list of atoms in each fragment.
        mol : pyscf.gto.Mole
            The molecule object.
        coeff_ao_lo : numpy.ndarray
            The coefficient matrix to transform atomic orbitals 
            to local orbitals.
        ovlp_ao : numpy.ndarray
            The overlap matrix of atomic orbitals.

    Returns:
        w2_frag_lo : numpy.ndarray
            The weight of local orbitals on each fragment.
    """

    if ovlp_ao is None:
        ovlp_ao = mol.intor_symmetric('int1e_ovlp')
    assert ovlp_ao is not None

    nfrag = len(frag_atms_list)
    natm = mol.natm
    nao = mol.nao
    nlo = coeff_ao_lo.shape[1]

    w2_atm_lo  = lo_weight_on_atom(mol, coeff_ao_lo, ovlp_ao)
    w2_frag_lo = numpy.zeros((nfrag, nlo))

    assert coeff_ao_lo.shape == (nao, nlo)
    assert ovlp_ao.shape == (nao, nao)
    assert w2_atm_lo.shape == (natm, nlo)

    for ifrag, frag_atms in enumerate(frag_atms_list):
        w2_frag_lo[ifrag] = numpy.sum(w2_atm_lo[frag_atms], axis=0)

    return w2_frag_lo
    

def partition_lo_to_atms(mol : pyscf.gto.Mole = None, 
                         coeff_ao_lo : numpy.ndarray = None,
                         w2_atm_lo : numpy.ndarray = None,
                         min_weight : float = 0.9,):
    """A function to partition local orbitals to atoms.

    Parameters:
        mol : pyscf.gto.Mole
            The molecule object.
        coeff_ao_lo : numpy.ndarray
            The coefficient matrix to transform atomic orbitals
            to local orbitals.
        w2_atm_lo : numpy.ndarray
            The weight of local orbitals on each atom.
        min_weight : float
            The minimum weight of local orbitals on each atom.
            If the max weight is smaller than min_weight, the
            local orbital will not be assigned to the atom.

    Returns:
        lo_idx_on_atm_list : list
            The list of local orbital indices on each atom.
    """

    if w2_atm_lo is None:
        w2_atm_lo = lo_weight_on_atom(mol, coeff_ao_lo, None)

    natm = mol.natm
    nlo  = coeff_ao_lo.shape[1]
    assert w2_atm_lo.shape == (natm, nlo)

    lo_idx_on_atm_list = [[] for iatm in range(natm)]

    for lo in range(nlo):
        iatm = numpy.argmax(w2_atm_lo[:, lo])

        if w2_atm_lo[iatm, lo] > min_weight:
            lo_idx_on_atm_list[iatm].append(lo)
            print("LO %3d is assigned to atom %3d with weight % 6.4f" % (lo, iatm, w2_atm_lo[iatm, lo]))
        else:
            print('Warning: The weight of local orbital %d on atom %d is % 6.4f, which is smaller than the tolerance % 6.4f!' % (lo, iatm, w2_atm_lo[iatm, lo], tol))

    return lo_idx_on_atm_list

def partition_lo_to_frags(frag_atms_list : list,
                          mol : pyscf.gto.Mole = None, 
                          coeff_ao_lo : numpy.ndarray = None, 
                          w2_frag_lo : numpy.ndarray = None,
                          min_weight : float = 0.9,):
    """A function to partition local orbitals to fragments.

    Parameters:
        frag_atms_list : list
            The list of atoms in each fragment.
        mol : pyscf.gto.Mole
            The molecule object.
        coeff_ao_lo : numpy.ndarray
            The coefficient matrix to transform atomic orbitals
            to local orbitals.
        w2_frag_lo : numpy.ndarray
            The weight of local orbitals on each atom.
        min_weight : float
            The minimum weight of local orbitals on each atom.
            If the max weight is smaller than min_weight, the
            local orbital will not be assigned to the atom.

    Returns:
        lo_idx_on_frag_list : list
            The list of local orbital indices on each fragment.
    """

    if w2_frag_lo is None:
        w2_frag_lo = lo_weight_on_frag(frag_atms_list, mol, coeff_ao_lo, None)

    nfrag = len(frag_atms_list)
    nlo  = coeff_ao_lo.shape[1]
    assert w2_frag_lo.shape == (nfrag, nlo)

    lo_idx_on_frag_list = [[] for ifrag in range(nfrag)]

    for lo in range(nlo):
        ifrag = numpy.argmax(w2_frag_lo[:, lo])

        if w2_frag_lo[ifrag, lo] > min_weight:
            lo_idx_on_frag_list[ifrag].append(lo)
            print("LO %3d is assigned to fragment %3d, weight = %6.4f" % (lo, ifrag, w2_frag_lo[ifrag, lo]))
        else:
            print('Warning: The weight of local orbital %d on fragment %d is % 6.4f, which is smaller than the tolerance % 6.4f!' % (lo, ifrag, w2_frag_lo[ifrag, lo], min_weight))

    return lo_idx_on_frag_list

partition_lo_to_imps = partition_lo_to_frags