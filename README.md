# pydmet
- Fragmet is a set of LO indices, which contains two parts, one is the impurity part (defined by the user), 
  the other is the environment part (obtained by the `build_env` function, which will exclude the virtual
  and core orbitals). The `build_emb_basis` function will stack the impurity orbitals and construct the 
  bath orbitals from the environment part.
- Fragment is also used to refer to the orbitals in the embedding space, which contains the impurity EOs
  (shall be the same as impurity LOs), and the bath EOs (constructed from the environment LOs).
- solver: the solver attribute could be a list of solvers or a childclass of `SolverMixin` (see `solver.py`).
  The solver is used to solve the impurity problem. The `DMET` class will call the `solve` function of the 
  solver to solve the impurity problem. The `DMET` class will also call the `update` function of the solver 
  to update the solver after the impurity problem is solved.
- The DMET is classified as the scf classes in pyscf, the mean filed object will be created and maintained 
  inside the DMET object. Using KS DFT as the mean field is not recommended, as the energy formula might
  not be correct.
- If the impurities are choosen to be a partition of the whole system, the rest part of the orbitals could
  also form a fragment with HF solver, (seems to be a trivial case, not sure if is the case for the k-point 
  version), while the rest part corresponds to a small bath, with the naturally converged HF solution from
  DMET object, no integral transformation is needed, just transorm the density matrix to this embedding 
  basis.
- The child classes shall not repeat the method names, for example,
```python
class DMET(object): # in rhf.py
    pass

class RHF(DMET): # in rhf.py
    pass

class KRHF(RHF): # in krhf.py
    pass

class MoleculeRHF(RHF): # in rhf.py
    pass

class LatticeRHF(RHF): # in rhf.py
    pass

class LatticeKRHF(LatticeRHF, KRHF): # in krhf.py
    pass

class ModelRHF(RHF): # in rhf.py
    pass

class ModelKRHF(ModelRHF, KRHF): # in krhf.py
    pass
```