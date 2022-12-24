# pydmet
We follow [code standards of pyscf](https://pyscf.org/code-rule.html#), and recommend to refer to
[google python style guide](https://google.github.io/styleguide/pyguide.html).
## Write Test Cases
- Unit tests: These tests focus on individual units of code, such as functions or methods. These tests shall ensure the robustness of both simple functions and more complex drivers between version changes. Unit tests are usually included in the same file as the code they are testing.
```
.
├── code_you_want_to_test.py
└── test
    └── test_code_you_want_to_test.py
```
- The test runner `pytest` is used to make it easy to run all of your tests at once, [`pytest`](https://docs.pytest.org/en/7.2.x/) command will run all of the tests in the `./tests` directory.
- Each test shall contain some assertions, which will be evaluated to determine if the test passes or fails. The `assert` statement is used to make assertions about the code being tested. If the assertion is true, the test passes. If the assertion is false, the test fails.
- End-to-end tests: These tests simulate the behavior of a user interacting with your program, and are used to ensure that the program is working correctly from start to finish. End-to-end tests are located in the `./examples` directory.
- Examples for modules should be placed in the appropriate directory inside the `/examples ` directory. While the examples should be light enough to run on a modest personal computer, the examples should not be trivial. Instead, the point of the examples is to showcase the functionality of the module.

## Notes from Junjie
- Some _standard_ test cases for the program, like three water molecules, or a small
lattice, and Hubbard models. As the three water case may not be a very good system to perform
DMET, other suggestions are welcome.
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