class SolverMixin(object):
    """Base class for solvers"""
    max_cycle = 100
    conv_tol = 1e-8

    pass

class CCSD(SolverMixin):
    def __init__(self):
        self.tmp_dir     = None
        self.restart_dir = None

    def build(self, hemb):
        pass

    def 