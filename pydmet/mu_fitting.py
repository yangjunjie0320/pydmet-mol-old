class MuFittingMixin(object):
    verbose  = 0
    stdout   = sys.stdout
    tmp_dir  = None

    conv_tol = 1e-8
    conv_tol_grad = 1e-5
    max_cycle = 50


class BrentQ(MuFittingMixin):
    pass

class Newton(MuFittingMixin):
    pass