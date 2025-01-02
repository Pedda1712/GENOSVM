"""
Sample code automatically generated on 2025-01-01 16:42:02

by geno from www.geno-project.org

from input

parameters
  matrix K symmetric
  scalar c
  vector y
variables
  vector a
min
  0.5*(a.*y)'*K*(a.*y)-sum(a)
st
  a >= 0
  a <= c
  y'*a == 0


The generated code is provided "as is" without warranty of any kind.
"""

from __future__ import division, print_function, absolute_import

from math import inf
from timeit import default_timer as timer
try:
    from genosolver import minimize, check_version
    USE_GENO_SOLVER = True
except ImportError:
    from scipy.optimize import minimize
    USE_GENO_SOLVER = False
    WRN = 'WARNING: GENO solver not installed. Using SciPy solver instead.\n' + \
          'Run:     pip install genosolver'
    print('*' * 63)
    print(WRN)
    print('*' * 63)



class GenoNLP:
    def __init__(self, K, c, y, np):
        self.np = np
        self.K = K
        self.c = c
        self.y = y
        assert isinstance(K, self.np.ndarray)
        dim = K.shape
        assert len(dim) == 2
        self.K_rows = dim[0]
        self.K_cols = dim[1]
        if isinstance(c, self.np.ndarray):
            dim = c.shape
            assert dim == (1, )
            self.c = c[0]
        self.c_rows = 1
        self.c_cols = 1
        assert isinstance(y, self.np.ndarray)
        dim = y.shape
        assert len(dim) == 1
        self.y_rows = dim[0]
        self.y_cols = 1
        self.a_rows = self.K_cols
        self.a_cols = 1
        self.a_size = self.a_rows * self.a_cols
        # the following dim assertions need to hold for this problem
        assert self.K_cols == self.K_rows == self.a_rows == self.y_rows

    def getLowerBounds(self):
        bounds = []
        bounds += [0] * self.a_size
        return self.np.array(bounds)

    def getUpperBounds(self):
        bounds = []
        bounds += [min(self.c, inf)] * self.a_size
        return self.np.array(bounds)

    def getStartingPoint(self):
        self.aInit = self.np.zeros((self.a_rows, self.a_cols))
        return self.aInit.reshape(-1)

    def variables(self, _x):
        a = _x
        return a

    def fAndG(self, _x):
        a = self.variables(_x)
        t_0 = (a * self.y)
        t_1 = (self.K).dot(t_0)
        f_ = ((0.5 * (t_0).dot(t_1)) - self.np.sum(a))
        g_0 = (((0.5 * (t_1 * self.y)) - self.np.ones(self.a_rows)) + (0.5 * ((self.K.T).dot(t_0) * self.y)))
        g_ = g_0
        return f_, g_

    def functionValueEqConstraint000(self, _x):
        a = self.variables(_x)
        f = (self.y).dot(a)
        return f

    def gradientEqConstraint000(self, _x):
        a = self.variables(_x)
        g_ = (self.y)
        return g_

    def jacProdEqConstraint000(self, _x, _v):
        a = self.variables(_x)
        gv_ = ((_v * self.y))
        return gv_

def solve(K, c, y, np, max_iter=1000, verbose=False):
    NLP = GenoNLP(K, c, y, np)
    x0 = NLP.getStartingPoint()
    lb = NLP.getLowerBounds()
    ub = NLP.getUpperBounds()
    options = {'eps_pg' : 1E-4,
               'constraint_tol' : 1E-4,
               'max_iter' : max_iter,
               'm' : 10,
               'ls' : 0,
               'verbose' : 5  if verbose else 0# Set it to 0 to fully mute it.
              }
    if USE_GENO_SOLVER:
        # Check if installed GENO solver version is sufficient.
        check_version('0.1.0')
        constraints = ({'type' : 'eq',
                        'fun' : NLP.functionValueEqConstraint000,
                        'jacprod' : NLP.jacProdEqConstraint000})
        result = minimize(NLP.fAndG, x0, lb=lb, ub=ub, options=options,
                      constraints=constraints, np=np)
    else:
        constraints = ({'type' : 'eq',
                        'fun' : NLP.functionValueEqConstraint000,
                        'jac' : NLP.gradientEqConstraint000})
        result = minimize(NLP.fAndG, x0, jac=True, method='SLSQP',
                          bounds=list(zip(lb, ub)),
                          constraints=constraints)

    a = NLP.variables(result.x)
    return result, a

