"""The same SciPy BFGS algorithm with its inverse-Hessian update expanded.

SciPy evaluates (I-rho*s*y') H (I-rho*y*s') + rho*s*s' using two dense
matrix products. Expanding those rank-one factors gives the identical BFGS
update using matrix-vector products and outer products, O(p^2) versus O(p^3).
No likelihood, line search, stopping rule, initialization, or option changes.
The installed SciPy function supplies all other code; no dependency is modified.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import inspect

import numpy as np
import scipy
from scipy.optimize import _optimize

UPDATE_VERSION = "scipy-bfgs-expanded-rank-two-v1"
ORIGINAL_UPDATE = '''Hk = np.dot(A1, np.dot(Hk, A2)) + (rhok * sk[:, np.newaxis] *
                                                 sk[np.newaxis, :])'''
EXPANDED_UPDATE = '''Hy = np.dot(Hk, yk)
        yH = np.dot(yk, Hk)
        Hk = (Hk - rhok * (np.outer(sk, yH) + np.outer(Hy, sk))
              + (rhok + rhok**2 * np.dot(yk, Hy)) * np.outer(sk, sk))'''


@lru_cache(maxsize=1)
def _implementation():
    original = inspect.getsource(_optimize._minimize_bfgs)
    if original.count(ORIGINAL_UPDATE) != 1:
        raise RuntimeError("Installed SciPy BFGS update differs; verify the new source before using the algebraic optimization")
    source = original.replace(ORIGINAL_UPDATE, EXPANDED_UPDATE)
    # A1/A2 are unused after expansion; avoid constructing two dense identities.
    source = source.replace("        A1 = I - sk[:, np.newaxis] * yk[np.newaxis, :] * rhok\n", "")
    source = source.replace("        A2 = I - yk[:, np.newaxis] * sk[np.newaxis, :] * rhok\n", "")
    namespace = dict(vars(_optimize))
    exec(compile(source, "<scipy-bfgs-expanded-rank-two>", "exec"), namespace)
    return namespace["_minimize_bfgs"]


def rank_two_bfgs(fun, x0, args=(), jac=None, hess=None, hessp=None,
                  bounds=None, constraints=(), callback=None, **options):
    # scipy.minimize passes method-independent options to callable methods.
    tolerance = options.pop("tol", None)
    if tolerance is not None:
        options.setdefault("gtol", tolerance)
    return _implementation()(fun, x0, args=args, jac=jac, callback=callback, **options)


def provenance() -> dict:
    return {"version": UPDATE_VERSION, "scipy_version": scipy.__version__,
            "original_function_sha256": hashlib.sha256(inspect.getsource(_optimize._minimize_bfgs).encode()).hexdigest(),
            "change": "algebraic expansion of the identical BFGS rank-two inverse-Hessian update; all other SciPy code/options unchanged"}
