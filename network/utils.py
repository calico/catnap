import os
import ctypes


def set_mkl_threads():
    try:
        import mkl
        mkl.set_num_threads(1)
        return 0
    except Exception:
        pass

    for name in ["libmkl_rt.so", "libmkl_rt.dylib", "mkl_Rt.dll"]:
        try:
            mkl_rt = ctypes.CDLL(name)
            mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(1)))
            return 0
        except Exception:
            pass

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

set_mkl_threads()

import TVGL as tvgl
import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering


def cluster_nodes(affinity, n_clusters=15):
    spec_cluster = SpectralClustering(n_clusters=n_clusters,
                                      assign_labels="discretize",
                                      affinity='precomputed',
                                      random_state=0).fit(affinity)
    return spec_cluster


def affinity_to_distance(affinity, gamma=2):
    return np.exp(-gamma * affinity ** 2)


def estimate_covariance(df):
    cov = np.sin((np.pi / 2) * df.corr(method='kendall'))
    return cov.values


def get_inferred_precision_matrices(cov_matrices, features,
                                    lmbda=0.5, beta=1.0):
    thetaSet = tvgl.TVGL(cov_matrices, lmbda, beta, indexOfPenalty=5,
                         verbose=True)
    return [pd.DataFrame(theta, columns=features, index=features) for
            theta in thetaSet]
