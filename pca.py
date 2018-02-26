'''
Principal Components Analysis
'''

import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np


def set_trace():
    """A Poor mans break point"""
    # without this in iPython debugger can generate strange characters.
    from IPython.core.debugger import Pdb
    Pdb().set_trace(sys._getframe().f_back)


def perform_pca(x, pca_opt):
    components = pca_opt
    x_std = StandardScaler().fit_transform(x)
    model = PCA(n_components=components)
    x_reduced = model.fit_transform(x_std)
    var_explained = model.explained_variance_ratio_.cumsum()
    print components, 'components explains ', var_explained[-1] * 100, '% variance'
    # get_covariance_matrix(x_std)
    return x_reduced
