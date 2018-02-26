'''
Principal Components Analysis - Optimal Number of Components
'''

import sys
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
import seaborn as sns


def set_trace():
    """A Poor mans break point"""
    # without this in iPython debugger can generate strange characters.
    from IPython.core.debugger import Pdb
    Pdb().set_trace(sys._getframe().f_back)


# Choosing number of Principal Components
def graph_pca(x):
    x_std = StandardScaler().fit_transform(x)
    var_exp_list = []
    max_comp = 750
    components = range(1, max_comp)
    for component in components:
        model = PCA(n_components=component)
        x_reduced = model.fit_transform(x_std)
        explained_var = model.explained_variance_ratio_.cumsum()[-1]
        var_exp_list.append(explained_var)
    df = pd.DataFrame(var_exp_list, columns=['Variance Explained'])
    df['Components'] = range(1, max_comp)
    sns.lmplot('Components', 'Variance Explained', data=df, fit_reg=False)
    plt.savefig('C:\Thesis111217\CriminalClassifier\Outputs\pca_graph.png')
    pca_opt = next(i for i, v in enumerate(var_exp_list) if v > 1)
    return pca_opt





 
 




