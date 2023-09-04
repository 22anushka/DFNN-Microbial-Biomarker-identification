# Source code: https://github.com/LankyCyril/pyvenn

!pip install venn

from venn import venn
%matplotlib inline
from matplotlib.pyplot import subplots, savefig
from itertools import chain, islice
from string import ascii_uppercase
from numpy.random import choice
import pandas as pd

method_list = ['MAGMA-DFNN', 'S_SE_MIC-DFNN', 'SPARCC-DFNN', 'SPIEC_EASI-DFNN']
sparcc_fname ='/content/var_crcsparcc(1)_withcols.csv'
spiec_fname ='/content/spieceasi(2)_features.csv'
merge_fname='/content/mergedfeatures.csv'
magma_fname='/content/featurelist.csv'
num_rows = 400

df_magma = pd.read_csv(magma_fname,index_col=False,  names = ['features'], nrows=num_rows)
df_s_se_mic = pd.read_csv(merge_fname, index_col=False, names = ['features'], nrows=num_rows)
df_sparcc = pd.read_csv(sparcc_fname, index_col=False, names = ['features'], nrows=num_rows)
df_spiec_easi = pd.read_csv(spiec_fname, index_col=False, names = ['features'], nrows=num_rows)
df_list = [df_magma, df_s_se_mic, df_sparcc, df_spiec_easi]

dataset_dict = {
    name: set(df['features']) # set of top - selected features
    for name, df in zip(method_list, df_list)
}

venn(dataset_dict, fmt="{size:.1f}", cmap="viridis", fontsize=8, legend_loc="upper left")