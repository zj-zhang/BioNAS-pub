'''
split train, val and testing
Zijun Zhang
7.18.2018
'''

import pandas as pd
import numpy as np
np.random.seed(111)

HELD_OUT_CHROM = 'chr8'
VAL_RATIO = 0.1

df = pd.read_table('label_matrix.gene_bins_100bp.with_peaks.bed')
df.index = df.ID
df.drop(['ID'], axis=1, inplace=True)

rowSum_pos = np.apply_along_axis(np.sum, 1, df)
df = df.iloc[rowSum_pos>4]

testing_idx = [x for x in df.index if x.startswith(HELD_OUT_CHROM)]
df.loc[testing_idx].to_csv("label_matrix.testing_%s.bed.gz"%HELD_OUT_CHROM, \
	sep='\t', header=True, compression='gzip')
df.drop(testing_idx, axis=0, inplace=True)

df_idx = df.index.values
val_size = int(len(df_idx)*VAL_RATIO)
val_idx = np.random.choice(df_idx, size=val_size, replace=False)
train_idx = np.setdiff1d(df_idx, val_idx)

df.loc[val_idx].to_csv("label_matrix.val.bed.gz", \
	sep='\t', header=True, compression='gzip')
df.loc[train_idx].to_csv("label_matrix.train.bed.gz", \
	sep='\t', header=True, compression='gzip')
