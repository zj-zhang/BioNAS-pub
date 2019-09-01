'''
make h5 files for training blocks of the
eCLIP model
Zijun Zhang
7.18.2018 
revised 3.30.2019: add h5 file for faster i/o in memory
'''

import h5py
#from pygr import seqdb
import pysam
import pandas as pd
import numpy as np

FLANKING_SIZE = 100

def read_label(fn='label_matrix.gene_bins_100bp.with_peaks.bed'):
	df = pd.read_table(fn)
	df.index = df.ID 
	df.drop(['ID'], axis=1, inplace=True)
	return df

def fetch_seq(chr, start, end, strand, genome):
	def reverse_complement(dna):
		complement = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A', 'N':'N'}
		return ''.join([complement[base] for base in dna[::-1]])
	try:
		seq = genome.fetch(chr, start, end)
		if strand=='-':
			seq = reverse_complement(seq)
	except:
		raise Exception('pysam cannot fetch sequeces')
	return seq


def fetch_seq_pygr(chr, start, end, strand, genome):
	"""Fetch a genmoic sequence upon calling `pygr`
	`genome` is initialized and stored in self.genome
	Args:
		chr (str): chromosome
		start (int): start locus
		end (int): end locuse
		strand (str): either '+' or '-'
	Returns:
		seq (str): genomic sequence in upper cases
	"""
	try:
		seq = genome[chr][start:end]
		if strand == "-":
			seq = -seq
	except:
		raise Exception('pygr cannot fetch sequences')
	return str(seq).upper()


def seq_to_matrix(seq):
	"""Convert a list of characters to an N by 4 integer matrix
	Args:
		seq (list): list of characters A,C,G,T,-,=;
			heterozygous locus is separated by semiconlon
			'-' is filler for heterozygous indels;
			'=' is homozygous deletions
		padding (int): length for padding
	Returns:
		seq_mat (np.array): numpy matrix for the sequences
	"""
	letter_to_index = {'A':0, 'C':1, 'G':2, 'T':3}
	mat_len = len(seq)
	mat = np.zeros((mat_len, 4))
	n_letter = len(seq)
	for i in range(n_letter):
		try:
			letter = seq[i]
			if letter in letter_to_index:
				mat[i, letter_to_index[letter] ] += 2
			elif letter=='N':
				mat[i, : ] += 0.5
		except KeyError:
			print(i, letter, seq[i])
	return mat

def matrix_to_seq(mat):
	index_to_letter = {0:'A', 1:'C', 2:'G', 3:'T'}
	seq = ''.join([ 
		index_to_letter[ np.where(mat[i,:]>0)[0][0] ]
		for i in range(mat.shape[0])
		] )
	return seq


def get_generator(label_df, 
	genome_fn='/u/nobackup/yxing/NOBACKUP/frankwoe/hg19/hg19.noRand.fa', 
	batch_size=32,
	y_idx=None):
	#genome = seqdb.SequenceFileDB(genome_fn)
	genome = pysam.FastaFile(genome_fn)
	df_idx = np.array(list(label_df.index.values))
	while True:
		np.random.shuffle(df_idx)
		try:
			for i in range(0, len(df_idx), batch_size):
				X_batch = []
				Y_batch = []
				this = df_idx[i:(i+batch_size)]
				for idx in this:
					chrom, start, end, strand = idx.split(':')
					left = int(start) - FLANKING_SIZE
					right = int(end) + FLANKING_SIZE
					seq = fetch_seq(chrom, left, right, strand, genome)
					x_j = seq_to_matrix(seq)
					if y_idx is None:
						y_j = label_df.loc[idx].values
					else:
						y_j = label_df.loc[idx].values[y_idx]
					X_batch.append(x_j)
					Y_batch.append(y_j)
				X_batch = np.array(X_batch)
				Y_batch = np.array(Y_batch)
				yield (X_batch, Y_batch)
		except StopIteration:
			pass

def store_data(label_fn, 
	save_fn,
	genome_fn='/u/nobackup/yxing/NOBACKUP/frankwoe/hg19/hg19.noRand.fa', 
	flanking_size=450,
	y_idx=None):
	label_df = read_label(label_fn)
	#genome = seqdb.SequenceFileDB(genome_fn)
	genome = pysam.FastaFile(genome_fn)
	df_idx = np.array(list(label_df.index.values))
	X_batch = []
	Y_batch = []
	#this = df_idx[i:(i+batch_size)]
	for idx in df_idx:
		chrom, start, end, strand = idx.split(':')
		left = int(start) - flanking_size
		right = int(end) + flanking_size
		seq = fetch_seq(chrom, left, right, strand, genome)
		x_j = seq_to_matrix(seq)
		if y_idx is None:
			y_j = label_df.loc[idx].values
		else:
			y_j = label_df.loc[idx].values[y_idx]
		X_batch.append(x_j)
		Y_batch.append(y_j)
	X_batch = np.array(X_batch, dtype="int32")
	Y_batch = np.array(Y_batch, dtype="int32")
	with h5py.File(save_fn, 'w') as f:
		f.create_dataset("X", data=X_batch)
		f.create_dataset("Y", data=Y_batch)
	return 0