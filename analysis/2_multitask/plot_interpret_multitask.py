
from pkg_resources import resource_filename
import os
from BioNAS.Interpret.sequence_model import *
from BioNAS.Interpret.interpret import *
from BioNAS.Controller.state_space import State
from BioNAS.KFunctions.K_func import *
from BioNAS.utils import plots

from examples.multitask_conv1d_state_space import get_state_space

try:
	import cPickle as pickle
except:
	import pickle

import os
import numpy as np
import pandas as pd
from collections import OrderedDict

input_state = State('Input', shape=(200,4))
output_state = State('Dense', units=3, activation='sigmoid')
model_compile_dict = { 'loss':'binary_crossentropy', 'optimizer':'adam', 'metrics':['acc']}
state_space = get_state_space()
working_dir = "tmp_interpret_multitask"
if not os.path.isdir(working_dir):
	os.makedirs(working_dir)

hist_file_list = ["/home/zhangz4/workspace/src/BioNAS/tmp_multitask_200_3/train_history.csv",
		"/home/zhangz4/workspace/src/BioNAS/tmp_multitask_300_1/train_history.csv"]

def read_hist():
	# read history
	hist = io.read_history(hist_file_list)
	hist.ID = hist.ID-1
	return hist


def get_models(hist):
	# 200_3 best config
	#high_kn_conditions = OrderedDict({
	#	'L1': ('==', str(state_space[0][0])),
	#	'L2': ('==', str(state_space[1][1])),
	#	'L3': ('==', str(state_space[2][0])),
	#	'L4': ('==', str(state_space[3][2])),
	#})
	# 300_1 best config
	high_kn_conditions = OrderedDict({
		'L1': ('==', str(state_space[0][0])),
		'L2': ('==', str(state_space[1][1])),
		'L3': ('==', str(state_space[2][1])),
		'L4': ('==', str(state_space[3][0])),
	})

	high_kn_idx = get_hist_index_by_conditions(high_kn_conditions, hist)
	rest_idx = get_hist_index_by_conditions(high_kn_conditions, hist, True)
	rest_hist = hist.iloc[rest_idx]
	q = hist.iloc[high_kn_idx].loss
	p = rest_hist.loss
	low_kn_idx = match_quantity(q, p, num_sample_per_slice=10, slice=10, replace=False)

	print(np.percentile(hist.iloc[high_kn_idx].loss, q=[10,30,50,70,90]))
	print(np.percentile(rest_hist.iloc[low_kn_idx].loss, q=[10,30,50,70,90]))
	print(np.percentile(hist.iloc[high_kn_idx].knowledge, q=[10,30,50,70,90]))
	print(np.percentile(rest_hist.iloc[low_kn_idx].knowledge, q=[10,30,50,70,90]))
	
	h_kn = hist.iloc[high_kn_idx].knowledge
	l_kn = rest_hist.iloc[low_kn_idx].knowledge
	pickle.dump((h_kn, l_kn), open('%s/unmasked_kn.pkl'%working_dir, 'wb'), -1)

	h_loss = hist.iloc[high_kn_idx].loss
	l_loss = rest_hist.iloc[low_kn_idx].loss
	pickle.dump((h_loss, l_loss), open('%s/val_loss.pkl'%working_dir, 'wb'), -1)

	high_kn_model_dict = get_models_from_hist(high_kn_idx, hist, input_state, output_state, state_space, model_compile_dict)
	low_kn_model_dict = get_models_from_hist(low_kn_idx, rest_hist, input_state, output_state, state_space, model_compile_dict)

	return high_kn_model_dict, low_kn_model_dict



def get_motif_Kfunc():
    '''Test function for building Knowledge K-function for Motifs.
	A Motif K-function (mkf) takes two arguments: higher temperature will erase difference, while lower temperature will
	enlarge differences; Lambda_regularizer specifies the penalty strength for having more filters to explain a given set
	of knowledge. 
	'''
    mkf = Motif_K_function(temperature=0.1, Lambda_regularizer=0.01)
    mkf.knowledge_encoder(['MYC_known1'], resource_filename('BioNAS.resources', 'rbp_motif/encode_motifs.txt.gz'), False)
    return mkf	


def read_data():
	# read in held-out data
	x_heldout = []
	for seqId, seq in data_parser.fastaReader( resource_filename("BioNAS.resources", "simdata/DensityEmbedding_motifs-MYC_known1_min-1_max-1_mean-1_zeroProb-0p0_seqLength-200_numSeqs-1000.fa.gz")):
		x_heldout.append(data_parser.seq_to_matrix(seq))
	x_heldout = np.array(x_heldout, dtype="int32")
	return x_heldout


def get_knowledge_on_test(mkf, high_kn_model_dict, low_kn_model_dict, x_heldout, y_heldout):
	h_kn = [mkf(high_kn_model_dict[x], None) for x in high_kn_model_dict]
	l_kn = [mkf(low_kn_model_dict[x], None) for x in low_kn_model_dict]
	h_loss = [high_kn_model_dict[x].evaluate(x_heldout, y_heldout)[0] for x in high_kn_model_dict]
	l_loss = [low_kn_model_dict[x].evaluate(x_heldout, y_heldout)[0] for x in low_kn_model_dict]

	pickle.dump((h_kn, l_kn), open('%s/masked_kn.pkl'%working_dir, 'wb'), -1)
	pickle.dump((h_loss, l_loss), open('%s/test_loss.pkl'%working_dir, 'wb'), -1)	



def plot_CDF(h_kn, l_kn, h_loss, l_loss, kn_fn, loss_fn, legend_off=False, xlab_kn='Knowledge', xlab_loss='Loss'):
	legend = ['BioNAS optimum, n=%i'%len(h_kn), 'Matching background, n=%i'%len(l_kn)]
	plots.multi_distplot_sns(
		[h_kn, l_kn], legend, 
		save_fn=kn_fn,
		xlab=xlab_kn, ylab='Cumulative Density',
		legend_off = legend_off,
		kde=True,
		hist=True, norm_hist=True,
		hist_kws={"cumulative":True},
		kde_kws={"cumulative":True})

	plots.multi_distplot_sns(
		[h_loss, l_loss], legend, 
		save_fn=loss_fn,
		xlab=xlab_loss, ylab='Cumulative Density',
		legend_off = legend_off,
		kde=True,
		hist=True, norm_hist=True,
		hist_kws={"cumulative":True},
		kde_kws={"cumulative":True})


def run():
	file_list = [
		'%s/unmasked_kn.pkl'%working_dir,
		'%s/val_loss.pkl'%working_dir,
		'%s/masked_kn.pkl'%working_dir,
		'%s/test_loss.pkl'%working_dir
		]
	if not all([os.path.isfile(x) for x in file_list]):
		# read the history
		hist = read_hist()

		# read testing data
		mkf = get_motif_Kfunc()
		x_heldout = read_data()
		y_heldout = np.zeros((x_heldout.shape[0], 3))
		y_heldout[:,2] = 1

		# re-build models
		high_kn_model_dict, low_kn_model_dict = get_models(hist)
		
		# get testing model population difference
		get_knowledge_on_test(mkf, high_kn_model_dict, low_kn_model_dict, x_heldout, y_heldout)
	
	unmasked_h_kn, unmasked_l_kn = pickle.load(open(file_list[0], 'rb'))
	val_h_loss, val_l_loss = pickle.load(open(file_list[1], 'rb'))
	plot_CDF(unmasked_h_kn, unmasked_l_kn, val_h_loss, val_l_loss,
		"%s/unmasked_kn.pdf"%working_dir,
		"%s/val_loss.pdf"%working_dir,
		legend_off = True,
		xlab_kn='Unmasked Knowledge',
		xlab_loss='Validation Loss'
		)	

	masked_h_kn, masked_l_kn = pickle.load(open(file_list[2], 'rb'))
	test_h_loss, test_l_loss = pickle.load(open(file_list[3], 'rb'))
	plot_CDF(masked_h_kn, masked_l_kn, test_h_loss, test_l_loss,
		"%s/masked_kn.pdf"%working_dir,
		"%s/test_loss.pdf"%working_dir,
		legend_off = True,
		xlab_kn='Masked Knowledge',
		xlab_loss='Testing Loss'
		)


if __name__ == '__main__':
	run()