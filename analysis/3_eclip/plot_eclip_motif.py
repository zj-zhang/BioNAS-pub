
from pkg_resources import resource_filename
import os
from BioNAS.Controller.state_space import State
from BioNAS.utils import plots,io,motif
from eclip_model_space import get_state_space, input_state, output_state, model_compile_dict
from multitask_eclip import get_motif_Kfunc
from BioNAS.Interpret.interpret import get_hist_index_by_conditions, match_quantity, get_models_from_hist_by_load

try:
	import cPickle as pickle
except:
	import pickle

import os
import numpy as np
import pandas as pd
from collections import OrderedDict

state_space = get_state_space()
working_dir = "tmp_eclip"

hist_file_list = ["/mnt/isilon/xing_lab/zhangz4/src/BioNAS/examples/eclip/tmp_eclip/train_history.csv"]


def read_hist():
	# read history
	hist = io.read_history(hist_file_list)
	hist.ID = hist.ID-1
	return hist


def get_models(hist):
	# best config
	high_kn_conditions = OrderedDict({
		'L1': ('==', str(state_space[0][-2])),
		'L2': ('==', str(state_space[1][-1])),
		'L3': ('==', str(state_space[2][1])),
		'L4': ('==', str(state_space[3][0])),
		'L5': ('==', str(state_space[4][-2])),
		'L6': ('==', str(state_space[5][-1])),
		'L7': ('==', str(state_space[6][-1])),
		'L8': ('==', str(state_space[7][2])),
	})

	high_kn_idx = get_hist_index_by_conditions(high_kn_conditions, hist)
	rest_idx = get_hist_index_by_conditions(high_kn_conditions, hist, True)
	rest_hist = hist.iloc[rest_idx]
	q = hist.iloc[high_kn_idx].loss
	p = rest_hist.loss
	low_kn_idx = match_quantity(
		q, p, 
		num_sample_per_slice=4, 
		slice=20, replace=False)

	print(np.percentile(hist.iloc[high_kn_idx].loss, q=[10,30,50,70,90]))
	print(np.percentile(rest_hist.iloc[low_kn_idx].loss, q=[10,30,50,70,90]))
	print(np.percentile(hist.iloc[high_kn_idx].knowledge, q=[10,30,50,70,90]))
	print(np.percentile(rest_hist.iloc[low_kn_idx].knowledge, q=[10,30,50,70,90]))
	
	#high_kn_model_dict = get_models_from_hist_by_load(high_kn_idx[0:5], hist, input_state, output_state, state_space, model_compile_dict)
	#low_kn_model_dict = get_models_from_hist_by_load(low_kn_idx[0:5], rest_hist, input_state, output_state, state_space, model_compile_dict)

	#return high_kn_model_dict, low_kn_model_dict

	mean_high_kn = np.mean(hist.iloc[high_kn_idx].knowledge)
	std_high_kn = np.std(hist.iloc[high_kn_idx].knowledge)
	left = mean_high_kn - std_high_kn
	right = mean_high_kn + std_high_kn
	np.random.seed(1) # for reproducible
	ids = hist.iloc[high_kn_idx].query("knowledge > %f and knowledge < %f"%(left, right)).ID
	high_kn_idx_pick = np.random.choice(ids, 1)[0]
	
	#high_kn_idx_pick = hist.iloc[high_kn_idx]['knowledge'].idxmin()
	model = get_models_from_hist_by_load([high_kn_idx_pick], hist)[high_kn_idx_pick]
	return model


def plot_motif(model, mkf):
	print(mkf(model, None))
	score_dict, weight_dict = mkf.get_matched_model_weights()

	sorted_motif_scores = sorted(score_dict.items(), key=lambda kv: kv[1])

	# top 80: 20%
	motif_to_plot = [x[0] for x in sorted_motif_scores[0:80] if mkf.W_knowledge[x[0]].shape[0]>5]
	tmp = []; mset = set()
	for m in motif_to_plot:
		try:
			n, v = m.split('_')
		except:
			n = m
		if n in mset:
			continue
		else:
			tmp.append(m)
			mset.add(n)
	motif_to_plot = tmp
	if not os.path.isdir('model_motifs'):
		os.mkdir('model_motifs')

	for m in motif_to_plot:
		motif.draw_dnalogo_Rscript(weight_dict[m].T, 'model_motifs/%s.model.pdf'%m)
		motif.draw_dnalogo_Rscript(mkf.W_knowledge[m].T, 'model_motifs/%s.motif.pdf'%m)



def run():
	hist = read_hist()
	model = get_models(hist)
	mkf = get_motif_Kfunc()
	plot_motif(model, mkf)



if __name__ == '__main__':
	run()