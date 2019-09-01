
from plot_interpret_multitask import read_hist, get_motif_Kfunc, state_space, input_state, output_state, model_compile_dict
from BioNAS.utils.motif import draw_dnalogo_matplot, draw_dnalogo_Rscript
from BioNAS.Interpret.interpret import get_hist_index_by_conditions, match_quantity, get_models_from_hist
import os
import pickle
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt

working_dir = "tmp_interpret_multitask"
file_list = [
	'%s/unmasked_kn.pkl'%working_dir,
	'%s/val_loss.pkl'%working_dir,
	'%s/masked_kn.pkl'%working_dir,
	'%s/test_loss.pkl'%working_dir
	]
test_h_loss, test_l_loss = pickle.load(open(file_list[3], 'rb'))
masked_h_kn, masked_l_kn = pickle.load(open(file_list[2], 'rb'))


def get_models(hist):
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
	
	high_kn_model_dict = get_models_from_hist(high_kn_idx, hist, input_state, output_state, state_space, model_compile_dict)
	low_kn_model_dict = get_models_from_hist(low_kn_idx, rest_hist, input_state, output_state, state_space, model_compile_dict)

	return high_kn_model_dict, low_kn_model_dict


def get_plot_model(hist, h_mod_dict, l_mod_dict):
	print([i for i in range(100) if test_l_loss[i]<0.165 and masked_l_kn[i]>0.35])
	# [14, 58, 59]
	l1 = list(l_mod_dict.keys())[14]  # 97, bad
	l2 = list(l_mod_dict.keys())[59]  # 90, worst
	h1 = list(h_mod_dict.keys())[1] # 615, good

	#target_hist = hist.iloc[ [h1, l1, l2] ]
	target_dict = {
		'high.iloc': h1, 'medium.iloc': l1, 'low.iloc': l2,
		'high.idx': 1, 'medium.idx': 14, 'low.idx': 59,
		}
	return target_dict



def run():

	hist = read_hist()
	h_mod_dict, l_mod_dict = get_models(hist)
	target_dict = get_plot_model(hist, h_mod_dict, l_mod_dict)
	mkf = get_motif_Kfunc()

	# high:
	k1 = mkf(h_mod_dict[target_dict['high.iloc']], None)
	# confirm the model by re-calculate motif knowledge,
	# and compare to previously stored data
	assert k1 == masked_h_kn[target_dict['high.idx']] 
	l1 = test_h_loss[target_dict['high.idx']]
	s1, w1 = mkf.get_matched_model_weights()
	draw_dnalogo_Rscript(w1['MYC_known1'].T, 'high.pdf')

	# medium
	k2 = mkf(l_mod_dict[target_dict['medium.iloc']], None)
	assert k2 == masked_l_kn[target_dict['medium.idx']]
	l2 = test_l_loss[target_dict['medium.idx']]
	s2, w2 = mkf.get_matched_model_weights()
	draw_dnalogo_Rscript(w2['MYC_known1'].T, 'medium.pdf')

	# low
	k3 = mkf(l_mod_dict[target_dict['low.iloc']], None)
	assert k3 == masked_l_kn[target_dict['low.idx']]
	l3 = test_l_loss[target_dict['low.idx']]
	s3, w3 = mkf.get_matched_model_weights()
	draw_dnalogo_Rscript(w3['MYC_known1'].T, 'low.pdf')

	# draw the gold standard
	draw_dnalogo_Rscript(mkf.W_knowledge['MYC_known1'].T, 'truth.pdf')

	# write out data
	target_hist = hist.iloc[ [target_dict['high.iloc'], target_dict['medium.iloc'], target_dict['low.iloc']] ]
	target_hist['masked_knowledge'] = [k1, k2, k3]
	target_hist['test_loss'] = [l1, l2, l3]
	target_hist.to_csv('archs_info.csv', sep='\t')

	# save the models
	h_mod_dict[target_dict['high.iloc']].save('high_kn_model.h5')
	l_mod_dict[target_dict['medium.iloc']].save('mid_kn_model.h5')
	l_mod_dict[target_dict['low.iloc']].save('low_kn_model.h5')



if __name__ == '__main__':
	run()