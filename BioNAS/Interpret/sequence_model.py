# -*- coding: UTF-8 -*-

"""Utils for interpret and explain the 
genomic sequence-based models

"""

from ..utils import motif as motif_fn, io, data_parser
from ..Controller import model as model_fn
from ..MockBlackBox import simple_conv1d_space
import sys
import numpy as np
import pandas as pd
import time
from sklearn import metrics
from collections import defaultdict


def read_motif(motif_name, motif_file, is_log_motif):
	motif_dict = motif_fn.load_binding_motif_pssm(motif_file, is_log_motif)
	target_motif = motif_dict[motif_name]
	return target_motif


def scan_motif(seq, motif):
	""" match_score = sum of individual site likelihood * site weight
	site weight = 1 / site entropy

	TODO:
		add weights for each sites, down-weigh un-informative sites
	"""
	motif_len = motif.shape[0]
	seq_len = seq.shape[0]
	match_score = np.zeros(seq_len-motif_len+1)
	motif[np.where(motif==0)] = 1e-10
	for i in range(seq_len - motif_len + 1):
		this_seq = seq[i:(i+motif_len)]
		#ms = np.sum(this_seq*motif)
		ms = 0
		for s, m in zip(this_seq, motif):
			idx = np.where(s!=0)[0] 
			ms += np.sum(s[idx]*np.log10(m[idx]))
		match_score[i] = ms
	return match_score


def normalize_sequence(seq):
	norm_seq = seq / (np.sum(seq)/seq.shape[0])
	normalizer = (np.sum(seq)/seq.shape[0])
	return norm_seq, normalizer


def matrix_to_seq(mat):
	seq = ''
	letter_dict = {0:'A', 1:'C', 2:'G', 3:'T'}
	for i in range(mat.shape[0]):
		idx = np.where(mat[i]>0)[0].flatten()
		if len(idx) == 1:
			seq += letter_dict[idx[0]]
		else:			
			seq += 'N'
	return seq


def saturate_permute_pred(seq, model, normalizer, lambda_pred):
	"""permute every nucleotide to every letter on every position, and 
	record the change in prediction score as vanilla substract perturbed
	"""
	vanilla_pred = lambda_pred(model.predict(np.expand_dims(seq*normalizer,axis=0)))
	pseq_pred_change = np.zeros(seq.shape)
	for i in range(seq.shape[0]):
		for j in range(seq.shape[1]):
			pseq = seq.copy()
			pseq[i,:] = 0
			pseq[i,j] = 1
			pseq_pred = lambda_pred(model.predict(np.expand_dims(pseq*normalizer, axis=0)))
			pseq_pred_change[i,j] = vanilla_pred - pseq_pred
	return pseq_pred_change


def saturate_permute_motif(seq, motif, normalizer):
	"""permute every nucleotide to every letter on every position, and 
	record the change in max motif match score as vanilla substract perturbed
	"""
	vanilla_match_score = scan_motif(seq, motif)
	vanilla_max = np.max(vanilla_match_score)
	pseq_motif_change = np.zeros(seq.shape)
	for i in range(seq.shape[0]):
		for j in range(seq.shape[1]):
			pseq = seq.copy()
			pseq[i,:] = 0
			pseq[i,j] = 1
			pseq_max = np.max(scan_motif(pseq, motif))
			pseq_motif_change[i,j] = vanilla_max - pseq_max
	#pseq_motif_change[np.where(seq>0)] = np.nan
	return pseq_motif_change


def evaluate_permute_acc_single_seq(pred_change, motif_change, seq, disrupt_cutoff=np.log10(10), nochange_cutoff=np.log10(2), auc_scorer=metrics.roc_auc_score):
	"""measure the consistency of model perturbation with motif perturbation
	in a single input sequence, i.e. local prioritization of genome perturbations
	
	Note:
		for reduce sites, need to reverse the sign when computing AUROC/AUPR
	"""
	#disrupt_idx = np.where( motif_change>=disrupt_cutoff )
	#nochange_idx = np.where( (motif_change > -enhance_reduce_cutoff) & (motif_change < enhance_reduce_cutoff) & (seq==0) )
	#nochange_idx = np.where( (motif_change==0) & (seq==0) )
	#enhance_idx = np.where( (motif_change> enhance_reduce_cutoff ) & (motif_change<disrupt_cutoff) )
	#reduce_idx = np.where(motif_change < -enhance_reduce_cutoff )
	disrupt_idx = np.where(motif_change>=disrupt_cutoff)
	nochange_idx = np.where( (np.abs(motif_change) <= nochange_cutoff) & (seq==0) )

	try:
		auc_disrupt = auc_scorer(
			y_true = np.concatenate([np.ones(len(disrupt_idx[0])), np.zeros(len(nochange_idx[0])) ]),
			y_score = np.concatenate([pred_change[disrupt_idx].flatten(), pred_change[nochange_idx].flatten() ])
			)
	except ValueError:
		auc_disrupt = np.nan
	#try:
	#	auc_enhance = auc_scorer(
	#		y_true = np.concatenate([np.ones(len(enhance_idx[0])), np.zeros(len(nochange_idx[0])) ]),
	#		y_score = np.concatenate([pred_change[enhance_idx].flatten(), pred_change[nochange_idx].flatten() ])
	#		)
	#except ValueError:
	#	auc_enhance = np.nan
	#try:
	#	auc_reduce = auc_scorer(
	#		y_true = np.concatenate([np.ones(len(reduce_idx[0])), np.zeros(len(nochange_idx[0])) ]),
	#		y_score = np.concatenate([-pred_change[reduce_idx].flatten(), -pred_change[nochange_idx].flatten() ])
	#		)
	#except ValueError:
	#	auc_reduce = np.nan

	eval_dict = {
		"disrupt": pred_change[disrupt_idx],
		"nochange": pred_change[nochange_idx],
		#"enhance": pred_change[enhance_idx],
		#"reduce": pred_change[reduce_idx],
		"auc_disrupt": auc_disrupt,
		#"auc_enhance": auc_enhance,
		#"auc_reduce": auc_reduce
	}
	return eval_dict


def evaluate_permute_acc_aggregate(model_idx, model_performance_dict, data_idx_list, auc_scorer=metrics.roc_auc_score):
	"""DEPRECATED: measure the consistency of model perturbation with motif perturbation
	by aggregating all input sequences, i.e. genome-wide prioritization of 
	perturbed DNAs
	
	Note:
		for reduce sites, need to reverse the sign when computing AUROC/AUPR
	"""	
	disrupt = []
	nochange = []
	#enhance = []
	#repress = []
	for i in data_idx_list:
		disrupt.extend(model_performance_dict[(model_idx, i)]['disrupt'])
		nochange.extend(model_performance_dict[(model_idx, i)]['nochange'])
		#enhance.extend(model_performance_dict[(model_idx, i)]['enhance'])
		#repress.extend(model_performance_dict[(model_idx, i)]['reduce'])

	auc_disrupt = auc_scorer(
		y_true = np.concatenate([np.ones(len(disrupt)), np.zeros(len(nochange)) ]),
		y_score = np.concatenate([disrupt, nochange ])
		)
	#auc_enhance = auc_scorer(
	#	y_true = np.concatenate([np.ones(len(enhance)), np.zeros(len(nochange)) ]),
	#	y_score = np.concatenate([ enhance, nochange ])
	#	)
	#auc_reduce = auc_scorer(
	#	y_true = np.concatenate([np.ones(len(repress)), np.zeros(len(nochange)) ]),
	#	y_score = -np.concatenate([ repress, nochange ])
	#	)

	#return {'auc_disrupt': auc_disrupt, 'auc_enhance': auc_enhance, 'auc_reduce': auc_reduce}
	return {'auc_disrupt': auc_disrupt}


def get_motif_goldstandard(X_seq, motif):
	"""Given a set of sequence inputs, systematically perturb
	every letter every position, and record as gold-standard
	"""

	motif_change_dict = {}
	for i in range(X_seq.shape[0]):
		sys.stdout.write("%i "%i)
		sys.stdout.flush()
		x, normalizer = normalize_sequence(X_seq[i])
		motif_change = saturate_permute_motif(x, motif, normalizer)
		motif_change_dict[i] = motif_change
	return motif_change_dict


def models_sensitivity_motif(model_dict, x_seq, motif_change_dict, auc_scorer=metrics.roc_auc_score, lambda_pred=lambda x:x.flatten(), **kwargs):
	"""evaluate the sensitivity for a set of models based on input x and motif-change as 
	gold-standard.
	Sensitivity defined as responsiveness and robustness of models for perturbations in input 
	features.

	Returns:
		defaultdict(dict) : model index -> {eval_attr: eval_val}

	Note:
		for reduce sites, need to reverse the sign for auc_scorer
	"""
	agg_auc_scorer = lambda a,b: auc_scorer(
			y_true = np.concatenate([np.ones(len(a)), np.zeros(len(b)) ]),
			y_score = np.concatenate([a, b ])
		)
	# not picklable; do not use
	#model_performance_dict = defaultdict(lambda:defaultdict(list))
	model_performance_dict = {}
	model_count = 0
	total_model_count = len(model_dict)
	for model_idx in model_dict:
		model_count += 1
		sys.stdout.write("%i/%i analyzing model %i."%(model_count, total_model_count, model_idx))
		sys.stdout.flush()
		seq_count = 0
		start_time = time.time()
		model_performance_dict[model_idx] = defaultdict(list)
		for seq_idx in motif_change_dict:
			seq_count += 1
			if seq_count/float(len(motif_change_dict))>0.1:
				sys.stdout.write('.')
				sys.stdout.flush()
				seq_count = 0
			model = model_dict[model_idx]
			motif_change = motif_change_dict[seq_idx]
			x1, normalizer = normalize_sequence(x_seq[seq_idx])
			pred_change = saturate_permute_pred(x1, model, normalizer, lambda_pred)
			eval_dict = evaluate_permute_acc_single_seq(pred_change, motif_change, x1, auc_scorer=auc_scorer, **kwargs)
			# extend lists
			model_performance_dict[model_idx]['disrupt'].extend(eval_dict['disrupt'])
			model_performance_dict[model_idx]['nochange'].extend(eval_dict['nochange'])
			#model_performance_dict[model_idx]['enhance'].extend(eval_dict['enhance'])
			#model_performance_dict[model_idx]['reduce'].extend(eval_dict['reduce'])
			# append measurements
			model_performance_dict[model_idx]['auc_disrupt'].append(eval_dict['auc_disrupt'])
			#model_performance_dict[model_idx]['auc_enhance'].append(eval_dict['auc_enhance'])
			#model_performance_dict[model_idx]['auc_reduce'].append(eval_dict['auc_reduce'])

		model_performance_dict[model_idx]['auc_agg_disrupt'] = \
			agg_auc_scorer(
				model_performance_dict[model_idx]['disrupt'],
				model_performance_dict[model_idx]['nochange']
			)
		#model_performance_dict[model_idx]['auc_agg_enhance'] = \
		#	agg_auc_scorer(
		#		model_performance_dict[model_idx]['enhance'],
		#		model_performance_dict[model_idx]['nochange']
		#	)
		#model_performance_dict[model_idx]['auc_agg_reduce'] = \
		#	agg_auc_scorer(
		#		- np.array(model_performance_dict[model_idx]['reduce']),
		#		model_performance_dict[model_idx]['nochange']
		#	)
		elapsed_time = time.time() - start_time
		sys.stdout.write("used %.3fs\n"%elapsed_time)
		sys.stdout.flush()
	return model_performance_dict


def rescore_sensitivity_motif(model_performance_dict, auc_scorer=metrics.roc_auc_score):
	"""re-score
	Note:
		for reduce sites, need to reverse the sign for auc_scorer
	"""
	agg_auc_scorer = lambda a,b: auc_scorer(
			y_true = np.concatenate([np.ones(len(a)), np.zeros(len(b)) ]),
			y_score = np.concatenate([a, b ])
		)
	# not picklable; do not use
	#model_performance_dict = defaultdict(lambda:defaultdict(list))
	model_count = 0
	total_model_count = len(model_performance_dict)
	for model_idx in model_performance_dict:
		model_count += 1
		sys.stdout.write("%i/%i analyzing model %i."%(model_count, total_model_count, model_idx))
		sys.stdout.flush()
		seq_count = 0
		start_time = time.time()
		model_performance_dict[model_idx]['auc_agg_disrupt'] = \
			agg_auc_scorer(
				model_performance_dict[model_idx]['disrupt'],
				model_performance_dict[model_idx]['nochange']
			)
		model_performance_dict[model_idx]['auc_agg_enhance'] = \
			agg_auc_scorer(
				model_performance_dict[model_idx]['enhance'],
				model_performance_dict[model_idx]['nochange']
			)
		model_performance_dict[model_idx]['auc_agg_reduce'] = \
			agg_auc_scorer(
				- np.array(model_performance_dict[model_idx]['reduce']),
				model_performance_dict[model_idx]['nochange']
			)
		model_performance_dict[model_idx]['auc_agg_disrupt_enhance'] = \
			agg_auc_scorer(
				np.concatenate(
					[
						model_performance_dict[model_idx]['disrupt'],
						model_performance_dict[model_idx]['enhance']
					]),
				model_performance_dict[model_idx]['nochange']
			)

		elapsed_time = time.time() - start_time
		sys.stdout.write("used %.3fs\n"%elapsed_time)
		sys.stdout.flush()
	return model_performance_dict


def summarize_sensitivity_motif(model_performance_dict):
	"""summarize a set of models performance into global/local
	motif accuracy
	"""
	local_disrupt_auc = []
	local_enhance_auc = []
	local_reduce_auc = []
	global_disrupt_auc = []
	global_enhance_auc = []
	global_reduce_auc = []
	for model_idx in model_performance_dict:
		local_disrupt_auc.extend(model_performance_dict[model_idx]['auc_disrupt'])
		local_enhance_auc.extend(model_performance_dict[model_idx]['auc_enhance'])
		local_reduce_auc.extend(model_performance_dict[model_idx]['auc_reduce'])
		global_enhance_auc.append(model_performance_dict[model_idx]['auc_agg_enhance'])
		global_reduce_auc.append(model_performance_dict[model_idx]['auc_agg_reduce'])
		global_disrupt_auc.append(model_performance_dict[model_idx]['auc_agg_disrupt'])

	local_disrupt_auc = np.array(local_disrupt_auc)[~np.isnan(local_disrupt_auc)]
	local_enhance_auc = np.array(local_enhance_auc)[~np.isnan(local_enhance_auc)]
	local_reduce_auc = np.array(local_reduce_auc)[~np.isnan(local_reduce_auc)]
	global_disrupt_auc = np.array(global_disrupt_auc)[~np.isnan(global_disrupt_auc)]
	global_enhance_auc = np.array(global_enhance_auc)[~np.isnan(global_enhance_auc)]
	global_reduce_auc = np.array(global_reduce_auc)[~np.isnan(global_reduce_auc)]

	df = pd.DataFrame({
		'auc': np.concatenate([
			local_disrupt_auc, local_enhance_auc, local_reduce_auc, 
			global_disrupt_auc, global_enhance_auc, global_reduce_auc
			]),
		'type': \
			['local']*(local_disrupt_auc.shape[0] + local_enhance_auc.shape[0]+local_reduce_auc.shape[0]) + \
			['global']*(global_disrupt_auc.shape[0] + global_enhance_auc.shape[0]+global_reduce_auc.shape[0]),
		'group': \
			['disrupt']*local_disrupt_auc.shape[0] + ['enhance']*local_enhance_auc.shape[0] + ['reduce']*local_reduce_auc.shape[0] + \
			['disrupt']*global_disrupt_auc.shape[0] + ['enhance']*global_enhance_auc.shape[0] + ['reduce']*global_reduce_auc.shape[0]
		})

	#return {'local_disrupt': local_disrupt_auc, 'global_disrupt': global_disrupt_auc, 
	#	'local_enhance': local_enhance_auc, 'local_reduce': local_reduce_auc, 
	#	'global_enhance': global_enhance_auc, 'global_reduce': global_reduce_auc}
	return df
