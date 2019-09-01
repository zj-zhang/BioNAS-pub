

import pandas as pd
import numpy as np
import os
import json
from .plots import sma
from ..Controller.state_space import get_layer_shortname

def read_history_set(fn_list):
	tmp = []
	for fn in fn_list:
		t = pd.read_table(fn, sep=",", header=None)
		t['dir'] = os.path.dirname(fn)
		tmp.append(t)
	d = pd.concat(tmp)
	return d


def read_history(fn_list):
	#d = pd.read_table('train_history.1.csv', sep=",", header=None)
	d = read_history_set(fn_list)
	d.columns = ['ID', 'metrics', 'reward'] + ['L%i'%i for i in range(1, d.shape[1]-3)] + ['dir']
	acc = []
	k = []
	l = []
	for i in range(d.shape[0]):
		tmp = d.iloc[i, 1].split(',')
		acc.append(float(tmp[0][1:]))
		k.append(float(tmp[1]))
		l.append(float(tmp[2][:-1]))

	d['acc'] = acc
	d['knowledge'] = k
	d['loss'] = l
	d.drop(columns=['reward', 'metrics'], inplace=True)
	print("read %i history files, total %i lines"%(len(fn_list), d.shape[0]))
	return d


def read_action_weights(fn):
	"""read 'weight_data.json' and derive the max likelihood
	architecture for each run. 'weight_data.json' stores the weight probability
	by function `save_action_weights` for a bunch of independent mock BioNAS
	optimization runs.
	"""
	data = json.load(open(fn, 'r'))
	archs = []
	tmp = list(data['L0'].keys())
	B = len(data['L0'][tmp[0]])
	for b in range(B):
		this_arch = []
		for l in range(len(data)):
			this_layer = {k:data["L%i"%l][k][b][-1] for k in data["L%i"%l]}
			this_arch.append(
				max(this_layer, key=this_layer.get)
				)
		archs.append(tuple(this_arch))
	return archs

def save_action_weights(probs_list, state_space, working_dir):
	"""
	probs_list: list of probability at each time step
	output a series of graphs each plotting weight of options of each layer over time

	"""
	save_path = os.path.join(working_dir, 'weight_data.json')
	if not os.path.exists(save_path):
		df = {}
		for layer, state_list in enumerate(state_space):
			df['L'+str(layer)] = dict()
			for k in state_list:
				t = get_layer_shortname(k)
				df['L' + str(layer)][t] = []
	else:
		df = json.load(open(save_path, 'r+'))

	data_per_layer = list(zip(*probs_list))
	for layer, state_list in enumerate(state_space):

		data = data_per_layer[int(layer)]
		data = [p.squeeze().tolist() for p in data]
		data_per_type = list(zip(*data))
		for i, d in enumerate(data_per_type):
			t = state_list[i].Layer_type
			k = state_list[i]
			t = get_layer_shortname(k)
			df['L'+str(layer)][t].append(sma(d).tolist())

	json.dump(df, open(save_path, 'w'))


def save_stats(loss_and_metrics_list, working_dir):
	save_path = os.path.join(working_dir, 'nas_training_stats.json')
	if not os.path.exists(save_path):
		df = {'Knowledge':[], 'Accuracy':[], 'Loss':[]}
	else:
		df = json.load(open(save_path, 'r+'))

	keys = list(loss_and_metrics_list[0].keys())
	data = [list(loss_and_metrics.values()) for loss_and_metrics in loss_and_metrics_list]
	data_per_cat = list(zip(*data))
	k_data = data_per_cat[keys.index('knowledge')]
	acc_data = data_per_cat[keys.index('acc')]
	loss_data = data_per_cat[keys.index('loss')]
	df['Knowledge'].append(list(k_data)) 
	df['Accuracy'].append(list(acc_data))
	df['Loss'].append(list(loss_data))
	
	json.dump(df, open(save_path, 'w'))

