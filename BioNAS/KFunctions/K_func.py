#!/usr/bin/env python
# -*- coding: UTF-8 -*-

'''
ZZJ
Nov. 16, 2018
'''
from __future__ import print_function
import numpy as np
from .K_math import *

class K_function(object):
	'''
	Scafold of K-function
	'''

	def __init__(self):
		self.W_model = None
		self.W_knowledge = None
		self._build_K_func()

	def __call__(self, model, data, **kwargs):
		self.model_encoder(model, data, **kwargs)
		return self.K_fn(self.W_model, self.W_knowledge, **kwargs)

	def __str__(self):
		return 'K-function for Interpretable Model Learning'

	def model_encoder(self, model, data, **kwargs):
		'''encode $\hat{W}$ from model
		'''
		raise NotImplementedError

	def knowledge_encoder(self, **kwargs):
		'''encode $\tilde{W}$ from existing knowledge
		'''
		raise NotImplementedError

	def _build_K_func(self, **kwargs):
		def K_fn(W_model, W_knowledge, **kwargs):
			return None
		self.K_fn = K_fn

	def get_K_val(self, **kwargs):
		return self.K_fn(self.W_model, self.W_knowledge, **kwargs)



class Motif_K_function(K_function):
	def __init__(self, temperature, Lambda_regularizer, is_multiGPU_model=False):
		super(Motif_K_function, self).__init__()
		self.temperature = temperature
		self.Lambda_regularizer = Lambda_regularizer
		self.is_multiGPU_model = is_multiGPU_model

	def __call__(self, model, data):
		'''Motif_K_function is independent of the data
		'''
		self.model_encoder(model, None)
		return self.K_fn(self.W_model, self.W_knowledge, self.Lambda_regularizer)

	def __str__(self):
		return 'Motif K-function for Interpretable Model Learning'

	def model_encoder(self, model, data):
		if self.is_multiGPU_model:
			multigpu_models = [ model.layers[i] for i in range(len(model.layers)) if model.layers[i].name.startswith('model')]
			assert len(multigpu_models)==1
			layer_dict = { multigpu_models[0].layers[i].name:multigpu_models[0].layers[i] for i in range(len(multigpu_models[0].layers))}
		else:
			layer_dict = { model.layers[i].name:model.layers[i] for i in range(len(model.layers))}
		W = layer_dict['conv1'].get_weights()[0]
		# W dimenstion: filter_len, num_channel(=4), num_filters for Conv1D
		# for Conv2D: num_channel/filter_height(=4), filter_len/filter_len, 1, num_filters
		if len(W.shape) == 4:
			W = np.squeeze(W, axis=2)
			W = np.moveaxis(W, [0,1], [1,0])
		# either way, num_filters is the last dim
		num_filters = W.shape[-1]
		W_prob = np.zeros((W.shape[2], W.shape[0], W.shape[1]))
		beta = 1. / self.temperature
		for i in range(num_filters):
			w = W[:, :, i].copy()
			for j in range(w.shape[0]):
				w[j, :] = np.exp(beta*w[j, :])
				w[j, :] /= np.sum(w[j, :])
			W_prob[i] = w
		self.W_model = W_prob
		return self

	def knowledge_encoder(self, motif_name_list, motif_file, is_log_motif):
		from ..utils import motif
		motif_dict = motif.load_binding_motif_pssm(motif_file, is_log_motif)
		self.W_knowledge = { motif_name: motif_dict[motif_name] for motif_name in motif_name_list}
		return self

	def _build_K_func(self):
		def K_fn(W_model, W_knowledge, Lambda_regularizer):
			score_dict = {x:float('inf') for x in W_knowledge}
			for i in range(len(W_model)):
				w = W_model[i]
				for motif in W_knowledge:
					d = compare_motif_diff_size(W_knowledge[motif], w)
					if score_dict[motif] > d:
						score_dict[motif] = d
			# K = KL + lambda * ||W||
			K = np.mean([x for x in score_dict.values()]) + Lambda_regularizer * W_model.shape[0]
			return K
		self.K_fn = K_fn

	def get_matched_model_weights(self):
		motif_dict = self.W_knowledge
		score_dict = {x:float('inf') for x in motif_dict}
		weight_dict = {}
		for i in range(len(self.W_model)):
			w = self.W_model[i]
			for motif in motif_dict:
				d = compare_motif_diff_size(motif_dict[motif], w)
				if score_dict[motif] > d:
					score_dict[motif] = d
					weight_dict[motif] = w
		return score_dict, weight_dict


