# -*- coding: UTF-8 -*-

'''
State space for controller
ZZJ
Nov. 17, 2018
'''

from __future__ import print_function
import numpy as np
import time
import pprint
from collections import defaultdict


def get_layer_shortname(layer):
	if layer.Layer_type == 'conv1d':
		sn = "%s_f%s_k%s"%(layer.Layer_type, layer.Layer_attributes['filters'], layer.Layer_attributes['kernel_size'])
	
	elif layer.Layer_type == 'denovo':
		sn = "%s_f%s_k%s"%('regconv2d', layer.Layer_attributes['filters'], layer.Layer_attributes['kernel_size'])

	elif layer.Layer_type == 'dense':
		sn = "%s_u%s"%(layer.Layer_type, layer.Layer_attributes['units'])
	
	elif layer.Layer_type == 'maxpool1d' or layer.Layer_type == 'avgpool1d':
		sn = layer.Layer_type
	
	elif layer.Layer_type == 'flatten' or layer.Layer_type == 'identity' or layer.Layer_type == 'globalmaxpool1d' or layer.Layer_type == 'globalavgpool1d':
		sn = layer.Layer_type

	elif layer.Layer_type == 'sfc':
		sn = layer.Layer_type

	else:
		sn = str(layer)
	return sn


class State(object):
	def __init__(self, Layer_type, **kwargs):
		Layer_type = Layer_type.lower()
		assert Layer_type in [
				'conv1d', 'maxpool1d', 'avgpool1d',
				'conv2d', 'maxpool2d', 'avgpool2d',
				'lstm',
				'dense', 'input', 'identity',
				'dropout', 'sparsek_vec', 'batchnorm',
				'flatten', 'globalavgpool1d', 'globalavgpool2d', 'globalmaxpool1d', 'globalmaxpool1d'
				'data', 'denovo', 'sfc'
			]
		self.Layer_type = Layer_type
		self.Layer_attributes = kwargs

	def __str__(self):
		return "{}:{}".format(self.Layer_type, self.Layer_attributes)

	def __eq__(self, other):
		return self.Layer_type == other.Layer_type and self.Layer_attributes == other.Layer_attributes

	def __hash__(self):
		unroll_attr =  ((x,self.Layer_attributes[x]) for x in self.Layer_attributes)
		return hash( (self.Layer_type, unroll_attr) )


class StateSpace:
	'''
	State Space manager
	Provides utility functions for holding "states" / "actions" that the controller
	must use to train and predict.
	Also provides a more convenient way to define the search space
	'''
	def __init__(self):
	    self.state_space = defaultdict(list)

	def __str__(self):
		return "StateSpace with {} layers and {} total combinations".format( len(self.state_space), self.get_space_size() )

	def __len__(self):
		return len(self.state_space)

	def __getitem__(self, layer_id):
		if layer_id < 0:
			layer_id = len(self.state_space) + layer_id
		if not layer_id in self.state_space:
			raise IndexError('layer_id out of range')
		return self.state_space[layer_id]

	def __setitem__(self, layer_id, layer_states):
		self.add_layer(layer_id, layer_states)

	def get_space_size(self):
		size_ = 1
		for i in self.state_space:
			size_ *= len(self.state_space[i])
		return size_

	def add_state(self, layer_id, state):
		self.state_space[layer_id].append(state)

	def delete_state(self, layer_id, state_id):
		del self.state_space[layer_id][state_id]

	def add_layer(self, layer_id, layer_states=None):
		if layer_states is None:
			self.state_space[layer_id] = []
		else:
			self.state_space[layer_id] = layer_states
		return self._check_space_integrity()

	def delete_layer(self, layer_id):
		del self.state_space[layer_id]
		return self._check_space_integrity()

	def _check_space_integrity(self):
		return len(self.state_space)-1 == max(self.state_space.keys())

	def print_state_space(self):
		for i in range(len(self.state_space)):
			print("Layer {}".format(i))
			print("\n".join(["  "+str(x) for x in self.state_space[i] ]))
			print('-'*10)
		return

	def get_random_model_states(self):
		model_states = []
		for i in range(len(self.state_space)):
			model_states.append( np.random.choice(self.state_space[i]) )
		return model_states
