# -*- coding: UTF-8 -*-

'''Special operators not in standard Keras library from publications
'''


from keras import backend as K
from . import regularizers, constraints

try:
	from keras import initializations
except:
	from keras import initializers as initializations

from keras.layers.core import Layer
import numpy as np

class SeparableFC(Layer):
	'''A Fully-Connected layer with a weights tensor that is
		the product of a matrix W_pos, for learning spatial correlations,
		and a matrix W_chan, for learning cross-channel correlations.
	# Author:
		Anshul Kundaje Lab
	# Source
		https://github.com/kundajelab/keras/blob/master/keras/layers/convolutional.py
		Accessed 2018.12.05

	# Arguments
		output_dim: the number of output neurons
		symmetric: if weights are to be symmetric along length, set to True
		smoothness_penalty: penalty to be applied to difference
			of adjacent weights in the length dimensions
		smoothness_l1: if smoothness penalty is to be computed in terms of the
			the absolute difference, set to True
			otherwise, penalty is computed in terms of the squared difference
		smoothness_second_diff: if smoothness penalty is to be applied to the
			difference of the difference, set to True
			otherwise, penalty is applied to the first difference
		curvature_constraint: constraint to be enforced on the second differences
			of the positional weights matrix
					
	# Input shape
		3D tensor with shape: `(samples, steps, features)`.
		
	# Output shape
		2D tensor with shape: `(samples, output_features)`.

	# Example
		model.add(keras.layers.convolutional.SeparableFC(symmetric=True,
												 smoothness_second_diff=True,
												 output_dim=1000,
												 smoothness_penalty=10.0,
												 smoothness_l1=True))
	'''
	def __init__(self, output_dim, symmetric,
					smoothness_penalty = None,
					smoothness_l1 = False,
					smoothness_second_diff = True,
					curvature_constraint = None, **kwargs):
		super(SeparableFC, self).__init__(**kwargs)
		self.output_dim = output_dim
		self.symmetric = symmetric
		self.smoothness_penalty = smoothness_penalty
		self.smoothness_l1 = smoothness_l1
		self.smoothness_second_diff = smoothness_second_diff
		self.curvature_constraint = curvature_constraint

	def build(self, input_shape):
		import numpy as np
		self.original_length = input_shape[1]
		if (self.symmetric == False):
			self.length = input_shape[1]
		else:
			self.odd_input_length = input_shape[1]%2.0 == 1
			self.length = int(input_shape[1]/2.0 + 0.5)
		self.num_channels = input_shape[2]
		#self.init = (lambda shape, name: initializations.uniform(
		#	shape, np.sqrt(
		#	np.sqrt(2.0/(self.length*self.num_channels+self.output_dim))),
		#	name))
		
		# Fix bug in Keras 2
		self.init = lambda shape=None: initializations.uniform(
			(self.output_dim, self.length), 
			-np.sqrt(np.sqrt(2.0/(self.length*self.num_channels+self.output_dim))),
			np.sqrt(np.sqrt(2.0/(self.length*self.num_channels+self.output_dim)))
			)

		self.W_pos = self.add_weight(
			shape = (self.output_dim, self.length),
			name='{}_W_pos'.format(self.name), 
			#initializer=self.init,
			initializer='random_uniform',
			constraint=(None if self.curvature_constraint is None else
				constraints.CurvatureConstraint(
					self.curvature_constraint)),
			regularizer=(None if self.smoothness_penalty is None else
				regularizers.SepFCSmoothnessRegularizer(
					self.smoothness_penalty,
					self.smoothness_l1,
					self.smoothness_second_diff)))
		self.W_chan = self.add_weight(
			shape = (self.output_dim, self.num_channels),
			name='{}_W_chan'.format(self.name), 
			#initializer=self.init,
			initializer='random_uniform',
			trainable=True)
		self.built = True

	def get_output_shape_for(self, input_shape):
		return (input_shape[0], self.output_dim)
	
	def call(self, x, mask=None):
		if (self.symmetric == False):
			W_pos = self.W_pos
		else:
			W_pos = K.concatenate(
				tensors=[self.W_pos,
				self.W_pos[:,::-1][:,(1 if self.odd_input_length else 0):]],
				axis=1)
		W_output = K.expand_dims(W_pos, 2) * K.expand_dims(self.W_chan, 1)
		W_output = K.reshape(W_output,
		  (self.output_dim, self.original_length*self.num_channels))
		x = K.reshape(x,
		  (-1, self.original_length*self.num_channels))
		output = K.dot(x, K.transpose(W_output))
		return output 

	def get_config(self):
		config = {'output_dim': self.output_dim,
				  'symmetric': self.symmetric,
				  'smoothness_penalty': self.smoothness_penalty,
				  'smoothness_l1': self.smoothness_l1,
				  'smoothness_second_diff': self.smoothness_second_diff,
				  'curvature_constraint': self.curvature_constraint}
		base_config = super(SeparableFC, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))


class denovo_ConvMotif(Layer):
	'''
	Author:
		Matthew Ploenzke, Raphael Irrizarry
	Source:
		https://github.com/mPloenzke/learnMotifs/blob/master/R/layer_denovoMotif.R
		Accessed 2018.12.05
	'''
	def __init__(self, output_dim, filters,
					filter_len,
					lambda_pos=0,
					lambda_filter=0,
					lambda_l1=0,
					lambda_offset=0,
					strides = (1, 1),
					padding = "valid",
					activation = 'sigmoid',
					use_bias = True,
					kernel_initializer = None, #initializer_random_uniform(0,.5),
					bias_initializer = "zeros",
					kernel_regularizer = None, #total_regularizer,
					kernel_constraint = None, #info_constraint,
					bias_constraint = None, #negative_constraint,
					input_shape = None,
					name = 'deNovo_conv', trainable = None, **kwargs):

		super(denovo_ConvMotif, self).__init__(**kwargs)
		raise NotImplementedError()

	def build(self, input_shape):
		'''
		# R Code:
			lambda_filter <<- lambda_filter
			lambda_l1 <<- lambda_l1
			lambda_pos <<- lambda_pos
			filter_len <<- filter_len
			keras::create_layer(keras:::keras$layers$Conv2D, object, list(filters = as.integer(filters),
															kernel_size = keras:::as_integer_tuple(c(4,filter_len)), strides = keras:::as_integer_tuple(strides),
															padding = padding, data_format = NULL, dilation_rate = c(1L, 1L),
															activation = activation, use_bias = use_bias, kernel_initializer = kernel_initializer,
															bias_initializer = bias_initializer, kernel_regularizer = kernel_regularizer,
															bias_regularizer = keras::regularizer_l1(l=lambda_offset), activity_regularizer = NULL,
															kernel_constraint = kernel_constraint, bias_constraint = bias_constraint,
															input_shape = keras:::normalize_shape(input_shape), batch_input_shape = keras:::normalize_shape(NULL),
															batch_size = keras:::as_nullable_integer(NULL), dtype = NULL,
															name = name, trainable = trainable, weights = NULL))
		'''
		self.built=True
