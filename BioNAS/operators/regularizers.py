# -*- coding: UTF-8 -*-

'''
placeholder for custom regularizers (e.g. SFC.SepFCSmoothnessRegularizer)
'''

from __future__ import absolute_import
from keras import backend as K
from keras.regularizers import *

class SepFCSmoothnessRegularizer(Regularizer):
	""" Specific to SeparableFC
		Applies penalty to length-wise differences in W_pos.
		
	# Arguments
		smoothness: penalty to be applied to difference
			of adjacent weights in the length dimension
		l1: if smoothness penalty is to be computed in terms of the
			the absolute difference, set to True
			otherwise, penalty is computed in terms of the squared difference
		second_diff: if smoothness penalty is to be applied to the
			difference of the difference, set to True
			otherwise, penalty is applied to the first differencesmoothness:
	"""

	def __init__(self, smoothness, l1=True, second_diff=False):
		self.smoothness = float(smoothness)
		self.l1 = l1
		self.second_diff = second_diff

	def __call__(self, x):
		diff1 = x[:,1:] - x[:,:-1]
		diff2 = diff1[:,1:] - diff1[:,:-1]
		if self.second_diff == True:
			diff = diff2
		else:
			diff = diff1
		if self.l1 == True:
			return K.mean(K.abs(diff))*self.smoothness
		else:
			return K.mean(K.square(diff))*self.smoothness

	def get_config(self):
		return {'name': self.__class__.__name__,
				'smoothness': float(self.smoothness),
				'l1': bool(self.l1),
				'second_diff': bool(self.second_diff)}
