from __future__ import print_function
import numpy as np

class Simulator:
	'''
	:param n: int, number of samples
	:param p: int, number of features
	:param beta_a: list-like, effect sizes for additive
	:param beta_i: list-like, effect sizes for interaction
	'''
	def __init__(self, n, p, beta_a, beta_i):
		self.n = n
		self.p = p
		self.beta_a = np.array(beta_a).astype('float32')
		self.beta_i = np.array(beta_i).astype('float32')

	def sample_effect(self, drop_a, drop_i):
		'''TODO: sample random effect sizes
		:param drop_a: probability of masking of additive
		:param drop_i: prob. for masking of interaction
		'''
		self.beta_a = np.random.normal
		self.beta_i = np.random.normal

	def sample_data(self):
		'''
		:rtype (X,y): a tuple of X and y 
		'''
		from sklearn.preprocessing import PolynomialFeatures
		X = np.array(np.random.randint(low=0, high=3, size=self.n * self.p)).reshape(self.n, self.p).astype('float32')
		X_s = PolynomialFeatures(2, interaction_only=False, include_bias=False).fit_transform(X)
		beta = np.concatenate([self.beta_a, self.beta_i])
		y = X_s.dot(beta) + np.random.normal(loc=0, scale=1, size=self.n)
		return (X, y)
