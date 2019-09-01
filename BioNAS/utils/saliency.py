# -*- coding: UTF-8 -*-
from __future__ import print_function
import numpy as np

def approx_grad(model, X, epsilon=0.01):
	nr, nc = X.shape
	grad = np.zeros(nc)
	for i in range(nc):
		X_tmp = np.copy(X)
		X_tmp[:, i] += epsilon
		y_plus = model.predict(X_tmp)
		X_tmp[:, i] -= 2*epsilon
		y_minus = model.predict(X_tmp)
		grad[i] = np.mean((y_plus - y_minus)/2./epsilon)
	return grad

def approx_grad_array(model, X, epsilon=0.01):
	nr, nc = X.shape
	grad_array = np.zeros((nr, nc))
	for i in range(nc):
		X_tmp = np.copy(X)
		X_tmp[:, i] += epsilon
		y_plus = model.predict(X_tmp).flatten()
		X_tmp[:, i] -= 2*epsilon
		y_minus = model.predict(X_tmp).flatten()
		grad_array[:, i] = (y_plus - y_minus)/2./epsilon
	return grad_array