import numpy as np


def multinomial_KL_divergence(P, Q):
	'''compute the KL-divergence for two metrics of identical shape
	:param P: n by 4 array, reference prob.
	:param Q: array, target prob.
	:rtype float: distance measured by KL-divergence
	'''
	assert P.shape == Q.shape
	d = 0
	for j in range(P.shape[0]):
		idx = np.where(P[j,:]!=0)[0]
		d += np.sum( P[j,idx] * (np.log(P[j,idx]) - np.log(Q[j,idx])) )
	return d


def compare_motif_diff_size(P, Q):
	'''find maximum match between two metrics
	P and Q with different sizes
	'''
	best_d = float('inf')
	# by padding, Q is always wider than P
	P_half_len = int(np.ceil(P.shape[0]/2.))
	Q_pad = np.concatenate(
		[
			np.ones((P_half_len,Q.shape[1]))/Q.shape[1], 
			Q,
			np.ones((P_half_len, Q.shape[1]))/Q.shape[1]
		], axis=0)

	# find best match of P in Q
	for i in range(0, Q_pad.shape[0]-P.shape[0]+1):
		d = multinomial_KL_divergence(P, Q_pad[i:(i+len(P))])
		if d < best_d:
			best_d = d

	best_d /= float(P.shape[0])
	return best_d
