# -*- coding: utf8 -*-

'''Read a set of train_history.csv files, and 
make a mock manager that returns a list of mock
validation metrics given any child net architecture
'''
import os
import keras.backend as K
import pandas as pd
import numpy as np
from ..Controller.manager import NetworkManager
from ..utils.io import read_history_set, read_history


def get_mock_reward(model_states, train_history_df, metric):
	model_states_str = [str(x) for x in model_states]; 
	idx_bool = np.array([train_history_df['L%i'%(i+1)]==model_states_str[i] for i in range(len(model_states_str))])
	index = np.apply_along_axis(func1d=lambda x: all(x), axis=0, arr=idx_bool); 
	if np.sum(index)==0:
		#mu, sd = np.mean(train_history_df[metric]), np.std(train_history_df[metric])
		#return np.abs(np.random.normal(loc=mu, scale=sd))
		#return train_history_df[metric].iloc[np.random.choice(train_history_df.index)]
		raise Exception("cannot find config in history: \n {}".format("\n".join(str(model_states_str[i]) for i in range(len(model_states_str)) ) ) )
	else:
		#mu, sd = np.mean(train_history_df[metric].iloc[index]), np.std(train_history_df[metric].iloc[index])
		return train_history_df[metric].iloc[np.random.choice(np.where(index)[0])]


def get_mock_reward_fn(model_states, train_history_df, lbd=1.):
	Lambda = lbd
	#train_history_df = read_history(['train_history.1.csv', 'train_history.2.csv'])
	metric = ['loss', 'knowledge', 'acc']
	mock_reward = get_mock_reward(model_states, train_history_df, metric)
	this_reward = -(mock_reward['loss'] + Lambda * mock_reward['knowledge'])
	loss_and_metrics = [mock_reward['loss'], mock_reward['acc']]
	reward_metrics = {'knowledge': mock_reward['knowledge']}
	return this_reward, loss_and_metrics, reward_metrics


class MockManager(NetworkManager):
	'''
	Helper class to manage the generation of subnetwork training given a dataset
	'''
	def __init__(self,
				history_fn_list,
				model_compile_dict,
				train_data=None, 
				validation_data=None, 
				input_state=None,
				output_state=None,
				model_fn=None, 
				reward_fn=None, 
				post_processing_fn=None, 
				working_dir='.',
				Lambda = 1.,
				acc_beta=0.8, 
				clip_rewards=0.0,
				verbose=0):
		#super(MockManager, self).__init__()
		self._lambda = Lambda
		self.model_compile_dict = model_compile_dict
		self.train_history_df = read_history(history_fn_list)
		self.clip_rewards = clip_rewards
		self.verbose = verbose
		self.working_dir = working_dir
		if not os.path.exists(self.working_dir):
			os.makedirs(self.working_dir)

		self.beta = acc_beta
		self.beta_bias = acc_beta
		self.moving_reward = 0.0


	def get_rewards(self, trial, model_states):
		this_reward, loss_and_metrics, reward_metrics = get_mock_reward_fn(model_states, self.train_history_df, self._lambda)
		# evaluate the model by `reward_fn`
		#this_reward, loss_and_metrics, reward_metrics = self.reward_fn(model, (X_val, y_val))
		loss = loss_and_metrics.pop(0)
		loss_and_metrics = { str(self.model_compile_dict['metrics'][i]):loss_and_metrics[i] for i in range(len(loss_and_metrics))}
		loss_and_metrics['loss'] = loss
		if reward_metrics:
			loss_and_metrics.update(reward_metrics)

		# do any post processing, 
		# e.g. save child net, plot training history, plot scattered prediction.
		#if self.post_processing_fn:
		#	val_pred = model.predict(X_val)
		#	self.post_processing_fn(
		#			trial=trial, 
		#			model=model, 
		#			hist=hist, 
		#			data=self.validation_data, 
		#			pred=val_pred, 
		#			loss_and_metrics=loss_and_metrics, 
		#			working_dir=self.working_dir
		#	)

		# compute the reward based on Exponentially-Weighted Average (EWA)
		# moving averaged rewards
		reward = (this_reward - self.moving_reward)

		# if rewards are clipped, clip them in the range -0.05 to 0.05
		if self.clip_rewards:
			reward = np.clip(reward, -0.05, 0.05)

		# update moving accuracy with bias correction for 1st update
		if self.beta > 0.0 and self.beta < 1.0:
			self.moving_reward = self.beta * self.moving_reward + (1 - self.beta) * this_reward
			self.moving_reward = self.moving_reward / (1. - self.beta_bias)
			self.beta_bias = 0

			reward = np.clip(reward, -0.1, 0.1)

		if self.verbose:
			print()
			print("Manager: EWA Accuracy = ", self.moving_reward)

		# clean up resources and GPU memory
		#network_sess.close()
		#del model
		#del hist
		return this_reward, loss_and_metrics


