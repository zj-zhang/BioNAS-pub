# -*- coding: UTF-8 -*-

'''Reward function for weighted sum of Loss and Knowledge
ZZJ
Nov. 18, 2018
'''

class Knowledge_Reward:
	def __init__(self, knowledge_function, Lambda):
		self.knowledge_function = knowledge_function
		self.Lambda = Lambda

	def __call__(self, model, data):
		X, y = data
		loss_and_metrics = model.evaluate(X, y)
		# Loss function values will always be the first value
		L = loss_and_metrics[0]
		K = self.knowledge_function(model=model, data=data)
		reward_metrics = {'knowledge': K}
		return -(L + self.Lambda * K), loss_and_metrics, reward_metrics