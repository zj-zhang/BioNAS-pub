# -*- coding: UTF-8 -*-

import sys
from pkg_resources import resource_filename

from BioNAS.Controller.controller import *
from BioNAS.Controller.state_space import *
from BioNAS.MockBlackBox.mock_manager import *
from BioNAS.Controller.train import *
from BioNAS.MockBlackBox.simple_conv1d_space import get_state_space


def get_controller(state_space):
	'''Test function for building controller network. A controller is a LSTM cell that predicts the next

	layer given the previous layer and all previous layers (as stored in the hidden cell states). The 
	controller model is trained by policy gradients as in reinforcement learning.
	Args:
		state_space: the State_Space object to search optimal layer compositions from.
		controller_units: larger number of units defines more complex controller network.
		embedding_dim: embed state_space to a unified input dimension.
		optimizer: optimization alogrithm for controller net
		exploration_rate: the prob. of random sampling from state_space instead of sampling from controller.
	'''
	with tf.device("/cpu:0"):
		controller = Controller(
			state_space,
			controller_units=50,
			embedding_dim=8,
			optimizer='adam',
			discount_factor=0.0,
			clip_val=0.2,
			kl_threshold=0.01,
			train_pi_iter=100,
			lr_pi=0.1,
			buffer_size=15, ## num of episodes saved
			batch_size=5
			)
	return controller



def get_mock_manager(history_fn_list, Lambda=1., wd='./tmp_mock'):
	'''Test function for building a mock manager. A mock manager 
	returns a loss and knowledge instantly based on previous 
	training history.
	Args:
		train_history_fn_list: a list of 
	'''
	manager = MockManager(
			history_fn_list = history_fn_list,
			model_compile_dict = { 'loss':'binary_crossentropy', 'optimizer':'adam', 'metrics':['acc']},
			working_dir=wd,
			Lambda=Lambda,
 			verbose=0
 		)
	return manager


def get_environment(controller, manager, should_plot, logger=None, wd='./tmp_mock/'):
	'''Test function for getting a training environment for controller.
	Args:
		controller: a built controller net
		manager: a manager is a function that manages child-networks. Manager is built upon `model_fn` and `reward_fn`.
		max_trials: maximum number of child-net generated
		working_dir: specifies the working dir where all files will be generated in.
		resume_prev_run: restore the controller parameters and states from a preivous run
		logger: a Logging file handler; creates a new logger if None.
	'''
	env = ControllerTrainEnvironment(
		controller,
		manager,
		max_episode=200,
		max_step_per_ep=3,
		logger=logger,
		resume_prev_run=False,
		should_plot=should_plot,
		working_dir=wd
	)
	return env

def train_simple_controller(should_plot=False, logger=None, Lambda=1., wd='./tmp_mock/'):
	# first get state_space
	state_space = get_state_space()


	# init network manager

	hist_file_list = [resource_filename("BioNAS.resources", "mock_black_box/tmp_%i/train_history.csv"%i) for i in range(1,21) ]

	manager = get_mock_manager(hist_file_list, Lambda=Lambda, wd=wd)

	# get controller
	controller = get_controller(state_space)

	# get the training environment
	env = get_environment(controller, manager, should_plot, logger, wd=wd)

	# train one step
	env.train()

	return

if __name__ == '__main__':
	logger = setup_logger()
	B, Lambda, wd = sys.argv[1:]
	B = int(B)
	Lambda = 10**(float(Lambda))
	for t in range(B):
		train_simple_controller(t==(B-1), logger, Lambda, wd)
