# -*- coding: UTF-8 -*-

import BioNAS.tensorflow_mem

from BioNAS.Controller.controller import *
from BioNAS.Controller.model import *
from BioNAS.Controller.manager import *
from BioNAS.Controller.reward import *
from BioNAS.Controller.train import *
from BioNAS.Controller import post_processing
from BioNAS.KFunctions.K_func import Motif_K_function
from BioNAS.utils import data_parser

import sys


def get_model_from_random_state(state_space, input_state = State('Input', shape=(200,4)), output_state = State('Dense', units=1, activation='sigmoid')):
	'''A test function for `model_fn`, model building function that builds a Keras.Model object
	from a list of State objests, and an input State and Output State.
	Args:
		state_space: a state_space object where a random list of states to be sampled from
		input_state: a state object that specifies the input data/tensor shape. Must be 'Input' type of state.
		output_state: a state object that specifies the output shape.
	Returns:
		a Keras.Model object built from the given arguments.
	'''
	assert input_state.Layer_type == 'input'
	model_states = state_space.get_random_model_states()	
	model_compile_dict = { 'loss':'binary_crossentropy', 'optimizer':'adam', 'metrics':['acc']}
	model = build_sequential_model(model_states, input_state, output_state, model_compile_dict)
	return model


def get_motif_Kfunc():
	'''Test function for building Knowledge K-function for Motifs.
	A Motif K-function (mkf) takes two arguments: higher temperature will erase difference, while lower temperature will
	enlarge differences; Lambda_regularizer specifies the penalty strength for having more filters to explain a given set
	of knowledge. 
	'''
	target_rbp_df = pd.concat([
		#pd.read_table(os.path.join(os.path.dirname(__file__), 'asprin', 'HepG2_asprin_snps.tsv')),
		pd.read_table(os.path.join('.', 'asprin', 'HepG2_asprin_snps.tsv')),
		#pd.read_table(os.path.join(os.path.dirname(__file__), 'asprin', 'K562_asprin_snps.tsv')),
		pd.read_table(os.path.join('.', 'asprin', 'K562_asprin_snps.tsv')),
		])

	rbp=list(set(target_rbp_df.rbp))
	from BioNAS.utils import motif
	from pkg_resources import resource_filename

	motif_dict = motif.load_binding_motif_pssm(resource_filename('BioNAS.resources', 'rbp_motif/human_rbp_pwm.txt'), True)
	target_rbp_motifs = [x for x in motif_dict if any([x.startswith(y) for y in rbp])]
	
	mkf = Motif_K_function(temperature=0.1, Lambda_regularizer=0.00)

	mkf.knowledge_encoder(
		target_rbp_motifs, 
		resource_filename('BioNAS.resources', 'rbp_motif/human_rbp_pwm.txt'), 
		True)
	return mkf


def get_reward_func(mkf):
	'''Test function for building a Reward function that combines Loss (L) and Knowledge (K) by 
	Reward(a) = L + Lambda * K.
	In this case the K-function is parsed in as an instance of Motif K Function. Lambda is the weight
	for K.
	'''
	reward_fn = Knowledge_Reward(mkf, Lambda=1.)
	return reward_fn


def get_data():
	'''Test function for reading data from a set of FASTA sequences. Read Positive and Negative FASTA files, and
	convert to 4 x N matrices.
	'''
	import h5py
	train_store = h5py.File(os.path.join('.', 'encode_data', 'label_matrix.train.h5'), 'r')
	X_train = train_store['X'].value
	Y_train = train_store['Y'].value
	test_store = h5py.File(os.path.join('.', 'encode_data', 'label_matrix.val.h5'), 'r')
	X_test = test_store['X'].value
	Y_test = test_store['Y'].value
	return (X_train, y_train), (X_test, y_test)


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
    controller = Controller(
        state_space,
        controller_units=50,
        embedding_dim=10,
        optimizer='adam',
        discount_factor=0.1,
        clip_val=0.2,
        beta=0.0,
        kl_threshold=0.01,
        train_pi_iter=100,
        lr_pi=1e-1,
        buffer_size=15 ## num of trajectories saved
    )
    return controller


def get_manager(train_data, validation_data, model_fn, reward_fn, wd):
	'''Test function for building manager. A manager interacts with the controller to get child-net
	model layers, manipulates data around the generated child-network to evaluate performance, and
	feedback the reward to controller.
	Args:
		train_data: a tuple of X,y; or a generator
		validation_data: a tuple of X,y
		model_fn: model building functions
		reward_fn: reward function
		post_processing_fn: do any post-processing after a trained child-net, e.g. plot training 
			history, or storing best model weights.
	'''
	manager = NetworkManager(train_data, validation_data,
			input_state = State('Input', shape=(200,4)), 
			output_state = State('Dense', units=3, activation='sigmoid'), 
			working_dir = wd,
			model_compile_dict = { 'loss':'binary_crossentropy', 'optimizer':'adam', 'metrics':['acc']},
			model_fn = model_fn, 
			reward_fn = reward_fn, 
			post_processing_fn = post_processing.post_processing_general, 
 			epochs=100, verbose=0,
 			child_batchsize=500
 		)
	return manager


def get_environment(controller, manager, wd):
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
		max_episode=300,
		max_step_per_ep=1,
		logger=None,
		resume_prev_run=False,
		should_plot=True,
		working_dir=wd
	)
	return env


def train_simple_controller(wd):
	'''Main entry for this example. Calls all test functions and run the controller training in the training
	environment.
	'''

	# first get state_space
	state_space = get_state_space()
	# then build reward function from K_function
	mkf = get_motif_Kfunc()
	reward_fn = get_reward_func(mkf)

	# set alias for model_fn
	model_fn = build_sequential_model

	# read the data
	train_data, validation_data = get_data()

	# init network manager
	manager = get_manager(train_data, validation_data, model_fn, reward_fn, wd)

	# test a step
	model_states = state_space.get_random_model_states()
	#res = manager.get_rewards(
	#		trial = 0, 
	#		model_states = model_states, 
	#	)

	# get controller
	controller = get_controller(state_space)

	# get the training environment
	env = get_environment(controller, manager, wd)

	# train one step
	env.train()
	return

if __name__ == '__main__':
	try:
		wd = sys.argv[1]
	except:
		wd = './tmp_eclip'
	print("workding_dir = " + str(wd))
	if not os.path.isdir(wd):
		os.makedirs(wd)
	train_simple_controller(wd)
