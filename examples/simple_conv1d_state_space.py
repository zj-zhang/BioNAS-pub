# -*- coding: UTF-8 -*-

#import BioNAS.tensorflow_mem as tf_mem
#tf_mem.KTF.set_session(tf_mem.get_session(gpu_fraction=0.1))

from BioNAS.Controller.controller import *
from BioNAS.Controller.state_space import *
from BioNAS.Controller.model import *
from BioNAS.Controller.manager import *
from BioNAS.Controller.reward import *
from BioNAS.Controller.train import *
from BioNAS.Controller import post_processing
from BioNAS.KFunctions.K_func import *
from BioNAS.utils import data_parser

from keras.optimizers import Adam
from keras import regularizers
from BioNAS.MockBlackBox.simple_conv1d_space import get_state_space, get_data


def get_model_from_random_state(state_space, input_state=State('Input', shape=(200, 4)),
                                output_state=State('Dense', units=1, activation='sigmoid', kernel_regularizer=regularizers.l1(1e-3))):
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
    model_compile_dict = {'loss': 'binary_crossentropy', 'optimizer': 'adam', 'metrics': ['acc']}
    model = build_sequential_model(model_states, input_state, output_state, model_compile_dict)
    return model


def get_motif_Kfunc():
    '''Test function for building Knowledge K-function for Motifs.
	A Motif K-function (mkf) takes two arguments: higher temperature will erase difference, while lower temperature will
	enlarge differences; Lambda_regularizer specifies the penalty strength for having more filters to explain a given set
	of knowledge. 
	'''
    mkf = Motif_K_function(temperature=0.1, Lambda_regularizer=0.01)
    mkf.knowledge_encoder(['MYC_known1'], './BioNAS/resources/rbp_motif/encode_motifs.txt.gz', False)
    return mkf


def get_reward_func(mkf):
    '''Test function for building a Reward function that combines Loss (L) and Knowledge (K) by
	Reward(a) = L + Lambda * K.
	In this case the K-function is parsed in as an instance of Motif K Function. Lambda is the weight
	for K.
	'''
    reward_fn = Knowledge_Reward(mkf, Lambda=1.)
    return reward_fn


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
        embedding_dim=8,
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


def get_manager(train_data, validation_data, model_fn, reward_fn, wd='./tmp'):
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
                             input_state=State('Input', shape=(200, 4)),
                             output_state=State('Dense', units=1, activation='sigmoid'),
                             working_dir=wd,
                             model_compile_dict={'loss': 'binary_crossentropy', 'optimizer': 'adam',
                                                 'metrics': ['acc']},
                             model_fn=model_fn,
                             reward_fn=reward_fn,
                             post_processing_fn=post_processing.post_processing_general,
                             epochs=100, verbose=0,
                             child_batchsize=500
                             )
    return manager


def get_environment(controller, manager, should_plot, wd='./tmp'):
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
        logger=None,
        resume_prev_run=False,
        should_plot=should_plot,
        working_dir=wd
        )
    return env


def train_simple_controller(ismaster=True):
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
    manager = get_manager(train_data, validation_data, model_fn, reward_fn)

    # test a step
    model_states = state_space.get_random_model_states()
    # res = manager.get_rewards(
    #		trial = 0,
    #		model_states = model_states,
    #	)

    # get controller
    controller = get_controller(state_space)

    # get the training environment
    env = get_environment(controller, manager, ismaster)

    # train one step
    env.train()
    return


if __name__ == '__main__':
    train_simple_controller()
