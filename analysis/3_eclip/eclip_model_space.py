
from BioNAS.Controller.state_space import StateSpace, State
from BioNAS.KFunctions.K_func import Motif_K_function

import sys
import pandas as pd

input_state = State('Input', shape=(1000,4))
output_state = State('Dense', units=333, activation='sigmoid')
model_compile_dict = { 
	'loss':'binary_crossentropy', 
	'optimizer':'adam', 
	#'metrics':['acc', 'top_k_categorical_accuracy']}
	'metrics':['acc']}


def get_state_space():
	'''State_space is the place we define all possible operations (called `States`) on each layer to stack a neural net. 
	The state_space is defined in a layer-by-layer manner, i.e. first define the first layer (layer 0), then layer 1, 
	so on and so forth. See below for how to define all possible choices for a given layer.
	Note:
		Use key-word arguments to define layer-specific attributes. 
		Adding `Identity` state to a layer is basically omitting a layer by performing no operations.
	Args:
		None
	Returns:
		a pre-defined state_space object
	'''
	state_space = StateSpace()
	state_space.add_layer(0, [
		State('conv1d', filters=100, kernel_size=20, kernel_initializer='glorot_uniform', activation='relu', name="conv1"),
		State('conv1d', filters=100, kernel_size=14, kernel_initializer='glorot_uniform', activation='relu', name="conv1"),
		State('conv1d', filters=100, kernel_size=8, kernel_initializer='glorot_uniform', activation='relu', name="conv1"),
		State('conv1d', filters=300, kernel_size=20, kernel_initializer='glorot_uniform', activation='relu', name="conv1"),
		State('conv1d', filters=300, kernel_size=14, kernel_initializer='glorot_uniform', activation='relu', name="conv1"),
		State('conv1d', filters=300, kernel_size=8, kernel_initializer='glorot_uniform', activation='relu', name="conv1"),
		State('conv1d', filters=500, kernel_size=20, kernel_initializer='glorot_uniform', activation='relu', name="conv1"),
		State('conv1d', filters=500, kernel_size=14, kernel_initializer='glorot_uniform', activation='relu', name="conv1"),
		State('conv1d', filters=500, kernel_size=8, kernel_initializer='glorot_uniform', activation='relu', name="conv1"),		
	])
	state_space.add_layer(1, [
		State('Identity'),
		State('maxpool1d', pool_size=4, strides=4), 
		State('avgpool1d', pool_size=4, strides=4),
	])
	state_space.add_layer(2, [
		State('Identity'),
		State('conv1d', filters=100, kernel_size=20, kernel_initializer='glorot_uniform', activation='relu', name="conv2"),
		State('conv1d', filters=100, kernel_size=14, kernel_initializer='glorot_uniform', activation='relu', name="conv2"),
		State('conv1d', filters=100, kernel_size=8, kernel_initializer='glorot_uniform', activation='relu', name="conv2"),
		State('conv1d', filters=300, kernel_size=20, kernel_initializer='glorot_uniform', activation='relu', name="conv2"),
		State('conv1d', filters=300, kernel_size=14, kernel_initializer='glorot_uniform', activation='relu', name="conv2"),
		State('conv1d', filters=300, kernel_size=8, kernel_initializer='glorot_uniform', activation='relu', name="conv2"),
		State('conv1d', filters=500, kernel_size=20, kernel_initializer='glorot_uniform', activation='relu', name="conv2"),
		State('conv1d', filters=500, kernel_size=14, kernel_initializer='glorot_uniform', activation='relu', name="conv2"),
		State('conv1d', filters=500, kernel_size=8, kernel_initializer='glorot_uniform', activation='relu', name="conv2"),		
	])
	state_space.add_layer(3, [
		State('Identity'),
		State('maxpool1d', pool_size=4, strides=4), 
		State('avgpool1d', pool_size=4, strides=4),
	])
	state_space.add_layer(4, [
		State('Identity'),
		State('conv1d', filters=100, kernel_size=20, kernel_initializer='glorot_uniform', activation='relu', name="conv3"),
		State('conv1d', filters=100, kernel_size=14, kernel_initializer='glorot_uniform', activation='relu', name="conv3"),
		State('conv1d', filters=100, kernel_size=8, kernel_initializer='glorot_uniform', activation='relu', name="conv3"),
		State('conv1d', filters=300, kernel_size=20, kernel_initializer='glorot_uniform', activation='relu', name="conv3"),
		State('conv1d', filters=300, kernel_size=14, kernel_initializer='glorot_uniform', activation='relu', name="conv3"),
		State('conv1d', filters=300, kernel_size=8, kernel_initializer='glorot_uniform', activation='relu', name="conv3"),
		State('conv1d', filters=500, kernel_size=20, kernel_initializer='glorot_uniform', activation='relu', name="conv3"),
		State('conv1d', filters=500, kernel_size=14, kernel_initializer='glorot_uniform', activation='relu', name="conv3"),
		State('conv1d', filters=500, kernel_size=8, kernel_initializer='glorot_uniform', activation='relu', name="conv3"),		
	])
	state_space.add_layer(5, [
		State('Identity'),
		State('maxpool1d', pool_size=4, strides=4), 
		State('avgpool1d', pool_size=4, strides=4),
	])
	state_space.add_layer(6, [
		State('Flatten'), 
		State('GlobalMaxPool1D'), 
		State('GlobalAvgPool1D'),
		State('SFC', output_dim=50, symmetric=True, smoothness_penalty=1., smoothness_l1=True, smoothness_second_diff=True, curvature_constraint=10., name='sfc')
	])
	state_space.add_layer(7,[
		State('Identity'),
		State('sparsek_vec'),
		State('dropout', rate=0.2)
	])
	state_space.add_layer(8, [
		State('Dense', units=100, activation='relu'), 
		State('Dense', units=300, activation='relu'), 
		State('Dense', units=500, activation='relu'), 
		State('Identity') 
	])	
	return state_space
