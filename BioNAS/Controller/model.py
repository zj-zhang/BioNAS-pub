# -*- coding: UTF-8 -*-


from keras.models import Sequential, Model
from keras.layers import Input, Dense, Conv2D, GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPooling2D, AveragePooling2D, Flatten
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D, AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, merge, Lambda, Permute
from keras.layers.recurrent import LSTM, GRU
from keras.callbacks import ModelCheckpoint, EarlyStopping
from BioNAS.operators.custom_filters import Layer_deNovo, SeparableFC
from BioNAS.operators.sparsek import sparsek_vec

import os
import ast

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import keras.backend as K

def get_layer(x,state):
    if state.Layer_type == 'dense':
        return Dense(**state.Layer_attributes)(x)

    elif state.Layer_type == 'sfc':
        return SeparableFC(**state.Layer_attributes)(x)
    
    elif state.Layer_type == 'input':
        return Input(**state.Layer_attributes)

    elif state.Layer_type == 'conv1d':
        return Conv1D(**state.Layer_attributes)(x)

    elif state.Layer_type == 'denovo':
        x = Lambda(lambda x: K.expand_dims(x))(x)
        x = Permute(dims=(2,1,3))(x)
        x = Layer_deNovo(**state.Layer_attributes)(x)
        x = Lambda(lambda x: K.squeeze(x, axis=1))(x)
        return x 

    elif state.Layer_type == 'sparsek_vec':
        x = Lambda(sparsek_vec)(x)
        return x
    
    elif state.Layer_type == 'maxpool1d':
        return MaxPooling1D(**state.Layer_attributes)(x)
    
    elif state.Layer_type == 'avgpool1d':
        return AveragePooling1D(**state.Layer_attributes)(x)
    
    elif state.Layer_type == 'lstm':
        return LSTM(**state.Layer_attributes)(x)

    elif state.Layer_type == 'flatten':
        return Flatten()(x)

    elif state.Layer_type == 'globalavgpool1d':
        return GlobalAveragePooling1D()(x)

    elif state.Layer_type == 'globalmaxpool1d':
        return GlobalMaxPooling1D()(x)

    elif state.Layer_type == 'dropout':
        return Dropout(**state.Layer_attributes)(x)
    
    elif state.Layer_type == 'identity':
        return Lambda(lambda x:x)(x)
    
    else:
        raise Exception('Layer_type "%s" is not understood'%layer_type)


def build_sequential_model(model_states, input_state, output_state, model_compile_dict):
    '''
    Args:
        model_states: a list of operators sampled from operator space
        input_operator: specifies the input tensor
        output_state: specifies the output tensor, e.g. Dense(1, activation='sigmoid')
        model_compile_dict: a dict of `loss`, `optimizer` and `metrics`
    Returns:
        Keras.Model instance
    '''
    inp = get_layer(None, input_state) 
    x = inp
    for state in model_states:
        x = get_layer(x, state)
    out = get_layer(x, output_state)
    model = Model(inputs=inp, outputs=out)
    model.compile(**model_compile_dict)
    return model


def build_multiGPU_sequential_model(model_states, input_state, output_state, model_compile_dict, gpus=4, **kwargs):
    try:
        from keras.utils import multi_gpu_model
    except:
        raise Exception("multi gpu not supported in keras. check your version.")
    vanilla_model = build_sequential_model(model_states, input_state, output_state, model_compile_dict)
    model = multi_gpu_model(vanilla_model, gpus=gpus, **kwargs)
    model.compile(**model_compile_dict)
    return model


def build_sequential_model_from_string(model_states_str, input_state, output_state, state_space, model_compile_dict):
    """build a sequential model from a string of states
    """
    assert len(model_states_str) == len(state_space)
    str_to_state = [ [str(state) for state in state_space[i]] for i in range(len(state_space)) ]
    try:
        model_states = [ state_space[i][str_to_state[i].index(model_states_str[i])] for i in range(len(state_space)) ]
    except ValueError:
        raise Exception("model_states_str not found in state-space")
    return build_sequential_model(model_states, input_state, output_state, model_compile_dict)


def build_multiGPU_sequential_model_from_string(model_states_str, input_state, output_state, state_space, model_compile_dict):
    """build a sequential model from a string of states
    """
    assert len(model_states_str) == len(state_space)
    str_to_state = [ [str(state) for state in state_space[i]] for i in range(len(state_space)) ]
    try:
        model_states = [ state_space[i][str_to_state[i].index(model_states_str[i])] for i in range(len(state_space)) ]
    except ValueError:
        raise Exception("model_states_str not found in state-space")
    return build_multiGPU_sequential_model(model_states, input_state, output_state, model_compile_dict)


class ModelBuilder(object):
    '''Scaffold of Model Builder
    '''
    def __init__(self, model_type, input_shape, output_shape, transfer_weights):
        self.model_type = model_type
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.transfer_weights = transfer_weights

    def __call__(self, model_states):
        raise NotImplementedError


