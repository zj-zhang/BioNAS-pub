import keras.backend as K
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Lambda, Flatten, Activation
from keras.initializers import uniform
import tensorflow as tf

import numpy as np

def expand(x):
    return K.expand_dims(x, 1)


def build_actor(inputs, rnn_units, input_dim, maxlen, state_space, scope, trainable=True):
    
    rnn = LSTM(rnn_units, return_state=True, stateful=False, name=scope+'/NAScell', trainable=trainable)


    outputs = [] # pi
    action = []
    embeds = []

    state = None
    expand_layer = Lambda(expand, output_shape=lambda s: (s[0], 1, s[1]), name=scope+'/expand_layer')

    for t in range(maxlen):
        if t == 0:
            input_t = inputs
            input_t = Dense(input_dim, activation='linear', use_bias=False, name=scope+'/embed_input', trainable=trainable)(input_t)
        else:
            input_t = embeds[-1]
            input_t = expand_layer(input_t)
        num_choices = len(state_space[t])
        output_t, h, c = rnn(input_t, initial_state=state)#K.shape(inputs[0])[0])
        state = h, c
        logits = Dense(num_choices, name=scope+'/action_%i' % t, trainable=trainable)(output_t)
        output_t_prob = Activation('softmax', name=scope+'/sofmax_%i'%t)(logits)
        a = Lambda(lambda L: K.squeeze(tf.multinomial(L,1), axis=1))(logits)
        a = Lambda(lambda L: K.one_hot(K.cast(L,'int32'), num_classes=num_choices))(a)
        
        action.append(a)
        embed_t = Dense(input_dim, activation='linear', name=scope+'/embed_%i' % t, trainable=trainable)(output_t_prob)
        #embed_t = Dense(input_dim, activation='linear', name=scope+'/embed_%i' % t, trainable=trainable)(a)
        outputs.append(output_t_prob)
        embeds.append(embed_t)

    return outputs, action



def build_sample_actor(inputs, rnn_units, input_dim, maxlen, state_space, scope, trainable=True):
    
    rnn = LSTM(rnn_units, return_state=True, stateful=False, name=scope+'/NAScell', trainable=trainable)

    outputs = [] # pi
    action = []
    embeds = []

    state = None
    expand_layer = Lambda(expand, output_shape=lambda s: (s[0], 1, s[1]), name=scope+'/expand_layer')

    for t in range(maxlen):
        if t == 0:
            input_t = inputs
            input_t = Dense(input_dim, activation='linear', use_bias=False, name=scope+'/embed_input', trainable=trainable)(input_t)
        else:
            input_t = embeds[-1]
            input_t = expand_layer(input_t)
        num_choices = len(state_space[t])
        output_t, h, c = rnn(input_t, initial_state=state)#K.shape(inputs[0])[0])
        state = h, c
        logits = Dense(num_choices, name=scope+'/logits_%i' % t, trainable=trainable)(output_t)
        output_t_prob = Activation('softmax', name=scope+'/sofmax_%i'%t)(logits)
        a = Lambda(lambda L: K.squeeze(tf.multinomial(L,1), axis=1), name=scope+'/multinomial_%i'%t)(logits)
        a = Lambda(lambda L: K.one_hot(K.cast(L,'int32'), num_classes=num_choices), name=scope+'/onehot_%i'%t)(a)
        
        action.append(a)
        embed_t = Dense(input_dim, activation='linear', name=scope+'/embed_%i' % t, trainable=trainable)(a)
        outputs.append(output_t_prob)
        embeds.append(embed_t)

    return outputs, action


def build_train_actor(inputs, rnn_units, input_dim, maxlen, state_space, scope, trainable=True):

    rnn = LSTM(rnn_units, return_state=True, stateful=False, name=scope+'/NAScell', trainable=trainable)

    outputs = [] # pi
    embeds = []

    state = None
    expand_layer = Lambda(expand, output_shape=lambda s: (s[0], 1, s[1]), name=scope+'/expand_layer')

    for t in range(maxlen):
        if t == 0:
            input_t = inputs[0]
            input_t = Dense(input_dim, activation='linear', use_bias=False, name=scope+'/embed_input', trainable=trainable)(input_t)
        else:
            input_t = embeds[-1]
        num_choices = len(state_space[t])
        output_t, h, c = rnn(input_t, initial_state=state)#K.shape(inputs[0])[0])
        state = h, c
        logits = Dense(num_choices, name=scope+'/logits_%i' % t, trainable=trainable)(output_t)
        output_t_prob = Activation('softmax', name=scope+'/sofmax_%i'%t)(logits)
        
        embed_t = Dense(input_dim, activation='linear', name=scope+'/embed_%i' % t, trainable=trainable)(inputs[t+1])
        outputs.append(output_t_prob)
        embeds.append(embed_t)

    return outputs


def build_critic(state_inputs, scope):

    state_inputs = Lambda(lambda L: K.squeeze(L, axis=1))(state_inputs)
    L = Dense(64, activation='relu', name=scope + '/layer1')(state_inputs)
    L = Dense(64, activation='relu', name=scope + '/layer2')(L)
    #L = Dense(32, activation='tanh', name=scope + '/layer3')(L)
    #L = Dense(16, activation='relu', name=scope + '/layer4')(L)
    value = Dense(1, activation='linear', name=scope + '/value')(L)
    return value
