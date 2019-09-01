'''
build a multi-layer CNN model
Zijun Zhang
7.18.2018
'''

import os
import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.optimizers import SGD,Adam
import keras

def get_session(gpu_fraction=0.35):
        '''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

        num_threads = os.environ.get('OMP_NUM_THREADS')
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

        if num_threads:
            return tf.Session(config=tf.ConfigProto(
                gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
        else:
            return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

KTF.set_session(get_session())

from keras.models import Model
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D, AveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, merge, Lambda
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import regularizers
import keras.backend as K

NUM_OUTPUT = 333
DEFAULT_WEIGHTS = np.ones((NUM_OUTPUT,2))
DEFAULT_WEIGHTS[:,1] = 20


def sparsek_vec(x):
	convmap = x
	shape = tf.shape(convmap)
	nb = shape[0]
	nl = shape[1]
	tk = tf.cast(0.2 * tf.cast(nl, tf.float32), tf.int32)

	convmapr= K.reshape(convmap, tf.stack([nb,-1]))# convert input to [batch, -1]

	th, _ = tf.nn.top_k(convmapr, tk) # nb*k
	th1 = tf.slice(th,[0,tk-1],[-1,1]) # nb*1 get kth threshold
	thr = tf.reshape(th1, tf.stack([nb, 1]))
	drop = tf.where(convmap < thr,\
	                    tf.zeros([nb,nl], tf.float32), tf.ones([nb,nl], tf.float32))
	convmap=convmap*drop
	    
	return convmap

def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
    return weighted_loss

def build_model(sequ_len=300, num_output=NUM_OUTPUT, class_weights=None):
	if class_weights is None:
		class_weights = np.ones((num_output, 2))
	## sequence input
	input = Input(shape=(sequ_len,4), name='sequence_input')
	conv1 = Convolution1D(filters=150, kernel_size=6, padding="valid",
		activation="sigmoid",
		use_bias=False,
		strides=1, #W_regularizer = regularizers.l1(0.001),
		kernel_initializer='glorot_normal',
		name='conv1')(input)
	pool1 = AveragePooling1D(pool_size=2, strides=2)(conv1)
	conv2 = Convolution1D(filters=200, kernel_size=6, padding="valid",
		activation="relu",
		strides=1,
		kernel_initializer='glorot_normal',
		name='conv2')(pool1)
	pool2 = MaxPooling1D(pool_size=2, strides=2)(conv2)
	conv3 = Convolution1D(filters=400, kernel_size=6, padding="valid",
		activation="relu",
		strides=1,
		kernel_initializer='glorot_normal',
		name='conv3')(pool2)
	#pool3 = MaxPooling1D(pool_size=4, strides=4)(conv3)


	# fully-connected layers
	#flat = Flatten()(pool2)
	#flat = BatchNormalization()(flat)
	#mlp1 = Dense(units=100, activation='relu')( flat ) 
	#mlp1 = Dropout(0.3)(mlp1)

	# global pooling layer
	mlp1 = GlobalMaxPooling1D()(conv3)
	mlp1 = Lambda(sparsek_vec)(mlp1)

	output = Dense(units=num_output, name='output', activation='sigmoid')(mlp1)
	# compile
	#sgd = SGD(lr=0.01, momentum=0.05, decay=0.99, nesterov=True)
	#adam = Adam(lr=0.005, decay=0.01)
	model = Model(input=input, output=output)
	model.compile( loss='binary_crossentropy', 
		#loss=get_weighted_loss(class_weights),
		optimizer='adam',
		metrics=['acc', keras.metrics.top_k_categorical_accuracy])
				#keras.metrics.sparse_top_k_categorical_accuracy])
	
	return model
