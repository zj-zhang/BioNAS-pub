# -*- coding: UTF-8 -*-

import os
import shutil
import gc
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import numpy as np
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint
import tensorflow as tf
from .model import *
from ..utils.plots import plot_training_history
from .post_processing import *


class NetworkManager:
    '''
    Helper class to manage the generation of subnetwork training given a dataset
    '''
    def __init__(self, 
                train_data, 
                validation_data, 
                input_state,
                output_state,
                model_compile_dict,
                model_fn, 
                reward_fn, 
                post_processing_fn, 
                working_dir='.',
                save_full_model=False,
                train_data_size=None,
                epochs=5, 
                child_batchsize=128, 
                acc_beta=0.8, 
                clip_rewards=0.0,
                tune_data_feeder=False,
                verbose=0):
        '''
        Manager which is tasked with creating subnetworks, training them on a dataset, and retrieving
        rewards in the term of accuracy, which is passed to the controller RNN.

        Args:
            dataset: a tuple of 4 arrays (X_train, y_train, X_val, y_val)
            input_state: tuple - specify the input shape to `model_fn`
            output_state: str - parsed to `get_layer` for a fixed output layer
            model_fn: a function for creating Keras.Model; takes model_states, input_state, output_state, model_compile_dict
            reward_fn: a function for computing Reward; takes two arguments, model and data
            post_processing_fn: a function for processing/plotting trained child model
            epochs: number of epochs to train the subnetworks
            child_batchsize: batchsize of training the subnetworks
            acc_beta: exponential weight for the accuracy
            clip_rewards: float - to clip rewards in [-range, range] to prevent
                large weight updates. Use when training is highly unstable.
            
        '''
        self.train_data = train_data
        self.validation_data = validation_data
        self.working_dir = working_dir
        if not os.path.exists(self.working_dir):
                os.makedirs(self.working_dir)
        
        self.save_full_model = save_full_model
        self.train_data_size = len(train_data[0]) if train_data_size is None else train_data_size
        self.epochs = epochs
        self.batchsize = child_batchsize
        self.clip_rewards = clip_rewards
        self.tune_data_feeder = tune_data_feeder
        self.verbose = verbose
        self.input_state = input_state
        self.output_state = output_state
        self.model_compile_dict = model_compile_dict

        self.model_fn = model_fn
        self.reward_fn = reward_fn
        self.post_processing_fn = post_processing_fn

        self.beta = acc_beta
        self.beta_bias = acc_beta
        self.moving_reward = 0.0
        self.entropy_record = []
        # if tune data feeder, must NOT provide train data
        assert self.tune_data_feeder == (self.train_data is None)

    def get_rewards(self, trial, model_states):
        '''
        Creates a subnetwork given the model_states predicted by the controller RNN,
        trains it on the provided dataset, and then returns a reward.

        Args:
            model_states: a list of parsed model_states obtained via an inverse mapping
                from the StateSpace. It is in a specific order as given below:

                Consider 4 states were added to the StateSpace via the `add_state`
                method. Then the `model_states` array will be of length 4, with the
                values of those states in the order that they were added.

                If number of layers is greater than one, then the `model_states` array
                will be of length `4 * number of layers` (in the above scenario).
                The index from [0:4] will be for layer 0, from [4:8] for layer 1,
                etc for the number of layers.

                These action values are for direct use in the construction of models.

        Returns:
            a reward for training a model with the given model_states
        '''
        train_graph = tf.Graph()
        train_sess = tf.Session(graph=train_graph)
        with train_graph.as_default(), train_sess.as_default():

            if self.tune_data_feeder:
                feeder_size = model_states[0].Layer_attributes['size']
                if self.verbose: print("feeder_size", feeder_size)

            #with tf.device('GPU:%d' % (trial%3)):
            # generate a submodel given predicted model_states by `model_fn`
            #try:
            if self.tune_data_feeder:
                model, feeder =  self.model_fn(model_states, self.input_state, self.output_state, self.model_compile_dict)  # type: Model - compiled
            else:
                model = self.model_fn(model_states, self.input_state, self.output_state, self.model_compile_dict)  # type: Model - compiled

            #except ValueError:  # conv dim failure
            #    return -0.1, -np.inf

            # unpack the dataset
            X_val, y_val = self.validation_data[0:2]

            # model fitting
            if self.tune_data_feeder:  # tuning data generator feeder
                hist = model.fit_generator(feeder,
                            epochs = self.epochs,
                            steps_per_epoch = self.train_data_size//feeder_size,
                            verbose = self.verbose,
                            validation_data = (X_val, y_val),
                            callbacks = [ModelCheckpoint(os.path.join(self.working_dir,'temp_network.h5'),
                                                 monitor='val_loss', verbose=self.verbose,
                                                     save_best_only=True),
                                        EarlyStopping(monitor='val_loss', patience=5, verbose=self.verbose)],
                            use_multiprocessing=True,
                            max_queue_size=8)
            else:
                if type(self.train_data) == tuple or type(self.train_data) == list:
                # is a tuple/list of np array
                    X_train, y_train = self.train_data

                    # train the model using Keras methods
                    print("	Start training model...")
                    hist = model.fit(X_train, y_train,
                            batch_size = self.batchsize,
                            epochs = self.epochs,
                            verbose = self.verbose,
                            validation_data=(X_val, y_val),
                            callbacks=[ModelCheckpoint(os.path.join(self.working_dir,'temp_network.h5'),
                                            monitor='val_loss', verbose=self.verbose,
                                            save_best_only=True),
                                       EarlyStopping(monitor='val_loss', patience=5, verbose=self.verbose)]
                                       )
                else:
                # is a generator
                    hist = model.fit_generator(self.train_data,
                            epochs = self.epochs,
                            steps_per_epoch = self.train_data_size//self.batchsize,
                            verbose = self.verbose,
                            validation_data=(X_val, y_val),
                            callbacks=[
                                    ModelCheckpoint(os.path.join(self.working_dir,'temp_network.h5'),
                                                     monitor='val_loss',
                                                     verbose=self.verbose,
                                                     save_best_only=True),
                                    EarlyStopping(monitor='val_loss', patience=5, verbose=self.verbose)],
                            use_multiprocessing=True,
                            max_queue_size=8)

            # load best performance epoch in this training session
            model.load_weights(os.path.join(self.working_dir,'temp_network.h5'))

            # evaluate the model by `reward_fn`
            this_reward, loss_and_metrics, reward_metrics = self.reward_fn(model, (X_val, y_val))
            loss = loss_and_metrics.pop(0)
            loss_and_metrics = { str(self.model_compile_dict['metrics'][i]):loss_and_metrics[i] for i in range(len(loss_and_metrics))}
            loss_and_metrics['loss'] = loss
            if reward_metrics:
                loss_and_metrics.update(reward_metrics)

            # do any post processing,
            # e.g. save child net, plot training history, plot scattered prediction.
            if self.post_processing_fn:
                val_pred = model.predict(X_val)
                self.post_processing_fn(
                        trial=trial,
                        model=model,
                        hist=hist,
                        data=self.validation_data,
                        pred=val_pred,
                        loss_and_metrics=loss_and_metrics,
                        working_dir=self.working_dir,
                        save_full_model=self.save_full_model
                    )

        # clean up resources and GPU memory
        #network_sess.close()
        del model
        del hist
        gc.collect()
        return this_reward, loss_and_metrics
