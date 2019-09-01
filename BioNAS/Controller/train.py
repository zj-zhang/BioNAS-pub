# -*- coding: UTF-8 -*-

import numpy as np
import csv
import logging
import datetime
import shutil
import os
import warnings

from keras import backend as K
from keras.utils import to_categorical
from keras.optimizers import *

from .controller import *
from ..utils.plots import plot_stats, plot_environment_entropy, plot_controller_performance, plot_action_weights, plot_stats2
from ..utils.io import save_action_weights, save_stats


def setup_logger(working_dir='.'):
    # setup logger
    logger = logging.getLogger('controller')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(working_dir, 'log.controller.txt'))
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -\n %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def get_controller_states(model):
    return [K.get_value(s) for s, _ in model.state_updates]


def set_controller_states(model, states):
    for (d, _), s in zip(model.state_updates, states):
        K.set_value(d, s)


def get_controller_history(fn='train_history.csv'):
    with open(fn, 'r') as f:
        csvreader = csv.reader(f)
        for row in csvreader:
            trial = row[0]
    return int(trial)


def compute_entropy(prob_states):
	ent = 0
	for prob in prob_states:
		ent += np.sum([-p*np.log2(p) for p in prob])
	return ent

class ControllerTrainEnvironment(object):
    '''ControllerNetEnvironment: employs `controller` model and `manager` to mange data and reward,
    creates a reinforcement learning environment
    '''

    def __init__(self, controller, manager, max_episode, max_step_per_ep, logger, resume_prev_run, should_plot,
             working_dir='.', entropy_converge_epsilon=0.01, verbose=0):
        self.controller = controller
        self.manager = manager
        self.max_episode = max_episode
        self.max_step_per_ep = max_step_per_ep
        self.start_ep = 0
        self.should_plot = should_plot
        self.working_dir = working_dir
        self.total_reward = 0
        self.entropy_record = []
        self.entropy_converge_epsilon = entropy_converge_epsilon

        self.last_actionState_size = len(self.controller.state_space[-1])

        if resume_prev_run:
            self.restore()
        else:
            self.clean()
        self.resume_prev_run = resume_prev_run
        self.logger = logger if logger else setup_logger(working_dir)
        if os.path.realpath(manager.working_dir) != os.path.realpath(self.working_dir):
            warnings.warn("manager working dir and environment working dir are different.")

    def restore(self):
        controller_states = np.load(os.path.join(self.working_dir, 'controller_states.npy'))
        set_controller_states(self.controller.model, controller_states)
        self.controller.model.load_weights(os.path.join(self.working_dir, 'controller_weights.h5'))
        self.start_ep = get_controller_history(os.path.join(self.working_dir, 'train_history.csv'))

    def clean(self):
        bak_weights_dir = os.path.join(self.working_dir,
                                       'weights_bak_%s' % datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        if os.path.isdir(os.path.join(self.working_dir, 'weights')):
            shutil.move(os.path.join(self.working_dir, 'weights'), bak_weights_dir)

        movable_files = [
            'buffers.txt',
            'log.controller.txt',
            'train_history.csv',
            'train_history.png',
            'entropy.png',
            'controller_states.npy',
            'controller_weights.h5',
        ]
        movable_files += [x for x in os.listdir(self.working_dir) if x.startswith("weight_at_layer_") and x.endswith(".png")]
        for file in movable_files:
            file = os.path.join(self.working_dir, file)
            if os.path.exists(file):
                shutil.move(file, bak_weights_dir)
        os.makedirs(os.path.join(self.working_dir, 'weights'))
        self.controller.remove_files(movable_files, self.working_dir)

    def reset(self):
        #x = np.random.normal(size=(1, 1, self.last_actionState_size))
        #return x
        x = np.random.uniform(0, 5, (1, 1, self.last_actionState_size))
        x = np.exp(x) / np.sum(np.exp(x))
        return x

    def step(self, action_prob):
        ## returns a state given an action (prob list)
        return np.array(action_prob[-1]).reshape(1, 1, self.last_actionState_size)

    def train(self):
        '''Performs training for controller
        '''
        LOGGER = self.logger
        #summary_writer = tf.summary.FileWriter(os.path.join(self.working_dir))
        #summary = tf.Summary()

        action_probs_record = []

        loss_and_metrics_list = []
        global_step = self.start_ep * self.max_step_per_ep
        if self.resume_prev_run:
            f = open(os.path.join(self.working_dir, 'train_history.csv'), mode='a+')
        else:
            f = open(os.path.join(self.working_dir, 'train_history.csv'), mode='w')
        writer = csv.writer(f)

        for ep in range(self.start_ep, self.max_episode):

            ## reset env
            state = self.reset()
            ep_reward = 0
            loss_and_metrics_ep = {'knowledge': 0, 'acc': 0, 'loss': 0}

            ep_probs = []

            for step in range(self.max_step_per_ep):
                # value = self.controller.get_value(state)
                actions, probs = self.controller.get_action(state)  # get an action for the previous state
                self.entropy_record.append(compute_entropy(probs))
                next_state = self.step(probs)
                # next_value = self.controller.get_value(next_state)
                ep_probs.append(probs)
                # LOGGER.debug the action probabilities
                action_list = parse_action_str(actions, self.controller.state_space)
                LOGGER.debug("Predicted actions : {}".format([str(x) for x in action_list]))

                # build a model, train and get reward and accuracy from the network manager
                reward, loss_and_metrics = self.manager.get_rewards(
                    global_step, action_list)
                LOGGER.debug("Rewards : " + str(reward) + " Metrics : " + str(loss_and_metrics))

                ep_reward += reward
                for x in loss_and_metrics_ep.keys():
                    loss_and_metrics_ep[x] += loss_and_metrics[x]

                # actions and states are equivalent, save the state and reward
                self.controller.store(state, probs, actions, reward)#, value, next_value)

                # update trial
                global_step += 1
                state = next_state

                # write the results of this trial into a file
                data = [global_step, [loss_and_metrics[x] for x in sorted(loss_and_metrics.keys())],
                        reward]
                data.extend(action_list)
                writer.writerow(data)
                f.flush()
            
            loss_and_metrics_list.append({x:(v / self.max_step_per_ep) for x, v in loss_and_metrics_ep.items()})
            ## average probs over trejactory
            ep_p = [sum(p)/len(p) for p in zip(*ep_probs)]
            action_probs_record.append(ep_p)
            if ep >= self.controller.buffer.max_size-1:
                # train the controller on the saved state and the discounted rewards
                loss = self.controller.train(ep, self.working_dir)
                self.total_reward += np.sum(np.array(self.controller.buffer.lt_adbuffer[-1]).flatten())
                LOGGER.debug("Total reward : " + str(self.total_reward))
                LOGGER.debug("END episode %d: Controller loss : %0.6f" % (ep, loss))
                LOGGER.debug("-" * 10)
            else:
                LOGGER.debug("END episode %d: Buffering" % (ep))
                LOGGER.debug("-" * 10)
                #self.controller.buffer.finish_path(self.controller.state_space, ep, self.working_dir)

            #for x in sorted(loss_and_metrics_ep.keys()):
            #    summary.value.add(tag=x, simple_value=loss_and_metrics_ep[x] / self.max_step_per_ep)
            #    summary_writer.add_summary(summary, global_step)

            #summary_writer.flush()

            # save the controller states and weights
            np.save(os.path.join(self.working_dir, 'controller_states.npy'),
                    get_controller_states(self.controller.model))
            self.controller.model.save_weights(os.path.join(self.working_dir, 'controller_weights.h5'))

            # check the entropy record and stop training if no progress was made (less than entropy_converge_epsilon)
            #if ep >= self.max_episode//3 and np.std(self.entropy_record[-(self.max_step_per_ep):])<self.entropy_converge_epsilon:
            #    LOGGER.info("Controller converged at episode %i"%ep)
            #    break

        LOGGER.debug("Total Reward : %s" % self.total_reward)

        f.close()
        plot_controller_performance(os.path.join(self.working_dir, 'train_history.csv'),
            metrics_dict={k: v for k, v in zip(sorted(loss_and_metrics.keys()), range(len(loss_and_metrics)))},
            save_fn=os.path.join(self.working_dir, 'train_history.png'), N_sma=10)
        plot_environment_entropy(self.entropy_record,
            os.path.join(self.working_dir, 'entropy.png'))



        save_action_weights(action_probs_record, self.controller.state_space, self.working_dir)
        save_stats(loss_and_metrics_list, self.working_dir)

        if self.should_plot:
            plot_action_weights(self.working_dir)
            plot_stats2(self.working_dir)

        ## return converged config idx
        act_idx = []
        for p in ep_p:
                act_idx.append(np.argmax(p))
        return act_idx
