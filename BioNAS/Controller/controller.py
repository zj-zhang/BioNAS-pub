# -*- coding: UTF-8 -*-

'''
A Keras-implementation of NAS
Aug. 7, 2018
'''

import os
import keras.backend as K
import numpy as np
import random
from keras.models import Model
from keras.layers import Input, Lambda, Dense
from .actor_critic import build_actor, build_critic
from keras import optimizers
from keras.losses import mean_squared_error, binary_crossentropy, categorical_crossentropy
from keras.metrics import kullback_leibler_divergence


def get_optimizer(name: str, lr:float, decay:float=0.995):
    if name.lower() == 'adam':
        return optimizers.Adam(lr, decay=decay)
    elif name.lower() == 'adagrad':
        return optimizers.Adagrad(lr, decay=decay)
    elif name.lower() == 'rmsprop':
        return optimizers.RMSprop(lr, decay=decay)
    elif name.lower() == 'sgd':
        return optimizers.SGD(lr, decay=decay)
    else:
        raise ValueError('Optimizer not supported')


def one_hot_encoder(val, num_classes, offset=0):
    val = np.array(val, dtype=np.int32)
    val -= offset
    if not (val.shape):
      val = np.expand_dims(val, axis=0)
    assert all(val>=0) and all(val<num_classes)
    tmp = np.zeros((val.shape[0], num_classes))
    for i,j in enumerate(val):
      tmp[i, j] = 1
    return tmp


def parse_action_str(action_onehot, state_space):
    return [state_space[i][int(j)] for i in range(len(state_space)) for j in range(len(action_onehot[i][0])) if action_onehot[i][0][int(j)]==1 ]


def parse_old_action_str(old_prob):
    return


def get_indexer(t):
    return Lambda(lambda x, t: x[:, t, :], arguments={'t':t}, output_shape=lambda s: (s[0], s[2]))


def get_kl_divergence_n_entropy(old_prob, new_prob, old_onehot, new_onehot):
    """
    compute approx kl and entropy
    return kl, ent
    """
    #kl = [ K.log( K.sum(old_oh * old_p, axis=1 )) - K.log( K.sum(old_oh * new_p, axis=1 ) )  for old_p, new_p, old_oh in zip(old_prob, new_prob, old_onehot) ]
    kl = [ kullback_leibler_divergence(old_p, new_p)  for old_p, new_p, old_oh in zip(old_prob, new_prob, old_onehot) ]
    ent = [ binary_crossentropy(new_oh, new_p)  for new_p, new_oh in zip(new_prob, new_onehot) ]
    return K.mean(sum(kl)), K.mean(sum(ent))


def proximal_policy_optimization_loss(curr_prediction, curr_onehotpred, old_prediction, old_onehotpred, rewards, advantage, clip_val, beta):

    rewards = K.squeeze(rewards, axis=1)
    advantage = K.squeeze(advantage, axis=1)

    entropy = 0
    r = 1
    for t, (p, onehot, old_p, old_onehot) in enumerate(zip(curr_prediction, curr_onehotpred, old_prediction, old_onehotpred)):

        r = r * K.exp(K.log(K.sum(old_onehot * p, axis=1)) - K.log(K.sum(old_onehot * old_p, axis=1))) ## prob / old_prob w.r.t old action taken

        entropy += -K.mean( K.log( K.sum(onehot * p, axis=1) ) )  ## approx entropy

    surr_obj = K.mean( K.abs(1/(rewards+1e-8)) * K.minimum(r * advantage, ## some trick for scaling gradients by reward... K.abs(1/(rewards+1e-8))
                             K.clip(r, min_value=1 - clip_val, max_value=1 + clip_val) * advantage))

    return - surr_obj + beta * (- entropy) ## maximize surr_obj for learning and entropy for regularization


def critic_loss(dcreward, value):
    return mean_squared_error(K.squeeze(dcreward, axis=1), K.squeeze(value, axis=1))


class Buffer(object):
    def __init__(self, max_size, gamma):
        self.max_size = max_size
        self.gamma = gamma
        ## short term buffer storing single traj
        self.state_buffer = []
        self.action_buffer = []
        self.prob_buffer = []
        self.reward_buffer = []
        # self.td_buffer = []  ## temporal difference for calculating advantage

        ## long_term buffer
        self.lt_sbuffer = []
        self.lt_abuffer = []
        self.lt_pbuffer = []
        # self.lt_rbuffer = []
        self.lt_adbuffer = []
        self.lt_nrbuffer = [] ## lt buffer for non discounted reward
        self.lt_rmbuffer = [] ## reward mean buffer

    def store(self, state, prob, action, reward):#, value, next_value):
        self.state_buffer.append(state)
        self.prob_buffer.append(prob)
        self.reward_buffer.append(reward)
        # td = reward + self.gamma * next_value - value  ## temporal difference
        # self.td_buffer.append(td)
        self.action_buffer.append(action)

    def discount_rewards(self):
        '''Example behavior of discounted reward, given reward_buffer=[1,2,3]:
        if buffer.gamma=0.1:
            ([array([[1.23]]), array([[2.3]]), array([[3.]])], [1, 2, 3])
        if buffer.gamma=0.9:
            ([array([[5.23]]), array([[4.7]]), array([[3.]])], [1, 2, 3])
        '''
        discounted_rewards = []
        running_add = 0
        for i, r in enumerate(reversed(self.reward_buffer)):
            #if rewards[t] != 0:   # with this condition, a zero reward_t can get a non-zero
            #    running_add = 0   # discounted reward from its predecessors
            running_add = running_add * self.gamma + np.array([r])
            discounted_rewards.append(np.reshape(running_add, (running_add.shape[0], 1)))

        # mean = np.mean(discounted_rewards)
        # std = np.std(discounted_rewards)
        # discounted_rewards = (discounted_rewards - mean) / (std + 1e-8)
        # discounted_rewards = discounted_rewards.tolist()

        discounted_rewards.reverse()
        return discounted_rewards, self.reward_buffer

    # def get_advantage(self):
    #     discounted_advantage = []
    #
    #     running_add = 0
    #     for i, td in enumerate(reversed(self.td_buffer)):
    #         running_add = running_add * self.gamma + np.array([td])
    #         discounted_advantage.append(np.reshape(running_add, (running_add.shape[0], 1)))
    #
    #
    #     if len(discounted_advantage) > 1:
    #         mean = np.mean(discounted_advantage)
    #         std = np.std(discounted_advantage)
    #         discounted_advantage = (discounted_advantage - mean) / (std + 1e-8)
    #         discounted_advantage = discounted_advantage.tolist()
    #
    #     discounted_advantage.reverse()
    #
    #     return discounted_advantage

    def finish_path(self, state_space, global_ep, working_dir):
        # dump buffers to file and return processed buffer
        """
        return processed [state, ob_prob, ob_onehot, reward, advantage]
        """

        dcreward, reward = self.discount_rewards()
        # advantage = self.get_advantage()

        # get data from buffer
        state = np.concatenate( self.state_buffer , axis=0)
        old_prob = [np.concatenate(p, axis=0) for p in zip(*self.prob_buffer)]
        action_onehot = [np.concatenate(onehot, axis=0) for onehot in zip(*self.action_buffer)]
        r = np.concatenate(dcreward, axis=0)
        if not self.lt_rmbuffer:
            self.r_mean = r.mean()
        else:
            self.r_mean = self.r_mean*0.2 + 0.8*self.lt_rmbuffer[-1]  ## THIS EWA NEEDS TUNING

        nr = np.array(reward)[:,np.newaxis]

        ad = r - self.r_mean
        if ad.shape[1] > 1:
           ad = ad / (ad.std() + 1e-8)

        self.lt_sbuffer.append(state)
        self.lt_pbuffer.append(old_prob)
        self.lt_abuffer.append(action_onehot)
        # self.lt_rbuffer.append(r)
        self.lt_adbuffer.append(ad)
        self.lt_nrbuffer.append(nr)

        self.lt_rmbuffer.append(r.mean())

        with open(os.path.join(working_dir, 'buffers.txt'), mode='a+') as f:
            action_onehot = self.action_buffer[-1]
            action_readable_str = ','.join([str(x) for x in parse_action_str(action_onehot, state_space)])
            f.write("Episode:%d\tReward:%0.4f\tAction:%s\tProb:%s\tR_Mean:%s\n" % (
            global_ep, self.reward_buffer[-1], action_readable_str, self.prob_buffer[-1],
            self.lt_rmbuffer[-1]))

            print("Saved buffers to file `buffers.txt` !")

        if len(self.lt_pbuffer) > self.max_size:
            self.lt_sbuffer = self.lt_sbuffer[-self.max_size:]
            self.lt_pbuffer = self.lt_pbuffer[-self.max_size:]
            # self.lt_rbuffer = self.lt_rbuffer[-self.max_size:]
            self.lt_adbuffer = self.lt_adbuffer[-self.max_size:]
            self.lt_abuffer = self.lt_abuffer[-self.max_size:]
            self.lt_nrbuffer = self.lt_nrbuffer[-self.max_size:]
            self.lt_rmbuffer = self.lt_rmbuffer[-self.max_size:]

        self.state_buffer, self.prob_buffer, self.action_buffer, self.reward_buffer = [], [], [], [] #, self.td_buffer= [], [], [], [], []

    def get_data(self, bs, shuffle=True):
        """
        get a batched data
        size of buffer: (traj, traj_len, data_shape)
        """

        lt_sbuffer, lt_pbuffer, lt_abuffer, lt_adbuffer, lt_nrbuffer = self.lt_sbuffer, self.lt_pbuffer, \
                                                                       self.lt_abuffer, self.lt_adbuffer, self.lt_nrbuffer

        lt_sbuffer = np.concatenate(lt_sbuffer, axis=0)
        lt_pbuffer = [np.concatenate(p, axis=0) for p in zip(*lt_pbuffer)]
        lt_abuffer = [np.concatenate(a, axis=0) for a in zip(*lt_abuffer)]
        lt_adbuffer = np.concatenate(lt_adbuffer, axis=0)
        lt_nrbuffer = np.concatenate(lt_nrbuffer, axis=0)

        if shuffle:
            slice = np.random.choice(lt_sbuffer.shape[0], size=lt_sbuffer.shape[0])
            lt_sbuffer = lt_sbuffer[slice]
            lt_pbuffer = [p[slice] for p in lt_pbuffer]
            lt_abuffer = [a[slice] for a in lt_abuffer]
            lt_adbuffer = lt_adbuffer[slice]
            lt_nrbuffer = lt_nrbuffer[slice]

        for i in range(0, len(lt_sbuffer), bs):
            b = min(i+bs, len(lt_sbuffer))
            p_batch = [p[i:b,:] for p in lt_pbuffer]
            a_batch = [a[i:b,:] for a in lt_abuffer]
            yield lt_sbuffer[i:b,:,:], p_batch, a_batch, lt_adbuffer[i:b,:], \
                  lt_nrbuffer[i:b, :]



class Controller(object):
    '''
    example state_space for a 2-layer conv-net:
        state_space = [['conv3', 'conv5', 'conv7'], ['maxp2', 'avgp2'],
            ['conv3', 'conv5', 'conv7'], ['maxp2', 'avgp2']]
    '''
    def __init__(self, 
                 state_space, 
                 controller_units, 
                 embedding_dim=5, 
                 optimizer='rmsprop', 
                 discount_factor=0.9,
                 clip_val=0.2, ## for PPO clipping
                 beta=0.001,  ## for entropy regularization
                 kl_threshold=0.05, ## for early stopping
                 train_pi_iter=100, ## num of substeps for training
                 train_v_iter=80,
                 lr_pi=0.005,
                 lr_v=0.01,
                 buffer_size=50,
                 batch_size=5
                 ):
        self.state_space = state_space
        self.controller_units = controller_units
        self.embedding_dim = embedding_dim
        self.a_optimizer = get_optimizer(optimizer, lr_pi, 0.999)
        self.clip_val = clip_val
        self.beta = beta
        self.kl_threshold = kl_threshold
        self.global_controller_step = 0

        self.buffer = Buffer(buffer_size, discount_factor)


        self.kl_div = 0
        self.train_pi_iter = train_pi_iter
        self.batch_size = batch_size

        state_inputs = self._add_input()
        self._build_model(state_inputs)

    def _add_input(self):
        maxlen = len(self.state_space)
        last_output_dim = len(self.state_space.state_space[maxlen - 1])

        state_inputs = Input((1, last_output_dim), batch_shape=(None, 1, last_output_dim))  # states
        return state_inputs

    def _build_model(self, state_inputs):
        maxlen = len(self.state_space)
        input_dim = self.embedding_dim

        # build actor
        outputs, action = build_actor(state_inputs, self.controller_units, input_dim, maxlen, self.state_space,
                                      scope='actor', trainable=True)

        # placeholders
        maxlen = len(outputs)
        old_onehot_placeholder = [K.placeholder(shape=K.int_shape(outputs[t]),
                                                        name="old_onehot_%i" % t) for t in range(maxlen)]
        old_pred_placeholder = [K.placeholder(shape=K.int_shape(outputs[t]),
                                                        name="old_pred_%i" % t) for t in range(maxlen)]

        # discount_reward_placeholder = K.placeholder(shape=(None,1),
        #                                                  name="discount_reward")
        reward_placeholder = K.placeholder(shape=(None, 1),
                                                    name="normal_reward")

        advantage_placeholder = K.placeholder((None,1), name='advantage')  # advantages

        self.model = Model(inputs=state_inputs, outputs=outputs+action)

        # actor loss
        aloss = proximal_policy_optimization_loss(outputs, action, old_pred_placeholder, old_onehot_placeholder,
                                                  reward_placeholder, advantage_placeholder, self.clip_val, self.beta)

        kl_div, entropy = get_kl_divergence_n_entropy(old_pred_placeholder, outputs, old_onehot_placeholder, action)
        self.kl_div_fn = K.function(inputs=[state_inputs]+old_pred_placeholder+old_onehot_placeholder, outputs=[kl_div, entropy])

        aupdates = self.a_optimizer.get_updates(params=[w for w in self.model.trainable_weights if w.name.startswith('actor')],
                                   loss=aloss)

        self.atrain_fn = K.function(inputs=[state_inputs, reward_placeholder, advantage_placeholder]+
                                          old_pred_placeholder+old_onehot_placeholder,
                                   outputs=[aloss],
                                   updates=aupdates)


    def get_action(self, seed):
        maxlen = len(self.state_space)
        pred = self.model.predict(seed)
        prob = pred[:maxlen]
        onehot_action = pred[-maxlen:]

        return tuple(onehot_action), prob

    def store(self, state, prob, action, reward):

        self.buffer.store( state, prob, action, reward)


    def train(self, episode, working_dir):

        """
        called only when path finishes
        """

        self.buffer.finish_path(self.state_space, episode, working_dir)

        aloss = 0
        g_t = 0

        for epoch in range(self.train_pi_iter):

            t = 0
            kl_sum = 0
            ent_sum = 0
            
            # get data from buffer
            for s_batch, p_batch, a_batch, ad_batch, nr_batch in self.buffer.get_data(self.batch_size):

                feeds = [s_batch, nr_batch, ad_batch] + p_batch + a_batch
                aloss += self.atrain_fn(feeds)[0]

                curr_kl, curr_ent = self.kl_div_fn([s_batch] + p_batch + a_batch)
                kl_sum += curr_kl
                ent_sum += curr_ent

                t += 1
                g_t += 1

                if kl_sum/t > self.kl_threshold:
                    print("     Early stopping at step {} as KL(old || new) = ".format(g_t), kl_sum / t)
                    return aloss / g_t

            if epoch % (self.train_pi_iter//5) == 0:
                print('     Epoch: {} Actor Loss: {} KL(old || new): {} Entropy(new) = {}'.format(epoch, aloss / g_t, kl_sum / t, ent_sum / t))

        return aloss / g_t # + closs / self.train_v_iter

    def remove_files(self, files, working_dir='.'):
        for file in files:
            file = os.path.join(working_dir, file)
            if os.path.exists(file):
                os.remove(file) 
