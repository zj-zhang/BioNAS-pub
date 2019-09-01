import keras.backend as K
from keras.layers import Conv1D, Conv2D, Layer, Permute, Lambda

from keras.constraints import Constraint
from keras.initializers import normal, uniform
from keras.regularizers import l1
from keras.regularizers import Regularizer

class InfoConstraint (Constraint):
    def __call__(self, w):
        w = w * K.cast(K.greater_equal(w, 0), K.floatx())  # force nonnegative
        w = w * (2 / K.maximum(K.sum(w, axis=1, keepdims=True), 2))  # force sum less-than equal to two (bits)
        return w


class NegativeConstraint(Constraint):
    def __call__(self, w):
        w = w * K.cast(K.less_equal(w, 0), K.floatx())  # force negative
        return w

def filter_reg(w, lambda_filter, lambda_l1):
    filter_penalty = lambda_filter * K.sum(K.l2_normalize(K.sum(w, axis=0), axis=1))
    weight_penalty = lambda_l1 * K.sum(K.abs(w))
    return filter_penalty + weight_penalty

def pos_reg(w, lambda_pos, filter_len):
    location_lambda = K.cast(K.concatenate([K.arange(filter_len / 2, stop = 0, step = -1), K.arange(start=1, stop=(filter_len / 2+1))]), 'float32')*(lambda_pos / (filter_len / 2))
    location_penalty = K.sum(location_lambda * K.sum(K.abs(w), axis=(0, 2, 3)))
    return location_penalty

def total_reg(w, lambda_filter, lambda_l1, lambda_pos, filter_len):
    return filter_reg(w, lambda_filter, lambda_l1) + pos_reg(w, lambda_pos, filter_len)

def Layer_deNovo(filters, kernel_size, strides=1, padding='valid', activation='sigmoid', lambda_pos =3e-3,
                 lambda_l1=3e-3, lambda_filter = 1e-8, name='denovo'):
    return Conv2D( filters, (4,kernel_size), strides=strides, padding=padding, activation=activation,
            kernel_initializer=normal(0,0.5), bias_initializer='zeros',
            kernel_regularizer=lambda w: total_reg(w, lambda_filter, lambda_l1, lambda_pos, kernel_size),
            kernel_constraint=InfoConstraint(), bias_constraint=NegativeConstraint(),
            name=name)


####################################################################################

class CurvatureConstraint(Constraint):
    """Constrains the second differences of weights in W_pos.
    # Arguments
        m: the maximum curvature for the incoming weights.
    """
    def __init__(self, m=1.0):
        self.m = float(m)

    def __call__(self, p):
        import numpy as np
        mean_p = K.mean(p, axis=1)
        (num_output, length) = K.int_shape(p)
        diff1 = p[:, 1:] - p[:, :-1]
        mean_diff1 = K.mean(diff1, axis=1)
        diff2 = diff1[:, 1:] - diff1[:, :-1]
        desired_diff2 = K.clip(diff2, -1.0 * self.m, self.m)

        il1 = np.triu_indices(length - 2)
        mask1 = np.ones((num_output, length - 1, length - 2))
        mask1[:, il1[0], il1[1]] = 0.0
        kmask1 = K.variable(value=mask1)
        mat1 = kmask1 * K.repeat_elements(K.expand_dims(desired_diff2, 1), length - 1, 1)
        desired_diff1 = K.squeeze(K.squeeze(
            K.dot(mat1, K.ones((1, length - 2, num_output)))[:, :, :1, :1], axis=2), axis=2)
        desired_diff1 += K.repeat_elements(K.expand_dims(
            mean_diff1 - K.mean(desired_diff1, axis=1), -1), length - 1, axis=1)

        il2 = np.triu_indices(length - 1)
        mask2 = np.ones((num_output, length, length - 1))
        mask2[:, il2[0], il2[1]] = 0.0
        kmask2 = K.variable(value=mask2)
        mat2 = kmask2 * K.repeat_elements(K.expand_dims(desired_diff1, 1), length, 1)
        desired_p = K.squeeze(K.squeeze(
            K.dot(mat2, K.ones((1, length - 1, num_output)))[:, :, :1, :1], axis=2), axis=2)
        desired_p += K.repeat_elements(K.expand_dims(
            mean_p - K.mean(desired_p, axis=1), -1), length, axis=1)

        return desired_p

    def get_config(self):
        return {'name': self.__class__.__name__,
                'm': self.m}


class SepFCSmoothnessRegularizer(Regularizer):
    """ Specific to SeparableFC
        Applies penalty to length-wise differences in W_pos.
    # Arguments
        smoothness: penalty to be applied to difference
            of adjacent weights in the length dimension
        l1: if smoothness penalty is to be computed in terms of the
            the absolute difference, set to True
            if False, penalty is computed in terms of the squared difference
        second_diff: if smoothness penalty is to be applied to the
            difference of the difference, set to True
            if False, penalty is applied to the first difference
    """

    def __init__(self, smoothness, l1=True, second_diff=True):
        self.smoothness = float(smoothness)
        self.l1 = bool(l1)
        self.second_diff = bool(second_diff)

    def __call__(self, x):
        diff1 = x[:, 1:] - x[:, :-1]
        diff2 = diff1[:, 1:] - diff1[:, :-1]
        if self.second_diff is True:
            diff = diff2
        else:
            diff = diff1
        if self.l1 is True:
            return K.mean(K.abs(diff)) * self.smoothness
        else:
            return K.mean(K.square(diff)) * self.smoothness

    def get_config(self):
        return {'name': self.__class__.__name__,
                'smoothness': float(self.smoothness),
                'l1': bool(self.l1),
                'second_diff': bool(self.second_diff)}


class SeparableFC(Layer):
    '''A Fully-Connected layer with a weights tensor that is
       the product of a matrix W_pos, for learning spatial correlations,
       and a matrix W_chan, for learning cross-channel correlations.

    # Arguments
        output_dim: the number of output neurons
        symmetric: if weights are to be symmetric along length, set to True
        smoothness_penalty: penalty to be applied to difference
            of adjacent weights in the length dimensions
        smoothness_l1: if smoothness penalty is to be computed in terms of the
            the absolute difference, set to True
            otherwise, penalty is computed in terms of the squared difference
        smoothness_second_diff: if smoothness penalty is to be applied to the
            difference of the difference, set to True
            otherwise, penalty is applied to the first difference
        curvature_constraint: constraint to be enforced on the second differences
            of the positional weights matrix

    # Input shape
        3D tensor with shape: `(samples, steps, features)`.

    # Output shape
        2D tensor with shape: `(samples, output_features)`.
    '''

    def __init__(self, output_dim, symmetric,
                 smoothness_penalty=None,
                 smoothness_l1=False,
                 smoothness_second_diff=True,
                 curvature_constraint=None, **kwargs):
        super(SeparableFC, self).__init__(**kwargs)
        self.output_dim = output_dim
        self.symmetric = symmetric
        self.smoothness_penalty = smoothness_penalty
        self.smoothness_l1 = smoothness_l1
        self.smoothness_second_diff = smoothness_second_diff
        self.curvature_constraint = curvature_constraint

    def build(self, input_shape):
        import numpy as np
        self.original_length = input_shape[1]
        if (self.symmetric == False):
            self.length = input_shape[1]
        else:
            self.odd_input_length = input_shape[1] % 2.0 == 1
            self.length = int(input_shape[1] / 2.0 + 0.5)
        self.num_channels = input_shape[2]
        self.init = uniform( -np.sqrt(
                np.sqrt(2.0 / (self.length * self.num_channels + self.output_dim))),  np.sqrt(
                np.sqrt(2.0 / (self.length * self.num_channels + self.output_dim))))
        self.W_pos = self.add_weight(
            name='{}_W_pos'.format(self.name),
            shape=(self.output_dim, self.length),
            initializer=self.init,
            constraint=(None if self.curvature_constraint is None else
                        CurvatureConstraint(
                            self.curvature_constraint)),
            regularizer=(None if self.smoothness_penalty is None else
                         SepFCSmoothnessRegularizer(
                             self.smoothness_penalty,
                             self.smoothness_l1,
                             self.smoothness_second_diff)))
        self.W_chan = self.add_weight(
            shape=(self.output_dim, self.num_channels),
            name='{}_W_chan'.format(self.name), initializer=self.init,
            trainable=True)
        self.built = True

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)

    def call(self, x, mask=None):
        if (self.symmetric == False):
            W_pos = self.W_pos
        else:
            W_pos = K.concatenate(
                tensors=[self.W_pos,
                         self.W_pos[:, ::-1][:, (1 if self.odd_input_length else 0):]],
                axis=1)
        W_output = K.expand_dims(W_pos, 2) * K.expand_dims(self.W_chan, 1)
        W_output = K.reshape(W_output,
                             (self.output_dim, self.original_length * self.num_channels))
        x = K.reshape(x,
                      (-1, self.original_length * self.num_channels))
        output = K.dot(x, K.transpose(W_output))
        return output

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'symmetric': self.symmetric,
                  'smoothness_penalty': self.smoothness_penalty,
                  'smoothness_l1': self.smoothness_l1,
                  'smoothness_second_diff': self.smoothness_second_diff,
                  'curvature_constraint': self.curvature_constraint}
        base_config = super(SeparableFC, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 3
        output_shape = [list(input_shape)[0], self.output_dim]
        return tuple(output_shape)
