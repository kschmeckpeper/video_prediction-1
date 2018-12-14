import tensorflow as tf
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs

from video_prediction.ops import dense

import pdb
from video_prediction.ops import conv_pool2d, upsample_conv2d

class HebbConv2DGRUCell(tf.nn.rnn_cell.RNNCell):
    """2D Convolutional GRU cell with (optional) normalization.

    Modified from these:
    https://github.com/carlthome/tensorflow-convlstm-cell/blob/master/cell.py
    https://github.com/tensorflow/tensorflow/blob/r1.4/tensorflow/python/ops/rnn_cell_impl.py
    """
    def __init__(self, input_shape, filters, kernel_size,
                 activation_fn=tf.tanh,
                 normalizer_fn=None, separate_norms=True,
                 bias_initializer=None, reuse=None):
        super(HebbConv2DGRUCell, self).__init__(_reuse=reuse)
        self._input_shape = input_shape
        self._filters = filters
        self._kernel_size = list(kernel_size) if isinstance(kernel_size, (tuple, list)) else [kernel_size] * 2
        self._activation_fn = activation_fn
        self._normalizer_fn = normalizer_fn
        self._separate_norms = separate_norms
        self._bias_initializer = bias_initializer
        output_size = self._input_shape[:-1] + [self._filters]

        self.hebb_pre_flat_size = None
        size_hebb_state = output_size[0]//2*output_size[1]//2*output_size[2]//8
        self._state_size = [tf.TensorShape(output_size), tf.TensorShape([size_hebb_state, size_hebb_state])]
        print('GRU state size', self._state_size)

    @property
    def state_size(self):
        return self._size

    @property
    def output_size(self):
        return self._size

    def _norm(self, inputs, scope, bias_initializer):
        shape = inputs.get_shape()[-1:]
        gamma_init = init_ops.ones_initializer()
        beta_init = bias_initializer
        with vs.variable_scope(scope):
            # Initialize beta and gamma for use by normalizer.
            vs.get_variable("gamma", shape=shape, initializer=gamma_init)
            vs.get_variable("beta", shape=shape, initializer=beta_init)
        normalized = self._normalizer_fn(inputs, reuse=True, scope=scope)
        return normalized

    def _conv2d(self, inputs, output_filters, bias_initializer=tf.zeros_initializer, mode=None):
        input_shape = inputs.get_shape().as_list()
        kernel_shape = list(self._kernel_size) + [input_shape[-1], output_filters]
        kernel = vs.get_variable("kernel", kernel_shape, dtype=dtypes.float32,
                                 initializer=init_ops.truncated_normal_initializer(stddev=0.02))

        if mode == None:
            outputs = nn_ops.conv2d(inputs, kernel, [1] * 4, padding='SAME')
        elif mode =='downsmp':
            outputs = conv_pool2d(inputs, output_filters, self._kernel_size, strides = [2,2])
        elif mode =='upsmp':
            outputs = upsample_conv2d(inputs, output_filters, self._kernel_size, strides = [2,2])
        else:
            raise NotImplementedError

        if not self._normalizer_fn:
            bias = vs.get_variable('bias', [output_filters], dtype=dtypes.float32,
                                   initializer=bias_initializer)
            pdb.set_trace()
            outputs = nn_ops.bias_add(outputs, bias)
        return outputs

    def call(self, inputs, state_hebb):
        state, hebb = state_hebb

        bias_ones = self._bias_initializer
        if self._bias_initializer is None:
            bias_ones = init_ops.ones_initializer()
        with vs.variable_scope('gates'):
            inputs = array_ops.concat([inputs, state], axis=-1)
            concat = self._conv2d(inputs, 2 * self._filters, bias_ones)
            if self._normalizer_fn and not self._separate_norms:
                concat = self._norm(concat, "reset_update", bias_ones)
            r_, u_ = array_ops.split(concat, 2, axis=-1)
            if self._normalizer_fn and self._separate_norms:
                r_ = self._norm(r_, "reset", bias_ones)
                u_ = self._norm(u_, "update", bias_ones)
            r_, u_ = math_ops.sigmoid(r_), math_ops.sigmoid(u_)

        bias_zeros = self._bias_initializer
        if self._bias_initializer is None:
            bias_zeros = init_ops.zeros_initializer()
        with vs.variable_scope('candidate'):

            # r_state = tf.expand_dims(r_ * state, 1)
            r_state = r_ * state
            hebb_prime = self.Enc(r_state)

            hebb_shape = hebb.get_shape()
            alpha = tf.get_variable('alpha', [hebb_shape[1], hebb_shape[1]], dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))

            hebb_prime_tp1 = tf.matmul(hebb_prime[:,None], alpha[None]*hebb)
            hebb_prime_tp1 = tf.squeeze(hebb_prime_tp1)

            hebb_rs = self.Dec(hebb_prime_tp1)

            inputs_r_state = array_ops.concat([inputs, r_state], axis=-1)

            candidate = self._conv2d(inputs_r_state, self._filters, bias_zeros) + hebb_rs

            if self._normalizer_fn:
                candidate = self._norm(candidate, "state", bias_zeros)

        c_ = self._activation_fn(candidate)
        new_h = u_ * state + (1 - u_) * c_

        # hebbian update according to backpropamine
        eta = tf.get_variable('eta',  [hebb_shape[1]], dtype=tf.float32, initializer=tf.constant_initializer(value=0.01))

        new_hebb = (1-eta)*hebb + eta*tf.matmul(hebb_prime[:,:,None], hebb_prime_tp1[:,None])

        new_state = [new_h, new_hebb]
        return new_h, new_state


    def Dec(self, state):
        """
        :param target_shape :
        :param state : 1-dim compressed state tensor
        :return  3-dim state tensor
        """
        with vs.variable_scope('dec'):
            state = tf.reshape(state, self.hebb_pre_flat_size)
            with vs.variable_scope('conv1'):
                state = self._conv2d(state, state.get_shape().as_list()[-1]*4)
            with vs.variable_scope('conv2'):
                state = self._conv2d(state, state.get_shape().as_list()[-1]*2, mode='upsmp')
            return state

    def Enc(self, _state):
        """
        :param state:  3-dim state tensor
        :return: 1-dim compressed state tensor
        """
        with vs.variable_scope('enc'):
            with vs.variable_scope('conv1'):
                state = self._conv2d(_state, _state.get_shape().as_list()[-1]//2, mode='downsmp')
            with vs.variable_scope('conv2'):
                state = self._conv2d(state, state.get_shape().as_list()[-1]//4)

            self.hebb_pre_flat_size = state.get_shape().as_list()
            state = tf.reshape(state, [state.get_shape().as_list()[0], -1])
            return state
