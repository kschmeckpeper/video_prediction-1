from video_prediction.hebbian_lstm.vanilla_lstm import SimpleLSTMCell
import tensorflow as tf

from video_prediction.ops import dense
from video_prediction.hebbian_lstm.vanilla_lstm import SimpleLSTMCell



class CustomCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, inputs, reuse=None):
        super(CustomCell, self).__init__(_reuse=reuse)

        self._output_size = tf.TensorShape(100)
        self._state_size = [tf.TensorShape(100), tf.TensorShape(100)]

    def zero_state(self, batch_size, dtype):
        init_state = super(CustomCell, self).zero_state(batch_size, dtype)
        return init_state

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    def call(self, inputs, state):

        ndense = 100
        with tf.variable_scope('h1'):
            h1= dense(inputs, ndense)

        simplelstmcell = SimpleLSTMCell(ndense, 1)

        with tf.variable_scope('h2'):
            h2, state = simplelstmcell(h1, state)

        return dense(h2, 1), state


def make_mini_lstm(inputs, hparams=None):

    cell = CustomCell(inputs, hparams)
    outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32,
                                   swap_memory=False, time_major=True)
    return outputs