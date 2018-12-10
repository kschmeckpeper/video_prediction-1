from video_prediction.hebbian_lstm.vanilla_lstm import SimpleLSTMCell
import tensorflow as tf
import pdb
from video_prediction.ops import dense
from video_prediction.hebbian_lstm.vanilla_lstm import SimpleLSTMCell



class CustomCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, batch_size, inputs, context_frames, reuse=None):
        super(CustomCell, self).__init__(_reuse=reuse)

        self.inputs = inputs
        self._output_size = tf.TensorShape(1)

        self._state_size = tf.nn.rnn_cell.LSTMStateTuple(tf.TensorShape(100), tf.TensorShape(100))
        self._state_size  = {'t': tf.TensorShape([]),
                              'gen_outputs': tf.TensorShape(1),
                              'lstm_states': tf.nn.rnn_cell.LSTMStateTuple(tf.TensorShape(100), tf.TensorShape(100))}

        ## scheduled sampling
        # ground_truth_sampling_shape = [self.hparams.sequence_length - 1 - self.hparams.context_frames, batch_size]
        #
        # k = 1000
        # iter_num = tf.to_float(tf.train.get_or_create_global_step())
        # prob = (k / (k + tf.exp((iter_num) / k)))
        # prob = tf.cond(tf.less(iter_num, 0), lambda: 1.0, lambda: prob)
        #
        # self.log_probs = tf.log([1 - prob, prob])
        # ground_truth_sampling = tf.multinomial([self.log_probs] * batch_size, ground_truth_sampling_shape[0])
        # ground_truth_sampling = tf.cast(tf.transpose(ground_truth_sampling, [1, 0]), dtype=tf.bool)
        # # Ensure that eventually, the model is deterministically
        # # autoregressive (as opposed to autoregressive with very high probability).
        # ground_truth_sampling = tf.cond(tf.less(prob, 0.001),
        #                                 lambda: tf.constant(False, dtype=tf.bool, shape=ground_truth_sampling_shape),
        #                                 lambda: ground_truth_sampling)
        # ground_truth_context = tf.constant(True, dtype=tf.bool, shape=[context_frames, batch_size])

        self.ground_truth = tf.concat([tf.constant(True, dtype=tf.bool, shape=[context_frames, batch_size]),
                                       tf.constant(False, dtype=tf.bool, shape=[context_frames, batch_size])], axis=0)

    def zero_state(self, batch_size, dtype):
        init_state = super(CustomCell, self).zero_state(batch_size, dtype)
        return init_state

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    def call(self, in_dict, state):

        x_vals = in_dict['x_vals']
        y_gtruth = in_dict['y_gtruth']

        gen_outputs = state['gen_outputs']
        time = state['t']
        lstm_states = state['lstm_states']

        t = tf.to_int32(time[0])
        y_vals = tf.where(self.ground_truth[t], y_gtruth, gen_outputs)  # schedule sampling (if any)

        ndense = 100
        with tf.variable_scope('h1'):
            h1 = dense(tf.concat([x_vals, y_vals], axis=1), ndense)

        simplelstmcell = SimpleLSTMCell(ndense, ndense)

        with tf.variable_scope('h2'):
            h2, new_lstm_state = simplelstmcell(h1, lstm_states)

        gen_outputs = dense(h2, 1,)

        new_states = {'t': time + 1,
                      'gen_outputs': gen_outputs,
                      'lstm_states': new_lstm_state}

        return gen_outputs, new_states



class HebbianCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, batch_size, inputs, context_frames, reuse=None):
        super(HebbianCell, self).__init__(_reuse=reuse)

        self.inputs = inputs
        self._output_size = tf.TensorShape(1)

        self._state_size = tf.nn.rnn_cell.LSTMStateTuple(tf.TensorShape(100), tf.TensorShape(100))
        self._state_size  = {'t': tf.TensorShape([]),
                             'gen_outputs': tf.TensorShape(1),
                             'hebb': tf.TensorShape(100),
                             'lstm_states': tf.nn.rnn_cell.LSTMStateTuple(tf.TensorShape(100), tf.TensorShape(100))}

        self.ground_truth = tf.concat([tf.constant(True, dtype=tf.bool, shape=[context_frames, batch_size]),
                                       tf.constant(False, dtype=tf.bool, shape=[context_frames, batch_size])], axis=0)

    def zero_state(self, batch_size, dtype):
        init_state = super(HebbianCell, self).zero_state(batch_size, dtype)
        return init_state

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    def call(self, in_dict, state):

        x_vals = in_dict['x_vals']
        y_gtruth = in_dict['y_gtruth']

        gen_outputs = state['gen_outputs']
        time = state['t']
        lstm_states = state['lstm_states']
        hebb = state['hebb']

        t = tf.to_int32(time[0])
        y_vals = tf.where(self.ground_truth[t], y_gtruth, gen_outputs)  # schedule sampling (if any)

        ndense = 100
        with tf.variable_scope('h1'):
            h1 = dense(tf.concat([x_vals, y_vals], axis=1), ndense)

        simplelstmcell = SimpleLSTMCell(ndense, ndense)

        with tf.variable_scope('h2'):
            h2, new_lstm_state = simplelstmcell(h1, lstm_states)

        gen_outputs = dense(h2, 1,)

        new_states = {'t': time + 1,
                      'gen_outputs': gen_outputs,
                      'lstm_states': new_lstm_state}

        return gen_outputs, new_states





def make_mini_lstm(x_vals, y_gtruth, batch_size, T):

    cell = CustomCell(batch_size, x_vals, T // 2)
    x_vals = tf.transpose(x_vals, [1, 0, 2])
    y_gtruth = tf.transpose(y_gtruth, [1, 0, 2])

    in_dict = {'x_vals': x_vals, 'y_gtruth':y_gtruth}

    outputs, _ = tf.nn.dynamic_rnn(cell, in_dict, dtype=tf.float32,
                                   swap_memory=False, time_major=True)
    outputs = tf.transpose(outputs, [1,0,2])
    return outputs

# def make_mini_lstm(inputs, hparams=None):
#
#     simplelstmcell = SimpleLSTMCell(1, 1)
#
#     inputs = tf.transpose(inputs, [1,0,2])
#     outputs, out1 = tf.nn.dynamic_rnn(simplelstmcell, inputs, dtype=tf.float32,
#                                    swap_memory=False, time_major=True)
#     outputs = tf.transpose(outputs, [1,0,2])
#     return outputs
