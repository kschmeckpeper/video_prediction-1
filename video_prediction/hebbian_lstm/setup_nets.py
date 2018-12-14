from video_prediction.hebbian_lstm.vanilla_lstm import SimpleLSTMCell
import tensorflow as tf
import pdb
from video_prediction.ops import dense
from video_prediction.hebbian_lstm.vanilla_lstm import SimpleLSTMCell


from video_prediction.hebbian_lstm.vanilla_gru import SimpleGRUCell
from video_prediction.hebbian_lstm.hebbian_gru import HebbianGRUCell


class CustomLSTMwrapCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, batch_size, inputs, context_frames):
        super(CustomLSTMwrapCell, self).__init__()

        self.inputs = inputs
        self._output_size = tf.TensorShape(1)

        self.nlstm = nlstm = 100
        self._state_size  = {'t': tf.TensorShape([]),
                              'gen_outputs': tf.TensorShape(1),
                              'lstm_states': tf.nn.rnn_cell.LSTMStateTuple(tf.TensorShape(nlstm), tf.TensorShape(nlstm))}

        ## scheduled sampling
        # ground_truth_sampling_shape = [self.hparams.sequence_length - 1 - self.hparams.context_frames, batch_size]
        #
        # k = nlstm0
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
        init_state = super(CustomLSTMwrapCell, self).zero_state(batch_size, dtype)
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

        with tf.variable_scope('h1'):
            h1 = dense(tf.concat([x_vals, y_vals], axis=1), self.nlstm)

        simplelstmcell = SimpleLSTMCell(self.nlstm, self.nlstm)

        with tf.variable_scope('h2'):
            h2, new_lstm_state = simplelstmcell(h1, lstm_states)

        gen_outputs = dense(h2, 1,)
        new_states = {'t': time + 1,
                      'gen_outputs': gen_outputs,
                      'lstm_states': new_lstm_state}

        return gen_outputs, new_states



class CustomGRUwrapCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, batch_size, inputs, context_frames, hebb):
        super(CustomGRUwrapCell, self).__init__()

        self.inputs = inputs
        self._output_size = tf.TensorShape(1)

        self.nlstm = nlstm = 10 # 100
        self._state_size  = {'t': tf.TensorShape([]),
                              'gen_outputs': tf.TensorShape(1),
                              }   # size lstmstates, size hebbian

        self.use_hebb = hebb

        if self.use_hebb:
            rnn_state_size = [tf.TensorShape(nlstm), tf.TensorShape([nlstm, nlstm])]
        else:
            rnn_state_size = tf.TensorShape(nlstm)
        self._state_size['lstm_states'] = rnn_state_size

        self.ground_truth = tf.concat([tf.constant(True, dtype=tf.bool, shape=[context_frames, batch_size]),
                                       tf.constant(False, dtype=tf.bool, shape=[context_frames, batch_size])], axis=0)

    def zero_state(self, batch_size, dtype):
        init_state = super(CustomGRUwrapCell, self).zero_state(batch_size, dtype)
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

        with tf.variable_scope('h1'):
            h1 = dense(tf.concat([x_vals, y_vals], axis=1), self.nlstm)


        if self.use_hebb:
            grucell = HebbianGRUCell(self.nlstm, self.nlstm)
        else:
            grucell = SimpleGRUCell(self.nlstm, self.nlstm)

        with tf.variable_scope('h2'):
            h2, new_lstm_state = grucell(h1, lstm_states)

        gen_outputs = dense(h2, 1,)
        new_states = {'t': time + 1,
                      'gen_outputs': gen_outputs,
                      'lstm_states': new_lstm_state}

        return gen_outputs, new_states


def make_mini_lstm(x_vals, y_gtruth, batch_size, T, args):

    if args.expname == 'lstm':
        cell = CustomLSTMwrapCell(batch_size, x_vals, T // 2)
    elif args.expname == 'gru':
        cell = CustomGRUwrapCell(batch_size, x_vals, T // 2, hebb=False)
    elif args.expname == 'gru_hebb':
        cell = CustomGRUwrapCell(batch_size, x_vals, T // 2, hebb=True)
    else:
        raise NotImplementedError

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
