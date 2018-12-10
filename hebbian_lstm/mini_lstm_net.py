from hebbian_lstm.vanilla_lstm import SimpleLSTMCell



from video_prediction.ops import dense


def make_mini_lstm():




    cell = SimpleLSTMCell(inputs, hparams)
    outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32,
                                   swap_memory=False, time_major=True)