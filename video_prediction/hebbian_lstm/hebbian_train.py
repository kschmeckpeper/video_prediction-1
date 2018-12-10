import os
import pickle as pkl
import numpy as np
import tensorflow as tf
import imp
import sys
import pickle
import pdb
import matplotlib.pyplot as plt

import imp
from tensorflow.python.platform import app
from tensorflow.python.platform import flags
from video_prediction.hebbian_lstm.mini_lstm_net import make_mini_lstm


from datetime import datetime
import collections
# How often to record tensorboard summaries.
SUMMARY_INTERVAL = 100

# How often to run a batch through the validation model.
VAL_INTERVAL = 500

# How often to save a model checkpoint
SAVE_INTERVAL = 4000

FLAGS_ = flags.FLAGS
flags.DEFINE_string('hyper', '', 'hyperparameters configuration file')

T = 30
NDATA= 10000

def create_sine_data():


    inputs = np.zeros([NDATA, T])
    targets = np.zeros([NDATA, T])

    for i in range(NDATA):
        amp = np.random.random() * 10 - 5

        ph = 1 * np.random.random() * np.pi/2 + 1
        xx = 1 * np.random.random() * 4 - 2
        yy = 0 # 1 * np.random.random() * 4 - 2
        # ph = 0

        X = np.random.random() * 10 + 10
        inputs[i,:] = np.arange(0, T) / T * X - (X/2)

        targets[i, :] = amp * np.sin(ph*inputs[i, :] + xx) + yy

        # plt.figure()
        # plt.plot(inputs[i], targets[i])
        # plt.show()

    dict = {'inputs':inputs, 'targets': targets}
    pickle.dump(dict, open('toy_data/sine.pkl', 'wb'))


def main():

    batch_size = 32
    inp_dim = 1
    out_dim = 1

    learning_rate = 1e-3

    inputs_pl = tf.placeholder(tf.float32, [batch_size, T, inp_dim])
    gtruth_pl = tf.placeholder(tf.float32, [batch_size, T, out_dim])
    outputs = make_mini_lstm(inputs_pl)

    loss = tf.reduce_mean(tf.square(gtruth_pl - outputs))
    g_optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = g_optimizer.minimize(loss)

    num_iter = 10000


    ### load data
    dict = pickle.load(open('toy_data/sine.pkl', "rb"))


    sess = tf.Session()

    for itr in range(num_iter):
        # Generate new batch of data_files.

        ind = np.random.choice(np.arange(NDATA), batch_size)
        feed_dict = {inputs_pl:dict['inputs'][ind], gtruth_pl:dict['targets'][ind]}
        cost, _, summary_str = sess.run([loss, train_op], feed_dict)

        if (itr) % 10 ==0:
            tf.logging.info(str(itr) + ' ' + str(cost))

        # if (itr) % VAL_INTERVAL == 0:
        #     # Run through validation set.
        #     feed_dict = {model.iter_num: np.float32(itr),
        #                  model.train_cond: 0}
        #     [val_summary_str] = sess.run([model.val_summ_op], feed_dict)
        #     summary_writer.add_summary(val_summary_str, itr)

        # if (itr) % SAVE_INTERVAL == 0 and itr != 0:
        #     tf.logging.info('Saving model to' + conf['output_dir'])
        #     saving_saver.save(sess, conf['output_dir'] + '/model' + str(itr))

        # if (itr) % SUMMARY_INTERVAL == 2:
        #     summary_writer.add_summary(summary_str, itr)

if __name__ == '__main__':
    # create_sine_data()
    main()
