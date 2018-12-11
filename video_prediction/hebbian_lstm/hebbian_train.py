import os
import pickle as pkl
import numpy as np
import tensorflow as tf
import argparse
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
NTRAIN = 10000
NVAL = 1000

def create_sine_data():


    inputs = np.zeros([NTRAIN + NVAL, T])
    targets = np.zeros([NTRAIN + NVAL, T])

    for i in range(NTRAIN + NVAL):
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

    dict = {'inputs':inputs[:NTRAIN], 'targets': targets[:NTRAIN]}
    pickle.dump(dict, open('toy_data/sine_train.pkl', 'wb'))

    dict = {'inputs':inputs[NTRAIN:], 'targets': targets[NTRAIN:]}
    pickle.dump(dict, open('toy_data/sine_val.pkl', 'wb'))


def main():
    parser = argparse.ArgumentParser(description='Run benchmarks')
    parser.add_argument('savedir', type=str)
    args = parser.parse_args()


    batch_size = 32
    inp_dim = 1
    out_dim = 1

    xvals_pl = tf.placeholder(tf.float32, [batch_size, T, inp_dim])
    yvals_gtruth_pl = tf.placeholder(tf.float32, [batch_size, T, out_dim])
    outputs = make_mini_lstm(xvals_pl, yvals_gtruth_pl, batch_size, T)

    loss = tf.reduce_mean(tf.square(yvals_gtruth_pl - outputs))

    trainsum = tf.summary.scalar('trainloss', loss)
    valsum = tf.summary.scalar('valloss', loss)

    learning_rate = 1e-3
    g_optimizer = tf.train.AdamOptimizer(learning_rate)
    train_op = g_optimizer.minimize(loss)

    # g_gradvars = g_optimizer.compute_gradients(loss, var_list=vars)
    # train_op = g_optimizer.apply_gradients(g_gradvars)  # also increments global_step

    ### load data
    train_dict = pickle.load(open('toy_data/sine_train.pkl', "rb"))
    val_dict = pickle.load(open('toy_data/sine_val.pkl', "rb"))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    summary_writer = tf.summary.FileWriter('plots/', graph=sess.graph, flush_secs=10)

    num_iter = 10000
    for itr in range(num_iter):
        # Generate new batch of data_files.

        ind = np.random.choice(np.arange(NTRAIN), batch_size)
        feed_dict = {xvals_pl:train_dict['inputs'][ind][...,None], yvals_gtruth_pl:train_dict['targets'][ind][...,None]}
        cost, _, train_summ_str = sess.run([loss, train_op, trainsum], feed_dict)

        if (itr) % 10 ==0:
            print(str(itr) + ' ' + str(cost))
            summary_writer.add_summary(train_summ_str, itr)

        if (itr) % 100 ==0:
            ind = np.random.choice(np.arange(NVAL), batch_size)
            feed_dict = {xvals_pl:val_dict['inputs'][ind][...,None], yvals_gtruth_pl:val_dict['targets'][ind][...,None]}
            cost, outputs_vals, val_summ_str = sess.run([loss, outputs, valsum], feed_dict)
            plot(itr, outputs_vals, val_dict['inputs'][ind], val_dict['targets'][ind], save_dir=args.savedir)

            summary_writer.add_summary(val_summ_str, itr)

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


def plot(itr, outputs, inputs, gtruth, save_dir):

    outputs = outputs.squeeze()
    inputs = inputs.squeeze()
    gtruth = gtruth.squeeze()

    for i in range(3):
        plt.figure()
        plt.plot(inputs[i], outputs[i])
        plt.plot(inputs[i], gtruth[i])
        plt.savefig(save_dir + '/sine_itr{}_ex{}.png'.format(itr, i))
        plt.close()

if __name__ == '__main__':
    create_sine_data()
    main()
