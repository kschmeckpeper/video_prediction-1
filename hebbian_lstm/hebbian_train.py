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

def create_sine_data():

    ntraj= 10000

    inputs = np.zeros([ntraj, T])
    targets = np.zeros([ntraj, T])

    for i in range(ntraj):
        amp = np.random.random() * 10 - 5

        ph = 1 * np.random.random() * np.pi/2 + 1
        xx = 1 * np.random.random() * 4 - 2
        yy = 0 # 1 * np.random.random() * 4 - 2
        # ph = 0

        X = np.random.random() * 10 + 10
        inputs[i,:] = np.arange(0, T) / T * X - (X/2)

        targets[i, :] = amp * np.sin(ph*inputs + xx) + yy

        # plt.figure()
        # plt.plot(inputs[i], targets[i])
        # plt.show()

    dict = {'inputs':inputs, 'targets': targets}
    pickle.dump(dict, open('toydata/sine.pkl', 'wb'))


def main(unused_argv, conf_dict= None, flags=None):

    num_iter = 10000
    sess = tf.Session()

    for itr in range(num_iter):
        t_startiter = datetime.now()
        # Generate new batch of data_files.

        cost, _, summary_str = sess.run([model.loss, model.train_op, model.train_summ_op],
                                        feed_dict)

        if (itr) % 10 ==0:
            tf.logging.info(str(itr) + ' ' + str(cost))

        if (itr) % VAL_INTERVAL == 0:
            # Run through validation set.
            feed_dict = {model.iter_num: np.float32(itr),
                         model.train_cond: 0}
            [val_summary_str] = sess.run([model.val_summ_op], feed_dict)
            summary_writer.add_summary(val_summary_str, itr)

        # if (itr) % SAVE_INTERVAL == 0 and itr != 0:
        #     tf.logging.info('Saving model to' + conf['output_dir'])
        #     saving_saver.save(sess, conf['output_dir'] + '/model' + str(itr))

        if (itr) % SUMMARY_INTERVAL == 2:
            summary_writer.add_summary(summary_str, itr)

if __name__ == '__main__':
    create_sine_data()
    # main()
