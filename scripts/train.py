from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import argparse
import errno
import itertools
import json
import math
import os
import random
import time
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

from video_prediction import datasets, models
from video_prediction.utils import ffmpeg_gif, tf_utils


def main():
    program_start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", "--input_dir", type=str, nargs='+',
                        help="either a directory containing subdirectories train, val, test, "
                             "etc, or a directory containing the tfrecords")
    parser.add_argument("--val_input_dirs", type=str, nargs='+', help="directories containing the tfrecords. default: [input_dir]")
    parser.add_argument("--train_input_dirs", type=str, nargs='+', help='directories containing the training datasets')


    parser.add_argument("--logs_dir", default='', help="ignored if output_dir is specified")
    parser.add_argument("--output_dir", help="output directory where json files, summary, model, gifs, etc are saved. "
                                             "default is logs_dir/model_fname, where model_fname consists of "
                                             "information from model and model_hparams")
    parser.add_argument("--checkpoint", help="directory with checkpoint or checkpoint name (e.g. checkpoint_dir/model-200000)")
    parser.add_argument("--no_resume", action='store_false', dest='resume', help='resume from lastest checkpoint in output_dir.')

    parser.add_argument("--conf", type=str, default='', help="folder with all config files")

    parser.add_argument("--dataset", type=str, nargs='+', help="dataset class name")
    parser.add_argument("--train_datasets", type=str, nargs='+', default=None, help="list of train dataset class names")
    parser.add_argument("--val_datasets", type=str, nargs='+', default=None, help="list of val dataset class names")

    parser.add_argument("--dataset_hparams", type=str, nargs='+', help="a string of comma separated list of dataset hyperparameters")
    parser.add_argument("--train_dataset_hparams", type=str, nargs='+', help="a string of comma seperated list of train dataset hyperparameters")
    parser.add_argument("--val_dataset_hparams", type=str, nargs='+', help="a string of comma seperated list of val dataset hyperparameters")

    parser.add_argument("--dataset_hparams_dict", type=str, nargs='+', help="a json file of dataset hyperparameters")
    parser.add_argument("--train_dataset_hparams_files", type=str, nargs='+', help='a list of json files for the train dataset hyperparameters')
    parser.add_argument("--val_dataset_hparams_files", type=str, nargs="+", help="a list of json files for the val dataset hyperparameters")

    parser.add_argument("--model", type=str, help="model class name")
    parser.add_argument("--model_hparams", type=str, help="a string of comma separated list of model hyperparameters")
    parser.add_argument("--model_hparams_dict", type=str, help="a json file of model hyperparameters")

    parser.add_argument("--train_batch_sizes", type=int, nargs='+', help="splits for the training datasets")

    parser.add_argument("--summary_freq", type=int, default=1000, help="save summaries (except for image and eval summaries) every summary_freq steps")
    parser.add_argument("--image_summary_freq", type=int, default=5000, help="save image summaries every image_summary_freq steps")
    parser.add_argument("--eval_summary_freq", type=int, default=0, help="save eval summaries every eval_summary_freq steps")
    parser.add_argument("--progress_freq", type=int, default=100, help="display progress every progress_freq steps")
    parser.add_argument("--metrics_freq", type=int, default=0, help="run and display metrics every metrics_freq step")
    parser.add_argument("--gif_freq", type=int, default=0, help="save gifs of predicted frames every gif_freq steps")
    parser.add_argument("--save_freq", type=int, default=5000, help="save model every save_freq steps, 0 to disable")

    parser.add_argument("--gpu_mem_frac", type=float, default=0, help="fraction of gpu memory to use")
    parser.add_argument("--seed", type=int)


    parser.add_argument("--timing_file", type=str, help="")

    parser.add_argument("--run_time", type=int, default=3600)
    
    args = parser.parse_args()
    if args.train_datasets is not None:
        assert len(args.train_datasets) == len(args.train_dataset_hparams_files) and len(args.train_datasets) == len(args.train_dataset_hparams)
        assert len(args.val_datasets) == len(args.val_dataset_hparams_files) and len(args.val_datasets) == len(args.val_dataset_hparams)
    elif len(args.dataset) == 1:
        args.dataset = args.dataset[0]
        if args.dataset_hparams is not None:
            args.dataset_hparams = args.dataset_hparams[0]
        if args.dataset_hparams_dict is not None:
            args.dataset_hparams_dict = args.dataset_hparams_dict[0]
    else:
        assert len(args.dataset) == len(args.dataset_hparams_dict) and len(args.dataset) == len(args.dataset_hparams_dict)

    logsdir = args.logs_dir

    if args.seed is not None:
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    if args.conf != '' and not isinstance(args.dataset, list) and args.train_datasets is None:
        dataset_hparams_file = args.conf + '/dataset_hparams.json'
        model_hparams_file = args.conf + '/model_hparams.json'
    elif args.conf != '':
        raise NotImplementedError("args.conf only supported with one dataset!")
    else:
        dataset_hparams_file = args.dataset_hparams_dict
        model_hparams_file = args.model_hparams_dict

    # if args.conf != '':
    #     logsdir = args.conf + '/' + args.dataset_hparams

    if args.output_dir is None:
        list_depth = 0
        model_fname = ''
        for t in ('model=%s,%s' % (args.model, args.model_hparams)):
            if t == '[':
                list_depth += 1
            if t == ']':
                list_depth -= 1
            if list_depth and t == ',':
                t = '..'
            if t in '=,':
                t = '.'
            if t in '[]':
                t = ''
            model_fname += t
        args.output_dir = os.path.join(logsdir, model_fname)

    if args.resume:
        print("args.checkpoint", args.checkpoint)
        if args.checkpoint:
            raise ValueError('resume and checkpoint cannot both be specified')
        args.checkpoint = args.output_dir
        print("Will attempt to resume from checkpoint at {}".format(args.checkpoint))

    model_hparams_dict = {}
    if model_hparams_file:
        with open(model_hparams_file) as f:
            model_hparams_dict.update(json.loads(f.read()))
    if args.checkpoint:
        checkpoint_dir = os.path.normpath(args.checkpoint)
        if not os.path.exists(checkpoint_dir):
            args.checkpoint = None
            checkpoint_dir = None
            print("cannot load checkpoint because none exists")
#            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoint_dir)
        else:
            if not os.path.isdir(args.checkpoint):
                checkpoint_dir, _ = os.path.split(checkpoint_dir)
            with open(os.path.join(checkpoint_dir, "options.json")) as f:
                print("loading options from checkpoint %s" % args.checkpoint)
                options = json.loads(f.read())
                args.dataset = args.dataset or options['dataset']
                args.model = args.model or options['model']
            try:
                with open(os.path.join(checkpoint_dir, "model_hparams.json")) as f:
                    model_hparams_dict.update(json.loads(f.read()))
                    model_hparams_dict.pop('num_gpus', None)  # backwards-compatibility
            except FileNotFoundError:
                print("model_hparams.json was not loaded because it does not exist")

    print('----------------------------------- Options ------------------------------------')
    for k, v in args._get_kwargs():
        print(k, "=", v)
    print('------------------------------------- End --------------------------------------')

    if args.train_datasets is not None and args.val_datasets is not None:
        print("loading seperate train and val datasets")

        train_dataset_hparams_dicts = [{} for _ in range(len(args.train_datasets))]
        for hparams_dict, params_file in zip(train_dataset_hparams_dicts, args.train_dataset_hparams_files):
            with open(params_file) as f:
                hparams_dict.update(json.loads(f.read()))

        val_dataset_hparams_dicts = [{} for _ in range(len(args.val_datasets))]
        for hparams_dict, params_file in zip(val_dataset_hparams_dicts, args.val_dataset_hparams_files):
            with open(params_file) as f:
                hparams_dict.update(json.loads(f.read()))

        train_datasets = []
        for dataset, hparams_dict, hparams_override, input_dir in zip(args.train_datasets,
                                                                      train_dataset_hparams_dicts,
                                                                      args.train_dataset_hparams,
                                                                      args.train_input_dirs):
            VideoDataset = datasets.get_dataset_class(dataset)
            print("Adding training dataset:", dataset, input_dir)
            train_datasets.append(VideoDataset(input_dir,
                                               mode='train',
                                               hparams_dict=hparams_dict,
                                               hparams=hparams_override))

        val_datasets = []
        for dataset, hparams_dict, hparams_override, input_dir in zip(args.val_datasets,
                                                                      val_dataset_hparams_dicts,
                                                                      args.val_dataset_hparams,
                                                                      args.val_input_dirs):
            print("Adding validation dataset:", dataset, input_dir)
            VideoDataset = datasets.get_dataset_class(dataset)
            val_datasets.append(VideoDataset(input_dir,
                                               mode='val',
                                               hparams_dict=hparams_dict,
                                               hparams=hparams_override))


    elif isinstance(args.dataset, list):
        print("dataset hparams file:", dataset_hparams_file)
        dataset_hparams_dicts = [{} for _ in range(len(args.dataset))]
        for hparams_dict, params_file in zip(dataset_hparams_dicts, dataset_hparams_file):
            with open(params_file) as f:
                hparams_dict.update(json.loads(f.read()))

        train_datasets, val_datasets = [], []
        val_input_dirs = args.val_input_dirs or args.input_dirs

        for dataset, hparams_dict, hparams_override, input_dir, val_input_dir in zip(args.dataset, dataset_hparams_dicts,
                                                                                     args.dataset_hparams, args.input_dirs,
                                                                                     val_input_dirs):
            VideoDataset = datasets.get_dataset_class(dataset)
            print("Dataset:", dataset)
            print("input_dir:", input_dir)
            print("val_input_dir", val_input_dir)
            train_datasets.append(VideoDataset(input_dir, mode='train',
                                              hparams_dict=hparams_dict, hparams=hparams_override))
            val_datasets.append(VideoDataset(val_input_dir, mode='val', hparams_dict=hparams_dict,
                                             hparams=hparams_override))
    else:
        dataset_hparams_dict = {}
        if dataset_hparams_file:
            with open(dataset_hparams_file) as f:
                dataset_hparams_dict.update(json.loads(f.read()))
        if args.checkpoint:
            try:
                with open(os.path.join(checkpoint_dir, "dataset_hparams.json")) as f:
                    dataset_hparams_dict.update(json.loads(f.read()))
            except FileNotFoundError:
                print("dataset_hparams.json was not loaded because it does not exist")

        VideoDataset = datasets.get_dataset_class(args.dataset)
        train_datasets = [VideoDataset(input_dir, mode='train', hparams_dict=dataset_hparams_dict, hparams=args.dataset_hparams)
                          for input_dir in args.input_dirs]
        val_input_dirs = args.val_input_dirs or args.input_dirs
        val_datasets = [VideoDataset(val_input_dir, mode='val', hparams_dict=dataset_hparams_dict, hparams=args.dataset_hparams)
                        for val_input_dir in val_input_dirs]
    # if len(val_input_dirs) > 1:
    #     if isinstance(val_datasets[-1], datasets.KTHVideoDataset):
    #         val_datasets[-1].set_sequence_length(40)
    #     else:
    #         val_datasets[-1].set_sequence_length(30)

    def override_hparams_dict(dataset):
        hparams_dict = dict(model_hparams_dict)
        hparams_dict['context_frames'] = dataset.hparams.context_frames
        hparams_dict['sequence_length'] = dataset.hparams.sequence_length
        hparams_dict['repeat'] = dataset.hparams.time_shift
        return hparams_dict

    VideoPredictionModel = models.get_model_class(args.model)
    # override hparams from first dataset for train model
    train_model = VideoPredictionModel(mode='train', hparams_dict=override_hparams_dict(train_datasets[0]), hparams=args.model_hparams)
    #if val_input_dirs == args.input_dirs:
    #    val_models = [VideoPredictionModel(mode='val', hparams_dict=override_hparams_dict(val_datasets[0]), hparams=args.model_hparams)]
    #else:
    val_models = [VideoPredictionModel(mode='val', hparams_dict=override_hparams_dict(val_dataset), hparams=args.model_hparams)
                      for val_dataset in val_datasets]

    batch_size = train_model.hparams.batch_size
    with tf.variable_scope('') as training_scope:
        if args.train_batch_sizes:
            assert len(args.train_batch_sizes) == len(train_datasets)
            assert sum(args.train_batch_sizes) == batch_size
            inputs, targets = zip(*[train_dataset.make_batch(bs)
                                    for train_dataset, bs in zip(train_datasets, args.train_batch_sizes)])
        else:
            assert batch_size % len(train_datasets) == 0
            inputs, targets = zip(*[train_dataset.make_batch(batch_size // 2) for train_dataset in train_datasets])
        inputs = nest.map_structure(lambda *x: tf.concat(x, axis=0), *inputs)
        targets = nest.map_structure(lambda *x: tf.concat(x, axis=0), *targets)

        train_model.build_graph(inputs, targets)
#    if val_input_dirs == args.input_dirs:
#        with tf.variable_scope(training_scope, reuse=True):
#            if args.train_batch_sizes:
#                inputs, targets = zip(*[train_dataset.make_batch(bs)
#                                        for train_dataset, bs in zip(train_datasets, args.train_batch_sizes)])
#            else:
#                inputs, targets = zip(*[train_dataset.make_batch(batch_size // 2) for train_dataset in train_datasets])
#            inputs = nest.map_structure(lambda *x: tf.concat(x, axis=0), *inputs)
#            targets = nest.map_structure(lambda *x: tf.concat(x, axis=0), *targets)
#            val_models[0].build_graph(inputs, targets)
#    else:
    for val_model, val_dataset in zip(val_models, val_datasets):
        with tf.variable_scope(training_scope, reuse=True):
            val_model.build_graph(*val_dataset.make_batch(batch_size))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    with open(os.path.join(args.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))
    with open(os.path.join(args.output_dir, "dataset_hparams.json"), "w") as f:
        f.write(json.dumps(train_datasets[0].hparams.values(), sort_keys=True, indent=4))  # save hparams from first dataset
    with open(os.path.join(args.output_dir, "model_hparams.json"), "w") as f:
        f.write(json.dumps(train_model.hparams.values(), sort_keys=True, indent=4))

    if args.gif_freq:
        val_model = val_models[0]
        val_tensors = OrderedDict()
        context_images = val_model.inputs['images'][:, :val_model.hparams.context_frames]
        val_tensors['gen_images_vis'] = tf.concat([context_images, val_model.gen_images], axis=1)
        if val_model.gen_images_enc is not None:
            val_tensors['gen_images_enc_vis'] = tf.concat([context_images, val_model.gen_images_enc], axis=1)
        val_tensors.update({name: tensor for name, tensor in val_model.inputs.items() if tensor.shape.ndims >= 4})
        val_tensors['targets'] = val_model.targets
        val_tensors.update({name: tensor for name, tensor in val_model.outputs.items() if tensor.shape.ndims >= 4})
        val_tensor_clips = OrderedDict([(name, tf_utils.tensor_to_clip(output)) for name, output in val_tensors.items()])

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=3)
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    image_summaries = set(tf.get_collection(tf_utils.IMAGE_SUMMARIES))
    eval_summaries = set(tf.get_collection(tf_utils.EVAL_SUMMARIES))
    eval_image_summaries = image_summaries & eval_summaries
    image_summaries -= eval_image_summaries
    eval_summaries -= eval_image_summaries
    if args.summary_freq:
        summary_op = tf.summary.merge(summaries)
    if args.image_summary_freq:
        image_summary_op = tf.summary.merge(list(image_summaries))
    if args.eval_summary_freq:
        eval_summary_op = tf.summary.merge(list(eval_summaries))
        eval_image_summary_op = tf.summary.merge(list(eval_image_summaries))

    if args.summary_freq or args.image_summary_freq or args.eval_summary_freq:
        summary_writer = tf.summary.FileWriter(args.output_dir)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem_frac)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    global_step = tf.train.get_or_create_global_step()
    max_steps = train_model.hparams.max_steps
    elapsed_times = []
    with tf.Session(config=config) as sess:
        print("parameter_count =", sess.run(parameter_count))

        sess.run(tf.global_variables_initializer())
        print("restoring from", args.checkpoint)
        train_model.restore(sess, args.checkpoint)
        print("restored from:", args.checkpoint)
        start_step = sess.run(global_step)
        # start at one step earlier to log everything without doing any training
        # step is relative to the start_step
        for step in range(-1, max_steps - start_step):
            if step == 0:
                start = time.time()

            def should(freq):
                return freq and ((step + 1) % freq == 0 or (step + 1) in (0, max_steps - start_step))

            fetches = {}
            if step >= 0:
                fetches["train_op"] = train_model.train_op

            if should(args.progress_freq):
                fetches['d_losses'] = train_model.d_losses
                fetches['g_losses'] = train_model.g_losses
                if isinstance(train_model.learning_rate, tf.Tensor):
                    fetches["learning_rate"] = train_model.learning_rate
            if should(args.metrics_freq):
                fetches['metrics'] = train_model.metrics
            if should(args.summary_freq):
                fetches["summary"] = summary_op
            if should(args.image_summary_freq):
                fetches["image_summary"] = image_summary_op
            if should(args.eval_summary_freq):
                fetches["eval_summary"] = eval_summary_op
                fetches["eval_image_summary"] = eval_image_summary_op

            run_start_time = time.time()
            results = sess.run(fetches)
            run_elapsed_time = time.time() - run_start_time
            elapsed_times.append(run_elapsed_time)
            if run_elapsed_time > 0.01:#1.5:
                print('session.run took %0.1fs' % run_elapsed_time)

            # print("average t_iter {} \n".format(np.mean(elapsed_times[-20:])))
            # if step == 88:
            #     with open(args.timing_file, 'w') as f:
            #         f.write("{}\n".format(np.mean(elapsed_times)))
            #     import sys; sys.exit("finished")

            if should(args.progress_freq) or should(args.summary_freq):
                if step >= 0:
                    elapsed_time = time.time() - start
                    average_time = elapsed_time / (step + 1)
                    images_per_sec = batch_size / average_time
                    remaining_time = (max_steps - (start_step + step)) * average_time

            if should(args.progress_freq):
                # global_step will have the correct step count if we resume from a checkpoint
                steps_per_epoch = math.ceil(sum([train_dataset.num_examples_per_epoch() for train_dataset in train_datasets]) / batch_size)
                train_epoch = math.ceil(global_step.eval() / steps_per_epoch)
                train_step = (global_step.eval() - 1) % steps_per_epoch + 1
                print("progress  global step %d  epoch %d  step %d" % (global_step.eval(), train_epoch, train_step))
                if step >= 0:
                    print("          image/sec %0.1f  remaining %dm (%0.1fh) (%0.1fd)" %
                          (images_per_sec, remaining_time / 60, remaining_time / 60 / 60, remaining_time / 60 / 60 / 24))

                for name, loss in itertools.chain(results['d_losses'].items(), results['g_losses'].items()):
                    print(name, loss)
                if isinstance(train_model.learning_rate, tf.Tensor):
                    print("learning_rate", results["learning_rate"])
            if should(args.metrics_freq):
                for name, metric in results['metrics']:
                    print(name, metric)

            if should(args.summary_freq):
                print("recording summary")
                summary_writer.add_summary(results["summary"], global_step.eval())
                if step >= 0:
                    try:
                        from tensorboard.summary import scalar_pb
                        for name, scalar in zip(['images_per_sec', 'remaining_hours'],
                                                [images_per_sec, remaining_time / 60 / 60]):
                            summary_writer.add_summary(scalar_pb(name, scalar), global_step.eval())
                    except ImportError:
                        pass

                print("done")
            if should(args.image_summary_freq):
                print("recording image summary")
                summary_writer.add_summary(
                    tf_utils.convert_tensor_to_gif_summary(results["image_summary"]), global_step.eval())
                print("done")
            if should(args.eval_summary_freq):
                print("recording eval summary")
                summary_writer.add_summary(results["eval_summary"], global_step.eval())
                summary_writer.add_summary(
                    tf_utils.convert_tensor_to_gif_summary(results["eval_image_summary"]), global_step.eval())
                print("done")



            if should(args.summary_freq) or should(args.image_summary_freq) or should(args.eval_summary_freq):
                summary_writer.flush()

            if should(args.save_freq) or time.time() - program_start_time > args.run_time:
                print("saving model to", args.output_dir)
                saver.save(sess, os.path.join(args.output_dir, "model"), global_step=global_step)
                print("done")
                if time.time() - program_start_time > args.run_time:
                    print("Exiting program because time limit has been reached")
                    return


            if should(args.gif_freq):
                image_dir = os.path.join(args.output_dir, 'images')
                if not os.path.exists(image_dir):
                    os.makedirs(image_dir)

                gif_clips = sess.run(val_tensor_clips)
                gif_step = global_step.eval()
                for name, clip in gif_clips.items():
                    filename = "%08d-%s.gif" % (gif_step, name)
                    print("saving gif to", os.path.join(image_dir, filename))
                    ffmpeg_gif.save_gif(os.path.join(image_dir, filename), clip, fps=4)
                    print("done")


if __name__ == '__main__':
    main()
