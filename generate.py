from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import errno
import json
import os
import random

import cv2
import numpy as np
import tensorflow as tf

from video_prediction import datasets, models
from video_prediction.utils.ffmpeg_gif import save_gif


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="either a directory containing subdirectories "
                                                                     "train, val, test, etc, or a directory containing "
                                                                     "the tfrecords")
    parser.add_argument("--results_dir", type=str, default='results', help="ignored if output_gif_dir is specified")
    parser.add_argument("--results_gif_dir", type=str, help="default is results_dir. ignored if output_gif_dir is specified")
    parser.add_argument("--results_png_dir", type=str, help="default is results_dir. ignored if output_png_dir is specified")
    parser.add_argument("--output_gif_dir", help="output directory where samples are saved as gifs. default is "
                                                 "results_gif_dir/model_fname")
    parser.add_argument("--output_png_dir", help="output directory where samples are saved as pngs. default is "
                                                 "results_png_dir/model_fname")
    parser.add_argument("--checkpoint", help="directory with checkpoint or checkpoint name (e.g. checkpoint_dir/model-200000)")

    parser.add_argument("--mode", type=str, choices=['val', 'test'], default='val', help='mode for dataset, val or test.')

    parser.add_argument("--dataset", type=str, help="dataset class name")
    parser.add_argument("--dataset_hparams", type=str, help="a string of comma separated list of dataset hyperparameters")
    parser.add_argument("--dataset_hparams_file", type=str)
    parser.add_argument("--model", type=str, help="model class name")
    parser.add_argument("--model_hparams", type=str, help="a string of comma separated list of model hyperparameters")

    parser.add_argument("--batch_size", type=int, default=8, help="number of samples in batch")
    parser.add_argument("--num_samples", type=int, help="number of samples in total (all of them by default)")
    parser.add_argument("--num_epochs", type=int, default=1)

    parser.add_argument("--num_stochastic_samples", type=int, default=5)
    parser.add_argument("--gif_length", type=int, help="default is sequence_length")
    parser.add_argument("--fps", type=int, default=4)

    parser.add_argument("--gpu_mem_frac", type=float, default=0, help="fraction of gpu memory to use")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--no_gif", action='store_true', default=False, help="don't store gif images")
    parser.add_argument("--no_im", action='store_true', default=False, help="don't store png images")
    parser.add_argument("--no_actions", action='store_true', default=False, help="don't store action images")

    args = parser.parse_args()

    if args.seed is not None:
        tf.set_random_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    args.results_gif_dir = args.results_gif_dir or args.results_dir
    args.results_png_dir = args.results_png_dir or args.results_dir
    dataset_hparams_dict = {}
    model_hparams_dict = {}
    if args.checkpoint:
        checkpoint_dir = os.path.normpath(args.checkpoint)
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), checkpoint_dir)
        if not os.path.isdir(args.checkpoint):
            checkpoint_dir, _ = os.path.split(checkpoint_dir)
        with open(os.path.join(checkpoint_dir, "options.json")) as f:
            print("loading options from checkpoint %s" % args.checkpoint)
            options = json.loads(f.read())
            args.dataset = args.dataset or options['dataset']
            args.model = args.model or options['model']
        if args.dataset_hparams_file is not None:
            with open(args.dataset_hparams_file) as f:
                dataset_hparams_dict = json.loads(f.read())
        else:
            try:
                with open(os.path.join(checkpoint_dir, "dataset_hparams.json")) as f:
                    dataset_hparams_dict = json.loads(f.read())
            except FileNotFoundError:
                print("dataset_hparams.json was not loaded because it does not exist")
        try:
            with open(os.path.join(checkpoint_dir, "model_hparams.json")) as f:
                model_hparams_dict = json.loads(f.read())
                model_hparams_dict.pop('num_gpus', None)  # backwards-compatibility
        except FileNotFoundError:
            print("model_hparams.json was not loaded because it does not exist")
        args.output_gif_dir = args.output_gif_dir or os.path.join(args.results_gif_dir, os.path.split(checkpoint_dir)[1])
        args.output_png_dir = args.output_png_dir or os.path.join(args.results_png_dir, os.path.split(checkpoint_dir)[1])
    else:
        if not args.dataset:
            raise ValueError('dataset is required when checkpoint is not specified')
        if not args.model:
            raise ValueError('model is required when checkpoint is not specified')
        args.output_gif_dir = args.output_gif_dir or os.path.join(args.results_gif_dir, 'model.%s' % args.model)
        args.output_png_dir = args.output_png_dir or os.path.join(args.results_png_dir, 'model.%s' % args.model)

    dataset_name = os.path.split(args.input_dir)
    while dataset_name[1] in ['train', 'test', 'val', '']:
        dataset_name = os.path.split(dataset_name[0])
    args.output_gif_dir = os.path.join(args.output_gif_dir, dataset_name[1])
    args.output_png_dir = os.path.join(args.output_png_dir, dataset_name[1])
    print("dataset:", dataset_name[1])

    print('----------------------------------- Options ------------------------------------')
    for k, v in args._get_kwargs():
        print(k, "=", v)
    print('------------------------------------- End --------------------------------------')

    VideoDataset = datasets.get_dataset_class(args.dataset)
    dataset = VideoDataset(args.input_dir, mode=args.mode, num_epochs=args.num_epochs, seed=args.seed,
                           hparams_dict=dataset_hparams_dict, hparams=args.dataset_hparams)

    def override_hparams_dict(dataset):
        hparams_dict = dict(model_hparams_dict)
        hparams_dict['context_frames'] = dataset.hparams.context_frames
        hparams_dict['sequence_length'] = dataset.hparams.sequence_length
        hparams_dict['repeat'] = dataset.hparams.time_shift
        return hparams_dict

    VideoPredictionModel = models.get_model_class(args.model)
    model = VideoPredictionModel(mode='test', hparams_dict=override_hparams_dict(dataset), hparams=args.model_hparams)

    print("hparams", model.hparams)

    if args.num_samples:
        if args.num_samples > dataset.num_examples_per_epoch():
            print('num_samples cannot be larger than the dataset')
            print("setting num_samples to dataset size:", dataset.num_examples_per_epoch())
            args.num_samples = dataset.num_examples_per_epoch()
        num_examples_per_epoch = args.num_samples
    else:
        num_examples_per_epoch = dataset.num_examples_per_epoch()
    if num_examples_per_epoch % args.batch_size != 0:
        raise ValueError('batch_size should evenly divide the dataset')

    inputs, targets = dataset.make_batch(args.batch_size)
#    if not isinstance(model, models.GroundTruthVideoPredictionModel):
        # remove ground truth data past context_frames to prevent accidentally using it
#        for k, v in inputs.items():
#            if k != 'actions':
#                inputs[k] = v[:, :model.hparams.context_frames]

    input_phs = {k: tf.placeholder(v.dtype, v.shape, '%s_ph' % k) for k, v in inputs.items()}

    with tf.variable_scope(''):
        model.build_graph(input_phs, targets)
    
    for output_dir in (args.output_gif_dir, args.output_png_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        with open(os.path.join(output_dir, "options.json"), "w") as f:
            f.write(json.dumps(vars(args), sort_keys=True, indent=4))
        with open(os.path.join(output_dir, "dataset_hparams.json"), "w") as f:
            f.write(json.dumps(dataset.hparams.values(), sort_keys=True, indent=4))
        with open(os.path.join(output_dir, "model_hparams.json"), "w") as f:
            f.write(json.dumps(model.hparams.values(), sort_keys=True, indent=4))

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_mem_frac)
    config = tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True)
    sess = tf.Session(config=config)

    model.restore(sess, args.checkpoint)
    print("outputs:", model.outputs.keys())
    sample_ind = 0
    mses = {}
    while True:
        print("inputs:", inputs.keys())
        if args.num_samples and sample_ind >= args.num_samples:
            break
        try:
            input_results = sess.run(inputs)
        except tf.errors.OutOfRangeError:
            break
        print("evaluation samples from %d to %d" % (sample_ind, sample_ind + args.batch_size))

        feed_dict = {input_ph: input_results[name] for name, input_ph in input_phs.items()}
        for stochastic_sample_ind in range(args.num_stochastic_samples):
            goals = [model.outputs['gen_images'],
                     model.outputs['decoded_actions'],
                     model.outputs['decoded_actions_inverse'],
                     model.metrics,
                     model.inputs]
            gen_images, auto_enc_actions, inv_actions, metrics, sess_inputs = sess.run(goals, feed_dict=feed_dict)
            
            print("actions", inputs['actions'].shape)
            print("auto_enc_actions", auto_enc_actions.shape)
            print("inv_actions", inv_actions.shape)
            for i, gen_images_ in enumerate(gen_images):
                gen_images_ = (gen_images_ * 255.0).astype(np.uint8)
#                print("actions:", inputs['actions'][i])
#                print("enc/dec action", auto_enc_actions[i])
#                print("inv actions", inv_actions[i])
                print("Mse:", str(metrics['mse']))
                if not args.no_gif:
                    gen_images_fname = 'gen_image_%05d_%02d.gif' % (sample_ind + i, stochastic_sample_ind)
                    mses[gen_images_fname] = str(metrics['mse'])
                    save_gif(os.path.join(args.output_gif_dir, gen_images_fname),
                            gen_images_[:args.gif_length] if args.gif_length else gen_images_, fps=args.fps)
                if not args.no_im:
                    gen_image_fname_pattern = 'gen_image_%%05d_%%02d_%%0%dd.png' % max(2, len(str(len(gen_images_) - 1)))
                    for t, gen_image in enumerate(gen_images_):
                        gen_image_fname = gen_image_fname_pattern % (sample_ind + i, stochastic_sample_ind, t)
                        gen_image = cv2.cvtColor(gen_image, cv2.COLOR_RGB2BGR)
                        cv2.imwrite(os.path.join(args.output_png_dir, gen_image_fname), gen_image)

                if not args.no_actions:
                    gt_ims = sess_inputs['images'][i]
                    print("gt_ims:", gt_ims)
                    action_fname = "action_image_%05d_%02d.png" % (sample_ind + i, stochastic_sample_ind)
                    im_width = gt_ims.shape[2]
                    action_image = np.ones((gt_ims.shape[1] * 3, im_width * (gt_ims.shape[0]), gt_ims.shape[3]))
                    print("images:", inputs['images'][i].shape)
                    print(action_image.shape)
                    for t in range(gt_ims.shape[0]):
                        action_image[:gt_ims.shape[1], t*im_width:(t+1)*im_width] = cv2.cvtColor(gt_ims[t], cv2.COLOR_RGB2BGR)

                    gt_actions = sess_inputs['actions'][i]
                    if model.hparams.rescale_actions:
                        gt_actions = gt_actions / np.array([0.07, 0.07, 0.5, 0.15])
                    for t in range(gt_actions.shape[0]):
                        arrow_image = np.ones((gt_ims.shape[1], gt_ims.shape[2], gt_ims.shape[3]))
                        center = (gt_ims.shape[2]//2, gt_ims.shape[1]//2)
                        scale = min(center[0], center[1])
                        head_size = 2
                        gt_end = (center[0] + int(gt_actions[t][1]*scale), center[1] + int(gt_actions[t][0]*scale))
                        print("gt_end", gt_end)
                        auto_enc_end = (center[0] + int(auto_enc_actions[i][t][1]*scale), center[1] + int(auto_enc_actions[i][t][0]*scale))
                        print("autoenc end:", auto_enc_end)
                        inv_end = (center[0] + int(inv_actions[i][t][1]*scale), center[1] + int(inv_actions[i][t][0]*scale))
                        print("inv_end", inv_end)
                        cv2.arrowedLine(arrow_image, center, gt_end, (1, 0, 0), 2) # In BGR color space
                        cv2.arrowedLine(arrow_image, center, auto_enc_end, (0, 1, 0), 1)
                        cv2.arrowedLine(arrow_image, center, inv_end, (0, 0, 1), 1)

                        action_image[gt_ims.shape[1]:2*gt_ims.shape[1], (t)*im_width + im_width//2:(t+1)*im_width+im_width//2] = arrow_image
                    for t in range(gt_actions.shape[0]):
                        arrow_image = np.ones((gt_ims.shape[1], gt_ims.shape[2], gt_ims.shape[3]))
                        center = (gt_ims.shape[2]//2, gt_ims.shape[1]//2)
                        scale = min(center[0], center[1])
                        head_size = 2
                        gt_end = (center[0] + int(gt_actions[t][2]*scale), center[1] + int(gt_actions[t][3]*scale))
                        print("gt_end", gt_end)
                        auto_enc_end = (center[0] + int(auto_enc_actions[i][t][2]*scale), center[1] + int(auto_enc_actions[i][t][3]*scale))
                        print("autoenc end:", auto_enc_end)
                        inv_end = (center[0] + int(inv_actions[i][t][2]*scale), center[1] + int(inv_actions[i][t][3]*scale))
                        print("inv_end", inv_end)
                        cv2.arrowedLine(arrow_image, center, gt_end, (1, 0, 0), 2) # In BGR color space
                        cv2.arrowedLine(arrow_image, center, auto_enc_end, (0, 1, 0), 1)
                        cv2.arrowedLine(arrow_image, center, inv_end, (0, 0, 1), 1)

                        action_image[2*gt_ims.shape[1]:3*gt_ims.shape[1], (t)*im_width + im_width//2:(t+1)*im_width+im_width//2] = arrow_image
                    cv2.imwrite(os.path.join(args.output_png_dir, action_fname), action_image * 255)

        sample_ind += args.batch_size
    print("mses:", mses)
    with open(os.path.join(args.output_gif_dir, "mses.json"), 'w') as f:
        f.write(json.dumps(mses))

if __name__ == '__main__':
    main()
