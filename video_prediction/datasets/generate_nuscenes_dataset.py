from nuscenes.nuscenes import NuScenes

from pyquaternion import Quaternion
import argparse
import numpy as np
import cv2
from PIL import Image
from os.path import join
import tensorflow as tf
import os
from tqdm import tqdm
import math


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def save_tf_record(save_path, images, actions, image_size):
    try:
        loaded_images = []
        for i in range(len(images)):        
            loaded_images.append([cv2.resize(cv2.imread(img), image_size, interpolation=cv2.INTER_AREA)[:, :, ::-1] for img in images[i]])
    except:
        return

    writer = tf.python_io.TFRecordWriter(save_path)

    for i in range(len(images)):

        featurized_images = [_bytes_feature(image.tostring()) for image in loaded_images[i]]
        featurized_actions = [_floats_feature(action) for action in actions[i]]

        feature = {}
        for j in range(len(images[i])):
            feature[str(j) + '/env/image_view0/encoded'] = featurized_images[j]
            feature[str(j) + '/policy/actions'] = featurized_actions[j]
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())

def  main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', default='/NAS/data/robotic_video/nuscenes/v1.0-mini', type=str)
    parser.add_argument('--dataset_version', default='v1.0-mini', type=str)
    parser.add_argument('--out_path', default="/NAS/data/robotic_video/nuscenes/mini_tf", type=str)
    parser.add_argument('--image_size', nargs='+', default=[64, 48], type=int)
    parser.add_argument('--traj_per_record', default=10, type=int)
    parser.add_argument('--sequence_length', default=20, type=int)
    parser.add_argument('--train_fraction', default=10, type=int)
    parser.add_argument('--starting_index', type=int, default=0)
    args = parser.parse_args()
    args.image_size = tuple(args.image_size)
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)
        os.makedirs(join(args.out_path, "boston", "val"))
        os.makedirs(join(args.out_path, "singapore", "val"))
        os.makedirs(join(args.out_path, "boston", "train"))
        os.makedirs(join(args.out_path, "singapore", "train"))

    nusc = NuScenes(version=args.dataset_version, dataroot=args.dataset_path, verbose=True)

    images = {'boston':[], 'singapore':[]}
    actions = {'boston':[], 'singapore':[]}

    for i, scene in tqdm(enumerate(nusc.scene), total=len(nusc.scene)):
        log = nusc.get("log", scene['log_token'])
        next_sample = scene['first_sample_token']
        prev_rot = None
        prev_trans = None
        prev_timestamp = None
        curr_images = []
        curr_actions = []

        while len(next_sample) > 0:
            sample = nusc.get("sample", next_sample)
            data = nusc.get("sample_data", sample['data']['CAM_FRONT'])
            ego_pose = nusc.get("ego_pose", data['ego_pose_token'])

            if prev_rot is not None:
                rot = Quaternion(ego_pose['rotation'])
                trans = np.array(ego_pose['translation']) - prev_trans
                trans = rot.inverse.rotate(trans)
                rot = Quaternion.absolute_distance(rot, prev_rot)
                timestamp = ego_pose['timestamp'] - prev_timestamp
            else:
                rot = 0
                trans = np.zeros(len(ego_pose['translation']))
                timestamp = 0
            prev_rot = Quaternion(ego_pose['rotation'])
            prev_trans = np.array(ego_pose['translation'])
            prev_timestamp = ego_pose['timestamp']

#            img = cv2.imread(join(args.dataset_path, data['filename']))
#            img_reshaped = cv2.resize(img, args.image_size, interpolation=cv2.INTER_AREA)
#            curr_images.append(img_reshaped)
            curr_images.append(join(args.dataset_path, data['filename']))
            curr_actions.append([rot, trans[0], trans[1]])

            if len(curr_images) == args.sequence_length:
                found = False
                for k in images.keys():
                    if k in log['location']:
                        images[k].append(curr_images)
                        actions[k].append(curr_actions)
                        found = True
                if not found:
                    print("unknown city")
                    print("log[location]", log['location'])
                    exit()
                curr_images = []
                curr_actions = []

            next_sample = sample['next']

    for k in images.keys():
        print(k, len(images[k]), len(actions[k]))
        print(len(images[k][0]), len(actions[k][0]))

        for i in tqdm(range(math.ceil(len(images[k]) / args.traj_per_record))):
            start = i*args.traj_per_record
            end = min((i + 1)* args.traj_per_record, len(images[k]))
            current_record = "traj_{}_to_{}.tfrecords".format(start + args.starting_index, end + args.starting_index)
            if i % args.train_fraction == 0:
                current_record = join(args.out_path, k, "val", current_record)
            else:
                current_record = join(args.out_path, k, "train", current_record)
            save_tf_record(current_record, images[k][start:end], actions[k][start:end], args.image_size)

if __name__ == '__main__':

    main()
