import argparse
import numpy as np
import cv2
from tqdm import tqdm
from os.path import isdir, isfile, join
from os import listdir
import os
import math

import tensorflow as tf

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def save_tf_record(save_path, images):
    writer = tf.python_io.TFRecordWriter(save_path)

    for i in range(len(images)):
        featurized_images = [_bytes_feature(image.tostring()) for image in images[i]]

        feature = {}
        for j in range(len(images[i])):
            feature[str(j) + '/env/image_view0/encoded'] = featurized_images[j]
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', default="/NAS/data/berkeley_deep_drive/bdd100k/videos", type=str)
    parser.add_argument('--output_path', default="/NAS/data/robotic_video/bdd_tf_2k", type=str)
    parser.add_argument('--image_size', nargs='+', default=[64, 48], type=int)
    parser.add_argument('--traj_per_record', default=10, type=int)
    parser.add_argument('--sequence_length', default=20, type=int)

    args = parser.parse_args()
    args.image_size = tuple(args.image_size)
    
    modes = [d for d in listdir(args.video_path) if isdir(join(args.video_path, d))]

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        

    count = 0
    for mode in modes:
        if not os.path.exists(join(args.output_path, mode)):
            os.makedirs(join(args.output_path, mode))
        print("mode:", mode)
        videos = [f for f in listdir(join(args.video_path, mode)) if isfile(join(args.video_path, mode, f)) and f[-4:] == ".mov"]

        index = 0
        for video_file in tqdm(videos):
            video_capture = cv2.VideoCapture(join(args.video_path, mode, video_file))
            frames = [[]]
            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if frame is None:
                    break
#                cv2.imwrite("frame_origin.png", frame)
#                print("Frame shape:", frame.shape)
#                frame = np.swapaxes(frame, 0, 1)
#                print("flipped shape:", frame.shape)
                cv2.imwrite("frame_examples/im{:05d}.png".format(index), frame)
                index += 1
                break
                resized_frame = cv2.resize(frame, args.image_size, interpolation=cv2.INTER_AREA)
#                cv2.imwrite("frame_resized.png", resized_frame)
#                exit()
                if len(frames[-1]) == args.sequence_length:
                    frames.append([])
                frames[-1].append(resized_frame)
            if len(frames[-1]) < args.sequence_length:
                frames = frames[:-1]
            continue
            for i in range(math.ceil(len(frames) / args.traj_per_record)):
                start = i*args.traj_per_record
                end = min((i + 1)* args.traj_per_record, len(frames))
 
                current_record = "traj_{:05d}_to_{:05d}.tfrecords".format(start + count, end + count)
                current_record = join(args.output_path, mode, current_record)
                save_tf_record(current_record, frames[start:end])
            count += end

        

if __name__ == '__main__':
    main()

