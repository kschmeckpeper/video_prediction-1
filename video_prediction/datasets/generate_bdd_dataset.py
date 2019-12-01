import argparse
import numpy as np
import cv2
from tqdm import tqdm
from os.path import isdir, isfile, join
from os import listdir
import os
import math
import time

import tensorflow as tf
from qtrotate import get_set_rotation

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
    parser.add_argument('--output_path_2', default=None, type=str)
    parser.add_argument('--image_size', nargs='+', default=[64, 48], type=int)
    parser.add_argument('--traj_per_record', default=10, type=int)
    parser.add_argument('--sequence_length', default=20, type=int)
    parser.add_argument('--start_count', default=0, type=int)
    parser.add_argument('--video_start_index', default=0, type=int)
    parser.add_argument('--time_remaining', default=-1, type=float)
    parser.add_argument('--prefix', default=None, type=str)
    args = parser.parse_args()
    save_video(args)

def save_video(args):
    if args.time_remaining > 0:
        start_time = time.time()
    sampling_frequency = 15

    args.image_size = tuple(args.image_size)
    
    modes = [d for d in listdir(args.video_path) if isdir(join(args.video_path, d))]

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        

    count = args.start_count
    video_index = 0


    for mode in modes:
        if not os.path.exists(join(args.output_path, mode)):
            os.makedirs(join(args.output_path, mode))
        print("mode:", mode)
        videos = [f for f in listdir(join(args.video_path, mode)) if isfile(join(args.video_path, mode, f)) and f[-4:] == ".mov"]

        for video_file in videos:
            video_index += 1
            if video_index < args.video_start_index:
                continue
            if video_index % 50 == 0:
                print("Video index:", video_index)
            video_capture = cv2.VideoCapture(join(args.video_path, mode, video_file))
            rotation = get_set_rotation(join(args.video_path, mode, video_file))
            if rotation == 0:
                continue
            frames = [[]]
            frames_2 = [[]]
            index = 0
            while video_capture.isOpened():
                index += 1
                ret, frame = video_capture.read()
                if frame is None:
                    break
                if index % sampling_frequency != 0:
                    continue
#                cv2.imwrite("frame_origin.png", frame)

                frame = np.swapaxes(frame, 0, 1)
                frame = np.fliplr(frame)
                if int(rotation) == 270:
                    rows, cols, _ = frame.shape
                    rot_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), 180, 1)
                    frame = cv2.warpAffine(frame, rot_matrix, (cols, rows))

#                cv2.imwrite("frame_examples/im{:05d}.png".format(index), frame)
                resized_frame = cv2.resize(frame, args.image_size, interpolation=cv2.INTER_AREA)
                if args.output_path_2 is not None:
                    resized_frame_2 = cv2.resize(frame, (64, 48), interpolation=cv2.INTER_AREA)
                else:
                    resized_frame_2 = None

                if len(frames[-1]) == args.sequence_length:
                    frames.append([])
                    frames_2.append([])
                frames[-1].append(resized_frame)
                frames_2[-1].append(resized_frame_2)
            if len(frames[-1]) < args.sequence_length:
                frames = frames[:-1]
                frames_2 = frames_2[:-1]
#            continue
            for i in range(math.ceil(len(frames) / args.traj_per_record)):
                start = i*args.traj_per_record
                end = min((i + 1)* args.traj_per_record, len(frames))
 
                current_record_name = "traj_{:05d}_to_{:05d}.tfrecords".format(start + count, end + count)
                if args.prefix is not None:
                    current_record_name = args.prefix + '_' + current_record_name
                current_record = join(args.output_path, mode, current_record_name)
                save_tf_record(current_record, frames[start:end])

                if args.output_path_2 is not None:
                    record_2 = join(args.output_path_2, mode, current_record_name)
                    save_tf_record(record_2, frames_2[start:end])

            count += end

            if args.time_remaining > 0:
                if time.time() - start_time > args.time_remaining:
                    print("time out", time.time() - start_time, args.time_remaining)
                    return False, video_index, count
        return True, video_index, count

        

if __name__ == '__main__':
    main()

