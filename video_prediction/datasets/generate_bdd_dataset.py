import argparse
import numpy as np
import cv2
from tqdm import tqdm

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _floats_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', default="/NAS/data/berkeley_deep_drive", type=str)
    parser.add_argument('--output_path', default="/NAS/data/robotic_video/bdd_tf", type=str)
    parser.add_argument('--image_size', nargs='+', default=[64, 48], type=int)
    parser.add_argument('--traj_per_record', default=10, type=int)
    parser.add_argument('--sequence_length', default=20, type=int)

    args = parser.parse_args()
    args.image_size = tuple(args.image_size)
    
    for mode in modes:
        print("mode:", mode)
        videos = [f for f in listdir(join(args.video_path, mode)) if isfile(join(args.video_path, mode, f)) and f[-4:] == ".mov"]
        print("videos:", videos)

        for video_file in tqdm(videos):
            video_capture = cv2.VideoCapture(join(args.video_path, mode, video_file))
            frames = []
            while video_capture.isOpened():
                ret, frame = video_capture.read()
                resized_frame = cv2.resize(frame, args.image_size, interpolation=cv2.INTER_AREA)
                frames.append(resized_frame)
            print("num frames:", len(frames))
            exit()
        

if __name__ == '__main__':
    
