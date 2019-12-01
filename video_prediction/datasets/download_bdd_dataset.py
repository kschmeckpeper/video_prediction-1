import os
import argparse
import subprocess
import json
import time
import shutil
from generate_bdd_dataset import save_video

start_time = time.time()
max_time = 3300

def update_checkpoint(path, name, progress, count=-1):
    try: 
        with open(path, 'r') as f:
            checkpoint_file = json.load(f)
    except:
        checkpoint_file = {}
    checkpoint_file[name] = progress
    if count > 0:
        checkpoint_file['tf_records_count'] = count
    with open(path, 'w') as f:
        json.dump(checkpoint_file, f, indent=4)
    if time.time() - start_time > max_time:
        exit()

def download_and_process(name, args):
    checkpoint = 'checkpoint.json'
    download_path = "/NAS/scratch/karls"
    if args.prefix is not None:
        checkpoint = args.prefix + "_" + checkpoint
        download_path = os.path.join(download_path, args.prefix)
        if not os.path.isdir(download_path):
            os.makedirs(download_path)

    try:
        with open(checkpoint, 'r') as f:
            checkpoint_file = json.load(f)
    except:
        checkpoint_file = {}
        checkpoint_file['tf_records_count'] = 0
    if name in checkpoint_file:
        progress = checkpoint_file[name]
    else:
        progress = {}


    if 'downloaded' not in progress:
        print("print starting download of:", name)
        subprocess.call(['wget', 'http://dl.yf.io/bdd-data/bdd100k/video_parts/'+name, '--directory-prefix', download_path])
        #time.sleep(1)
        progress['downloaded'] = True
        update_checkpoint(checkpoint, name, progress)

#    if 'moved' not in progress:
#        print("moving:", name)
#        shutil.move(name, "/NAS/scratch/karls/")
#        progress['moved'] = True
#        update_checkpoint(checkpoint, name, progress)

    if 'unzipped' not in progress:
        print("unzipping", name)
        subprocess.call(['unzip', os.path.join(download_path, name), '-d', download_path])
        progress['unzipped'] = True
        update_checkpoint(checkpoint, name, progress)

    if 'converted' not in progress or progress['converted']['finished'] == False:
        if 'converted' not in progress:
           video_index = 0
           progress['converted'] = {}
        else:
            video_index = progress['converted']['video_index']
        count = checkpoint_file['tf_records_count']
        print("converting", name, "starting at index", video_index)
        args.video_path = os.path.join(download_path, "bdd100k/videos")
        args.output_path = "/NAS/data/robotic_video/bdd_tf_100k_128x128"
        args.output_path_2 = "/NAS/data/robotic_video/bdd_tf_100k_64x48"
        args.image_size = [128, 128]
        args.start_count = count
        args.video_start_index = video_index
        args.time_remaining = max_time - (time.time() - start_time)
        args.traj_per_record = 10
        args.sequence_length = 20
        args.prefix = args.prefix
        finished, video_index, count = save_video(args)
        progress['converted']['finished'] = finished
        progress['converted']['video_index'] = video_index

        update_checkpoint(checkpoint, name, progress, count=count)

        if not finished:
            exit()

    if 'deleted' not in progress:
        print("deleting:", name)
        shutil.rmtree("/NAS/scratch/karls/bdd100k")
        os.remove("/NAS/scratch/karls/"+name)
        progress['deleted'] = True
        update_checkpoint(checkpoint, name, progress)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--mod", type=int, default=1)
    parser.add_argument("--mod_equals", type=int, default=0)
    args = parser.parse_args()

    dataset = {'train':(18, 70), 'val':(0, 10), 'test':(1, 20)}
    for mode in dataset:
        for i in range(dataset[mode][0], dataset[mode][1]):
            if i % args.mod == args.mod_equals:
                download_and_process("bdd100k_videos_{}_{:02d}.zip".format(mode, i), args)
            if time.time() - start_time > max_time:
                exit()

if __name__ == '__main__':
    main()
