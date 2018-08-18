import json
import argparse
import pdb
import os
import re
import pdb

def launch_job_func(hyper, options, int, test, name):

    data = {}
    start_dir = "/workspace/video_prediction/docker"

    data["aceName"] = "nv-us-west-2"
    data["command"] = \
    "cd /result && tensorboard --logdir . & \
     export VMPC_DATA_DIR=/mnt/pushing_data;\
     export TEN_DATA=/mnt/tensorflow_data;\
     export ALEX_DATA=/mnt/pretrained_models;\
     export RESULT_DIR=/result; " + \
     "cd /workspace/video_prediction; git checkout dev; git pull;" + \
     "cd {};".format(start_dir)

    data['dockerImageName'] = "ucb_rail8888/tf_mj1.5:latest"

    data["datasetMounts"] = [
        {"containerMountPoint": "/data/autograsp_newphysics_1", "id": 11701},
        {"containerMountPoint": "/data/autograsp_allobj_newphysics_1", "id": 11702},
        {"containerMountPoint": "/data/autograsp_bowls", "id": 11720},
                             ]

    ngpu = 1
    data["aceInstance"] = "ngcv{}".format(ngpu)
    if int== 'True':
        command = "/bin/sleep 360000"
        data["name"] = 'int' + name
    else:
        command = "CUDA_VISIBLE_DEVICES=0 python ../scripts/train.py " + " --conf " + hyper  + " " + options

    data["name"] = name
    data["command"] += command
    data["resultContainerMountPoint"] = "/result"
    data["publishedContainerPorts"] = [6006] #for tensorboard

    with open('autogen.json', 'w') as outfile:
        json.dump(data, outfile, indent=4)

    print('#######################')
    print('command', data["command"])

    if not bool(test):
        os.system("ngc batch run -f autogen.json")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='write json configuration for ngc')
    parser.add_argument('hyper', type=str, help='relative path to folder with hyperparams files', default="")
    parser.add_argument('--options', default='', type=str, help='options')
    parser.add_argument('--int', default='False', type=str, help='interactive')
    parser.add_argument('--name', default='', type=str, help='additional arguments')
    parser.add_argument('--test', default=0, type=int, help='testrun')
    args = parser.parse_args()

    launch_job_func(args.hyper, args.options, args.int, args.test, args.name)

