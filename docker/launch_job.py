import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='experiment')
parser.add_argument('command', type=str)
parser.add_argument('--num_gpus', type=int, default=1)
parser.add_argument('--int', type=str, default='False')
args = parser.parse_args()

print(args.command)

command_split = args.command.split(' ')

if command_split[0].startswith('CUDA_VISIBLE_DEVICES='):
    cuda_visible_devices = command_split[0].replace('CUDA_VISIBLE_DEVICES=', '')
    if cuda_visible_devices == '':
        num_gpus = 0
    else:
        num_gpus = len(cuda_visible_devices.split(','))
else:
    num_gpus = 1
assert num_gpus in (1, 2, 4, 8)

"""
wget http://people.eecs.berkeley.edu/~alexlee_gk/projects/savp/ucf101.tar; \
tar -xf ucf101.tar -C logs/ --no-same-owner; \
rm ucf101.tar; \
"""
"""
wget http://people.eecs.berkeley.edu/~alexlee_gk/projects/savp/gan.tar; \
mkdir logs/ucf101; \
tar -xf gan.tar -C logs/ucf101/ --no-same-owner; \
rm gan.tar; \
"""
"""
wget http://people.eecs.berkeley.edu/~alexlee_gk/projects/savp/vae_gan.tar; \
mkdir logs/ucf101; \
tar -xf vae_gan.tar -C logs/ucf101/ --no-same-owner; \
rm vae_gan.tar; \
"""


cmd = args.command

if args.int == 'True':
    cmd = "/bin/sleep 360000"

data = {}
data['dockerImageName'] = "ucb_rail8888/video_prediction_image:0.1"
data["aceName"] = "nv-us-west-2"
data["name"] = args.name
# git clone git@github.com:alexlee-gk/video_prediction.git /video_prediction; \
data["command"] = """\
git clone -b dev --single-branch https://github.com/febert/video_prediction-1.git /video_prediction-1; \
cd /video_prediction-1; \
pip install -r requirements.txt; \
ln -s /logs logs; \
ln -s /data/ag_scripted_longtraj data/ag_scripted_longtraj; \
ln -s /data/ag_reopen_records data/ag_reopen_records; \
tensorboard --logdir logs  & \
export PYTHONPATH=/video_prediction-1; \
{0}\
""".format(cmd)

data["datasetMounts"] = [
    {"containerMountPoint": "/data/ag_scripted_longtraj", "id": 10217},
    {"containerMountPoint": "/data/ag_reopen_records", "id": 10401}]

assert data["datasetMounts"]
data["resultContainerMountPoint"] = "/logs"
data["aceInstance"] = "ngcv%d" % num_gpus
data["publishedContainerPorts"] = [6006]

with open('autogen.json', 'w') as outfile:
    json.dump(data, outfile, sort_keys=True,
              indent=4, separators=(', ', ': '))

os.system("ngc batch run -f autogen.json")