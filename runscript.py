# strong scaling

import numpy as np
import os

ngpus = np.array([1,2,4,8])
for g in ngpus:
    indexlist = [str(i_gpu) for i_gpu in range(g)]
    gpustr = ','.join(indexlist)
    cmd = 'CUDA_VISIBLE_DEVICES={} python scripts/train.py --input_dir /mnt/pushing_data/cartgripper_updown_sact/train --dataset cartgripper --model savp --model_hparams_dict hparams/bair/ours_deterministic_l1/model_hparams.json --model_hparams tv_weight=0.001,transformation=flow,last_frames=2,generate_scratch_image=false,batch_size=16 --summary_freq 10 --timing_file time{}.txt'.format(gpustr, gpustr)

    print(cmd)
    # os.system(cmd)