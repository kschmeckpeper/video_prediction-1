# strong scaling

import numpy as np
import os

ngpus = np.array([1,2,4,8])
bsizes = 16*ngpus
for i, g in enumerate(ngpus):
    indexlist = [str(i_gpu) for i_gpu in range(g)]
    gpustr = ','.join(indexlist)
    bsize = bsizes[i]
    cmd = 'CUDA_VISIBLE_DEVICES={} python scripts/train.py --input_dir /mnt/pushing_data/cartgripper_updown_sact/train --dataset cartgripper --model savp --model_hparams_dict hparams/bair/ours_deterministic_l1/model_hparams.json --model_hparams tv_weight=0.001,transformation=flow,last_frames=2,generate_scratch_image=false,batch_size={} --summary_freq 10 --timing_file timeb{}_{}.txt'.format(gpustr, bsize, bsize, gpustr)

    print(cmd)
    os.system(cmd)