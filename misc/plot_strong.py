import matplotlib.pyplot as plt

import numpy as np
plt.figure()

b16_ngpus = np.array([1,2,4,8])
b32_ngpus = np.array([1,2,4,8])
b64_ngpus = np.array([2,4,8])
b128_ngpus = np.array([4,8])

bsizes = [16, 32, 64, 128]
ngpu_ranges = [b16_ngpus, b32_ngpus, b64_ngpus, b128_ngpus]

plt.figure()

colorlist = ['r', 'g', 'b', 'y']

for idata, bsize, nrng in zip(range(4), bsizes, ngpu_ranges):
    avgt = []
    for i, g in enumerate(nrng):
        indexlist = [str(i_gpu) for i_gpu in range(g)]
        gpustr = ','.join(indexlist)

        file = '../strong/b{}/time{}.txt'.format(bsize, gpustr)
        with open(file) as input_file:
            for line in input_file:
                num = float(line)
        avgt.append(num)

    plt.plot(nrng, avgt, '-{}s'.format(colorlist[idata]), label='Batchsize {}'.format(bsize))
    ideal = np.array([avgt[0]*nrng[0]/float(n) for n in nrng])
    plt.plot(nrng, ideal, '--{}s'.format(colorlist[idata]), label='Ideal batchsize {}'.format(bsize))

plt.legend(loc='upper left')

plt.xlabel('number of GPUs')
plt.ylabel('Average time per Iteration [s]')
plt.title("Strong scaling")
# plt.show()

plt.savefig("strongscaling.png")







