import matplotlib.pyplot as plt

import numpy as np
from matplotlib import rcParams
rcParams['axes.titlepad'] = 20

plt.figure()

ngpus = np.array([1,2,4,8])
bsizes = 16*ngpus
avgt = []

for i, g in enumerate(ngpus):
    indexlist = [str(i_gpu) for i_gpu in range(g)]
    gpustr = ','.join(indexlist)
    bsize = bsizes[i]

    file = '../weak/timeb{}_{}.txt'.format(bsize, gpustr)
    with open(file) as input_file:
        for line in input_file:
            num = float(line)
    avgt.append(num)


ideal = np.array([avgt[0] for _ in range(4)])

tper_ex = avgt[0]/bsizes[0]
serial = np.array([tper_ex*n for n in bsizes])

fig = plt.figure()
ax1 = fig.add_subplot(111)

ax1.plot(ngpus, avgt, '-bs', label='Parallel implementation')
ax1.plot(ngpus, ideal, '--ks', label='Ideal scaling')
ax1.plot(ngpus, serial, '--rs', label='Serial')
ax1.set_xlim(0, 8)

ax2 = ax1.twiny()

ax2.set_xticks(ngpus)
ax2.set_xticklabels(bsizes)
# ax2.cla()
ax1.legend()

ax1.set_xlabel('number of GPUs')
ax1.set_ylabel('Average time per Iteration [s]')
plt.title("Weak scaling")
# ttl = ax1.title
# ttl.set_position([.5, 1.25])

# plt.show()
plt.savefig('weakscaling.png')





