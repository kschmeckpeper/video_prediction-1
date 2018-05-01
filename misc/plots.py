import matplotlib.pyplot as plt

import numpy as np
plt.figure()

ngpus = np.array([1,2,4,8])
avgt = []

plt.bar(ngpus, avgt, 0.5, facecolor='b', alpha=0.5)
plt.xlabel('number of GPUs')
plt.ylabel('Average time per Iteration [s] (batch 16)')

plt.title("Strong scaling")




