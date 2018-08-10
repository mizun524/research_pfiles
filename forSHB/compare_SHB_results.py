# compare SHB's results

import numpy as np

from mapping_SHB_result import mapping_SHB_result

import matplotlib.pyplot as plt


# values
freq = np.array([2000,3500,3500,4000])
p = 3
order = 2
# figure
fig = plt.figure()
# ax
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
# plot
mapping_SHB_result(order, freq[0], p, 20, 20, ax1)
mapping_SHB_result(order, freq[1], p, 20, 20, ax2)
mapping_SHB_result(order, freq[2], p, 20, 20, ax3)
mapping_SHB_result(order, freq[3], p, 20, 20, ax4)
fig.tight_layout()
