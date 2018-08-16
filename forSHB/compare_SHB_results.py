# compare SHB's results

import numpy as np

from mapping_SHB_result import mapping_SHB_result

import matplotlib.pyplot as plt


# values
freq = np.array([1000,1000,1000,1000])
time = np.array([150,250,200,300,250,350,300,400])
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
mapping_SHB_result(order, freq[0], p, time[0], time[1], ax1)
mapping_SHB_result(order, freq[1], p, time[2], time[3], ax2)
mapping_SHB_result(order, freq[2], p, time[4], time[5], ax3)
mapping_SHB_result(order, freq[3], p, time[6], time[7], ax4)
fig.tight_layout()

# freq
#plt.savefig("images/SHB_res_p{}_o{}_t{}_f{}_{}.png".format(p, order, time[0], freq[0], freq[-1]))
# time
plt.savefig("images/SHB_res_p{}_o{}_f{}_t{}_{}.png".format(p, order, freq[0], time[0], time[-1]))
