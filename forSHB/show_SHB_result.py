# show mapping and decay curve
# values: place, freq, order, time

import numpy as np

from itertools import product

from mapping_SHB_result import mapping_SHB_result
from plot_SHB_result_decay import plot_SHB_result_decay

import matplotlib.pyplot as plt

# values

place_idx = np.arange(3) + 1
freq_idx = np.arange(16) * 500 + 500
order_idx = np.arange(5) + 2
t=20
p=3


for f, o in product(freq_idx, order_idx):
    # figure
    fig = plt.figure()
    # ax
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    mapping_SHB_result(o, f, p, t, t, ax1)
    plot_SHB_result_decay(o, f, p, t, t, ax2)
    fig.tight_layout()
    plt.savefig("images/res_p{}o{}f{}t{}-{}".format(p, o, f, t, t))

    plt.close()
