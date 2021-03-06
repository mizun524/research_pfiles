# show the SHB's result in the color map

import numpy as np
import numpy.matlib

import matplotlib.pyplot as plt


def mapping_SHB_result(order, freq, place, time_s, time_e, ax):
    # 1. import data
    fname = "beamp_o{}/beamp_{}Hz_P{:02d}.npz".format(order, freq, place)
    variable = np.load(fname, 'r')
    beamp_all = variable['beamp']
    # remove the noise
    beamp = beamp_all[:, :430]

    # 2. remove the decay
    # the mean of the beampattern
    beamp_mean = np.mean(beamp, axis=0)
    # decay curve E
    E = np.zeros_like(beamp_mean)
    E[0] = np.sum(np.power(beamp_mean, 2), axis=0)
    for t in range(len(E) - 1):
        E[t + 1] = E[t] - np.power(beamp_mean[t], 2)
    # remove the decay
    beamp_g = beamp / np.matlib.repmat(np.sqrt(E), beamp.shape[0], 1)

    # 3. adjust beampattern
    beamp_g_mean = np.mean(beamp_g[:, time_s:time_e + 1], axis=1)
    # observation dirs
    # azimuth every pi/60 rad (0~2pi)
    azi = np.arange(-np.pi, np.pi + np.pi / 60. * 0.1, np.pi / 60.)
    # elevation every pi/60 rad (0~pi)
    ele = np.arange(np.pi, 0 - np.pi / 60. * 0.1, -np.pi / 60.)
    # reshape for color mapping
    bp_c = np.reshape(beamp_g_mean, (len(ele), len(azi)), order='C')

    # 4. mapping
    X, Y = np.meshgrid(azi, ele)
    # colormap
    ax.pcolormesh(X, Y, np.hstack(
        (bp_c[:, 90::-1], bp_c[:, -1:90:-1])), cmap='jet')
    # colorbar
    # ax.colorbar(orientation='vertical')
    # labels
    ax.set_title('{}Hz time({}-{})'.format(freq, time_s * 128, time_e * 128))
    ax.set_xlabel('azimuth', fontsize=14)
    ax.set_ylabel('elevation', fontsize=14)
    ax.set_aspect('equal')

    # plot the correct direct sound position (x)
    if place == 1:
        ax.plot(8. * np.pi / 180., 85. * np.pi / 180., marker='x')
    elif place == 2:
        ax.plot(6.5 * np.pi / 180., 80. * np.pi / 180., marker='x')
    elif place == 3:
        ax.plot(4. * np.pi / 180., 78. * np.pi / 180., marker='x')

    # ax.savefig(
    #    "images/o{}_{}Hz_P{:02d}_{}_{}.png".format(order, freq, ip, time_s * 128, time_e * 128))

# test
if __name__ == '__main__':
    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    mapping_SHB_result(3, 4000, 3, 20, 20, ax1)
    mapping_SHB_result(3, 4000, 3, 100, 200, ax2)
    mapping_SHB_result(3, 4000, 3, 200, 300, ax3)
    mapping_SHB_result(3, 4000, 3, 300, 400, ax4)
    fig.tight_layout()
