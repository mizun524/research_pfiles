# show the beampattern in the color map

import numpy as np
import numpy.matlib

import matplotlib.pyplot as plt

# 0. values
order = 3
freq = [500, 1000, 2000, 4000, 6000, 8000]
place = [1]
time_s = 7
time_e = 7

for ip in place:
    for ifreq in freq:
        # 1. import data
        fname = "beamp_o{}/beamp_{}Hz_P{:02d}.npz".format(order, ifreq, ip)
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
        azi = np.arange(.0, 2. * np.pi + np.pi / 60. * 0.1, np.pi / 60.)
        # elevation every pi/60 rad (0~pi)
        ele = np.arange(.0, np.pi + np.pi / 60. * 0.1, np.pi / 60.)
        # reshape for color mapping
        bp_c = np.reshape(beamp_g_mean, (len(ele), len(azi)), order='C')
        bp_c = np.flipud(np.flipud(bp_c))

        # 4. mapping
        X, Y = np.meshgrid(azi, ele)
        fig = plt.figure()
        # colormap
        plt.pcolormesh(X, Y, np.hstack(
            (bp_c[-1::-1, 90::-1], bp_c[-1::-1, -1:90:-1])), cmap='hot')
        # colorbar
        plt.colorbar(orientation='vertical')
        # labels
        plt.title('{}Hz time({}-{})'.format(ifreq, time_s * 128, time_e * 128))
        plt.xlabel('azimuth', fontsize=14)
        plt.ylabel('elevation', fontsize=14)
        plt.axis('equal')

        # plot the correct direct sound position (x)
        if ip == 1:
            plt.plot(8. * np.pi / 180. + np.pi, 85. * np.pi / 180., marker='o')
        elif ip == 2:
            plt.plot(6.5 * np.pi / 180. + np.pi, 80. * np.pi / 180., marker='o')
        elif ip == 3:
            plt.plot(4. * np.pi / 180. + np.pi, 78. * np.pi / 180., marker='o')

        plt.savefig(
            "images/o{}_{}Hz_P{:02d}_{}_{}.png".format(order, ifreq, ip, time_s * 128, time_e * 128))
