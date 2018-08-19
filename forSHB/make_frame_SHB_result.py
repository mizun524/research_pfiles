# calculation to show the SHB's result in the color map (for animation)

import numpy as np
import numpy.matlib

import matplotlib.pyplot as plt


def make_frame_SHB_result(order, freq, place, time_s, time_e):
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

    return bp_c

# test
if __name__ == '__main__':
    order = 3
    freq = 3000
    place = 1
    time_s = 1
    time_e = 1
    f1 = make_frame_SHB_result(order, freq, place, time_s, time_e)
