# calculation the decay of SHB_result
# plot decay curve E

import numpy as np
import matplotlib.pyplot as plt


def plot_SHB_result_decay(order, freq, place, time_s, time_e, ax):
    # import data
    fname = "beamp_o{}/beamp_{}Hz_P{:02d}.npz".format(order, freq, place)
    variable = np.load(fname, 'r')
    beamp_all = variable['beamp']
    beamp = beamp_all[:, :430]  # remove the noise

    # the mean of the beampattern
    beamp_mean = np.mean(beamp, axis=0)
    # decay curve E
    E = np.zeros_like(beamp_mean)
    E[0] = np.sum(np.power(beamp_mean, 2), axis=0)
    for t in range(len(E) - 1):
        E[t + 1] = E[t] - np.power(beamp_mean[t], 2)
    E_lv = 10 * np.log10(E / E[0])

    # plot
    time = (np.arange(430) + 1.) * 128. / 48000. * 1000.
    ax.plot(time, E_lv)
    ax.plot(time[time_s:time_e], E_lv[time_s:time_e], 'r')
    t_s = np.ones(100) * time[time_s]
    t_e = np.ones(100) * time[time_e]
    E_line = np.linspace(-100., 5., 100)
    ax.plot(t_s, E_line, 'r--')
    ax.plot(t_e, E_line, 'r--')
    E_s = np.ones(100) * E_lv[time_s]
    E_e = np.ones(100) * E_lv[time_e]
    tE_s = np.linspace(0., time[time_s], 100)
    tE_e = np.linspace(0., time[time_e], 100)
    ax.plot(tE_s, E_s, 'g--')
    ax.plot(tE_e, E_e, 'g--')
    ax.set_xlim([0, 430. * 128. / 48000. * 1000.])
    ax.set_ylim([-100., 5.])
    ax.set_xlabel('time [ms]')
    ax.set_ylabel('E [dB]')
    ax.set_title("Energy decay curve (order{} {}Hz)".format(order, freq))


if __name__ == '__main__':
    import itertools

    order = np.arange(2, 7)
    freq = np.arange(0, 9000, 500)
    place = 3
    for iorder, ifreq in itertools.product(order, freq):
        # figure
        fig = plt.figure()
        # ax
        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        # plot
        plot_SHB_result_decay(iorder, ifreq, place, 100, 200, ax1)
        plot_SHB_result_decay(iorder, ifreq, place, 150, 250, ax2)
        plot_SHB_result_decay(iorder, ifreq, place, 200, 300, ax3)
        plot_SHB_result_decay(iorder, ifreq, place, 250, 350, ax4)
        fig.tight_layout()
        plt.savefig("images/plot_decay/E_p{}_o{}_f{}_t{}_{}".format(place,
                                                                    iorder, ifreq, time_s, time_e))
        plt.close()
