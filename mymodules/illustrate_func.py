# functions for illustration
#   mapping_frequency(hrtf, Fs)
#
#

import numpy as np
import matplotlib.pyplot as plt


def mapping_frequency(hrtf, Fs):
    hrir_len = hrtf.shape[1]
    # xticks : 1000, 2000, 4000, 8000, 16000
    xticks = np.array([(2**n) * 1000 / (Fs / hrir_len) for n in range(5)])
    # yticks : 0*5-175, 17*5-175, 35*5-175, 53*5-175, 71*5-175
    yticks = np.array([0, 17, 35, 53, 71])
    # plot hrtf
    fig = plt.subplot()
    plt.imshow(np.abs(hrtf[:, :int(hrir_len / 2) + 1]), aspect='auto', cmap='jet')
    plt.colorbar()
    ax = plt.gca()
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels(np.array([(2**n) for n in range(5)]))
    ax.set_yticklabels(yticks * 5 - 175)
    plt.xlabel('Frequency [kHz]')
    plt.ylabel('azimuth [°]')
    # save
    return fig


# mapping_frequency(hrtf, Fs)
if __name__ == '__main__':
    from scipy.fftpack import fft

    hrir_len = 512
    hrirL0 = np.zeros((72, hrir_len))
    hrirR0 = np.zeros((72, hrir_len))
    line = 0
    for azi in range(-175, 180 + 5, 5):
        filename = "/Users/mizun524/research/hrtfdata/imp_allSP_kemar/e0/L{}.dat".format(
            azi)
        hrirL0[line, :] = np.fromfile(filename, np.double)
        filename = "/Users/mizun524/research/hrtfdata/imp_allSP_kemar/e0/R{}.dat".format(
            azi)
        hrirR0[line, :] = np.fromfile(filename, np.double)
        line = line + 1
    hrtfL0 = fft(hrirL0, axis=1)
    hrtfR0 = fft(hrirR0, axis=1)
    fig = mapping_frequency(hrtfR0, 44100)
    plt.title('HRTF Left')
