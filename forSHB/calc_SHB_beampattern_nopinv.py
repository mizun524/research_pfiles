# calculate the beampattern of Spherical Harmonics Beamforming not using pinv
#   not beampattern but the spatial distribution(?)
#
# importdata:
#   "../ViRealMic/ViRealMic1_lvl.mat"  - for calibration
#   "../ViRealMic/ViRealMic.mat"  - microphone's position
#   "../ViRealMic/HardyHall_P01_FLT32_avg.wav"
#   "../ViRealMic/HardyHall_P02_FLT32_avg.wav"
#   "../ViRealMic/HardyHall_P03_FLT32_avg.wav"
#
# outputdata:
#   "variables_o{}/beamp_{}Hz_P{}.npz"
#
# Reference:
#   "Room reflections analysis with the use of spherical beamforming and wavelets"
#       Use not wavelets but STFT in this program.
#

import numpy as np
import numpy.matlib

from scipy import io
from scipy.fftpack import fft, ifft
from scipy.signal import stft
from scipy.special import sph_harm, spherical_jn, spherical_yn

import soundfile as sf

from mymodules import cart2sph, sph_harm_real

# 0. values
order = 2
freq_analyse = np.array([2000, 4000])


# 1. import data
# import lvlLin for calibration
mic_lvl_mat = io.loadmat(
    "../ViRealMic/ViRealMic1_lvl.mat")
lvlLin = np.array(mic_lvl_mat["lvlLin"])

# import the micophones arrangement
mPos_mat = io.loadmat("../ViRealMic/ViRealMic.mat")
mPos_cart = np.array(mPos_mat["Pmic"])
# cartesian to spherical coordinate
mPos_sph = np.zeros_like(mPos_cart)
mPos_sph[:, 0], mPos_sph[:, 1], mPos_sph[:, 2] = cart2sph(
    mPos_cart[:, 0], mPos_cart[:, 1], mPos_cart[:, 2])

# import mesured IRs (P1, P2, P3)
y1, Fs = sf.read("../ViRealMic/HardyHall_P01_FLT32_avg.wav")
y2, Fs = sf.read("../ViRealMic/HardyHall_P02_FLT32_avg.wav")
y3, Fs = sf.read("../ViRealMic/HardyHall_P03_FLT32_avg.wav")

# calibration (level adjustment among channels)
y1_c = np.zeros_like(y1)
y2_c = np.zeros_like(y2)
y3_c = np.zeros_like(y3)
for ich in range(y1.shape[1]):
    y1_c[:, ich] = y1[:, ich] * lvlLin[ich]
    y2_c[:, ich] = y2[:, ich] * lvlLin[ich]
    y3_c[:, ich] = y3[:, ich] * lvlLin[ich]


# 2. STFT
nperseg = 256
noverlap = 128
nfft = 256
flen = int(nfft / 2) + 1
tlen = int(np.floor(y1.shape[0] / noverlap)) + 1
s1 = np.zeros([flen, tlen, y1.shape[1]], dtype=complex)
s2 = np.zeros([flen, tlen, y2.shape[1]], dtype=complex)
s3 = np.zeros([flen, tlen, y3.shape[1]], dtype=complex)
for ich in range(y1.shape[1]):
    freq_f, time_f, s1[:, :, ich] = stft(
        y1_c[:, ich], Fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    freq_f, time_f, s2[:, :, ich] = stft(
        y2_c[:, ich], Fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)
    freq_f, time_f, s3[:, :, ich] = stft(
        y3_c[:, ich], Fs, nperseg=nperseg, noverlap=noverlap, nfft=nfft)


# 3. SHB
# calculate SH at the positions of microphones
Y_mic = np.array([sph_harm(m, n, mPos_sph[:, 0], mPos_sph[:, 1])
                  for n in range(order + 1) for m in range(-n, n + 1)])

# observation dirs
# azimuth every pi/60 rad (0~2pi)
azi = np.arange(.0, 2. * np.pi + np.pi / 60. * 0.1, np.pi / 60.)
# elevation every pi/60 rad (0~pi)
ele = np.arange(.0, np.pi + np.pi / 60. * 0.1, np.pi / 60.)
len_dirs = len(azi) * len(ele)
dirs_obs = np.zeros((len_dirs, 2))
# dirs_obs[:, 0](phi) : 0~2pi, 0~2pi, ..., 0~2pi
dirs_obs[:, 0] = np.matlib.repmat(azi, 1, len(ele))
# dirs_obs[:, 1](theta) : 0, 0, ..., 0, pi/60, ..., 2pi, 2pi
dirs_obs[:, 1] = np.array(
    [np.floor(n / len(azi)) * np.pi / 60 for n in range(len_dirs)])
# calculate SH at the observation points
Y_obs = np.array([sph_harm(m, n, dirs_obs[:, 0], dirs_obs[:, 1])
                  for n in range(order + 1) for m in range(-n, n + 1)])

# function for calculating mode strength


def calc_j(n, k, a):
    shankel = spherical_jn(n, k * a, derivative=False) + \
        1j * spherical_yn(n, k * a, derivative=False)
    shankel_d = spherical_jn(n, k * a, derivative=True) + \
        1j * spherical_yn(n, k * a, derivative=True)
    j = 4 * np.pi * np.power(1j, n) * (spherical_jn(n, k * a, derivative=False) -
                                       spherical_jn(n, k * a, derivative=True) / shankel_d * shankel)
    return j


# calculate the beampattern for each frequency
for ifreq in freq_analyse:
    # find the nearest value of freq_analyse
    idx = np.abs(freq_f - ifreq).argmin()
    ifreq_real = freq_f[idx]

    # mode strength b(ka)
    c = 347  # sound speed
    k = 2 * np.pi * ifreq_real / c
    b = np.array([calc_j(int(np.floor(np.sqrt(nn))), k, mPos_sph[0, 2])
                  for nn in range((order + 1)**2)])

    beamp1 = np.zeros([len_dirs, len(time_f)])
    beamp2 = np.zeros([len_dirs, len(time_f)])
    beamp3 = np.zeros([len_dirs, len(time_f)])
    for itime in range(len(time_f)):
        # SH coeffients for the w
        wnm = np.conjugate(Y_mic) / np.matlib.repmat(b, Y_mic.shape[1], 1).T
        # the weighting of each microphone w
        w = np.dot(wnm.T, Y_obs)
        # beampattern
        beamp1[:, itime] = np.abs(np.dot(s1[idx, itime, :], w))
        beamp2[:, itime] = np.abs(np.dot(s2[idx, itime, :], w))
        beamp3[:, itime] = np.abs(np.dot(s3[idx, itime, :], w))

    # save beampattern
    fname1 = "beamp_o{}/beamp_{}Hz_P01.npz".format(order, ifreq)
    np.savez(fname1, beamp=beamp1)
    fname2 = "beamp_o{}/beamp_{}Hz_P02.npz".format(order, ifreq)
    np.savez(fname2, beamp=beamp2)
    fname3 = "beamp_o{}/beamp_{}Hz_P03.npz".format(order, ifreq)
    np.savez(fname3, beamp=beamp3)
