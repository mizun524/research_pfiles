# show the SHB's beampattern by simulation

import numpy as np

from scipy.special import sph_harm, spherical_jn, spherical_yn

import matplotlib.pyplot as plt

# 0. values
order = 3
freq_analyse = np.array([8000])

# 1. calculate SH
# calculate SH at the desired direction
Y_mic = np.array([sph_harm(m, n, 0, np.pi / 2)
                  for n in range(order + 1) for m in range(-n, n + 1)])

# observation dirs
# azimuth every pi/60 rad (-pi~pi)
azi = np.arange(-np.pi, np.pi + np.pi / 60. * 0.1, np.pi / 60.)
# elevation every pi/60 rad (0~pi)
ele = np.arange(.0, np.pi + np.pi / 60. * 0.1, np.pi / 60.)
len_dirs = len(azi) * len(ele)
dirs_obs = np.zeros((len_dirs, 2))
# dirs_obs[:, 0](phi) : -pi~pi, -pi~pi, ..., -pi~pi
dirs_obs[:, 0] = np.matlib.repmat(azi, 1, len(ele))
# dirs_obs[:, 1](theta) : 0, 0, ..., 0, pi/60, ..., 2pi, 2pi
dirs_obs[:, 1] = np.array(
    [np.floor(n / len(azi)) * np.pi / 60 for n in range(len_dirs)])
# calculate SH at the observation points
Y_obs = np.array([sph_harm(m, n, dirs_obs[:, 0], dirs_obs[:, 1])
                  for n in range(order + 1) for m in range(-n, n + 1)])

# 2. Beamforming
# function for calculating mode strength


def calc_j(n, k, a):
    shankel = spherical_jn(n, k * a, derivative=False) + \
        1j * spherical_yn(n, k * a, derivative=False)
    shankel_d = spherical_jn(n, k * a, derivative=True) + \
        1j * spherical_yn(n, k * a, derivative=True)
    j = 4 * np.pi * np.power(1j, n) * (spherical_jn(n, k * a, derivative=False) -
                                       spherical_jn(n, k * a, derivative=True) / shankel_d * shankel)
    return j


for ifreq in freq_analyse:

    # mode strength b(ka)
    c = 347  # sound speed
    k = 2 * np.pi * ifreq / c
    b = np.array([calc_j(int(np.floor(np.sqrt(nn))), k, mPos_sph[0, 2])
                  for nn in range((order + 1)**2)])
    # beamforming direction
    pnm1 = Y_mic
    # SH coefficients for the beampattern
    bnm1 = pnm1 / b
    # beampattern
    beamp1 = np.abs(np.dot(bnm1, Y_obs))

    # 3. mapping
    # reshape
    bp_c = np.reshape(beamp1, (len(ele), len(azi)), order='C')
    # grid
    X, Y = np.meshgrid(azi, ele)
    fig = plt.figure()
    # colormap
    plt.pcolormesh(X, Y, bp_c, cmap='hot')
    # colorbar
    plt.colorbar(orientation='vertical')
    # labels
    plt.title('order{} {}Hz'.format(order, ifreq))
    plt.xlabel('azimuth', fontsize=14)
    plt.ylabel('elevation', fontsize=14)
    plt.axis('equal')
