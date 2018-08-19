# introduce microphone's inputs, calculate SHB's beampattern and show the result

import numpy as np

from scipy import io
from scipy.special import sph_harm, spherical_jn, spherical_yn

import matplotlib.pyplot as plt

import sys
sys.path.append('/Users/mizun524/research/pfiles')
from mymodules import cart2sph, sph_harm_real

# 0. values
order = 3
order_in = 40
freq = 4000
inc_dir = np.array([[0, np.pi / 2], [np.pi / 2, np.pi / 2]])
xdb = 0

# 1. import data
# import the micophones arrangement
mPos_mat = io.loadmat("../../ViRealMic/ViRealMic.mat")
mPos_cart = np.array(mPos_mat["Pmic"])
# cartesian to spherical coordinate
mPos_sph = np.zeros_like(mPos_cart)
mPos_sph[:, 0], mPos_sph[:, 1], mPos_sph[:, 2] = cart2sph(
    mPos_cart[:, 0], mPos_cart[:, 1], mPos_cart[:, 2])

# 2. introduce microphone's inputs
# function for calculating mode strength


def calc_j(n, k, a):
    shankel = spherical_jn(n, k * a, derivative=False) + \
        1j * spherical_yn(n, k * a, derivative=False)
    shankel_d = spherical_jn(n, k * a, derivative=True) + \
        1j * spherical_yn(n, k * a, derivative=True)
    j = 4 * np.pi * np.power(1j, n) * (spherical_jn(n, k * a, derivative=False) -
                                       spherical_jn(n, k * a, derivative=True) / shankel_d * shankel)
    return j


# mode strength b(ka)
c = 347  # sound speed
k = 2 * np.pi * freq / c
b = np.array([calc_j(int(np.floor(np.sqrt(nn))), k, mPos_sph[0, 2])
              for nn in range((order_in + 1)**2)])

# calculate SH at the positions of microphones
Y_mic_in = np.array([sph_harm(m, n, mPos_sph[:, 0], mPos_sph[:, 1])
                  for n in range(order_in + 1) for m in range(-n, n + 1)])

p_in = np.zeros((mPos_sph.shape[0]), dtype='complex')
for i in range(inc_dir.shape[0]):
    # calculate SH for incoming wave
    Y_in = np.array([sph_harm(m, n, inc_dir[i, 0], inc_dir[i, 1])
                     for n in range(order_in + 1) for m in range(-n, n + 1)])

    # calculate microphone's input each order and degree
    pnm_in = np.array([b * Y_mic_in[:, idirs] * np.conjugate(Y_in)
                    for idirs in range(mPos_sph.shape[0])])
    # calculate microphone's input
    if i==0:
        p_in += np.sum(pnm_in, axis=1)
    else:
        p_in += np.sum(pnm_in, axis=1) * np.power(10, xdb/20)

# add noise
p_noise = np.random.rand(64) * np.max(p_in) * 3
#p_in += p_noise

# 3. calculate SHB's beampattern
# observation dirs
# azimuth every pi/60 rad (0~2pi)
azi = np.arange(-np.pi, np.pi + np.pi / 60. * 0.1, np.pi / 60.)
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

# calculate SH at the positions of microphones
Y_mic = np.array([sph_harm(m, n, mPos_sph[:, 0], mPos_sph[:, 1])
                  for n in range(order + 1) for m in range(-n, n + 1)])

# mode strength b(ka)
c = 347  # sound speed
k = 2 * np.pi * freq / c
b = np.array([calc_j(int(np.floor(np.sqrt(nn))), k, mPos_sph[0, 2])
              for nn in range((order + 1)**2)])

# calculate the beampattern
pnm = np.dot(p_in, np.linalg.pinv(Y_mic))
# SH coefficients for the beampattern
bnm = pnm / b
# beampattern
beamp = np.abs(np.dot(bnm, Y_obs))

# 4. show the SHB's result
# reshape for color mapping
bp_c = np.reshape(beamp, (len(ele), len(azi)), order='C')

# mapping
X, Y = np.meshgrid(azi, ele)
# colormap
plt.pcolormesh(X, Y, bp_c, cmap='jet')
# colorbar
# ax.colorbar(orientation='vertical')
# labels
plt.title('order{} {}Hz'.format(order, freq))
plt.xlabel('azimuth', fontsize=14)
plt.ylabel('elevation', fontsize=14)
plt.colorbar()
plt.axis('equal')
plt.plot(inc_dir[:, 0] , inc_dir[:, 1], marker='x', markersize='14',linestyle='none')
plt.savefig("images/sim_2s_o{}_azi{}".format(order, int(inc_dir[1, 0]/np.pi*180)))
