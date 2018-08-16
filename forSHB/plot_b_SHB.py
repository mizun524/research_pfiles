# plot b(ka) in SHB

import numpy as np

from scipy.special import sph_harm, spherical_jn, spherical_yn

import matplotlib.pyplot as plt


def calc_j(n, k, a):
    shankel = spherical_jn(n, k * a, derivative=False) + \
        1j * spherical_yn(n, k * a, derivative=False)
    shankel_d = spherical_jn(n, k * a, derivative=True) + \
        1j * spherical_yn(n, k * a, derivative=True)
    j = 4 * np.pi * np.power(1j, n) * (spherical_jn(n, k * a, derivative=False) -
                                       spherical_jn(n, k * a, derivative=True) / shankel_d * shankel)
    return j


freq = np.linspace(100, 10000, 100)
c = 347   # sound speed
k = 2 * np.pi * freq / c
a = 0.05

plt.figure()
ax1 = plt.subplot(111)
#ax2 = plt.subplot(212)
for order in range(7):
    b = calc_j(order, k, a)
    ax1.plot(freq, np.abs(b), label='order{}'.format(order))
    #ax2.plot(freq, np.angle(b), label='order{}'.format(order))
ax1.legend()
plt.savefig("images/plot_b")
#ax2.legend()
