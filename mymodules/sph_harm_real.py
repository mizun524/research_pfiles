# calculate Real Spherical harmonics

from scipy.special import sph_harm
import cmath

# theta:azimuth phi:polar
def sph_harm_real(m, n, theta, phi):
    if m > 0:
        y = 1 / cmath.sqrt(2) * (sph_harm(m, n, theta, phi) +
                                 (-1)**m * sph_harm(-m, n, theta, phi))
    elif m == 0:
        y = sph_harm(0, n, theta, phi)
    elif m < 0:
        y = 1 / cmath.sqrt(2) / 1j * (sph_harm(m, n, theta, phi) -
                                      (-1)**m * sph_harm(-m, n, theta, phi))
    return y.real

if __name__ == '__main__':
    import numpy as np
    Y = np.array([sph_harm_real(m, n, cmath.pi/6, cmath.pi/3)
                    for n in range(3 + 1) for m in range(-n, n + 1)])
    print(Y)
    Y2 = np.array([sph_harm(m, n, cmath.pi/6, cmath.pi/3)
                    for n in range(3 + 1) for m in range(-n, n + 1)])
    print(Y2)
