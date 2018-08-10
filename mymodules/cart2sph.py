# transform from Cartesian to Spherical Coordinate
# func : phi, theta, r = cart2sph(x, y, z)
#
#reference:
#   https://en.wikipedia.org/wiki/Spherical_coordinate_system
#

import numpy as np

def cart2sph(x, y, z):

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)

    return phi, theta, r

if __name__ == '__main__':
    cdirs = np.array([-1, 0.001, 1])
    sdirs = cart2sph(cdirs[0], cdirs[1], cdirs[2])
    print(sdirs)
