# transform from Spherical to Cartesian Coordinate
# func : x, y, z = cart2sph(phi, theta, r)
#
#reference:
#   https://en.wikipedia.org/wiki/Spherical_coordinate_system
#

import numpy as np

def sph2cart(phi, theta, r):

    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)

    return x, y, z

if __name__ == '__main__':
    sdirs = np.array([np.pi/2, np.pi/2, 2])
    cdirs = sph2cart(sdirs[0], sdirs[1], sdirs[2])
    print(cdirs)
