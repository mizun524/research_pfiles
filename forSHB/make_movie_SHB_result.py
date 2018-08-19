# make a movie of SHB's result

import numpy as np

from make_frame_SHB_result import make_frame_SHB_result

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# value
p = 3
order = 4
freq = 4000
tlen=400

bp_c_0 = make_frame_SHB_result(order, freq, p, 0, 0)
G = np.zeros((bp_c_0.shape[0], bp_c_0.shape[1], tlen))
G.shape
maxG = np.zeros(tlen)

for i in range(tlen):
    bp_c = make_frame_SHB_result(order, freq, p, i, i)
    G[:, :, i] = np.hstack((bp_c[:, 90::-1], bp_c[:, -1:90:-1]))
    maxG[i] = np.max(G[:, :, i])
maxG_mean = np.mean(maxG)

# observation dirs
# azimuth every pi/60 rad (0~2pi)
azi = np.arange(-np.pi, np.pi + np.pi / 60. * 0.1, np.pi / 60.)
# elevation every pi/60 rad (0~pi)
ele = np.arange(np.pi, 0 - np.pi / 60. * 0.1, -np.pi / 60.)

X, Y = np.meshgrid(azi, ele)

fig, ax1 = plt.subplots(figsize=(4, 3))

cax = ax1.pcolormesh(X, Y, G[:-1, :-1, 0],vmin=0, vmax=maxG_mean, cmap='jet')
ax1.set_xlabel('azimuth', fontsize=14)
ax1.set_ylabel('elevation', fontsize=14)
ax1.set_aspect('equal')
fig.colorbar(cax)

def animate(i):
    cax.set_array(G[:-1, :-1, i].flatten())


# Save the animation
anim = FuncAnimation(fig, animate, interval=100, frames=400, repeat=True)
fig.show()
anim.save('testanimation.mp4', writer='ffmpeg')
