# convert from .dat files to .wav file
# 04-101, 05-B1, 06-EntranceHall

import numpy as np

import struct

import soundfile as sf


# values
room = 'EntranceHall'
pnum = 6
Fs = 48000
ch_num = 64
ir_len = 50000
ir_start = 0
mic_in = np.zeros((ir_len - ir_start, ch_num))

# import binary data
for ich in range(64):
    fname_in = "/Users/mizun524/research/64ch_IRs/{}/ir_0_{}.dat".format(
        room, ich)
    with open(fname_in, 'rb') as f:
        cont = f.read()
        data = struct.unpack('f' * (len(cont) // 4), cont)
    mic_in[:, ich] = data[ir_start:ir_len]

# export wave file
fname_out = '/Users/mizun524/research/64ch_IRs/{}_P{:02d}_FLT32_avg.wav'.format(
    room, pnum)
sf.write(fname_out, mic_in, Fs)
