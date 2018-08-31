# convert from .dat files to .wav file
# 04-101, 05-B1, 06-EntranceHall

import numpy as np

from scipy import io

import soundfile as sf


# values
pnum = 9
Fs = 48000
ch_num = 64
ir_len = 50000
ir_start = 5000
mic_in = np.zeros((131072, ch_num))

# import binary data
fname_in = "/Users/mizun524/research/20180822_HardyHall/IR/data_P{}_1_C_IR.mat".format(
    pnum)
IR_mat = io.loadmat(fname_in)
IR = np.array(IR_mat["ir"])
IR_64ch = IR[:,:ch_num]

# export wave file
fname_out = '/Users/mizun524/research/20180822_HardyHall/HardyHall_P{:02d}_FLT32_avg.wav'.format(
    pnum)
sf.write(fname_out, IR_64ch, Fs)
