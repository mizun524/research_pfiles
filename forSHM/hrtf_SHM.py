# Sparse Spherical Harmonics-based Modeling for HRTF
#
# Reerence
# "A Sparse Spherical Harmonic-Based Model in Subbands for Head-Related Transfer Functions."
#
# add func : sph_harm_real, calc_ITD

import numpy as np
from multiprocessing import Pool
import sys
sys.path.append('/Users/mizun524/others/pfiles')  # for Mac
# sys.path.append('c://Users//pfiles') #for Windows
from mymodules import sph_harm_real
from mymodules import illustrate_func
#from mymodules import calc_ITD

from scipy.fftpack import fft, ifft

from sklearn import linear_model as lm


# 0.values
hrir_len = 512  # HRIR's length
Fs = 44100  # Sampling rate
order = 40  # Spherical Harmonics order
core_num = 4  # for parallel processing

# 1.import HRTF
hrirL = np.zeros((37 * 72, hrir_len))
hrirR = np.zeros((37 * 72, hrir_len))
line = 0
for ele in range(-90, 90 + 5, 5):
    for azi in range(-175, 180 + 5, 5):
        filename = "/Users/mizun524/research/hrtfdata/imp_allSP_kemar/e{}/L{}.dat".format(
            ele, azi)  # for Mac
        # filename = "c://Users//hrtfdata//imp_allSP_kemar//e{}//L{}.dat".format(
        #    ele, azi) #for Windows
        hrirL[line, :] = np.fromfile(filename, np.double)
        filename = "/Users/mizun524/research/hrtfdata/imp_allSP_kemar/e{}/R{}.dat".format(
            ele, azi)  # for Mac
        # filename = "c://Users//hrtfdata//imp_allSP_kemar//e{}//R{}.dat".format(
        #    ele, azi) #for Windows
        hrirR[line, :] = np.fromfile(filename, np.double)
        line = line + 1

# 2.preprocessing
# 2.1.HRTF preprocessing
# 2.1.1.HRIR to HRTF
hrtfL = fft(hrirL, axis=1)
hrtfR = fft(hrirR, axis=1)

###
import matplotlib.pyplot as plt
plt.plot(np.real(np.abs(hrtfL[18*72+10,:256])))

###

# 2.1.2.minimum phase HRTF
hrtfL[:, 0] = np.real(hrtfL[:, 0]) + 1j
hrirL_min = ifft(np.log(np.abs(hrtfL)), axis=1)
hrirL_min[1, int(hrir_len / 2):] = 0
hrtfL_min_phase = fft(hrirL_min, axis=1)
hrtfL_min = np.real(hrtfL) * np.exp(np.imag(hrtfL_min_phase) * 1j)
hrtfR[:, 0] = np.real(hrtfR[:, 0]) + 1j
hrirR_min = ifft(np.log(np.abs(hrtfR)), axis=1)
hrirR_min[1, int(hrir_len / 2):] = 0
hrtfR_min_phase = fft(hrirR_min, axis=1)
hrtfR_min = np.real(hrtfR) * np.exp(np.imag(hrtfR_min_phase) * 1j)

# 2.1.3.average HRTF
# discard hrtf[:,0] (nearly 0)
hrtfL_avg = np.mean(
    20 * np.log10(np.abs(hrtfL_min[:, 1:int(hrir_len / 2) + 1])), axis=0)
hrtfR_avg = np.mean(
    20 * np.log10(np.abs(hrtfR_min[:, 1:int(hrir_len / 2) + 1])), axis=0)
hrtfL_p = np.zeros((hrtfL.shape[0], int(hrir_len / 2)))
hrtfR_p = np.zeros((hrtfR.shape[0], int(hrir_len / 2)))
line = 0
for line in range(hrtfL.shape[0]):
    hrtfL_p[line, :] = 20 * \
        np.log10(np.abs(hrtfL_min[line, 1:int(hrir_len / 2) + 1])) - hrtfL_avg
    hrtfR_p[line, :] = 20 * \
        np.log10(np.abs(hrtfR_min[line, 1:int(hrir_len / 2) + 1])) - hrtfR_avg
    line = line + 1

'''
# 2.2.calculate ITD

# wrapper_calc_ITD


_def wrapper_calc_ITD(args):
    return calc_ITD.calc_ITD(*args)


# parallel processing
if __name__ == '__main__':
    # wrapping hrirL, hrirR and Fs
    hrir_wrap = [[hrirL[line, :], hrirR[line, :], Fs]
                 for line in range(hrtfL.shape[0])]
    # processes=core_num
    p = Pool(processes=core_num)
    # ITDs = [ITD, delayL, delayR]
    ITDs = p.map(wrapper_calc_ITD, hrir_wrap)
    p.close()

delayL = np.zeros_like(hrtfL_p)
delayR = np.zeros_like(hrtfR_p)
for line in range(hrtfL.shape[0]):
    delayL[line, :] = ITDs[line][1]
    delayR[line, :] = ITDs[line][2]
'''

# 3.Spherical harmonics expansion
# Spherical Harmonics
Y = np.zeros((hrtfL.shape[0], (order + 1)**2))
line = 0
for ele in range(-90, 90 + 5, 5):
    phi = np.pi / 2 - ele / 180 * np.pi  # ele to phi(polar)
    for azi in range(-175, 180 + 5, 5):
        theta = azi / 180 * np.pi
        Y[line, :] = np.array([sph_harm_real.sph_harm_real(m, n, theta, phi)
                               for n in range(order + 1) for m in range(-n, n + 1)])
        line = line + 1
# Lasso

# wrapper_lassoCV


def wrapper_lassoCV(args):
    clf = lm.LassoCV()
    lasso = clf.fit(*args)
    return lasso.coef_


# parallel preprocessing
if __name__ == '__main__':
    # wrapping Y and hrtf(L or R)
    hrtfL_p_wrap = [[Y, hrtfL_p[:, freqind]]
                    for freqind in range(hrtfL_p.shape[1])]
    hrtfR_p_wrap = [[Y, hrtfR_p[:, freqind]]
                    for freqind in range(hrtfR_p.shape[1])]
    '''
    # wrapping Y and delay(L or R)
    delayL_wrap = [[Y, delayL[:, freqind]]
                   for freqind in range(delayL.shape[1])]
    delayR_wrap = [[Y, delayR[:, freqind]]
                   for freqind in range(delayR.shape[1])]
    '''
    # processes=core_num
    p = Pool(processes=core_num)
    # SH coefficients
    YcL = p.map(wrapper_lassoCV, hrtfL_p_wrap)
    YcL_array = np.array(YcL)
    YcR = p.map(wrapper_lassoCV, hrtfR_p_wrap)
    YcR_array = np.array(YcR)
    '''
    Yc_delayL = p.map(wrapper_lassoCV, delayL_wrap)
    Yc_delayR = p.map(wrapper_lassoCV, delayR_wrap)
    '''
    p.close()

# save Spherical Harmonics coefficients
np.savez('hrtf_Yc.npz', ycl=YcL_array, ycr=YcR_array)#, ycdl=Yc_delayL, ycdr=Yc_delayR)
# Spherical Harmonics expression
hrtfL_p_sshm = np.dot(Y, YcL_array.T)
hrtfR_p_sshm = np.dot(Y, YcR_array.T)
'''
delayL_sshm = np.dot(Y, Yc_delayL)
delayR_sshm = np.dot(Y, Yc_delayR)
'''


illustrate_func.mapping_frequency(hrtfR_p[18*72:19*72,:], Fs)
illustrate_func.mapping_frequency(np.hstack((hrtfR_p_sshm[18*72:19*72,:],hrtfR_p_sshm[18*72:19*72,:])), Fs)
illustrate_func.mapping_frequency(hrtfR_p_sshm[18*72:19*72,:], Fs)
