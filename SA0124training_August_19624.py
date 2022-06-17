from __future__ import division
import mne
import numpy as np
import scipy.signal
from scipy.signal import butter, lfilter
from matplotlib import pyplot
import math
from scipy.fftpack import fft, ifft
from scipy import signal
from scipy.signal import butter, lfilter, iirfilter
from scipy.signal import hilbert
from biosppy.signals import tools
import pandas as pd


def movingaverage(values, window_size):
    weights = (np.ones(window_size)) / window_size
    a = np.ones(1)
    return lfilter(weights, a, values)


csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/EEG_timewindowarr_SA0124_15s.csv', sep=',',
                         header=None)
t_window_arr = csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/EEGvariance_SA0124_15s_3h.csv', sep=',',
                         header=None)
Raw_variance_EEG = csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/EEGauto_SA0124_15s_3h.csv', sep=',', header=None)
Raw_auto_EEG = csv_reader.values

Raw_variance_EEG_arr = []
for item in Raw_variance_EEG:
    Raw_variance_EEG_arr.append(float(item))
Raw_auto_EEG_arr = []
for item in Raw_auto_EEG:
    Raw_auto_EEG_arr.append(float(item))

var_arr = []
for item in Raw_variance_EEG_arr:
    if item < 1e-8:
        var_arr.append(item)
    else:
        var_arr.append(var_arr[-1])
Raw_variance_EEG = var_arr

value_arr = []
for item in Raw_auto_EEG_arr:
    if item < 500:
        value_arr.append(item)
    else:
        value_arr.append(value_arr[-1])
Raw_auto_EEG = value_arr

csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_ch31_timewindowarr_SA0124_15s_3h.csv', sep=',',
                         header=None)
rri_t = csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_ch31_rawvariance_SA0124_15s_3h.csv', sep=',',
                         header=None)
RRI_var = csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_ch31_rawauto_SA0124_15s_3h.csv', sep=',',
                         header=None)
Raw_auto_RRI31 = csv_reader.values

rri_t_arr = []
for item in rri_t:
    rri_t_arr.append(2.98805 + float(item))
Raw_variance_RRI31_arr = []
for item in RRI_var:
    Raw_variance_RRI31_arr.append(float(item))
Raw_auto_RRI31_arr = []
for item in Raw_auto_RRI31:
    Raw_auto_RRI31_arr.append(float(item))


print(t_window_arr[0])
window_time_arr = t_window_arr[0:19624]
Raw_variance_EEG = Raw_variance_EEG[0:19624]
Raw_auto_EEG = Raw_auto_EEG[0:19624]

window_RRI_time_arr = rri_t_arr[0:19624]
Raw_variance_RRI31 = Raw_variance_RRI31_arr[0:19624]
Raw_auto_RRI31 = Raw_auto_RRI31_arr[0:19624]

medium_rhythm_var_arr_3 = movingaverage(Raw_variance_EEG, 240 * 6)
long_rhythm_var_arr = medium_rhythm_var_arr_3
var_trans = hilbert(long_rhythm_var_arr)
var_phase = np.angle(var_trans)
phase_long_EEGvariance_arr = var_phase
print(len(phase_long_EEGvariance_arr));

medium_rhythm_value_arr_3 = movingaverage(Raw_auto_EEG, 240 * 6)
long_rhythm_value_arr = medium_rhythm_value_arr_3
value_trans = hilbert(long_rhythm_value_arr)
value_phase = np.angle(value_trans)
phase_long_EEGauto_arr = value_phase
print(len(phase_long_EEGauto_arr));

medium_rhythm_RRIvar_arr_3 = movingaverage(Raw_variance_RRI31, 240 * 6)
long_rhythm_RRIvar_arr = medium_rhythm_RRIvar_arr_3
var_trans = hilbert(long_rhythm_RRIvar_arr)
var_phase = np.angle(var_trans)
phase_long_RRIvariance_arr = var_phase
print(len(phase_long_RRIvariance_arr));

medium_rhythm_RRIvalue_arr_3 = movingaverage(Raw_auto_RRI31, 240 * 6)
long_rhythm_RRIvalue_arr = medium_rhythm_RRIvalue_arr_3
value_trans = hilbert(long_rhythm_RRIvalue_arr)
value_phase = np.angle(value_trans)
phase_long_RRIauto_arr = value_phase
print(len(phase_long_RRIauto_arr));

seizure_timing_index = []
for k in range(len(window_time_arr)):
    if window_time_arr[k] < 9.19205 and window_time_arr[k + 1] >= 9.19205:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 18.9488833 and window_time_arr[k + 1] >= 18.9488833:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 24.16555 and window_time_arr[k + 1] >= 24.16555:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 32.9738833 and window_time_arr[k + 1] >= 32.9738833:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 45.149161 and window_time_arr[k + 1] >= 45.149161:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 55.0694389 and window_time_arr[k + 1] >= 55.0694389:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 67.5319389 and window_time_arr[k + 1] >= 67.5319389:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 80.90055 and window_time_arr[k + 1] >= 80.90055:
        seizure_timing_index.append(k)
print(seizure_timing_index)

#### combined probability calculation
bins_number = 18
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
pro_eegvars_time = []
pro_eegvars_time_false = []
for i in range(len(phase_long_EEGvariance_arr)):
    if phase_long_EEGvariance_arr[i] >= bins[0] and phase_long_EEGvariance_arr[i] < bins[1]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[1] and phase_long_EEGvariance_arr[i] < bins[2]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[2] and phase_long_EEGvariance_arr[i] < bins[3]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[3] and phase_long_EEGvariance_arr[i] < bins[4]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[4] and phase_long_EEGvariance_arr[i] < bins[5]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[5] and phase_long_EEGvariance_arr[i] < bins[6]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[6] and phase_long_EEGvariance_arr[i] <= bins[7]:
        pro_eegvars_time_false.append(0.065966558)
        pro_eegvars_time.append(0.25)
    elif phase_long_EEGvariance_arr[i] > bins[7] and phase_long_EEGvariance_arr[i] < bins[8]:
        pro_eegvars_time_false.append(0.187805873)
        pro_eegvars_time.append(0.125)
    elif phase_long_EEGvariance_arr[i] >= bins[8] and phase_long_EEGvariance_arr[i] <= bins[9]:
        pro_eegvars_time_false.append(0.214620718)
        pro_eegvars_time.append(0.125)
    elif phase_long_EEGvariance_arr[i] > bins[9] and phase_long_EEGvariance_arr[i] < bins[10]:
        pro_eegvars_time_false.append(0.318107667)
        pro_eegvars_time.append(0.375)
    elif phase_long_EEGvariance_arr[i] >= bins[10] and phase_long_EEGvariance_arr[i] <= bins[11]:
        pro_eegvars_time_false.append(0.176998369)
        pro_eegvars_time.append(0.125)
    elif phase_long_EEGvariance_arr[i] > bins[11] and phase_long_EEGvariance_arr[i] < bins[12]:
        pro_eegvars_time_false.append(0.008666395)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] > bins[12] and phase_long_EEGvariance_arr[i] < bins[13]:
        pro_eegvars_time_false.append(0.010195759)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[13] and phase_long_EEGvariance_arr[i] < bins[14]:
        pro_eegvars_time_false.append(0.017638662)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[14] and phase_long_EEGvariance_arr[i] < bins[15]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[15] and phase_long_EEGvariance_arr[i] < bins[16]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[16] and phase_long_EEGvariance_arr[i] < bins[17]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[17]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
print(pro_eegvars_time[1487]);print(pro_eegvars_time[3829]);print(pro_eegvars_time[5081]);print(pro_eegvars_time[7195]);
print(pro_eegvars_time[10117]);print(pro_eegvars_time[12498]);print(pro_eegvars_time[15489]);print(pro_eegvars_time[18697])
# pyplot.figure(figsize=(8,4))
# pyplot.plot(window_time_arr,pro_eegvars_time)
# pyplot.annotate('', xy=(9.19205, np.max(pro_eegvars_time)), xytext=(9.19205, np.max(pro_eegvars_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(18.9488833, np.max(pro_eegvars_time)), xytext=(18.9488833, np.max(pro_eegvars_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(24.16555, np.max(pro_eegvars_time)), xytext=(24.16555, np.max(pro_eegvars_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(32.9738833, np.max(pro_eegvars_time)), xytext=(32.9738833, np.max(pro_eegvars_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(45.149161, np.max(pro_eegvars_time)), xytext=(45.149161, np.max(pro_eegvars_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(55.0694389, np.max(pro_eegvars_time)), xytext=(55.0694389, np.max(pro_eegvars_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(67.5319389, np.max(pro_eegvars_time)), xytext=(67.5319389, np.max(pro_eegvars_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(80.90055, np.max(pro_eegvars_time)), xytext=(80.90055, np.max(pro_eegvars_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# # pyplot.hlines(0.125, window_time_arr[0],window_time_arr[-1],'r')
# pyplot.xlabel('Time(hours)', fontsize=14)
# pyplot.ylabel('Seizure probability', fontsize=14)
# locs, labels = pyplot.xticks(fontsize=14)
# locs, labels = pyplot.yticks(fontsize=14)
# pyplot.show()

pro_eegautos_time = []
pro_eegautos_time_false = []
for i in range(len(phase_long_EEGauto_arr)):
    if phase_long_EEGauto_arr[i] >= bins[0] and phase_long_EEGauto_arr[i] < bins[1]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[1] and phase_long_EEGauto_arr[i] < bins[2]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[2] and phase_long_EEGauto_arr[i] < bins[3]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[3] and phase_long_EEGauto_arr[i] < bins[4]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[4] and phase_long_EEGauto_arr[i] < bins[5]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[5] and phase_long_EEGauto_arr[i] < bins[6]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[6] and phase_long_EEGauto_arr[i] < bins[7]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[7] and phase_long_EEGauto_arr[i] < bins[8]:
        pro_eegautos_time_false.append(0.073664356)
        pro_eegautos_time.append(0.125)
    elif phase_long_EEGauto_arr[i] >= bins[8] and phase_long_EEGauto_arr[i] < bins[9]:
        pro_eegautos_time_false.append(0.396156199)
        pro_eegautos_time.append(0.375)
    elif phase_long_EEGauto_arr[i] >= bins[9] and phase_long_EEGauto_arr[i] <= bins[10]:
        pro_eegautos_time_false.append(0.507850734)
        pro_eegautos_time.append(0.5)
    elif phase_long_EEGauto_arr[i] > bins[10] and phase_long_EEGauto_arr[i] < bins[11]:
        pro_eegautos_time_false.append(0.007442904)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[11] and phase_long_EEGauto_arr[i] < bins[12]:
        pro_eegautos_time_false.append(0.004180261)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[12] and phase_long_EEGauto_arr[i] < bins[13]:
        pro_eegautos_time_false.append(0.00423124)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[13] and phase_long_EEGauto_arr[i] < bins[14]:
        pro_eegautos_time_false.append(0.006474307)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[14] and phase_long_EEGauto_arr[i] < bins[15]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[15] and phase_long_EEGauto_arr[i] < bins[16]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[16] and phase_long_EEGauto_arr[i] < bins[17]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[17]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)

print(pro_eegautos_time[1487]);print(pro_eegautos_time[3829]);print(pro_eegautos_time[5081]);print(pro_eegautos_time[7195]);
print(pro_eegautos_time[10117]);print(pro_eegautos_time[12498]);print(pro_eegautos_time[15489]);print(pro_eegautos_time[18697])
# pyplot.figure(figsize=(8,4))
# pyplot.plot(window_time_arr,pro_eegautos_time)
# pyplot.annotate('', xy=(9.19205, np.max(pro_eegautos_time)), xytext=(9.19205, np.max(pro_eegautos_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(18.9488833, np.max(pro_eegautos_time)), xytext=(18.9488833, np.max(pro_eegautos_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(24.16555, np.max(pro_eegautos_time)), xytext=(24.16555, np.max(pro_eegautos_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(32.9738833, np.max(pro_eegautos_time)), xytext=(32.9738833, np.max(pro_eegautos_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(45.149161, np.max(pro_eegautos_time)), xytext=(45.149161, np.max(pro_eegautos_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(55.0694389, np.max(pro_eegautos_time)), xytext=(55.0694389, np.max(pro_eegautos_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(67.5319389, np.max(pro_eegautos_time)), xytext=(67.5319389, np.max(pro_eegautos_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(80.90055, np.max(pro_eegautos_time)), xytext=(80.90055, np.max(pro_eegautos_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# # pyplot.hlines(0.125, window_time_arr[0],window_time_arr[-1],'r')
# pyplot.xlabel('Time(hours)', fontsize=14)
# pyplot.ylabel('Seizure probability', fontsize=14)
# locs, labels = pyplot.xticks(fontsize=14)
# locs, labels = pyplot.yticks(fontsize=14)
# pyplot.show()

pro_RRIvars_time = []
pro_RRIvars_time_false = []
for i in range(len(phase_long_RRIvariance_arr)):
    if phase_long_RRIvariance_arr[i] >= bins[0] and phase_long_RRIvariance_arr[i] < bins[1]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[1] and phase_long_RRIvariance_arr[i] < bins[2]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[2] and phase_long_RRIvariance_arr[i] < bins[3]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[3] and phase_long_RRIvariance_arr[i] < bins[4]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[4] and phase_long_RRIvariance_arr[i] < bins[5]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[5] and phase_long_RRIvariance_arr[i] < bins[6]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[6] and phase_long_RRIvariance_arr[i] < bins[7]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[7] and phase_long_RRIvariance_arr[i] <= bins[8]:
        pro_RRIvars_time_false.append(0.245309951)
        pro_RRIvars_time.append(0.25)
    elif phase_long_RRIvariance_arr[i] > bins[8] and phase_long_RRIvariance_arr[i] < bins[9]:
        pro_RRIvars_time_false.append(0.246074633)
        pro_RRIvars_time.append(0.25)
    elif phase_long_RRIvariance_arr[i] >= bins[9] and phase_long_RRIvariance_arr[i] <= bins[10]:
        pro_RRIvars_time_false.append(0.310766721)
        pro_RRIvars_time.append(0.375)
    elif phase_long_RRIvariance_arr[i] > bins[10] and phase_long_RRIvariance_arr[i] <= bins[11]:
        pro_RRIvars_time_false.append(0.145238581)
        pro_RRIvars_time.append(0.125)
    elif phase_long_RRIvariance_arr[i] > bins[11] and phase_long_RRIvariance_arr[i] < bins[12]:
        pro_RRIvars_time_false.append(0.037826264)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[12] and phase_long_RRIvariance_arr[i] < bins[13]:
        pro_RRIvars_time_false.append(0.00632137)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[13] and phase_long_RRIvariance_arr[i] < bins[14]:
        pro_RRIvars_time_false.append(0.00846248)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[14] and phase_long_RRIvariance_arr[i] < bins[15]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[15] and phase_long_RRIvariance_arr[i] < bins[16]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[16] and phase_long_RRIvariance_arr[i] < bins[17]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[17]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
print(pro_RRIvars_time[1487]);print(pro_RRIvars_time[3829]);print(pro_RRIvars_time[5081]);print(pro_RRIvars_time[7195]);
print(pro_RRIvars_time[10117]);print(pro_RRIvars_time[12498]);print(pro_RRIvars_time[15489]);print(pro_RRIvars_time[18697])
# pyplot.figure(figsize=(8,4))
# pyplot.plot(window_time_arr,pro_RRIvars_time)
# pyplot.annotate('', xy=(9.19205, np.max(pro_RRIvars_time)), xytext=(9.19205, np.max(pro_RRIvars_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(18.9488833, np.max(pro_RRIvars_time)), xytext=(18.9488833, np.max(pro_RRIvars_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(24.16555, np.max(pro_RRIvars_time)), xytext=(24.16555, np.max(pro_RRIvars_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(32.9738833, np.max(pro_RRIvars_time)), xytext=(32.9738833, np.max(pro_RRIvars_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(45.149161, np.max(pro_RRIvars_time)), xytext=(45.149161, np.max(pro_RRIvars_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(55.0694389, np.max(pro_RRIvars_time)), xytext=(55.0694389, np.max(pro_RRIvars_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(67.5319389, np.max(pro_RRIvars_time)), xytext=(67.5319389, np.max(pro_RRIvars_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(80.90055, np.max(pro_RRIvars_time)), xytext=(80.90055, np.max(pro_RRIvars_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# # pyplot.hlines(0.125, window_time_arr[0],window_time_arr[-1],'r')
# pyplot.xlabel('Time(hours)', fontsize=14)
# pyplot.ylabel('Seizure probability', fontsize=14)
# locs, labels = pyplot.xticks(fontsize=14)
# locs, labels = pyplot.yticks(fontsize=14)
# pyplot.show()

pro_RRIautos_time = []
pro_RRIautos_time_false = []
for i in range(len(phase_long_RRIauto_arr)):
    if phase_long_RRIauto_arr[i] >= bins[0] and phase_long_RRIauto_arr[i] < bins[1]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[1] and phase_long_RRIauto_arr[i] < bins[2]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[2] and phase_long_RRIauto_arr[i] < bins[3]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[3] and phase_long_RRIauto_arr[i] < bins[4]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[4] and phase_long_RRIauto_arr[i] < bins[5]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[5] and phase_long_RRIauto_arr[i] < bins[6]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[6] and phase_long_RRIauto_arr[i] < bins[7]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[7] and phase_long_RRIauto_arr[i] <= bins[8]:
        pro_RRIautos_time_false.append(0.038081158)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] > bins[8] and phase_long_RRIauto_arr[i] <= bins[9]:
        pro_RRIautos_time_false.append(0.497756933)
        pro_RRIautos_time.append(0.75)
    elif phase_long_RRIauto_arr[i] > bins[9] and phase_long_RRIauto_arr[i] <= bins[10]:
        pro_RRIautos_time_false.append(0.422359299)
        pro_RRIautos_time.append(0.25)
    elif phase_long_RRIauto_arr[i] > bins[10] and phase_long_RRIauto_arr[i] < bins[11]:
        pro_RRIautos_time_false.append(0.02406199)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[11] and phase_long_RRIauto_arr[i] < bins[12]:
        pro_RRIautos_time_false.append(0.006576264)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[12] and phase_long_RRIauto_arr[i] < bins[13]:
        pro_RRIautos_time_false.append(0.004486134)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[13] and phase_long_RRIauto_arr[i] < bins[14]:
        pro_RRIautos_time_false.append(0.006678222)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[14] and phase_long_RRIauto_arr[i] < bins[15]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[15] and phase_long_RRIauto_arr[i] < bins[16]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[16] and phase_long_RRIauto_arr[i] < bins[17]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[17]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)

print(pro_RRIautos_time[1487]);print(pro_RRIautos_time[3829]);print(pro_RRIautos_time[5081]);print(pro_RRIautos_time[7195]);
print(pro_RRIautos_time[10117]);print(pro_RRIautos_time[12498]);print(pro_RRIautos_time[15489]);print(pro_RRIautos_time[18697]);
print(window_time_arr[0]);
print(window_time_arr[-1]);

# pyplot.figure(figsize=(8,4))
# pyplot.plot(window_time_arr,pro_RRIautos_time)
# pyplot.annotate('', xy=(9.19205, np.max(pro_RRIautos_time)), xytext=(9.19205, np.max(pro_RRIautos_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(18.9488833, np.max(pro_RRIautos_time)), xytext=(18.9488833, np.max(pro_RRIautos_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(24.16555, np.max(pro_RRIautos_time)), xytext=(24.16555, np.max(pro_RRIautos_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(32.9738833, np.max(pro_RRIautos_time)), xytext=(32.9738833, np.max(pro_RRIautos_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(45.149161, np.max(pro_RRIautos_time)), xytext=(45.149161, np.max(pro_RRIautos_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(55.0694389, np.max(pro_RRIautos_time)), xytext=(55.0694389, np.max(pro_RRIautos_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(67.5319389, np.max(pro_RRIautos_time)), xytext=(67.5319389, np.max(pro_RRIautos_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(80.90055, np.max(pro_RRIautos_time)), xytext=(80.90055, np.max(pro_RRIautos_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# # pyplot.hlines(0.25, window_time_arr[0],window_time_arr[-1],'r')
# pyplot.xlabel('Time(hours)', fontsize=14)
# pyplot.ylabel('Seizure probability', fontsize=14)
# locs, labels = pyplot.xticks(fontsize=14)
# locs, labels = pyplot.yticks(fontsize=14)
# pyplot.show()







Pseizureeegvar = 0.000407664;
Pnonseizureeegvar = 0.999592336;

Pcombined = []
for m in range(len(pro_eegvars_time)):
    P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIvars_time[m]*pro_RRIautos_time[m]
    P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m])
    Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_eegvars_time[m]*pro_eegautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_RRIvars_time[m]*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

pyplot.figure(figsize=(8, 4))
ax = pyplot.subplot(111)
pyplot.plot(window_time_arr, Pcombined,'darkblue')
# pyplot.title('Combined probability', fontsize=15)
pyplot.annotate('', xy=(9.19205, np.max(Pcombined)), xytext=(9.19205, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(18.9488833, np.max(Pcombined)), xytext=(18.9488833, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(24.16555, np.max(Pcombined)), xytext=(24.16555, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(32.9738833, np.max(Pcombined)), xytext=(32.9738833, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(45.149161, np.max(Pcombined)), xytext=(45.149161, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(55.0694389, np.max(Pcombined)), xytext=(55.0694389, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(67.5319389, np.max(Pcombined)), xytext=(67.5319389, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(80.90055, np.max(Pcombined)), xytext=(80.90055, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.tight_layout()
# pyplot.xlim(window_time_arr[0], window_time_arr[-1])
pyplot.hlines(2.3961124493674196e-06, window_time_arr[0],window_time_arr[-1],'r')
pyplot.hlines(0.3*2.3961124493674196e-06, window_time_arr[0],window_time_arr[-1],'orange')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
pyplot.xlabel('Time (hours)', fontsize=16)
pyplot.ylabel('Probability', fontsize=16)
pyplot.title('Critical slowing model', fontsize=16)
locs, labels = pyplot.xticks(fontsize=14)
locs, labels = pyplot.yticks([0,0.00001,0.00002],fontsize=16)
pyplot.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
pyplot.show()

print(Pcombined[1487]);print(Pcombined[3829]);print(Pcombined[5081]);
print(Pcombined[7195]);print(Pcombined[10117]);print(Pcombined[12498]);
print(Pcombined[15489]);print(Pcombined[18697])

# index=[]
# n1=0;n2=0;n3=0;n4=0;n5=0;n6=0;n7=0;n8=0
# for i, j in enumerate(Pcombined):
#     if j >= 0.6*1.3455125787746595e-05:
#         index.append(i)
#         if 9.19205-window_time_arr[i]<=1 and 9.19205-window_time_arr[i]>=0:
#             n1=n1+1
#         if 18.9488833 - window_time_arr[i] < 1 and 18.9488833 - window_time_arr[i] >= 0:
#             n2 = n2 + 1
#         if 24.16555 - window_time_arr[i] < 1 and 24.16555 - window_time_arr[i] >= 0:
#             n3 = n3 + 1
#         if 32.9738833 - window_time_arr[i] < 1 and 32.9738833 - window_time_arr[i] >= 0:
#             n4 = n4 + 1
#         if 45.149161 - window_time_arr[i] < 1 and 45.149161 - window_time_arr[i] >= 0:
#             n5 = n5 + 1
#         if 55.0694389 - window_time_arr[i] < 1 and 55.0694389 - window_time_arr[i] >= 0:
#             n6 = n6 + 1
#         if 67.5319389 - window_time_arr[i] < 1 and 67.5319389 - window_time_arr[i] >= 0:
#             n7 = n7 + 1
#         if  80.90055 - window_time_arr[i] < 1 and  80.90055 - window_time_arr[i] >= 0:
#             n8 = n8 + 1
# print(n1);print(n2);print(n3);print(n4);print(n5);print(n6);print(n7);print(n8);
# print(n1+n2+n3+n4+n5+n6+n7+n8)
# print(len(index))






### add circadian
t=np.linspace(2.98805+0.00416667,2.98805+0.00416667+0.00416667*(len(Raw_variance_EEG_arr)-1),len(Raw_variance_EEG_arr))
print(len(t));
print(t[0]);print(t[19080]);print(t[-1]);
a=np.where(t<7.73806+2.98805)
print(a)
print(t[1856]);print(t[1857])
t[0:1857]=t[0:1857]-2.98805+16.26194
t[1857:]=t[1857:]-7.73806-2.98805
print(t[1857]);
print(t);print(type(t));print(t[0])

time_feature_arr=[]
for i in range(len(t)):
    if t[i]>24:
        time_feature_arr.append(t[i] - (t[i] // 24) * 24)
    else:
        time_feature_arr.append(t[i])
print(time_feature_arr)


# print(time_feature_arr[1487]);print(time_feature_arr[3829]);
# print(time_feature_arr[5081]);print(time_feature_arr[7195]);
# print(time_feature_arr[10117]);print(time_feature_arr[12498]);
# print(time_feature_arr[15489]);print(time_feature_arr[18697]);

bins_number = 18
bins = np.linspace(0, 24, bins_number + 1)
pro_circadian_time=[]
pro_circadian_time_false=[]
for i in range(len(time_feature_arr)):
    if time_feature_arr[i] >= bins[0] and time_feature_arr[i] <= bins[1]:
        pro_circadian_time_false.append(0.05033557)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] > bins[1] and time_feature_arr[i] < bins[2]:
        pro_circadian_time_false.append(0.05033557)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[2] and time_feature_arr[i] < bins[3]:
        pro_circadian_time_false.append(0.05033557)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[3] and time_feature_arr[i] < bins[4]:
        pro_circadian_time_false.append(0.05033557)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[4] and time_feature_arr[i] < bins[5]:
        pro_circadian_time_false.append(0.05033557)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[5] and time_feature_arr[i] <= bins[6]:
        pro_circadian_time_false.append(0.05033557)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] > bins[6] and time_feature_arr[i] < bins[7]:
        pro_circadian_time_false.append(0.050230705)
        pro_circadian_time.append(0.25)
    elif time_feature_arr[i] >= bins[7] and time_feature_arr[i] <= bins[8]:
        pro_circadian_time_false.append(0.050283138)
        pro_circadian_time.append(0.125)
    elif time_feature_arr[i] > bins[8] and time_feature_arr[i] < bins[9]:
        pro_circadian_time_false.append(0.05033557)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[9] and time_feature_arr[i] < bins[10]:
        pro_circadian_time_false.append(0.05033557)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[10] and time_feature_arr[i] < bins[11]:
        pro_circadian_time_false.append(0.050283138)
        pro_circadian_time.append(0.125)
    elif time_feature_arr[i] >= bins[11] and time_feature_arr[i] < bins[12]:
        pro_circadian_time_false.append(0.05033557)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[12] and time_feature_arr[i] < bins[13]:
        pro_circadian_time_false.append(0.063810822)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[13] and time_feature_arr[i] < bins[14]:
        pro_circadian_time_false.append(0.067114094)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[14] and time_feature_arr[i] < bins[15]:
        pro_circadian_time_false.append(0.067114094)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[15] and time_feature_arr[i] < bins[16]:
        pro_circadian_time_false.append(0.067061661)
        pro_circadian_time.append(0.125)
    elif time_feature_arr[i] >= bins[16] and time_feature_arr[i] < bins[17]:
        pro_circadian_time_false.append(0.066956795)
        pro_circadian_time.append(0.375)
    elif time_feature_arr[i] >= bins[17] and time_feature_arr[i] <= bins[18]:
        pro_circadian_time_false.append(0.064125419)
        pro_circadian_time.append(0)


pro_circadian_time=pro_circadian_time[0:19624]
pyplot.figure(figsize=(8,4))
ax = pyplot.subplot(111)
pyplot.plot(window_time_arr,pro_circadian_time, 'darkblue')
pyplot.annotate('', xy=(9.19205, np.max(pro_circadian_time)), xytext=(9.19205, np.max(pro_circadian_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(18.9488833, np.max(pro_circadian_time)), xytext=(18.9488833, np.max(pro_circadian_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(24.16555, np.max(pro_circadian_time)), xytext=(24.16555, np.max(pro_circadian_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(32.9738833, np.max(pro_circadian_time)), xytext=(32.9738833, np.max(pro_circadian_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(45.149161, np.max(pro_circadian_time)), xytext=(45.149161, np.max(pro_circadian_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(55.0694389, np.max(pro_circadian_time)), xytext=(55.0694389, np.max(pro_circadian_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(67.5319389, np.max(pro_circadian_time)), xytext=(67.5319389, np.max(pro_circadian_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(80.90055, np.max(pro_circadian_time)), xytext=(80.90055, np.max(pro_circadian_time) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.hlines(0.125, window_time_arr[0],window_time_arr[-1],'r')
pyplot.hlines(0.125*0.3, window_time_arr[0],window_time_arr[-1],'orange')
# pyplot.hlines(0.125*0.6, window_time_arr[0],window_time_arr[-1],'orange')
# pyplot.hlines(0.125*1.2, window_time_arr[0],window_time_arr[-1],'orange')
# pyplot.hlines(0.125*2, window_time_arr[0],window_time_arr[-1],'orange')
pyplot.xlabel('Time (hours)', fontsize=16)
pyplot.title('Circadian model', fontsize=16)
pyplot.ylabel('Probability', fontsize=16)
locs, labels = pyplot.xticks(fontsize=16)
locs, labels = pyplot.yticks([0,0.2,0.4],fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
pyplot.show()








Pcombined=[]
for m in range(len(pro_eegvars_time)):
    P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIvars_time[m]*pro_RRIautos_time[m]*pro_circadian_time[m]
    P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
    Pcombined.append(P1/(P1+P2))

# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_circadian_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_circadian_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_RRIvars_time[m]*pro_RRIautos_time[m]*pro_circadian_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
#     Pcombined.append(P1/(P1+P2))

pyplot.figure(figsize=(8, 4))
ax = pyplot.subplot(111)
pyplot.plot(window_time_arr, Pcombined,'darkblue')
pyplot.annotate('', xy=(9.19205, np.max(Pcombined)), xytext=(9.19205, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(18.9488833, np.max(Pcombined)), xytext=(18.9488833, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(24.16555, np.max(Pcombined)), xytext=(24.16555, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(32.9738833, np.max(Pcombined)), xytext=(32.9738833, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(45.149161, np.max(Pcombined)), xytext=(45.149161, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(55.0694389, np.max(Pcombined)), xytext=(55.0694389, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(67.5319389, np.max(Pcombined)), xytext=(67.5319389, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(80.90055, np.max(Pcombined)), xytext=(80.90055, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.tight_layout()
# pyplot.xlim(window_time_arr[0], window_time_arr[-1])
pyplot.hlines(5.978085536184582e-07, window_time_arr[0],window_time_arr[-1],'r')
pyplot.hlines(0.3*5.978085536184582e-07, window_time_arr[0],window_time_arr[-1],'orange')
pyplot.xlabel('Time (hours)', fontsize=16)
pyplot.ylabel('Probability', fontsize=16)
pyplot.title('Combine model', fontsize=16)
locs, labels = pyplot.xticks(fontsize=14)
locs, labels = pyplot.yticks([0,0.000002,0.000004],fontsize=16)
pyplot.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
pyplot.show()

print(Pcombined[1487]);print(Pcombined[3829]);print(Pcombined[5081]);
print(Pcombined[7195]);print(Pcombined[10117]);print(Pcombined[12498]);
print(Pcombined[15489]);print(Pcombined[18697])


# index=[]
# n1=0;n2=0;n3=0;n4=0;n5=0;n6=0;n7=0;n8=0
# for i, j in enumerate(Pcombined):
#     if j >= 2*1.5973262415242407e-06:
#         index.append(i)
#         if 9.19205-window_time_arr[i]<=1 and 9.19205-window_time_arr[i]>=0:
#             n1=n1+1
#         if 18.9488833 - window_time_arr[i] < 1 and 18.9488833 - window_time_arr[i] >= 0:
#             n2 = n2 + 1
#         if 24.16555 - window_time_arr[i] < 1 and 24.16555 - window_time_arr[i] >= 0:
#             n3 = n3 + 1
#         if 32.9738833 - window_time_arr[i] < 1 and 32.9738833 - window_time_arr[i] >= 0:
#             n4 = n4 + 1
#         if 45.149161 - window_time_arr[i] < 1 and 45.149161 - window_time_arr[i] >= 0:
#             n5 = n5 + 1
#         if 55.0694389 - window_time_arr[i] < 1 and 55.0694389 - window_time_arr[i] >= 0:
#             n6 = n6 + 1
#         if 67.5319389 - window_time_arr[i] < 1 and 67.5319389 - window_time_arr[i] >= 0:
#             n7 = n7 + 1
#         if  80.90055 - window_time_arr[i] < 1 and  80.90055 - window_time_arr[i] >= 0:
#             n8 = n8 + 1
# print(n1);print(n2);print(n3);print(n4);print(n5);print(n6);print(n7);print(n8);
# print(n1+n2+n3+n4+n5+n6+n7+n8)
# print(len(index))