from __future__ import division
import mne
import numpy as np
import scipy.signal
from scipy.signal import butter, lfilter
from matplotlib import pyplot
import math
from scipy.fftpack import fft, ifft
from scipy import signal
from scipy.signal import butter, lfilter, iirfilter, filtfilt
from scipy.signal import hilbert
from biosppy.signals import tools
import pandas as pd
from matplotlib import rc
from sklearn.metrics import auc

def butter_bandpass(lowcut, highcut, fs, order=7):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


# def butter_bandpass_filter(data, lowcut, highcut, fs, order=7):
#     b, a = butter_bandpass(lowcut, highcut, fs, order=order)
#     y = lfilter(b, a, data)
#     return y

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def Implement_Notch_Filter(fs, band, freq, ripple, order, filter_type, data):
    nyq = fs / 2.0
    low = freq - band / 2.0
    high = freq + band / 2.0
    low = low / nyq
    high = high / nyq
    b, a = iirfilter(order, [low, high], rp=ripple, btype='bandstop', analog=False, ftype=filter_type)
    filtered_data = filtfilt(b, a, data)
    return filtered_data


def split(arr, size):
    arrs = []
    while len(arr) > size:
        pice = arr[:size]
        arrs.append(pice)
        arr = arr[int(size):]
    arrs.append(arr)
    return arrs


def movingaverage(values, window_size):
    weights = (np.ones(window_size)) / window_size
    a = np.ones(1)
    return lfilter(weights, a, values)




cycle_list=[1/60,1/12,1,6,12,24] ## 1min, 5min, 1h, 6h, 12h, 24h
cycle = 6

csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD0290/Cz_EEGvariance_QLD0290_15s_3h.csv', sep=',',
                         header=None)
Raw_variance_EEG = csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD0290/Cz_EEGauto_QLD0290_15s_3h.csv', sep=',',
                         header=None)
Raw_auto_EEG = csv_reader.values

Raw_variance_EEG_arr = []
for item in Raw_variance_EEG:
    Raw_variance_EEG_arr.append(float(item))
Raw_auto_EEG_arr = []
for item in Raw_auto_EEG:
    Raw_auto_EEG_arr.append(float(item))
t = np.linspace(0 + 0.00416667, 0 + 0.00416667 + 0.00416667 * (len(Raw_variance_EEG_arr) - 1),
                len(Raw_variance_EEG_arr))
t_window_arr = t

print(len(t_window_arr));
print(t_window_arr[0]);
print(t_window_arr[-1] - t_window_arr[0]);
print(t_window_arr[19450]);
print(t_window_arr[19450] - t_window_arr[0]);
print(t_window_arr[-1] - t_window_arr[19450]);


window_time_arr = t_window_arr
# pyplot.plot(window_time_arr, Raw_variance_EEG_arr, 'grey', alpha=0.5)
# pyplot.ylabel('Voltage', fontsize=13)
# pyplot.xlabel('Time (hours)', fontsize=13)
# pyplot.title('raw EEG variance in QLD0290', fontsize=13)
# pyplot.show()
var_arr = []
for item in Raw_variance_EEG_arr:
    if item < 1e-8:
        var_arr.append(item)
    else:
        var_arr.append(var_arr[-1])
Raw_variance_EEG = var_arr
# pyplot.plot(window_time_arr, Raw_variance_EEG, 'grey', alpha=0.5)
# pyplot.ylabel('Voltage', fontsize=13)
# pyplot.xlabel('Time (hours)', fontsize=13)
# pyplot.title('EEG variance in QLD0290', fontsize=13)
# pyplot.show()

seizure_timing_index = []
for k in range(len(window_time_arr)):
    if window_time_arr[k] < 22.79056 and window_time_arr[k + 1] >= 22.79056:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 43.05778 and window_time_arr[k + 1] >= 43.05778:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 48.68167 and window_time_arr[k + 1] >= 48.68167:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 66.425277 and window_time_arr[k + 1] >= 66.425277:
        seizure_timing_index.append(k)
    # if window_time_arr[k] < 93.82222 and window_time_arr[k + 1] >= 93.82222:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 117.80083 and window_time_arr[k + 1] >= 117.80083:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 121.36472 and window_time_arr[k + 1] >= 121.36472:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 140.3647 and window_time_arr[k + 1] >= 140.3647:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 142.20333 and window_time_arr[k + 1] >= 142.20333:
    #     seizure_timing_index.append(k)
print(seizure_timing_index)




# # # # # # # ### EEG variance
window_time_arr = t_window_arr[0:19450]
Raw_variance_EEG = Raw_variance_EEG[0:19450]
# window_time_arr=t_window_arr
# Raw_variance_EEG=Raw_variance_EEG

# long_rhythm_var_arr = movingaverage(Raw_variance_EEG, 5760)
# medium_rhythm_var_arr = movingaverage(Raw_variance_EEG, 240)
# medium_rhythm_var_arr_2 = movingaverage(Raw_variance_EEG, 240 * 3)
# medium_rhythm_var_arr_3 = movingaverage(Raw_variance_EEG, 240 * 6)
# medium_rhythm_var_arr_4 = movingaverage(Raw_variance_EEG, 240 * 12)
short_rhythm_var_arr_plot = movingaverage(Raw_variance_EEG, int(240 * cycle))


long_rhythm_var_arr = short_rhythm_var_arr_plot * (10 ** 12)
var_trans = hilbert(long_rhythm_var_arr)
var_trans_nomal = []
for m in var_trans:
    var_trans_nomal.append(m / abs(m))
SIvarlong = sum(var_trans_nomal) / len(var_trans_nomal)
print(SIvarlong)
seizure_phase = []
for item in seizure_timing_index:
    seizure_phase.append(var_trans_nomal[item])
SIvarlongseizure = sum(seizure_phase) / len(seizure_phase)
print(SIvarlongseizure)
var_phase = np.angle(var_trans)
phase_long_EEGvariance_arr = var_phase
seizure_phase_var_long = []
for item in seizure_timing_index:
    seizure_phase_var_long.append(phase_long_EEGvariance_arr[item])
print(seizure_phase_var_long)
n=0
for item in seizure_phase_var_long:
    if item <0:
        n=n+1
print(n/len(seizure_phase_var_long))
cs_pro_EEGvar=n/len(seizure_phase_var_long)

bins_number = 18
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
nEEGsvar, _, _ = pyplot.hist(phase_long_EEGvariance_arr, bins)
nEEGsvarsei, _, _ = pyplot.hist(seizure_phase_var_long, bins)
print(nEEGsvar)
print(nEEGsvarsei)
width = 2 * np.pi / bins_number
# params = dict(projection='polar')
# fig, ax2 = pyplot.subplots(subplot_kw=params)
# ax2.bar(bins[:bins_number], nEEGsvarsei / sum(nEEGsvarsei), width=width, color='grey', alpha=0.7, linewidth=2,
#         edgecolor='k')
# locs, labels = pyplot.yticks([0.1, 0.3, 0.5], ['0.1', '0.3', '0.5'], fontsize=16)
# ax2.set_title('EEG variance', fontsize=16)
# ax2.set_xticklabels(['0', '0.25$\pi$', '0.5$\pi$', '0.75$\pi$', '-$\pi$', '-0.75$\pi$', '-0.5$\pi$', '-0.25$\pi$', ],
#                     fontsize=16)
# pyplot.show()

nEEGsvarnonsei=nEEGsvar-nEEGsvarsei
pro_seizure_EEGvar=nEEGsvarsei/sum(nEEGsvarsei)
pro_nonseizure_EEGvar=nEEGsvarnonsei/sum(nEEGsvarnonsei)




# pyplot.plot(t_window_arr, Raw_auto_EEG_arr, 'grey', alpha=0.5)
# pyplot.xlabel('Time (hours)', fontsize=13)
# pyplot.title('raw EEG autocorrelation in QLD0290', fontsize=13)
# pyplot.show()
value_arr = []
for item in Raw_auto_EEG_arr:
    if item < 500:
        value_arr.append(item)
    else:
        value_arr.append(value_arr[-1])
Raw_auto_EEG_arr = value_arr
# pyplot.plot(t_window_arr, Raw_auto_EEG_arr, 'grey', alpha=0.5)
# pyplot.xlabel('Time (hours)', fontsize=13)
# pyplot.title('EEG autocorrelation in QLD0290', fontsize=13)
# pyplot.show()

Raw_auto_EEG = Raw_auto_EEG_arr[0:19450]
window_time_arr = t_window_arr[0:19450]
# Raw_auto_EEG=Raw_auto_EEG_arr
# window_time_arr=t_window_arr

# long_rhythm_value_arr = movingaverage(Raw_auto_EEG, 5760)
# medium_rhythm_value_arr = movingaverage(Raw_auto_EEG, 240)
# medium_rhythm_value_arr_2 = movingaverage(Raw_auto_EEG, 240 * 3)
# medium_rhythm_value_arr_3 = movingaverage(Raw_auto_EEG, 240 * 6)
# medium_rhythm_value_arr_4 = movingaverage(Raw_auto_EEG, 240 * 12)
short_rhythm_value_arr_plot = movingaverage(Raw_auto_EEG, int(240 * cycle))


long_rhythm_value_arr = short_rhythm_value_arr_plot
value_trans = hilbert(long_rhythm_value_arr)
value_trans_nomal = []
for m in value_trans:
    value_trans_nomal.append(m / abs(m))
SIvaluelong = sum(value_trans_nomal) / len(value_trans_nomal)
print(SIvaluelong)
seizure_phase = []
for item in seizure_timing_index:
    seizure_phase.append(value_trans_nomal[item])
SIvaluelongseizure = sum(seizure_phase) / len(seizure_phase)
print(SIvaluelongseizure)
value_phase = np.angle(value_trans)
phase_long_EEGauto_arr = value_phase
seizure_phase_value_long = []
for item in seizure_timing_index:
    seizure_phase_value_long.append(phase_long_EEGauto_arr[item])
print(seizure_phase_value_long)
n=0
for item in seizure_phase_value_long:
    if item <0:
        n=n+1
print(n/len(seizure_phase_value_long))
cs_pro_EEGauto=n/len(seizure_phase_value_long)

bins_number = 18
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
nEEGsauto, _, _ = pyplot.hist(phase_long_EEGauto_arr, bins)
nEEGsautosei, _, _ = pyplot.hist(seizure_phase_value_long, bins)
print(nEEGsauto)
print(nEEGsautosei)
# width = 2 * np.pi / bins_number
# params = dict(projection='polar')
# fig, ax2 = pyplot.subplots(subplot_kw=params)
# ax2.bar(bins[:bins_number], nEEGsautosei / sum(nEEGsautosei), width=width, color='grey', alpha=0.7, linewidth=2,
#         edgecolor='k')
# locs, labels = pyplot.yticks([0.1, 0.3, 0.5], ['0.1', '0.3', '0.5'], fontsize=16)
# ax2.set_title('EEG autocorrelation', fontsize=16)
# ax2.set_xticklabels(['0', '0.25$\pi$', '0.5$\pi$', '0.75$\pi$', '-$\pi$', '-0.75$\pi$', '-0.5$\pi$', '-0.25$\pi$', ],
#                     fontsize=16)
# pyplot.show()

nEEGsautononsei=nEEGsauto-nEEGsautosei
pro_seizure_EEGauto=nEEGsautosei/sum(nEEGsautosei)
pro_nonseizure_EEGauto=nEEGsautononsei/sum(nEEGsautononsei)





# # ### ECG data
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD0290/RRI_ch31_timewindowarr_QLD0290_15s_3h.csv',sep=',', header=None)
rri_t = csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD0290/RRI_ch31_rawvariance_QLD0290_15s_3h.csv',sep=',', header=None)
RRI_var = csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD0290/RRI_ch31_rawauto_QLD0290_15s_3h.csv', sep=',',header=None)
Raw_auto_RRI31 = csv_reader.values

rri_t_arr = []
for item in rri_t:
    rri_t_arr.append(0 + float(item))

Raw_variance_RRI31_arr = []
for item in RRI_var:
    Raw_variance_RRI31_arr.append(float(item))
Raw_auto_RRI31_arr = []
for item in Raw_auto_RRI31:
    Raw_auto_RRI31_arr.append(float(item))
print(len(Raw_variance_RRI31_arr))

# pyplot.plot(Raw_variance_RRI31_arr, 'grey', alpha=0.5)
# pyplot.xlabel('Time (hours)', fontsize=13)
# pyplot.title('RRI variance in QLD0290', fontsize=13)
# pyplot.show()
#
# pyplot.plot(Raw_auto_RRI31_arr, 'grey', alpha=0.5)
# pyplot.xlabel('Time (hours)', fontsize=13)
# pyplot.title('RRI autocorrelation in QLD0290', fontsize=13)
# pyplot.show()

# window_time_arr=t_window_arr
# Raw_variance_RRI31=Raw_variance_RRI31_arr
window_time_arr = t_window_arr[0:19450]
Raw_variance_RRI31 = Raw_variance_RRI31_arr[0:19450]

# long_rhythm_var_arr = movingaverage(Raw_variance_RRI31, 5760)
# medium_rhythm_var_arr = movingaverage(Raw_variance_RRI31, 240)
# medium_rhythm_var_arr_2 = movingaverage(Raw_variance_RRI31, 240 * 3)
# medium_rhythm_var_arr_3 = movingaverage(Raw_variance_RRI31, 240 * 6)
# medium_rhythm_var_arr_4 = movingaverage(Raw_variance_RRI31, 240 * 12)
short_rhythm_var_arr_plot = movingaverage(Raw_variance_RRI31, int(240 * cycle))


long_rhythm_var_arr = short_rhythm_var_arr_plot
var_trans = hilbert(long_rhythm_var_arr)
var_trans_nomal = []
for m in var_trans:
    var_trans_nomal.append(m / abs(m))
SIvarlong = sum(var_trans_nomal) / len(var_trans_nomal)
print(SIvarlong)
seizure_phase = []
for item in seizure_timing_index:
    seizure_phase.append(var_trans_nomal[item])
SIvarlongseizure = sum(seizure_phase) / len(seizure_phase)
print(SIvarlongseizure)
var_phase = np.angle(var_trans)
phase_whole_long = var_phase
seizure_phase_var_long = []
for item in seizure_timing_index:
    seizure_phase_var_long.append(phase_whole_long[item])
print(seizure_phase_var_long)
n=0
for item in seizure_phase_var_long:
    if item <0:
        n=n+1
print(n/len(seizure_phase_var_long))
cs_pro_RRIvar=n/len(seizure_phase_var_long)

bins_number = 18
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
nRRIsvar, _, _ = pyplot.hist(phase_whole_long, bins)
nRRIsvarsei, _, _ = pyplot.hist(seizure_phase_var_long, bins)
print(nRRIsvar)
print(nRRIsvarsei)
# params = dict(projection='polar')
# fig, ax2 = pyplot.subplots(subplot_kw=params)
# ax2.bar(bins[:bins_number], nRRIsvarsei / sum(nRRIsvarsei), width=width, color='grey', alpha=0.7, edgecolor='k',
#         linewidth=2)
# ax2.set_title('RRI variance', fontsize=16)
# locs, labels = pyplot.yticks([0.1, 0.3, 0.5], ['0.1', '0.3', '0.5'], fontsize=16)
# # ax2.set_rlim([0,0.002])
# ax2.set_xticklabels(['0', '0.25$\pi$', '0.5$\pi$', '0.75$\pi$', '-$\pi$', '-0.75$\pi$', '-0.5$\pi$', '-0.25$\pi$', ],
#                     fontsize=16)
# pyplot.show()

nRRIsvarnonsei=nRRIsvar-nRRIsvarsei
pro_seizure_RRIvar=nRRIsvarsei/sum(nRRIsvarsei)
pro_nonseizure_RRIvar=nRRIsvarnonsei/sum(nRRIsvarnonsei)






# Raw_auto_RRI31=Raw_auto_RRI31_arr
Raw_auto_RRI31 = Raw_auto_RRI31_arr[0:19450]

# long_rhythm_value_arr = movingaverage(Raw_auto_RRI31, 5760)
# medium_rhythm_value_arr = movingaverage(Raw_auto_RRI31, 240)
# medium_rhythm_value_arr_2 = movingaverage(Raw_auto_RRI31, 240 * 3)
# medium_rhythm_value_arr_3 = movingaverage(Raw_auto_RRI31, 240 * 6)
# medium_rhythm_value_arr_4 = movingaverage(Raw_auto_RRI31, 240 * 12)
short_rhythm_value_arr_plot = movingaverage(Raw_auto_RRI31, int(240 * cycle))


long_rhythm_value_arr = short_rhythm_value_arr_plot
value_trans = hilbert(long_rhythm_value_arr)
value_trans_nomal = []
for m in value_trans:
    value_trans_nomal.append(m / abs(m))
SIvaluelong = sum(value_trans_nomal) / len(value_trans_nomal)
print(SIvaluelong)
seizure_phase = []
for item in seizure_timing_index:
    seizure_phase.append(value_trans_nomal[item])
SIvaluelongseizure = sum(seizure_phase) / len(seizure_phase)
print(SIvaluelongseizure)
value_phase = np.angle(value_trans)
phase_whole_value_long = value_phase
seizure_phase_value_long = []
for item in seizure_timing_index:
    seizure_phase_value_long.append(phase_whole_value_long[item])
print(seizure_phase_value_long)
n=0
for item in seizure_phase_value_long:
    if item <0:
        n=n+1
print(n/len(seizure_phase_value_long))
cs_pro_RRIauto=n/len(seizure_phase_value_long)

bins_number = 18
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
nRRIsauto, _, _ = pyplot.hist(phase_whole_value_long, bins)
nRRIsautosei, _, _ = pyplot.hist(seizure_phase_value_long, bins)
print(nRRIsauto)
print(nRRIsautosei)
# width = 2 * np.pi / bins_number
# params = dict(projection='polar')
# fig, ax2 = pyplot.subplots(subplot_kw=params)
# ax2.bar(bins[:bins_number], nRRIsautosei / sum(nRRIsautosei), width=width, color='grey', alpha=0.7, edgecolor='k',
#         linewidth=2)
# ax2.set_title('RRI autocorrelation', fontsize=16)
# locs, labels = pyplot.yticks([0.1, 0.45, 0.8], ['0.1', '0.45', '0.8'], fontsize=16)
# ax2.set_xticklabels(['0', '0.25$\pi$', '0.5$\pi$', '0.75$\pi$', '-$\pi$', '-0.75$\pi$', '-0.5$\pi$', '-0.25$\pi$', ],
#                     fontsize=16)
# pyplot.show()
nRRIsautononsei=nRRIsauto-nRRIsautosei
pro_seizure_RRIauto=nRRIsautosei/sum(nRRIsautosei)
pro_nonseizure_RRIauto=nRRIsautononsei/sum(nRRIsautononsei)


t = np.linspace(0 + 0.00416667, 0 + 0.00416667 + 0.00416667 * (len(Raw_variance_EEG_arr) - 1),
                len(Raw_variance_EEG_arr))
window_time_arr = t
a = np.where(t < 9.4505556 + 0)
print(a);
print(t[2268]);print(t[2269]);
t[0:2268] = t[0:2268] - 0 + 14.5494444
t[2268:] = t[2268:] - 9.4505556+0 - 0
print(t[2268]);print(t[0])

time_feature_arr = []
for i in range(len(t)):
    if t[i] > 24:
        time_feature_arr.append(t[i] - (t[i] // 24) * 24)
    else:
        time_feature_arr.append(t[i])
seizure_time = [time_feature_arr[5468], time_feature_arr[10332], time_feature_arr[11682], time_feature_arr[15941],
                ]
print(seizure_time)

bins_number = 18
bins = np.linspace(0, 24, bins_number + 1)
ntimes, _, _ = pyplot.hist(time_feature_arr[0:19450], bins)
ntimesei, _, _ = pyplot.hist(seizure_time, bins)
print(ntimes)
print(ntimesei)

ncircadiannonsei=ntimes-ntimesei
pro_seizure_circadian=ntimesei/sum(ntimesei)
pro_nonseizure_circadian=ncircadiannonsei/sum(ncircadiannonsei)


# #### section 2 training training
medium_rhythm_var_arr_3 = movingaverage(Raw_variance_EEG, int(240 * cycle))
long_rhythm_var_arr = medium_rhythm_var_arr_3
var_trans = hilbert(long_rhythm_var_arr)
var_phase = np.angle(var_trans)
phase_long_EEGvariance_arr = var_phase
print(len(phase_long_EEGvariance_arr));
medium_rhythm_value_arr_3 = movingaverage(Raw_auto_EEG, int(240 * cycle))
long_rhythm_value_arr = medium_rhythm_value_arr_3
value_trans = hilbert(long_rhythm_value_arr)
value_phase = np.angle(value_trans)
phase_long_EEGauto_arr = value_phase
print(len(phase_long_EEGauto_arr));
medium_rhythm_RRIvar_arr_3 = movingaverage(Raw_variance_RRI31, int(240 * cycle))
long_rhythm_RRIvar_arr = medium_rhythm_RRIvar_arr_3
var_trans = hilbert(long_rhythm_RRIvar_arr)
var_phase = np.angle(var_trans)
phase_long_RRIvar_arr = var_phase
print(len(phase_long_RRIvar_arr));
medium_rhythm_RRIvalue_arr_3 = movingaverage(Raw_auto_RRI31, int(240 * cycle))
long_rhythm_RRIvalue_arr = medium_rhythm_RRIvalue_arr_3
value_trans = hilbert(long_rhythm_RRIvalue_arr)
value_phase = np.angle(value_trans)
phase_long_RRIauto_arr = value_phase
print(len(phase_long_RRIauto_arr));



# # combined probability calculation
bins_number = 18
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
pro_eegvars_time = []
pro_eegvars_time_false = []
for i in range(len(phase_long_EEGvariance_arr)):
    if phase_long_EEGvariance_arr[i] >= bins[0] and phase_long_EEGvariance_arr[i] < bins[1]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[0])
        pro_eegvars_time.append(pro_seizure_EEGvar[0])
    elif phase_long_EEGvariance_arr[i] >= bins[1] and phase_long_EEGvariance_arr[i] < bins[2]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[1])
        pro_eegvars_time.append(pro_seizure_EEGvar[1])
    elif phase_long_EEGvariance_arr[i] >= bins[2] and phase_long_EEGvariance_arr[i] < bins[3]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[2])
        pro_eegvars_time.append(pro_seizure_EEGvar[2])
    elif phase_long_EEGvariance_arr[i] >= bins[3] and phase_long_EEGvariance_arr[i] < bins[4]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[3])
        pro_eegvars_time.append(pro_seizure_EEGvar[3])
    elif phase_long_EEGvariance_arr[i] >= bins[4] and phase_long_EEGvariance_arr[i] < bins[5]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[4])
        pro_eegvars_time.append(pro_seizure_EEGvar[4])
    elif phase_long_EEGvariance_arr[i] >= bins[5] and phase_long_EEGvariance_arr[i] < bins[6]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[5])
        pro_eegvars_time.append(pro_seizure_EEGvar[5])
    elif phase_long_EEGvariance_arr[i] >= bins[6] and phase_long_EEGvariance_arr[i] <= bins[7]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[6])
        pro_eegvars_time.append(pro_seizure_EEGvar[6])
    elif phase_long_EEGvariance_arr[i] > bins[7] and phase_long_EEGvariance_arr[i] < bins[8]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[7])
        pro_eegvars_time.append(pro_seizure_EEGvar[7])
    elif phase_long_EEGvariance_arr[i] >= bins[8] and phase_long_EEGvariance_arr[i] <= bins[9]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[8])
        pro_eegvars_time.append(pro_seizure_EEGvar[8])
    elif phase_long_EEGvariance_arr[i] > bins[9] and phase_long_EEGvariance_arr[i] < bins[10]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[9])
        pro_eegvars_time.append(pro_seizure_EEGvar[9])
    elif phase_long_EEGvariance_arr[i] >= bins[10] and phase_long_EEGvariance_arr[i] <= bins[11]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[10])
        pro_eegvars_time.append(pro_seizure_EEGvar[10])
    elif phase_long_EEGvariance_arr[i] > bins[11] and phase_long_EEGvariance_arr[i] < bins[12]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[11])
        pro_eegvars_time.append(pro_seizure_EEGvar[11])
    elif phase_long_EEGvariance_arr[i] > bins[12] and phase_long_EEGvariance_arr[i] < bins[13]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[12])
        pro_eegvars_time.append(pro_seizure_EEGvar[12])
    elif phase_long_EEGvariance_arr[i] >= bins[13] and phase_long_EEGvariance_arr[i] < bins[14]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[13])
        pro_eegvars_time.append(pro_seizure_EEGvar[13])
    elif phase_long_EEGvariance_arr[i] >= bins[14] and phase_long_EEGvariance_arr[i] < bins[15]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[14])
        pro_eegvars_time.append(pro_seizure_EEGvar[14])
    elif phase_long_EEGvariance_arr[i] >= bins[15] and phase_long_EEGvariance_arr[i] < bins[16]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[15])
        pro_eegvars_time.append(pro_seizure_EEGvar[15])
    elif phase_long_EEGvariance_arr[i] >= bins[16] and phase_long_EEGvariance_arr[i] < bins[17]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[16])
        pro_eegvars_time.append(pro_seizure_EEGvar[16])
    elif phase_long_EEGvariance_arr[i] >= bins[17]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[17])
        pro_eegvars_time.append(pro_seizure_EEGvar[17])
print(pro_eegvars_time[5468]);print(pro_eegvars_time[10332]);print(pro_eegvars_time[11682]);print(pro_eegvars_time[15941]);


pro_eegautos_time = []
pro_eegautos_time_false = []
for i in range(len(phase_long_EEGauto_arr)):
    if phase_long_EEGauto_arr[i] >= bins[0] and phase_long_EEGauto_arr[i] < bins[1]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[0])
        pro_eegautos_time.append(pro_seizure_EEGauto[0])
    elif phase_long_EEGauto_arr[i] >= bins[1] and phase_long_EEGauto_arr[i] < bins[2]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[1])
        pro_eegautos_time.append(pro_seizure_EEGauto[1])
    elif phase_long_EEGauto_arr[i] >= bins[2] and phase_long_EEGauto_arr[i] < bins[3]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[2])
        pro_eegautos_time.append(pro_seizure_EEGauto[2])
    elif phase_long_EEGauto_arr[i] >= bins[3] and phase_long_EEGauto_arr[i] < bins[4]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[3])
        pro_eegautos_time.append(pro_seizure_EEGauto[3])
    elif phase_long_EEGauto_arr[i] >= bins[4] and phase_long_EEGauto_arr[i] < bins[5]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[4])
        pro_eegautos_time.append(pro_seizure_EEGauto[4])
    elif phase_long_EEGauto_arr[i] >= bins[5] and phase_long_EEGauto_arr[i] < bins[6]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[5])
        pro_eegautos_time.append(pro_seizure_EEGauto[5])
    elif phase_long_EEGauto_arr[i] >= bins[6] and phase_long_EEGauto_arr[i] <= bins[7]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[6])
        pro_eegautos_time.append(pro_seizure_EEGauto[6])
    elif phase_long_EEGauto_arr[i] > bins[7] and phase_long_EEGauto_arr[i] < bins[8]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[7])
        pro_eegautos_time.append(pro_seizure_EEGauto[7])
    elif phase_long_EEGauto_arr[i] >= bins[8] and phase_long_EEGauto_arr[i] <= bins[9]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[8])
        pro_eegautos_time.append(pro_seizure_EEGauto[8])
    elif phase_long_EEGauto_arr[i] > bins[9] and phase_long_EEGauto_arr[i] < bins[10]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[9])
        pro_eegautos_time.append(pro_seizure_EEGauto[9])
    elif phase_long_EEGauto_arr[i] >= bins[10] and phase_long_EEGauto_arr[i] <= bins[11]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[10])
        pro_eegautos_time.append(pro_seizure_EEGauto[10])
    elif phase_long_EEGauto_arr[i] > bins[11] and phase_long_EEGauto_arr[i] < bins[12]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[11])
        pro_eegautos_time.append(pro_seizure_EEGauto[11])
    elif phase_long_EEGauto_arr[i] > bins[12] and phase_long_EEGauto_arr[i] < bins[13]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[12])
        pro_eegautos_time.append(pro_seizure_EEGauto[12])
    elif phase_long_EEGauto_arr[i] >= bins[13] and phase_long_EEGauto_arr[i] < bins[14]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[13])
        pro_eegautos_time.append(pro_seizure_EEGauto[13])
    elif phase_long_EEGauto_arr[i] >= bins[14] and phase_long_EEGauto_arr[i] < bins[15]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[14])
        pro_eegautos_time.append(pro_seizure_EEGauto[14])
    elif phase_long_EEGauto_arr[i] >= bins[15] and phase_long_EEGauto_arr[i] < bins[16]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[15])
        pro_eegautos_time.append(pro_seizure_EEGauto[15])
    elif phase_long_EEGauto_arr[i] >= bins[16] and phase_long_EEGauto_arr[i] < bins[17]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[16])
        pro_eegautos_time.append(pro_seizure_EEGauto[16])
    elif phase_long_EEGauto_arr[i] >= bins[17]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[17])
        pro_eegautos_time.append(pro_seizure_EEGauto[17])
print(pro_eegautos_time[5468]);print(pro_eegautos_time[10332]);print(pro_eegautos_time[11682]);print(pro_eegautos_time[15941]);


pro_RRIvars_time = []
pro_RRIvars_time_false = []
for i in range(len(phase_long_RRIvar_arr)):
    if phase_long_RRIvar_arr[i] >= bins[0] and phase_long_RRIvar_arr[i] < bins[1]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[0])
        pro_RRIvars_time.append(pro_seizure_RRIvar[0])
    elif phase_long_RRIvar_arr[i] >= bins[1] and phase_long_RRIvar_arr[i] < bins[2]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[1])
        pro_RRIvars_time.append(pro_seizure_RRIvar[1])
    elif phase_long_RRIvar_arr[i] >= bins[2] and phase_long_RRIvar_arr[i] < bins[3]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[2])
        pro_RRIvars_time.append(pro_seizure_RRIvar[2])
    elif phase_long_RRIvar_arr[i] >= bins[3] and phase_long_RRIvar_arr[i] < bins[4]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[3])
        pro_RRIvars_time.append(pro_seizure_RRIvar[3])
    elif phase_long_RRIvar_arr[i] >= bins[4] and phase_long_RRIvar_arr[i] < bins[5]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[4])
        pro_RRIvars_time.append(pro_seizure_RRIvar[4])
    elif phase_long_RRIvar_arr[i] >= bins[5] and phase_long_RRIvar_arr[i] < bins[6]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[5])
        pro_RRIvars_time.append(pro_seizure_RRIvar[5])
    elif phase_long_RRIvar_arr[i] >= bins[6] and phase_long_RRIvar_arr[i] <= bins[7]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[6])
        pro_RRIvars_time.append(pro_seizure_RRIvar[6])
    elif phase_long_RRIvar_arr[i] > bins[7] and phase_long_RRIvar_arr[i] < bins[8]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[7])
        pro_RRIvars_time.append(pro_seizure_RRIvar[7])
    elif phase_long_RRIvar_arr[i] >= bins[8] and phase_long_RRIvar_arr[i] <= bins[9]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[8])
        pro_RRIvars_time.append(pro_seizure_RRIvar[8])
    elif phase_long_RRIvar_arr[i] > bins[9] and phase_long_RRIvar_arr[i] < bins[10]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[9])
        pro_RRIvars_time.append(pro_seizure_RRIvar[9])
    elif phase_long_RRIvar_arr[i] >= bins[10] and phase_long_RRIvar_arr[i] <= bins[11]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[10])
        pro_RRIvars_time.append(pro_seizure_RRIvar[10])
    elif phase_long_RRIvar_arr[i] > bins[11] and phase_long_RRIvar_arr[i] < bins[12]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[11])
        pro_RRIvars_time.append(pro_seizure_RRIvar[11])
    elif phase_long_RRIvar_arr[i] > bins[12] and phase_long_RRIvar_arr[i] < bins[13]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[12])
        pro_RRIvars_time.append(pro_seizure_RRIvar[12])
    elif phase_long_RRIvar_arr[i] >= bins[13] and phase_long_RRIvar_arr[i] < bins[14]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[13])
        pro_RRIvars_time.append(pro_seizure_RRIvar[13])
    elif phase_long_RRIvar_arr[i] >= bins[14] and phase_long_RRIvar_arr[i] < bins[15]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[14])
        pro_RRIvars_time.append(pro_seizure_RRIvar[14])
    elif phase_long_RRIvar_arr[i] >= bins[15] and phase_long_RRIvar_arr[i] < bins[16]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[15])
        pro_RRIvars_time.append(pro_seizure_RRIvar[15])
    elif phase_long_RRIvar_arr[i] >= bins[16] and phase_long_RRIvar_arr[i] < bins[17]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[16])
        pro_RRIvars_time.append(pro_seizure_RRIvar[16])
    elif phase_long_RRIvar_arr[i] >= bins[17]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[17])
        pro_RRIvars_time.append(pro_seizure_RRIvar[17])
print(pro_RRIvars_time[5468]);print(pro_RRIvars_time[10332]);print(pro_RRIvars_time[11682]);print(pro_RRIvars_time[15941]);


pro_RRIautos_time = []
pro_RRIautos_time_false = []
for i in range(len(phase_long_RRIauto_arr)):
    if phase_long_RRIauto_arr[i] >= bins[0] and phase_long_RRIauto_arr[i] < bins[1]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[0])
        pro_RRIautos_time.append(pro_seizure_RRIauto[0])
    elif phase_long_RRIauto_arr[i] >= bins[1] and phase_long_RRIauto_arr[i] < bins[2]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[1])
        pro_RRIautos_time.append(pro_seizure_RRIauto[1])
    elif phase_long_RRIauto_arr[i] >= bins[2] and phase_long_RRIauto_arr[i] < bins[3]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[2])
        pro_RRIautos_time.append(pro_seizure_RRIauto[2])
    elif phase_long_RRIauto_arr[i] >= bins[3] and phase_long_RRIauto_arr[i] < bins[4]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[3])
        pro_RRIautos_time.append(pro_seizure_RRIauto[3])
    elif phase_long_RRIauto_arr[i] >= bins[4] and phase_long_RRIauto_arr[i] < bins[5]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[4])
        pro_RRIautos_time.append(pro_seizure_RRIauto[4])
    elif phase_long_RRIauto_arr[i] >= bins[5] and phase_long_RRIauto_arr[i] < bins[6]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[5])
        pro_RRIautos_time.append(pro_seizure_RRIauto[5])
    elif phase_long_RRIauto_arr[i] >= bins[6] and phase_long_RRIauto_arr[i] <= bins[7]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[6])
        pro_RRIautos_time.append(pro_seizure_RRIauto[6])
    elif phase_long_RRIauto_arr[i] > bins[7] and phase_long_RRIauto_arr[i] < bins[8]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[7])
        pro_RRIautos_time.append(pro_seizure_RRIauto[7])
    elif phase_long_RRIauto_arr[i] >= bins[8] and phase_long_RRIauto_arr[i] <= bins[9]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[8])
        pro_RRIautos_time.append(pro_seizure_RRIauto[8])
    elif phase_long_RRIauto_arr[i] > bins[9] and phase_long_RRIauto_arr[i] < bins[10]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[9])
        pro_RRIautos_time.append(pro_seizure_RRIauto[9])
    elif phase_long_RRIauto_arr[i] >= bins[10] and phase_long_RRIauto_arr[i] <= bins[11]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[10])
        pro_RRIautos_time.append(pro_seizure_RRIauto[10])
    elif phase_long_RRIauto_arr[i] > bins[11] and phase_long_RRIauto_arr[i] < bins[12]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[11])
        pro_RRIautos_time.append(pro_seizure_RRIauto[11])
    elif phase_long_RRIauto_arr[i] > bins[12] and phase_long_RRIauto_arr[i] < bins[13]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[12])
        pro_RRIautos_time.append(pro_seizure_RRIauto[12])
    elif phase_long_RRIauto_arr[i] >= bins[13] and phase_long_RRIauto_arr[i] < bins[14]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[13])
        pro_RRIautos_time.append(pro_seizure_RRIauto[13])
    elif phase_long_RRIauto_arr[i] >= bins[14] and phase_long_RRIauto_arr[i] < bins[15]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[14])
        pro_RRIautos_time.append(pro_seizure_RRIauto[14])
    elif phase_long_RRIauto_arr[i] >= bins[15] and phase_long_RRIauto_arr[i] < bins[16]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[15])
        pro_RRIautos_time.append(pro_seizure_RRIauto[15])
    elif phase_long_RRIauto_arr[i] >= bins[16] and phase_long_RRIauto_arr[i] < bins[17]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[16])
        pro_RRIautos_time.append(pro_seizure_RRIauto[16])
    elif phase_long_RRIauto_arr[i] >= bins[17]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[17])
        pro_RRIautos_time.append(pro_seizure_RRIauto[17])
print(pro_RRIautos_time[5468]);print(pro_RRIautos_time[10332]);print(pro_RRIautos_time[11682]);print(pro_RRIautos_time[15941]);






Pseizureeegvar = 4/19450;
Pnonseizureeegvar = (19450-4)/19450;
t = np.linspace(0 + 0.00416667, 0 + 0.00416667 + 0.00416667 * (len(Raw_variance_EEG) - 1), len(Raw_variance_EEG))
window_time_arr = t

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIvars_time[m]*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))


# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_RRIvars_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_RRIvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))

Pcombined = []
for m in range(len(pro_RRIvars_time)):
    P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]
    P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m])
    Pcombined.append(P1/(P1+P2))


# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_eegvars_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_eegautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_RRIvars_time)):
#     P1=pro_RRIvars_time[m]*Pseizureeegvar*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_RRIvars_time)):
#     P1=Pseizureeegvar*pro_RRIvars_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_RRIautos_time)):
#     P1=Pseizureeegvar*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))


pyplot.figure(figsize=(12, 5))
pyplot.plot(window_time_arr, Pcombined)
pyplot.title('Combined probability in QLD0290', fontsize=15)
pyplot.annotate('', xy=(22.79056, np.max(Pcombined)), xytext=(22.79056, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(43.05778, np.max(Pcombined)), xytext=(43.05778, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(48.68167, np.max(Pcombined)), xytext=(48.68167, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.annotate('', xy=(66.425277, np.max(Pcombined)), xytext=(66.425277, np.max(Pcombined) + 0.00000000001),
                arrowprops=dict(facecolor='black', shrink=0.05))
pyplot.tight_layout()
pyplot.xlabel('Time(h)', fontsize=15)
pyplot.ylabel('seizure probability', fontsize=15)
pyplot.show()
pro=[]
for item in seizure_timing_index:
    pro.append(float(Pcombined[item]))
    print(Pcombined[item])
print(pro)
Th1=np.min(pro)
print(Th1)


Sen=[]; FPR=[];
seizure_count=len(seizure_timing_index)
Pcombined_train=Pcombined
Pcombined_train=split(Pcombined_train,240)
print(len(Pcombined_train))
index=[]
for i in range(len(Pcombined_train)):
    for item in Pcombined_train[i]:
        if item >= Th1:
            index.append(240*i+0)
# print(window_time_arr[index])
a=np.unique(window_time_arr[index])
# print(a);
print(len(a))
time_arr=[22.79056,43.05778,48.68167,66.425277]
k1=0
n_arr=[]
pretime=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k1=k1+1
            n_arr.append(n)
            pretime.append(m - n)
print(k1)
time_arr=[22.79056,43.05778,48.68167,66.425277]
k2=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k2=k2+1
            n_arr.append(n)
print(k2)
Sen.append(k1/seizure_count)
FPR.append((len(a)-k2)/len(Pcombined_train))
index=[]
for i in range(len(Pcombined_train)):
    for item in Pcombined_train[i]:
        if item >= 0.3*Th1:
            index.append(240*i+0)
# print(window_time_arr[index])
a=np.unique(window_time_arr[index])
# print(a);
print(len(a))
time_arr=[22.79056,43.05778,48.68167,66.425277]
k1=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k1=k1+1
            n_arr.append(n)
            pretime.append(m - n)
print(k1)
time_arr=[22.79056,43.05778,48.68167,66.425277]
k2=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k2=k2+1
            n_arr.append(n)
print(k2)
Sen.append(k1/seizure_count)
FPR.append((len(a)-k2)/len(Pcombined_train))
index=[]
for i in range(len(Pcombined_train)):
    for item in Pcombined_train[i]:
        if item >= 0.6*Th1:
            index.append(240*i+0)
# print(window_time_arr[index])
a=np.unique(window_time_arr[index])
# print(a);
print(len(a))
time_arr=[22.79056,43.05778,48.68167,66.425277]
k1=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k1=k1+1
            n_arr.append(n)
            pretime.append(m - n)
print(k1)
time_arr=[22.79056,43.05778,48.68167,66.425277]
k2=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k2=k2+1
            n_arr.append(n)
print(k2)
Sen.append(k1/seizure_count)
FPR.append((len(a)-k2)/len(Pcombined_train))
index=[]
for i in range(len(Pcombined_train)):
    for item in Pcombined_train[i]:
        if item >= 1.2*Th1:
            index.append(240*i+0)
# print(window_time_arr[index])
a=np.unique(window_time_arr[index])
# print(a);
print(len(a))
time_arr=[22.79056,43.05778,48.68167,66.425277]
k1=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k1=k1+1
            n_arr.append(n)
            pretime.append(m - n)
print(k1)
time_arr=[22.79056,43.05778,48.68167,66.425277]
k2=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k2=k2+1
            n_arr.append(n)
print(k2)
Sen.append(k1/seizure_count)
FPR.append((len(a)-k2)/len(Pcombined_train))
index=[]
for i in range(len(Pcombined_train)):
    for item in Pcombined_train[i]:
        if item >= 2*Th1:
            index.append(240*i+0)
# print(window_time_arr[index])
a=np.unique(window_time_arr[index])
# print(a);
print(len(a))
time_arr=[22.79056,43.05778,48.68167,66.425277]
k1=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k1=k1+1
            n_arr.append(n)
            pretime.append(m - n)
print(k1)
time_arr=[22.79056,43.05778,48.68167,66.425277]
k2=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k2=k2+1
            n_arr.append(n)
print(k2)
Sen.append(k1/seizure_count)
FPR.append((len(a)-k2)/len(Pcombined_train))
# print(pretime)
print(np.mean(pretime));
print(Sen);print(FPR);
print(np.mean(Sen));print(np.mean(FPR));
Sen.append(0);Sen.append(1);
FPR.append(0);FPR.append(1);
print('computed AUC using sklearn.metrics.auc: {}'.format(auc(sorted(FPR),sorted(Sen))))






t = np.linspace(0 + 0.00416667, 0 + 0.00416667 + 0.00416667 * (len(Raw_variance_EEG_arr) - 1),
                len(Raw_variance_EEG_arr))
a = np.where(t < 9.4505556 + 0)
t[0:2268] = t[0:2268] - 0 + 14.5494444
t[2268:] = t[2268:] - 9.4505556+0 - 0
time_feature_arr = []
for i in range(len(t)):
    if t[i] > 24:
        time_feature_arr.append(t[i] - (t[i] // 24) * 24)
    else:
        time_feature_arr.append(t[i])

bins_number = 18
bins = np.linspace(0, 24, bins_number + 1)
pro_circadian_time = []
pro_circadian_time_false = []
for i in range(len(time_feature_arr)):
    if time_feature_arr[i] >= bins[0] and time_feature_arr[i] <= bins[1]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[0])
        pro_circadian_time.append(pro_seizure_circadian[0])
    elif time_feature_arr[i] > bins[1] and time_feature_arr[i] < bins[2]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[1])
        pro_circadian_time.append(pro_seizure_circadian[1])
    elif time_feature_arr[i] >= bins[2] and time_feature_arr[i] < bins[3]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[2])
        pro_circadian_time.append(pro_seizure_circadian[2])
    elif time_feature_arr[i] >= bins[3] and time_feature_arr[i] < bins[4]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[3])
        pro_circadian_time.append(pro_seizure_circadian[3])
    elif time_feature_arr[i] >= bins[4] and time_feature_arr[i] < bins[5]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[4])
        pro_circadian_time.append(pro_seizure_circadian[4])
    elif time_feature_arr[i] >= bins[5] and time_feature_arr[i] <= bins[6]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[5])
        pro_circadian_time.append(pro_seizure_circadian[5])
    elif time_feature_arr[i] > bins[6] and time_feature_arr[i] < bins[7]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[6])
        pro_circadian_time.append(pro_seizure_circadian[6])
    elif time_feature_arr[i] >= bins[7] and time_feature_arr[i] <= bins[8]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[7])
        pro_circadian_time.append(pro_seizure_circadian[7])
    elif time_feature_arr[i] > bins[8] and time_feature_arr[i] < bins[9]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[8])
        pro_circadian_time.append(pro_seizure_circadian[8])
    elif time_feature_arr[i] >= bins[9] and time_feature_arr[i] < bins[10]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[9])
        pro_circadian_time.append(pro_seizure_circadian[9])
    elif time_feature_arr[i] >= bins[10] and time_feature_arr[i] < bins[11]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[10])
        pro_circadian_time.append(pro_seizure_circadian[10])
    elif time_feature_arr[i] >= bins[11] and time_feature_arr[i] < bins[12]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[11])
        pro_circadian_time.append(pro_seizure_circadian[11])
    elif time_feature_arr[i] >= bins[12] and time_feature_arr[i] < bins[13]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[12])
        pro_circadian_time.append(pro_seizure_circadian[12])
    elif time_feature_arr[i] >= bins[13] and time_feature_arr[i] < bins[14]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[13])
        pro_circadian_time.append(pro_seizure_circadian[13])
    elif time_feature_arr[i] >= bins[14] and time_feature_arr[i] < bins[15]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[14])
        pro_circadian_time.append(pro_seizure_circadian[14])
    elif time_feature_arr[i] >= bins[15] and time_feature_arr[i] < bins[16]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[15])
        pro_circadian_time.append(pro_seizure_circadian[15])
    elif time_feature_arr[i] >= bins[16] and time_feature_arr[i] < bins[17]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[16])
        pro_circadian_time.append(pro_seizure_circadian[16])
    elif time_feature_arr[i] >= bins[17] and time_feature_arr[i] <= bins[18]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[17])
        pro_circadian_time.append(pro_seizure_circadian[17])
print(pro_circadian_time[5468]);print(pro_circadian_time[10332]);print(pro_circadian_time[11682]);print(pro_circadian_time[15941]);

pro= [pro_circadian_time[5468],pro_circadian_time[10332],pro_circadian_time[11682],pro_circadian_time[15941]];


## section 3 froecast
t = np.linspace(0, 0 + 0.00416667 * (len(Raw_variance_EEG_arr) - 1), len(Raw_variance_EEG_arr))
t_window_arr = t
fore_arr_EEGvars = []
for k in range(81, 82):
    variance_arr = Raw_variance_EEG_arr[0:(19450 + 240 * k)]
    long_rhythm_var_arr = movingaverage(variance_arr, 240 *6)
    pyplot.figure(figsize=(6, 3))
    pyplot.title('EEG variance')
    pyplot.ylabel('Voltage ($\mathregular{v^2}$)')
    pyplot.xlabel('Time(h)')
    pyplot.plot(t_window_arr[240 * 24:(19450 + 240 * k)], long_rhythm_var_arr[240 * 24:], 'orange')
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD0290/newpar72p1d1q0_Cz_forecast81hsignal_3hcycle_EEGvar_QLD0290.csv',
                         sep=',', header=None)
forecast_var_EEG = csv_reader.values
forecast_var_EEG_arr = []
for item in forecast_var_EEG:
    forecast_var_EEG_arr = forecast_var_EEG_arr + list(item)
t = np.linspace(t_window_arr[19450], t_window_arr[19450] + 0.1666667 * (len(forecast_var_EEG_arr) - 1),
                len(forecast_var_EEG_arr))
pyplot.plot(t, forecast_var_EEG_arr, 'k', label='forecast EEG var')
pyplot.legend()
pyplot.show()

fore_arr_EEGauto = []
for k in range(81, 82):
    auto_arr = Raw_auto_EEG_arr[0:(19450 + 240 * k)]
    long_rhythm_auto_arr = movingaverage(auto_arr, 240 * 6)
    pyplot.figure(figsize=(6, 3))
    pyplot.title('EEG autocorrelation')
    pyplot.xlabel('time(h)')
    pyplot.plot(t_window_arr[240 * 24:(19450 + 240 * k)], long_rhythm_auto_arr[240 * 24:], 'orange')
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD0290/cycles6h_Cz_forecast81hsignal_3hcycle_EEGauto_QLD0290.csv', sep=',',
    header=None)
forecast_auto_EEG = csv_reader.values
forecast_auto_EEG_arr = []
for item in forecast_auto_EEG:
    forecast_auto_EEG_arr = forecast_auto_EEG_arr + list(item)
t = np.linspace(t_window_arr[19450], t_window_arr[19450] + 0.1666667 * (len(forecast_auto_EEG_arr) - 1),
                len(forecast_auto_EEG_arr))
pyplot.plot(t, forecast_auto_EEG_arr, 'k', label='forecast EEG auto')
pyplot.legend()
pyplot.show()


fore_arr_RRIvars = []
for k in range(81, 82):
    variance_arr = Raw_variance_RRI31_arr[0:(19450 + 240 * k)]
    long_rhythm_var_arr = movingaverage(variance_arr, 240 * 6)
    pyplot.figure(figsize=(6, 3))
    pyplot.title('RRI variance')
    pyplot.ylabel('Second ($\mathregular{s^2}$)')
    pyplot.xlabel('Time(h)')
    pyplot.plot(t_window_arr[240 * 24:(19450 + 240 * k)], long_rhythm_var_arr[240 * 24:], 'orange')
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD0290/newpar72p1d0q0_ch31_forecast81hsignal_3hcycle_RRIvar_QLD0290.csv', sep=',',header=None)
forecast_var_RRI31 = csv_reader.values
forecast_var_RRI31_arr = []
for item in forecast_var_RRI31:
    forecast_var_RRI31_arr = forecast_var_RRI31_arr + list(item)
t = np.linspace(t_window_arr[19450], t_window_arr[19450] + 0.1666667 * (len(forecast_var_RRI31_arr) - 1),
                len(forecast_var_RRI31_arr))
pyplot.plot(t, forecast_var_RRI31_arr, 'k', label='forecast RRI var')
pyplot.legend()
pyplot.show()

fore_arr_RRIautos = []
save_data_RRIautos = []
for k in range(81, 82):
    auto_arr = Raw_auto_RRI31_arr[0:19450 + 240 * k]
    long_rhythm_auto_arr = movingaverage(auto_arr, 240 * 6)
    pyplot.figure(figsize=(6, 3))
    pyplot.title('RRI autocorrelation')
    pyplot.xlabel('Time(h)')
    pyplot.plot(t_window_arr[240 * 24:19450 + 240 * k], long_rhythm_auto_arr[240 * 24:], 'orange')
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD0290/cycles6h_ch31_forecast81hsignal_3hcycle_RRIauto_QLD0290.csv', sep=',',
    header=None)
forecast_auto_RRI31 = csv_reader.values
forecast_auto_RRI31_arr = []
for item in forecast_auto_RRI31:
    forecast_auto_RRI31_arr = forecast_auto_RRI31_arr + list(item)
t = np.linspace(t_window_arr[19450], t_window_arr[19450] + 0.1666667 * (len(forecast_auto_RRI31_arr) - 1),
                len(forecast_auto_RRI31_arr))
pyplot.plot(t, forecast_auto_RRI31_arr, 'k', label='forecast RRI auto')
pyplot.legend()
pyplot.show()
# print(len(forecast_var_EEG_arr));print(len(forecast_auto_EEG_arr));print(len(forecast_var_RRI31_arr));print(len(forecast_auto_RRI31_arr));





# ### predict, forecast data
var_trans = hilbert(forecast_var_EEG_arr)
var_phase = np.angle(var_trans)
rolmean_short_EEGvar = var_phase
var_trans = hilbert(forecast_auto_EEG_arr)
var_phase = np.angle(var_trans)
rolmean_short_EEGauto = var_phase
var_trans = hilbert(forecast_var_RRI31_arr)
var_phase = np.angle(var_trans)
rolmean_short_RRIvar = var_phase
var_trans = hilbert(forecast_auto_RRI31_arr)
var_phase = np.angle(var_trans)
rolmean_short_RRIauto = var_phase
print(len(rolmean_short_EEGvar));print(len(rolmean_short_EEGauto));
print(len(rolmean_short_RRIvar));print(len(rolmean_short_RRIauto));

bins_number = 18
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
pro_eegvars_time = []
pro_eegvars_time_false = []
for i in range(len(rolmean_short_EEGvar)):
    if rolmean_short_EEGvar[i] >= bins[0] and rolmean_short_EEGvar[i] < bins[1]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[0])
        pro_eegvars_time.append(pro_seizure_EEGvar[0])
    elif rolmean_short_EEGvar[i] >= bins[1] and rolmean_short_EEGvar[i] < bins[2]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[1])
        pro_eegvars_time.append(pro_seizure_EEGvar[1])
    elif rolmean_short_EEGvar[i] >= bins[2] and rolmean_short_EEGvar[i] < bins[3]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[2])
        pro_eegvars_time.append(pro_seizure_EEGvar[2])
    elif rolmean_short_EEGvar[i] >= bins[3] and rolmean_short_EEGvar[i] < bins[4]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[3])
        pro_eegvars_time.append(pro_seizure_EEGvar[3])
    elif rolmean_short_EEGvar[i] >= bins[4] and rolmean_short_EEGvar[i] < bins[5]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[4])
        pro_eegvars_time.append(pro_seizure_EEGvar[4])
    elif rolmean_short_EEGvar[i] >= bins[5] and rolmean_short_EEGvar[i] < bins[6]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[5])
        pro_eegvars_time.append(pro_seizure_EEGvar[5])
    elif rolmean_short_EEGvar[i] >= bins[6] and rolmean_short_EEGvar[i] <= bins[7]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[6])
        pro_eegvars_time.append(pro_seizure_EEGvar[6])
    elif rolmean_short_EEGvar[i] > bins[7] and rolmean_short_EEGvar[i] < bins[8]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[7])
        pro_eegvars_time.append(pro_seizure_EEGvar[7])
    elif rolmean_short_EEGvar[i] >= bins[8] and rolmean_short_EEGvar[i] <= bins[9]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[8])
        pro_eegvars_time.append(pro_seizure_EEGvar[8])
    elif rolmean_short_EEGvar[i] > bins[9] and rolmean_short_EEGvar[i] < bins[10]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[9])
        pro_eegvars_time.append(pro_seizure_EEGvar[9])
    elif rolmean_short_EEGvar[i] >= bins[10] and rolmean_short_EEGvar[i] <= bins[11]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[10])
        pro_eegvars_time.append(pro_seizure_EEGvar[10])
    elif rolmean_short_EEGvar[i] > bins[11] and rolmean_short_EEGvar[i] < bins[12]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[11])
        pro_eegvars_time.append(pro_seizure_EEGvar[11])
    elif rolmean_short_EEGvar[i] > bins[12] and rolmean_short_EEGvar[i] < bins[13]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[12])
        pro_eegvars_time.append(pro_seizure_EEGvar[12])
    elif rolmean_short_EEGvar[i] >= bins[13] and rolmean_short_EEGvar[i] < bins[14]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[13])
        pro_eegvars_time.append(pro_seizure_EEGvar[13])
    elif rolmean_short_EEGvar[i] >= bins[14] and rolmean_short_EEGvar[i] < bins[15]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[14])
        pro_eegvars_time.append(pro_seizure_EEGvar[14])
    elif rolmean_short_EEGvar[i] >= bins[15] and rolmean_short_EEGvar[i] < bins[16]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[15])
        pro_eegvars_time.append(pro_seizure_EEGvar[15])
    elif rolmean_short_EEGvar[i] >= bins[16] and rolmean_short_EEGvar[i] < bins[17]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[16])
        pro_eegvars_time.append(pro_seizure_EEGvar[16])
    elif rolmean_short_EEGvar[i] >= bins[17]:
        pro_eegvars_time_false.append(pro_nonseizure_EEGvar[17])
        pro_eegvars_time.append(pro_seizure_EEGvar[17])
print(len(pro_eegvars_time))
pro_eegautos_time = []
pro_eegautos_time_false = []
for i in range(len(rolmean_short_EEGauto)):
    if rolmean_short_EEGauto[i] >= bins[0] and rolmean_short_EEGauto[i] < bins[1]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[0])
        pro_eegautos_time.append(pro_seizure_EEGauto[0])
    elif rolmean_short_EEGauto[i] >= bins[1] and rolmean_short_EEGauto[i] < bins[2]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[1])
        pro_eegautos_time.append(pro_seizure_EEGauto[1])
    elif rolmean_short_EEGauto[i] >= bins[2] and rolmean_short_EEGauto[i] < bins[3]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[2])
        pro_eegautos_time.append(pro_seizure_EEGauto[2])
    elif rolmean_short_EEGauto[i] >= bins[3] and rolmean_short_EEGauto[i] < bins[4]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[3])
        pro_eegautos_time.append(pro_seizure_EEGauto[3])
    elif rolmean_short_EEGauto[i] >= bins[4] and rolmean_short_EEGauto[i] < bins[5]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[4])
        pro_eegautos_time.append(pro_seizure_EEGauto[4])
    elif rolmean_short_EEGauto[i] >= bins[5] and rolmean_short_EEGauto[i] < bins[6]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[5])
        pro_eegautos_time.append(pro_seizure_EEGauto[5])
    elif rolmean_short_EEGauto[i] >= bins[6] and rolmean_short_EEGauto[i] <= bins[7]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[6])
        pro_eegautos_time.append(pro_seizure_EEGauto[6])
    elif rolmean_short_EEGauto[i] > bins[7] and rolmean_short_EEGauto[i] < bins[8]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[7])
        pro_eegautos_time.append(pro_seizure_EEGauto[7])
    elif rolmean_short_EEGauto[i] >= bins[8] and rolmean_short_EEGauto[i] <= bins[9]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[8])
        pro_eegautos_time.append(pro_seizure_EEGauto[8])
    elif rolmean_short_EEGauto[i] > bins[9] and rolmean_short_EEGauto[i] < bins[10]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[9])
        pro_eegautos_time.append(pro_seizure_EEGauto[9])
    elif rolmean_short_EEGauto[i] >= bins[10] and rolmean_short_EEGauto[i] <= bins[11]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[10])
        pro_eegautos_time.append(pro_seizure_EEGauto[10])
    elif rolmean_short_EEGauto[i] > bins[11] and rolmean_short_EEGauto[i] < bins[12]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[11])
        pro_eegautos_time.append(pro_seizure_EEGauto[11])
    elif rolmean_short_EEGauto[i] > bins[12] and rolmean_short_EEGauto[i] < bins[13]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[12])
        pro_eegautos_time.append(pro_seizure_EEGauto[12])
    elif rolmean_short_EEGauto[i] >= bins[13] and rolmean_short_EEGauto[i] < bins[14]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[13])
        pro_eegautos_time.append(pro_seizure_EEGauto[13])
    elif rolmean_short_EEGauto[i] >= bins[14] and rolmean_short_EEGauto[i] < bins[15]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[14])
        pro_eegautos_time.append(pro_seizure_EEGauto[14])
    elif rolmean_short_EEGauto[i] >= bins[15] and rolmean_short_EEGauto[i] < bins[16]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[15])
        pro_eegautos_time.append(pro_seizure_EEGauto[15])
    elif rolmean_short_EEGauto[i] >= bins[16] and rolmean_short_EEGauto[i] < bins[17]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[16])
        pro_eegautos_time.append(pro_seizure_EEGauto[16])
    elif rolmean_short_EEGauto[i] >= bins[17]:
        pro_eegautos_time_false.append(pro_nonseizure_EEGauto[17])
        pro_eegautos_time.append(pro_seizure_EEGauto[17])
print(len(pro_eegautos_time))
pro_RRIvars_time = []
pro_RRIvars_time_false = []
for i in range(len(rolmean_short_RRIvar)):
    if rolmean_short_RRIvar[i] >= bins[0] and rolmean_short_RRIvar[i] < bins[1]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[0])
        pro_RRIvars_time.append(pro_seizure_RRIvar[0])
    elif rolmean_short_RRIvar[i] >= bins[1] and rolmean_short_RRIvar[i] < bins[2]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[1])
        pro_RRIvars_time.append(pro_seizure_RRIvar[1])
    elif rolmean_short_RRIvar[i] >= bins[2] and rolmean_short_RRIvar[i] < bins[3]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[2])
        pro_RRIvars_time.append(pro_seizure_RRIvar[2])
    elif rolmean_short_RRIvar[i] >= bins[3] and rolmean_short_RRIvar[i] < bins[4]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[3])
        pro_RRIvars_time.append(pro_seizure_RRIvar[3])
    elif rolmean_short_RRIvar[i] >= bins[4] and rolmean_short_RRIvar[i] < bins[5]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[4])
        pro_RRIvars_time.append(pro_seizure_RRIvar[4])
    elif rolmean_short_RRIvar[i] >= bins[5] and rolmean_short_RRIvar[i] < bins[6]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[5])
        pro_RRIvars_time.append(pro_seizure_RRIvar[5])
    elif rolmean_short_RRIvar[i] >= bins[6] and rolmean_short_RRIvar[i] <= bins[7]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[6])
        pro_RRIvars_time.append(pro_seizure_RRIvar[6])
    elif rolmean_short_RRIvar[i] > bins[7] and rolmean_short_RRIvar[i] < bins[8]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[7])
        pro_RRIvars_time.append(pro_seizure_RRIvar[7])
    elif rolmean_short_RRIvar[i] >= bins[8] and rolmean_short_RRIvar[i] <= bins[9]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[8])
        pro_RRIvars_time.append(pro_seizure_RRIvar[8])
    elif rolmean_short_RRIvar[i] > bins[9] and rolmean_short_RRIvar[i] < bins[10]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[9])
        pro_RRIvars_time.append(pro_seizure_RRIvar[9])
    elif rolmean_short_RRIvar[i] >= bins[10] and rolmean_short_RRIvar[i] <= bins[11]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[10])
        pro_RRIvars_time.append(pro_seizure_RRIvar[10])
    elif rolmean_short_RRIvar[i] > bins[11] and rolmean_short_RRIvar[i] < bins[12]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[11])
        pro_RRIvars_time.append(pro_seizure_RRIvar[11])
    elif rolmean_short_RRIvar[i] > bins[12] and rolmean_short_RRIvar[i] < bins[13]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[12])
        pro_RRIvars_time.append(pro_seizure_RRIvar[12])
    elif rolmean_short_RRIvar[i] >= bins[13] and rolmean_short_RRIvar[i] < bins[14]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[13])
        pro_RRIvars_time.append(pro_seizure_RRIvar[13])
    elif rolmean_short_RRIvar[i] >= bins[14] and rolmean_short_RRIvar[i] < bins[15]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[14])
        pro_RRIvars_time.append(pro_seizure_RRIvar[14])
    elif rolmean_short_RRIvar[i] >= bins[15] and rolmean_short_RRIvar[i] < bins[16]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[15])
        pro_RRIvars_time.append(pro_seizure_RRIvar[15])
    elif rolmean_short_RRIvar[i] >= bins[16] and rolmean_short_RRIvar[i] < bins[17]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[16])
        pro_RRIvars_time.append(pro_seizure_RRIvar[16])
    elif rolmean_short_RRIvar[i] >= bins[17]:
        pro_RRIvars_time_false.append(pro_nonseizure_RRIvar[17])
        pro_RRIvars_time.append(pro_seizure_RRIvar[17])
print(len(pro_RRIvars_time))
pro_RRIautos_time = []
pro_RRIautos_time_false = []
for i in range(len(rolmean_short_RRIauto)):
    if rolmean_short_RRIauto[i] >= bins[0] and rolmean_short_RRIauto[i] < bins[1]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[0])
        pro_RRIautos_time.append(pro_seizure_RRIauto[0])
    elif rolmean_short_RRIauto[i] >= bins[1] and rolmean_short_RRIauto[i] < bins[2]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[1])
        pro_RRIautos_time.append(pro_seizure_RRIauto[1])
    elif rolmean_short_RRIauto[i] >= bins[2] and rolmean_short_RRIauto[i] < bins[3]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[2])
        pro_RRIautos_time.append(pro_seizure_RRIauto[2])
    elif rolmean_short_RRIauto[i] >= bins[3] and rolmean_short_RRIauto[i] < bins[4]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[3])
        pro_RRIautos_time.append(pro_seizure_RRIauto[3])
    elif rolmean_short_RRIauto[i] >= bins[4] and rolmean_short_RRIauto[i] < bins[5]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[4])
        pro_RRIautos_time.append(pro_seizure_RRIauto[4])
    elif rolmean_short_RRIauto[i] >= bins[5] and rolmean_short_RRIauto[i] < bins[6]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[5])
        pro_RRIautos_time.append(pro_seizure_RRIauto[5])
    elif rolmean_short_RRIauto[i] >= bins[6] and rolmean_short_RRIauto[i] <= bins[7]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[6])
        pro_RRIautos_time.append(pro_seizure_RRIauto[6])
    elif rolmean_short_RRIauto[i] > bins[7] and rolmean_short_RRIauto[i] < bins[8]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[7])
        pro_RRIautos_time.append(pro_seizure_RRIauto[7])
    elif rolmean_short_RRIauto[i] >= bins[8] and rolmean_short_RRIauto[i] <= bins[9]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[8])
        pro_RRIautos_time.append(pro_seizure_RRIauto[8])
    elif rolmean_short_RRIauto[i] > bins[9] and rolmean_short_RRIauto[i] < bins[10]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[9])
        pro_RRIautos_time.append(pro_seizure_RRIauto[9])
    elif rolmean_short_RRIauto[i] >= bins[10] and rolmean_short_RRIauto[i] <= bins[11]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[10])
        pro_RRIautos_time.append(pro_seizure_RRIauto[10])
    elif rolmean_short_RRIauto[i] > bins[11] and rolmean_short_RRIauto[i] < bins[12]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[11])
        pro_RRIautos_time.append(pro_seizure_RRIauto[11])
    elif rolmean_short_RRIauto[i] > bins[12] and rolmean_short_RRIauto[i] < bins[13]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[12])
        pro_RRIautos_time.append(pro_seizure_RRIauto[12])
    elif rolmean_short_RRIauto[i] >= bins[13] and rolmean_short_RRIauto[i] < bins[14]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[13])
        pro_RRIautos_time.append(pro_seizure_RRIauto[13])
    elif rolmean_short_RRIauto[i] >= bins[14] and rolmean_short_RRIauto[i] < bins[15]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[14])
        pro_RRIautos_time.append(pro_seizure_RRIauto[14])
    elif rolmean_short_RRIauto[i] >= bins[15] and rolmean_short_RRIauto[i] < bins[16]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[15])
        pro_RRIautos_time.append(pro_seizure_RRIauto[15])
    elif rolmean_short_RRIauto[i] >= bins[16] and rolmean_short_RRIauto[i] < bins[17]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[16])
        pro_RRIautos_time.append(pro_seizure_RRIauto[16])
    elif rolmean_short_RRIauto[i] >= bins[17]:
        pro_RRIautos_time_false.append(pro_nonseizure_RRIauto[17])
        pro_RRIautos_time.append(pro_seizure_RRIauto[17])
print(len(pro_RRIautos_time))

Pseizureeegvar = 4/19450;
Pnonseizureeegvar = (19450-4)/19450;

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIvars_time[m]*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

Pcombined = []
for m in range(len(pro_eegvars_time)):
    P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]
    P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m])
    Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_RRIvars_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_RRIvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_eegvars_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_eegautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_RRIvars_time)):
#     P1=pro_RRIvars_time[m]*Pseizureeegvar*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_RRIvars_time)):
#     P1=Pseizureeegvar*pro_RRIvars_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined = []
# for m in range(len(pro_RRIautos_time)):
#     P1=Pseizureeegvar*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))


pyplot.figure(figsize=(8, 4))
RRI_timewindow_arr = t
pyplot.plot(RRI_timewindow_arr, Pcombined)
pyplot.annotate('',xy=(93.82222,np.max(Pcombined)),xytext=(93.82222,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
pyplot.annotate('',xy=(117.80083,np.max(Pcombined)),xytext=(117.80083,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
pyplot.annotate('',xy=(121.36472,np.max(Pcombined)),xytext=(121.36472,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='k',shrink=0.05))
pyplot.annotate('',xy=(140.3647,np.max(Pcombined)),xytext=(140.3647,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
pyplot.annotate('',xy=(142.20333,np.max(Pcombined)),xytext=(142.20333,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
pyplot.hlines(Th1, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'r')
pyplot.title('Forecast seizures in QLD0290')
pyplot.xlabel('Time(h)')
pyplot.ylabel('Seizure probability')
pyplot.show()


## seizure clusters
Pcombined_X=Pcombined
Pcombined=split(Pcombined,6)
print(len(Pcombined))

Sen=[]; FPR=[];
seizure_count_test=len([93.82222,117.80083,121.36472,140.3647])

index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= Th1:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[93.82222,117.80083,121.36472,140.3647]
k1=0
n_arr=[]
pretime=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k1=k1+1
            n_arr.append(n)
            pretime.append(m-n)
print(k1)
time_arr = [93.82222,117.80083,121.36472,140.3647,142.20333]
k2=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k2=k2+1
            n_arr.append(n)
print(k2)
Sen.append(k1/seizure_count_test)
FPR.append((len(a)-k2)/len(Pcombined))
index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= 0.3*Th1:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[93.82222,117.80083,121.36472,140.3647]
k1=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k1=k1+1
            n_arr.append(n)
            pretime.append(m - n)
print(k1)
time_arr = [93.82222,117.80083,121.36472,140.3647,142.20333]
k2=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k2=k2+1
            n_arr.append(n)
print(k2)
Sen.append(k1/seizure_count_test)
FPR.append((len(a)-k2)/len(Pcombined))
index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= 0.6*Th1:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[93.82222,117.80083,121.36472,140.3647]
k1=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k1=k1+1
            n_arr.append(n)
            pretime.append(m - n)
print(k1)
time_arr = [93.82222,117.80083,121.36472,140.3647,142.20333]
k2=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k2=k2+1
            n_arr.append(n)
print(k2)
Sen.append(k1/seizure_count_test)
FPR.append((len(a)-k2)/len(Pcombined))
index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= 1.2*Th1:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[93.82222,117.80083,121.36472,140.3647]
k1=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k1=k1+1
            n_arr.append(n)
            pretime.append(m - n)
print(k1)
time_arr =[93.82222,117.80083,121.36472,140.3647,142.20333]
k2=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k2=k2+1
            n_arr.append(n)
print(k2)
Sen.append(k1/seizure_count_test)
FPR.append((len(a)-k2)/len(Pcombined))
index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= 2*Th1:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[93.82222,117.80083,121.36472,140.3647]
k1=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k1=k1+1
            n_arr.append(n)
            pretime.append(m - n)
print(k1)
time_arr = [93.82222,117.80083,121.36472,140.3647,142.20333]
k2=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k2=k2+1
            n_arr.append(n)
print(k2)
Sen.append(k1/seizure_count_test)
FPR.append((len(a)-k2)/len(Pcombined))
print(pretime)
print(np.mean(pretime))

print(Sen);print(FPR);
print(np.mean(Sen));print(np.mean(FPR));
Sen.append(0);Sen.append(1);
FPR.append(0);FPR.append(1);
print('computed AUC using sklearn.metrics.auc: {}'.format(auc(sorted(FPR),sorted(Sen))))
AUC=auc(sorted(FPR),sorted(Sen))


Pcombined = split(Pcombined_X, 6)
print(len(Pcombined))
time_arr_arr=[]
AUC_cs_arr=[]
for i in range(5000):
    time_arr = np.random.uniform(low=t_window_arr[19450], high=t_window_arr[-1], size=4)
    time_arr_arr.append(time_arr)
    time_arr=np.sort(time_arr)

    index = []
    for i in range(len(Pcombined)):
        for item in Pcombined[i]:
            if item >= Th1:
                index.append(6 * i + 0)
    # print(RRI_timewindow_arr[index])
    a1 = np.unique(RRI_timewindow_arr[index])
    # print(a1);
    # print(len(a1))
    k1 = 0
    n_arr = []
    for m in time_arr:
        for n in a1:
            if m - n <= 1 and m - n >= 0:
                k1 = k1 + 1
                n_arr.append(n)
    # print(k1)
    index = []
    for i in range(len(Pcombined)):
        for item in Pcombined[i]:
            if item >= 0.3 * Th1:
                index.append(6 * i + 0)
    # print(RRI_timewindow_arr[index])
    a2 = np.unique(RRI_timewindow_arr[index])
    # print(a2);
    # print(len(a2))
    k2 = 0
    n_arr = []
    for m in time_arr:
        for n in a2:
            if m - n <= 1 and m - n >= 0:
                k2 = k2 + 1
                n_arr.append(n)
    # print(k2)
    index = []
    for i in range(len(Pcombined)):
        for item in Pcombined[i]:
            if item >= 0.6 * Th1:
                index.append(6 * i + 0)
    # print(RRI_timewindow_arr[index])
    a3 = np.unique(RRI_timewindow_arr[index])
    # print(a3);
    # print(len(a3))
    k3 = 0
    n_arr = []
    for m in time_arr:
        for n in a3:
            if m - n <= 1 and m - n >= 0:
                k3 = k3 + 1
                n_arr.append(n)
    # print(k3)
    index = []
    for i in range(len(Pcombined)):
        for item in Pcombined[i]:
            if item >= 1.2 * Th1:
                index.append(6 * i + 0)
    # print(RRI_timewindow_arr[index])
    a4 = np.unique(RRI_timewindow_arr[index])
    # print(a);
    # print(len(a4))
    k4 = 0
    n_arr = []
    for m in time_arr:
        for n in a4:
            if m - n <= 1 and m - n >= 0:
                k4 = k4 + 1
                n_arr.append(n)
    # print(k4)
    index = []
    for i in range(len(Pcombined)):
        for item in Pcombined[i]:
            if item >= 2 * Th1:
                index.append(6 * i + 0)
    # print(RRI_timewindow_arr[index])
    a5 = np.unique(RRI_timewindow_arr[index])
    # print(a5);
    # print(len(a5))
    k5 = 0
    n_arr = []
    for m in time_arr:
        for n in a5:
            if m - n <= 1 and m - n >= 0:
                k5 = k5 + 1
                n_arr.append(n)
    # print(k5)
    Sen1 = k1 / len(time_arr);
    Sen2 = k2 / len(time_arr);
    Sen3 = k3 / len(time_arr);
    Sen4 = k4 / len(time_arr);
    Sen5 = k5 / len(time_arr);
    FPR1 = (len(a1) - k1) / len(Pcombined);
    FPR2 = (len(a2) - k2) / len(Pcombined);
    FPR3 = (len(a3) - k3) / len(Pcombined);
    FPR4 = (len(a4) - k4) / len(Pcombined);
    FPR5 = (len(a5) - k5) / len(Pcombined);
    Sen_arr_CS = [0, Sen1, Sen2, Sen3, Sen4, Sen5, 1]
    FPR_arr_CS = [0, FPR1, FPR2, FPR3, FPR4, FPR5, 1]
    from sklearn.metrics import auc

    AUC_cs = auc(np.sort(FPR_arr_CS), np.sort(Sen_arr_CS))
    # print(AUC_cs)
    AUC_cs_arr.append(AUC_cs)

# print(AUC_cs_arr)
# print(time_arr_arr)

# np.savetxt("C:/Users/wxiong/Documents/PHD/2011.1/QLD0290/chance/AUC_EEGECGm288m72_12h_QLD0290_2022.csv", AUC_cs_arr, delimiter=",", fmt='%s')
#
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2011.1/QLD0290/chance/AUC_EEGECGm288m72_12h_QLD0290_2022.csv', sep=',',header=None)
# AUC_EEG = csv_reader.values
# AUC_EEG_arr = []
# for item in AUC_EEG:
#     AUC_EEG_arr.append(float(item))

n=0
for i in AUC_cs_arr:
    if i > AUC:
        n=n+1
print(n/len(AUC_cs_arr))





t1 = np.linspace(0 + 0.00416667, 0 + 0.00416667 + 0.00416667 * (len(Raw_variance_EEG_arr) - 1),
                 len(Raw_variance_EEG_arr))
a = np.where(t1 < 9.4505556+0 + 0)
t1[0:2268] = t1[0:2268] - 0 + 14.5494444
t1[2268:] = t1[2268:] - 9.4505556+0 - 0
time_feature_arr = []
for i in range(len(t1)):
    if t1[i] > 24:
        time_feature_arr.append(t1[i] - (t1[i] // 24) * 24)
    else:
        time_feature_arr.append(t1[i])
print(len(time_feature_arr))
time_arr = time_feature_arr[19450:]
print(len(time_arr))
new_arr = []
for j in range(0, int(19450/40)):
    new_arr.append(time_arr[40 * j])

bins_number = 18
bins = np.linspace(0, 24, bins_number + 1)
pro_circadian_time = []
pro_circadian_time_false = []
for i in range(len(new_arr)):
    if new_arr[i] >= bins[0] and new_arr[i] <= bins[1]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[0])
        pro_circadian_time.append(pro_seizure_circadian[0])
    elif new_arr[i] > bins[1] and new_arr[i] < bins[2]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[1])
        pro_circadian_time.append(pro_seizure_circadian[1])
    elif new_arr[i] >= bins[2] and new_arr[i] < bins[3]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[2])
        pro_circadian_time.append(pro_seizure_circadian[2])
    elif new_arr[i] >= bins[3] and new_arr[i] < bins[4]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[3])
        pro_circadian_time.append(pro_seizure_circadian[3])
    elif new_arr[i] >= bins[4] and new_arr[i] < bins[5]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[4])
        pro_circadian_time.append(pro_seizure_circadian[4])
    elif new_arr[i] >= bins[5] and new_arr[i] <= bins[6]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[5])
        pro_circadian_time.append(pro_seizure_circadian[5])
    elif new_arr[i] > bins[6] and new_arr[i] < bins[7]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[6])
        pro_circadian_time.append(pro_seizure_circadian[6])
    elif new_arr[i] >= bins[7] and new_arr[i] <= bins[8]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[7])
        pro_circadian_time.append(pro_seizure_circadian[7])
    elif new_arr[i] > bins[8] and new_arr[i] < bins[9]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[8])
        pro_circadian_time.append(pro_seizure_circadian[8])
    elif new_arr[i] >= bins[9] and new_arr[i] < bins[10]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[9])
        pro_circadian_time.append(pro_seizure_circadian[9])
    elif new_arr[i] >= bins[10] and new_arr[i] < bins[11]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[10])
        pro_circadian_time.append(pro_seizure_circadian[10])
    elif new_arr[i] >= bins[11] and new_arr[i] < bins[12]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[11])
        pro_circadian_time.append(pro_seizure_circadian[11])
    elif new_arr[i] >= bins[12] and new_arr[i] < bins[13]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[12])
        pro_circadian_time.append(pro_nonseizure_circadian[12])
    elif new_arr[i] >= bins[13] and new_arr[i] < bins[14]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[13])
        pro_circadian_time.append(pro_seizure_circadian[13])
    elif new_arr[i] >= bins[14] and new_arr[i] < bins[15]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[14])
        pro_circadian_time.append(pro_seizure_circadian[14])
    elif new_arr[i] >= bins[15] and new_arr[i] < bins[16]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[15])
        pro_circadian_time.append(pro_seizure_circadian[15])
    elif new_arr[i] >= bins[16] and new_arr[i] < bins[17]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[16])
        pro_circadian_time.append(pro_seizure_circadian[16])
    elif new_arr[i] >= bins[17] and new_arr[i] <= bins[18]:
        pro_circadian_time_false.append(pro_nonseizure_circadian[17])
        pro_circadian_time.append(pro_seizure_circadian[17])



Th_circadian=np.min(pro)
Pcombined=split(pro_circadian_time,6)
print(len(Pcombined))

Sen=[]; FPR=[];
seizure_count_test=len([93.82222,117.80083,121.36472,140.3647])

index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= Th_circadian:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[93.82222,117.80083,121.36472,140.3647]
k1=0
n_arr=[]
pretime=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k1=k1+1
            n_arr.append(n)
            pretime.append(m-n)
print(k1)
time_arr = [93.82222,117.80083,121.36472,140.3647,142.20333]
k2=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k2=k2+1
            n_arr.append(n)
print(k2)
Sen.append(k1/seizure_count_test)
FPR.append((len(a)-k2)/len(Pcombined))
index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= 0.3*Th_circadian:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[93.82222,117.80083,121.36472,140.3647]
k1=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k1=k1+1
            n_arr.append(n)
            pretime.append(m - n)
print(k1)
time_arr = [93.82222,117.80083,121.36472,140.3647,142.20333]
k2=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k2=k2+1
            n_arr.append(n)
print(k2)
Sen.append(k1/seizure_count_test)
FPR.append((len(a)-k2)/len(Pcombined))
index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= 0.6*Th_circadian:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[93.82222,117.80083,121.36472,140.3647]
k1=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k1=k1+1
            n_arr.append(n)
            pretime.append(m - n)
print(k1)
time_arr = [93.82222,117.80083,121.36472,140.3647,142.20333]
k2=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k2=k2+1
            n_arr.append(n)
print(k2)
Sen.append(k1/seizure_count_test)
FPR.append((len(a)-k2)/len(Pcombined))
index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= 1.2*Th_circadian:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[93.82222,117.80083,121.36472,140.3647]
k1=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k1=k1+1
            n_arr.append(n)
            pretime.append(m - n)
print(k1)
time_arr = [93.82222,117.80083,121.36472,140.3647,142.20333]
k2=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k2=k2+1
            n_arr.append(n)
print(k2)
Sen.append(k1/seizure_count_test)
FPR.append((len(a)-k2)/len(Pcombined))
index=[]
for i in range(len(Pcombined)):
    for item in Pcombined[i]:
        if item >= 2*Th_circadian:
            index.append(6*i+0)
print(RRI_timewindow_arr[index])
a=np.unique(RRI_timewindow_arr[index])
print(a); print(len(a))
time_arr=[93.82222,117.80083,121.36472,140.3647]
k1=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k1=k1+1
            n_arr.append(n)
            pretime.append(m - n)
print(k1)
time_arr = [93.82222,117.80083,121.36472,140.3647,142.20333]
k2=0
n_arr=[]
for m in time_arr:
    for n in a:
        if m-n<=1 and m-n>=0:
            k2=k2+1
            n_arr.append(n)
print(k2)
Sen.append(k1/seizure_count_test)
FPR.append((len(a)-k2)/len(Pcombined))
print(pretime)
print(np.mean(pretime))

print(Sen);print(FPR);
print(np.mean(Sen));print(np.mean(FPR));
Sen.append(0);Sen.append(1);
FPR.append(0);FPR.append(1);
print('computed AUC using sklearn.metrics.auc: {}'.format(auc(sorted(FPR),sorted(Sen))))
AUC=auc(sorted(FPR),sorted(Sen))




# Pcombined = split(pro_circadian_time, 6)
# print(len(Pcombined))
# time_arr_arr=[]
# AUC_cs_arr=[]
# for i in range(50000):
#     time_arr = np.random.uniform(low=t_window_arr[19450], high=t_window_arr[-1], size=4)
#     time_arr_arr.append(time_arr)
#     time_arr=np.sort(time_arr)
#
#     index = []
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= Th_circadian:
#                 index.append(6 * i + 0)
#     # print(RRI_timewindow_arr[index])
#     a1 = np.unique(RRI_timewindow_arr[index])
#     # print(a1);
#     # print(len(a1))
#     k1 = 0
#     n_arr = []
#     for m in time_arr:
#         for n in a1:
#             if m - n <= 1 and m - n >= 0:
#                 k1 = k1 + 1
#                 n_arr.append(n)
#     # print(k1)
#     # time_arr=[76.88111,97.923611,119.25278,119.44028]
#     # k11=0
#     # n_arr=[]
#     # for m in time_arr:
#     #     for n in a1:
#     #         if m-n<=1 and m-n>=0:
#     #             k11=k11+1
#     #             n_arr.append(n)
#     # print(k11)
#     index = []
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= 0.3 * Th_circadian:
#                 index.append(6 * i + 0)
#     # print(RRI_timewindow_arr[index])
#     a2 = np.unique(RRI_timewindow_arr[index])
#     # print(a2);
#     # print(len(a2))
#     k2 = 0
#     n_arr = []
#     for m in time_arr:
#         for n in a2:
#             if m - n <= 1 and m - n >= 0:
#                 k2 = k2 + 1
#                 n_arr.append(n)
#     # print(k2)
#     # time_arr=[76.88111,97.923611,119.25278,119.44028]
#     # k22=0
#     # n_arr=[]
#     # for m in time_arr:
#     #     for n in a2:
#     #         if m-n<=1 and m-n>=0:
#     #             k22=k22+1
#     #             n_arr.append(n)
#     # print(k22)
#     index = []
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= 0.6 * Th_circadian:
#                 index.append(6 * i + 0)
#     # print(RRI_timewindow_arr[index])
#     a3 = np.unique(RRI_timewindow_arr[index])
#     # print(a3);
#     # print(len(a3))
#     k3 = 0
#     n_arr = []
#     for m in time_arr:
#         for n in a3:
#             if m - n <= 1 and m - n >= 0:
#                 k3 = k3 + 1
#                 n_arr.append(n)
#     # print(k3)
#     # time_arr=[76.88111,97.923611,119.25278,119.44028]
#     # k33=0
#     # n_arr=[]
#     # for m in time_arr:
#     #     for n in a3:
#     #         if m-n<=1 and m-n>=0:
#     #             k33=k33+1
#     #             n_arr.append(n)
#     # print(k33)
#     index = []
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= 1.2 * Th_circadian:
#                 index.append(6 * i + 0)
#     # print(RRI_timewindow_arr[index])
#     a4 = np.unique(RRI_timewindow_arr[index])
#     # print(a);
#     # print(len(a4))
#     k4 = 0
#     n_arr = []
#     for m in time_arr:
#         for n in a4:
#             if m - n <= 1 and m - n >= 0:
#                 k4 = k4 + 1
#                 n_arr.append(n)
#     # print(k4)
#     # time_arr=[76.88111,97.923611,119.25278,119.44028]
#     # k44=0
#     # n_arr=[]
#     # for m in time_arr:
#     #     for n in a4:
#     #         if m-n<=1 and m-n>=0:
#     #             k44=k44+1
#     #             n_arr.append(n)
#     # print(k44)
#     index = []
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= 2 * Th_circadian:
#                 index.append(6 * i + 0)
#     # print(RRI_timewindow_arr[index])
#     a5 = np.unique(RRI_timewindow_arr[index])
#     # print(a5);
#     # print(len(a5))
#     k5 = 0
#     n_arr = []
#     for m in time_arr:
#         for n in a5:
#             if m - n <= 1 and m - n >= 0:
#                 k5 = k5 + 1
#                 n_arr.append(n)
#     # print(k5)
#     # time_arr=[76.88111,97.923611,119.25278,119.44028]
#     # k55=0
#     # n_arr=[]
#     # for m in time_arr:
#     #     for n in a:
#     #         if m-n<=1 and m-n>=0:
#     #             k55=k55+1
#     #             n_arr.append(n)
#     # print(k55)
#     Sen1 = k1 / len(time_arr);
#     Sen2 = k2 / len(time_arr);
#     Sen3 = k3 / len(time_arr);
#     Sen4 = k4 / len(time_arr);
#     Sen5 = k5 / len(time_arr);
#     FPR1 = (len(a1) - k1) / len(Pcombined);
#     FPR2 = (len(a2) - k2) / len(Pcombined);
#     FPR3 = (len(a3) - k3) / len(Pcombined);
#     FPR4 = (len(a4) - k4) / len(Pcombined);
#     FPR5 = (len(a5) - k5) / len(Pcombined);
#     Sen_arr_CS = [0, Sen1, Sen2, Sen3, Sen4, Sen5, 1]
#     FPR_arr_CS = [0, FPR1, FPR2, FPR3, FPR4, FPR5, 1]
#     from sklearn.metrics import auc
#
#     AUC_cs = auc(np.sort(FPR_arr_CS), np.sort(Sen_arr_CS))
#     # print(AUC_cs)
#     AUC_cs_arr.append(AUC_cs)
#
# n=0
# for i in AUC_cs_arr:
#     if i > AUC:
#         n=n+1
# print(n/len(AUC_cs_arr))