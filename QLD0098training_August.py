from __future__ import division
import mne
import numpy as np
import scipy.signal
from scipy.signal import butter, lfilter
from matplotlib import pyplot
import math
from scipy.fftpack import fft, ifft
from scipy import signal
from scipy.signal import butter, lfilter,iirfilter
from scipy.signal import hilbert
from biosppy.signals import tools
import pandas as pd

def movingaverage(values, window_size):
    weights = (np.ones(window_size))/window_size
    a=np.ones(1)
    return lfilter(weights,a,values)


csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/EEGauto_QLD0098_15s_3h.csv',sep=',',header=None)
Raw_auto_EEG= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/EEGauto_lag1_QLD0098_15s_3h.csv',sep=',',header=None)
# Raw_auto1_EEG= csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/EEGvariance_QLD0098_15s_3h.csv',sep=',',header=None)
Raw_variance_EEG= csv_reader.values

Raw_variance_EEG_arr=[]
for item in Raw_variance_EEG:
    Raw_variance_EEG_arr.append(float(item))
Raw_auto_EEG_arr=[]
for item in Raw_auto_EEG:
    Raw_auto_EEG_arr.append(float(item))
# Raw_auto1_EEG_arr=[]
# for item in Raw_auto1_EEG:
#     Raw_auto1_EEG_arr.append(float(item))


t=np.linspace(2.9219+0.00416667,2.9219+0.00416667+0.00416667*(len(Raw_variance_EEG_arr)-1),len(Raw_variance_EEG_arr))
window_time_arr=t
print(len(t))
print(t[-1]-t[0])
var_arr=[]
for item in Raw_variance_EEG_arr:
    if item<1e-8:
        var_arr.append(item)
    else:
        var_arr.append(var_arr[-1])
Raw_variance_EEG_arr=var_arr


value_arr=[]
for item in Raw_auto_EEG_arr:
    if item<500:
        value_arr.append(item)
    else:
        value_arr.append(value_arr[-1])
Raw_auto_EEG_arr=value_arr

seizure_timing_index = []
for k in range(len(window_time_arr)):
    if window_time_arr[k] < 6.9544 and window_time_arr[k + 1] >= 6.9544:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 31.2324556 and window_time_arr[k + 1] >= 31.2324556:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 55.0324556 and window_time_arr[k + 1] >= 55.0324556:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 79.63162 and window_time_arr[k+1]>=79.63162:
        seizure_timing_index.append(k)
    # if window_time_arr[k] < 109.51 and window_time_arr[k + 1] >= 109.51:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 127.774122 and window_time_arr[k + 1] >= 127.774122:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k] < 134.0119 and window_time_arr[k + 1] >= 134.0119:
    #     seizure_timing_index.append(k)
    # if window_time_arr[k]<152.234122 and window_time_arr[k+1]>=152.234122:
    #     seizure_timing_index.append(k)
print(seizure_timing_index)

# # ### EEG variance
window_time_arr=t[0:19080]
Raw_variance_EEG=Raw_variance_EEG_arr[0:19080]
medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG,240*6)
long_rhythm_var_arr=medium_rhythm_var_arr_3
var_trans=hilbert(long_rhythm_var_arr)
var_phase=np.angle(var_trans)
phase_long_EEGvariance_arr=var_phase
print(len(phase_long_EEGvariance_arr));

Raw_auto_EEG=Raw_auto_EEG_arr[0:19080]
medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG,240*6)
long_rhythm_value_arr=medium_rhythm_value_arr_3
var_trans=hilbert(long_rhythm_value_arr)
value_phase=np.angle(var_trans)
phase_long_EEGauto_arr=value_phase
print(len(phase_long_EEGauto_arr));


csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/ch2-1/RRI_ch21_timewindowarr_QLD0098_15s_3h.csv',sep=',',header=None)
rri_t= csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/ch2-1/RRI_ch21_rawvariance_QLD0098_15s_3h.csv',sep=',',header=None)
RRI_var= csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/ch2-1/RRI_ch21_rawauto_QLD0098_15s_3h.csv',sep=',',header=None)
Raw_auto_RRI31= csv_reader.values

# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_ch31_timewindowarr_QLD0098_15s_3h.csv',sep=',',header=None)
# rri_t= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_ch31_rawvariance_QLD0098_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_ch31_rawauto_QLD0098_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values



t_window_arr=[]
for item in rri_t:
    t_window_arr.append(2.9219+float(item))
print(t_window_arr[0])
Raw_variance_RRI31_arr=[]
for item in RRI_var:
    Raw_variance_RRI31_arr.append(float(item))
Raw_auto_RRI31_arr=[]
for item in Raw_auto_RRI31:
    Raw_auto_RRI31_arr.append(float(item))


Raw_variance_RRI31=Raw_variance_RRI31_arr[0:19080]
medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31,240*6)
long_rhythm_var_arr=medium_rhythm_var_arr_3
var_trans=hilbert(long_rhythm_var_arr)
var_phase=np.angle(var_trans)
phase_long_RRIvariance_arr=var_phase
print(len(phase_long_RRIvariance_arr));

Raw_auto_RRI31=Raw_auto_RRI31_arr[0:19080]
medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31,240*6)
long_rhythm_value_arr=medium_rhythm_value_arr_3
value_trans=hilbert(long_rhythm_value_arr)
value_phase=np.angle(value_trans)
phase_long_RRIauto_arr=value_phase
print(len(phase_long_RRIauto_arr));


#### combined probability calculation
bins_number = 18
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
print(bins)
pro_eegvars_time=[]
pro_eegvars_time_false=[]
for i in range(len(phase_long_EEGvariance_arr)):
    if phase_long_EEGvariance_arr[i] >= bins[0] and phase_long_EEGvariance_arr[i]< bins[1]:
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
        pro_eegvars_time_false.append(0.007443909)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[6] and phase_long_EEGvariance_arr[i] <= bins[7]:
        pro_eegvars_time_false.append(0.058397987)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] > bins[7] and phase_long_EEGvariance_arr[i] < bins[8]:
        pro_eegvars_time_false.append(0.205284127)
        pro_eegvars_time.append(0.5)
    elif phase_long_EEGvariance_arr[i] >= bins[8] and phase_long_EEGvariance_arr[i] <= bins[9]:
        pro_eegvars_time_false.append(0.323862445)
        pro_eegvars_time.append(0.25)
    elif phase_long_EEGvariance_arr[i] > bins[9] and phase_long_EEGvariance_arr[i] < bins[10]:
        pro_eegvars_time_false.append(0.15244286)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[10] and phase_long_EEGvariance_arr[i] <= bins[11]:
        pro_eegvars_time_false.append(0.160620675)
        pro_eegvars_time.append(0.25)
    elif phase_long_EEGvariance_arr[i] > bins[11] and phase_long_EEGvariance_arr[i] < bins[12]:
        pro_eegvars_time_false.append(0.046812749)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] > bins[12] and phase_long_EEGvariance_arr[i] < bins[13]:
        pro_eegvars_time_false.append(0.019815475)
        pro_eegvars_time.append(0)
    elif phase_long_EEGvariance_arr[i] >= bins[13] and phase_long_EEGvariance_arr[i] < bins[14]:
        pro_eegvars_time_false.append(0.025319774)
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
print(pro_eegvars_time[966]);print(pro_eegvars_time[6793]);print(pro_eegvars_time[12505]);print(pro_eegvars_time[18409]);



bins_number = 18
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
pro_eegautos_time=[]
pro_eegautos_time_false=[]
for i in range(len(phase_long_EEGauto_arr)):
    if phase_long_EEGauto_arr[i] >= bins[0] and phase_long_EEGauto_arr[i]< bins[1]:
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
        pro_eegautos_time_false.append(0.114384567)
        pro_eegautos_time.append(0.25)
    elif phase_long_EEGauto_arr[i] >= bins[8] and phase_long_EEGauto_arr[i] < bins[9]:
        pro_eegautos_time_false.append(0.36501363)
        pro_eegautos_time.append(0.5)
    elif phase_long_EEGauto_arr[i] >= bins[9] and phase_long_EEGauto_arr[i] <= bins[10]:
        pro_eegautos_time_false.append(0.429702244)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] > bins[10] and phase_long_EEGauto_arr[i] < bins[11]:
        pro_eegautos_time_false.append(0.065789474)
        pro_eegautos_time.append(0.25)
    elif phase_long_EEGauto_arr[i] >= bins[11] and phase_long_EEGauto_arr[i] < bins[12]:
        pro_eegautos_time_false.append(0.007706018)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[12] and phase_long_EEGauto_arr[i] < bins[13]:
        pro_eegautos_time_false.append(0.008282659)
        pro_eegautos_time.append(0)
    elif phase_long_EEGauto_arr[i] >= bins[13] and phase_long_EEGauto_arr[i] < bins[14]:
        pro_eegautos_time_false.append(0.009121409)
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
print(pro_eegautos_time[966]);print(pro_eegautos_time[6793]);print(pro_eegautos_time[12505]);print(pro_eegautos_time[18409]);




bins_number = 18
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
pro_RRIvars_time=[]
pro_RRIvars_time_false=[]
for i in range(len(phase_long_RRIvariance_arr)):
    if phase_long_RRIvariance_arr[i] >= bins[0] and phase_long_RRIvariance_arr[i]< bins[1]:
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
        pro_RRIvars_time_false.append(0.039211575)
        pro_RRIvars_time.append(0.25)
    elif phase_long_RRIvariance_arr[i] >= bins[7] and phase_long_RRIvariance_arr[i] <= bins[8]:
        pro_RRIvars_time_false.append(0.117634724)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] > bins[8] and phase_long_RRIvariance_arr[i] < bins[9]:
        pro_RRIvars_time_false.append(0.296131264)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[9] and phase_long_RRIvariance_arr[i] <= bins[10]:
        pro_RRIvars_time_false.append(0.391486685)
        pro_RRIvars_time.append(0.5)
    elif phase_long_RRIvariance_arr[i] > bins[10] and phase_long_RRIvariance_arr[i] <= bins[11]:
        pro_RRIvars_time_false.append(0.145113247)
        pro_RRIvars_time.append(0.25)
    elif phase_long_RRIvariance_arr[i] > bins[11] and phase_long_RRIvariance_arr[i] < bins[12]:
        pro_RRIvars_time_false.append(0.005609142)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[12] and phase_long_RRIvariance_arr[i] < bins[13]:
        pro_RRIvars_time_false.append(0.002883204)
        pro_RRIvars_time.append(0)
    elif phase_long_RRIvariance_arr[i] >= bins[13] and phase_long_RRIvariance_arr[i] < bins[14]:
        pro_RRIvars_time_false.append(0.005084923)
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

print(pro_RRIvars_time[966]);print(pro_RRIvars_time[6793]);print(pro_RRIvars_time[12505]);print(pro_RRIvars_time[18409]);



bins_number = 18
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
pro_RRIautos_time=[]
pro_RRIautos_time_false=[]
for i in range(len(phase_long_RRIauto_arr)):
    if phase_long_RRIauto_arr[i] >= bins[0] and phase_long_RRIauto_arr[i]< bins[1]:
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
        pro_RRIautos_time_false.append(0.038687356)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] > bins[8] and phase_long_RRIauto_arr[i] <= bins[9]:
        pro_RRIautos_time_false.append(0.523851961)
        pro_RRIautos_time.append(0.25)
    elif phase_long_RRIauto_arr[i] > bins[9] and phase_long_RRIauto_arr[i] <= bins[10]:
        pro_RRIautos_time_false.append(0.376756133)
        pro_RRIautos_time.append(0.5)
    elif phase_long_RRIauto_arr[i] > bins[10] and phase_long_RRIauto_arr[i] < bins[11]:
        pro_RRIautos_time_false.append(0.042304466)
        pro_RRIautos_time.append(0.25)
    elif phase_long_RRIauto_arr[i] >= bins[11] and phase_long_RRIauto_arr[i] < bins[12]:
        pro_RRIautos_time_false.append(0.008335081)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[12] and phase_long_RRIauto_arr[i] < bins[13]:
        pro_RRIautos_time_false.append(0.004665548)
        pro_RRIautos_time.append(0)
    elif phase_long_RRIauto_arr[i] >= bins[13] and phase_long_RRIauto_arr[i] < bins[14]:
        pro_RRIautos_time_false.append(0.005399455)
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
print(pro_RRIautos_time[966]);print(pro_RRIautos_time[6793]);print(pro_RRIautos_time[12505]);print(pro_RRIautos_time[18409]);



Pseizureeegvar=0.000209644;
Pnonseizureeegvar=0.999790356;

# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIvars_time[m]*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_eegvars_time[m]*pro_eegautos_time[m]*pro_RRIvars_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIvars_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_RRIvars_time[m]*pro_eegautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_eegautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))

Pcombined=[]
for m in range(len(pro_eegvars_time)):
    P1=Pseizureeegvar*pro_RRIvars_time[m]*pro_eegvars_time[m]
    P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_eegvars_time_false[m])
    Pcombined.append(P1/(P1+P2))



pyplot.figure(figsize=(12,5))
pyplot.plot(window_time_arr,Pcombined)
pyplot.title('combined probability in QLD0098',fontsize=15)
pyplot.annotate('',xy=(6.9544,np.max(Pcombined)),xytext=(6.9544,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.annotate('',xy=(31.2324556,np.max(Pcombined)),xytext=(31.2324556,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.annotate('',xy=(55.0324556,np.max(Pcombined)),xytext=(55.0324556,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.annotate('',xy=(79.63162,np.max(Pcombined)),xytext=(79.63162,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.tight_layout()
pyplot.xlim(window_time_arr[0],window_time_arr[-1])
pyplot.xlabel('Time(h)',fontsize=15)
pyplot.ylabel('seizure probability',fontsize=15)

# pyplot.hlines(1.3245288853236061e-05, window_time_arr[0],window_time_arr[-1],'r')
# pyplot.hlines(1.31643697818471e-05, window_time_arr[0],window_time_arr[-1],'r')
pyplot.show()
print(Pcombined[966]);
print(Pcombined[6793]);
print(Pcombined[12505]);
print(Pcombined[18409]);



### add circadain
t=np.linspace(2.9219+0.00416667,2.9219+0.00416667+0.00416667*(len(Raw_variance_EEG_arr)-1),len(Raw_variance_EEG_arr))
a=np.where(t<5.6497222+2.9219)
print(a);print(t[1354]);print(t[1355])
t[0:1355]=t[0:1355]-2.9219+18.3502778
t[1355:]=t[1355:]-5.6497222-2.9219
print(t[1355]);
print(t);print(type(t));print(t[0])

time_feature_arr=[]
for i in range(len(t)):
    if t[i]>24:
        time_feature_arr.append(t[i] - (t[i] // 24) * 24)
    else:
        time_feature_arr.append(t[i])
print(time_feature_arr)
print(time_feature_arr[966]);print(time_feature_arr[6793]);
print(time_feature_arr[12505]);print(time_feature_arr[18409]);


bins_number = 18
bins = np.linspace(0, 24, bins_number + 1)
pro_circadian_time=[]
pro_circadian_time_false=[]
for i in range(len(time_feature_arr)):
    if time_feature_arr[i] >= bins[0] and time_feature_arr[i] <= bins[1]:
        pro_circadian_time_false.append(0.067100021)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] > bins[1] and time_feature_arr[i] < bins[2]:
        pro_circadian_time_false.append(0.056877752)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[2] and time_feature_arr[i] < bins[3]:
        pro_circadian_time_false.append(0.050325016)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[3] and time_feature_arr[i] < bins[4]:
        pro_circadian_time_false.append(0.050325016)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[4] and time_feature_arr[i] < bins[5]:
        pro_circadian_time_false.append(0.050325016)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[5] and time_feature_arr[i] <= bins[6]:
        pro_circadian_time_false.append(0.050325016)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] > bins[6] and time_feature_arr[i] < bins[7]:
        pro_circadian_time_false.append(0.050325016)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[7] and time_feature_arr[i] <= bins[8]:
        pro_circadian_time_false.append(0.050325016)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] > bins[8] and time_feature_arr[i] < bins[9]:
        pro_circadian_time_false.append(0.050325016)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[9] and time_feature_arr[i] < bins[10]:
        pro_circadian_time_false.append(0.050325016)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[10] and time_feature_arr[i] < bins[11]:
        pro_circadian_time_false.append(0.050325016)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[11] and time_feature_arr[i] < bins[12]:
        pro_circadian_time_false.append(0.050325016)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[12] and time_feature_arr[i] < bins[13]:
        pro_circadian_time_false.append(0.050325016)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[13] and time_feature_arr[i] < bins[14]:
        pro_circadian_time_false.append(0.054256658)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[14] and time_feature_arr[i] < bins[15]:
        pro_circadian_time_false.append(0.067100021)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[15] and time_feature_arr[i] < bins[16]:
        pro_circadian_time_false.append(0.067100021)
        pro_circadian_time.append(0)
    elif time_feature_arr[i] >= bins[16] and time_feature_arr[i] < bins[17]:
        pro_circadian_time_false.append(0.066942755)
        pro_circadian_time.append(0.75)
    elif time_feature_arr[i] >= bins[17] and time_feature_arr[i] <= bins[18]:
        pro_circadian_time_false.append(0.067047599)
        pro_circadian_time.append(0.25)
print(len(pro_circadian_time))
print(len(pro_circadian_time_false))
print(pro_circadian_time[966]);print(pro_circadian_time[6793]);print(pro_circadian_time[12505]);print(pro_circadian_time[18409]);





# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIvars_time[m]*pro_RRIautos_time[m]*pro_circadian_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIvars_time[m]*pro_circadian_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIvars_time_false[m]*pro_circadian_time_false[m])
#     Pcombined.append(P1/(P1+P2))

# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=pro_RRIvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_circadian_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_eegautos_time_false[m]*pro_circadian_time_false[m])
#     Pcombined.append(P1/(P1+P2))

Pcombined=[]
for m in range(len(pro_eegvars_time)):
    P1=pro_RRIvars_time[m]*Pseizureeegvar*pro_eegvars_time[m]*pro_circadian_time[m]
    P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_eegvars_time_false[m]*pro_circadian_time_false[m])
    Pcombined.append(P1/(P1+P2))



print(len(Pcombined))
pyplot.figure(figsize=(12,5))
pyplot.plot(window_time_arr,Pcombined)
pyplot.title('combined probability in QLD0098',fontsize=15)
pyplot.annotate('',xy=(6.9544,np.max(Pcombined)),xytext=(6.9544,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.annotate('',xy=(31.2324556,np.max(Pcombined)),xytext=(31.2324556,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.annotate('',xy=(55.0324556,np.max(Pcombined)),xytext=(55.0324556,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.annotate('',xy=(79.63162,np.max(Pcombined)),xytext=(79.63162,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.tight_layout()
pyplot.xlim(window_time_arr[0],window_time_arr[-1])
pyplot.xlabel('Time(h)',fontsize=15)
pyplot.ylabel('seizure probability',fontsize=15)
# pyplot.hlines(8.191771441573039e-07, window_time_arr[0],window_time_arr[-1],'r')
# pyplot.hlines(3.2784621689673455e-06, window_time_arr[0],window_time_arr[-1],'r')

pyplot.show()
print(Pcombined[966]);
print(Pcombined[6793]);
print(Pcombined[12505]);
print(Pcombined[18409]);





# def time_in_risk(probability, seizure_idxs, low=None, high=None):
#     seizure_idxs = np.array(seizure_idxs)
#     if not low:
#         low = np.max(probability) / 3
#         high = 2 * low
#     # Use brute force approach with N possible threshold values
#     N = 10000
#     thresholds = np.logspace(start=0, stop=1, num=N) / (30*N)
#
#     opt_values = np.zeros(N)
#     perc_times = np.zeros(N)
#     perc_seizures = np.zeros(N)
#
#     if np.isnan(probability).all():
#         return 0, 0, 0
#
#     for idx, thr in enumerate(thresholds):
#         result, perc_time_in_high, perc_sz_in_high = f_time(probability=probability,seizure_idxs=seizure_idxs,thr=thr)
#         opt_values[idx] = result
#         perc_times[idx] = perc_time_in_high
#         perc_seizures[idx] = perc_sz_in_high
#
#     if np.max(opt_values) == 0:
#         print('No maximum found, using default values. Using default values.')
#
#         return low, 0, 0
#         threshold = low
#         time_in_high, sz_in_high = 0 # TODO: implement
#
#     max_idx = np.argwhere(opt_values == np.max(opt_values))[0]
#     threshold = thresholds[max_idx][0]
#     time_in_high = perc_times[max_idx][0]
#     seiz_in_high = perc_seizures[max_idx][0]
#
#     print('Maximum found... ', threshold)
#
#     return threshold, time_in_high, seiz_in_high
#
# def f_time(probability, seizure_idxs, thr):
#     # Initialise
#     prob_low = np.zeros(len(probability))
#     prob_high = np.zeros(len(probability))
#
#     # Compute in time & n. of seizures in 'low risk'
#     prob_low[np.where(probability < thr)] = 1
#     if np.sum(prob_low) != 0:
#         perc_time_in_low = len(np.argwhere(prob_low)) / len(probability)
#         # perc_time_in_low = round(perc_time_in_low * 100, 2)
#         if len(seizure_idxs) != 0:
#             sz_in_low = prob_low[seizure_idxs]
#             perc_sz_in_low = sum(sz_in_low) / len(seizure_idxs)
#             # perc_sz_in_low = round(perc_sz_in_low * 100, 2)
#         else:
#             sz_in_low, perc_sz_in_low = 0, 0
#     else:
#         sz_in_low, perc_sz_in_low, perc_time_in_low = 0, 0, 0
#
#     # Compute in time & n. of seizures in 'high risk'
#     prob_high[np.where(probability >= thr)] = 1
#     if np.sum(prob_high) != 0:
#         perc_time_in_high = len(np.argwhere(prob_high)) / len(probability)
#         # perc_time_in_high = round(perc_time_in_high * 100, 2)
#         if len(seizure_idxs) != 0:
#             sz_in_high = prob_high[seizure_idxs]
#             perc_sz_in_high = sum(sz_in_high) / len(seizure_idxs)
#             perc_sz_in_high = round(perc_sz_in_high * 100, 2)
#         else:
#             sz_in_high, perc_sz_in_high = 0, 0
#     else:
#         sz_in_high, perc_sz_in_high, perc_time_in_high = 0, 0, 0
#
#     F = perc_time_in_low * perc_sz_in_high
#
#     return F, perc_time_in_high, perc_sz_in_high
#
#
#
# probability=Pcombined
# seizure_idxs=[966,6793,12505,18409]
# threshold, time_in_high, seiz_in_high=time_in_risk(probability, seizure_idxs)
# print(threshold)
# print(time_in_high)
# print(seiz_in_high)