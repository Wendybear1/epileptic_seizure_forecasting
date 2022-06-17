from __future__ import division
import mne
import numpy as np
import scipy.signal
from scipy.signal import butter, lfilter
from matplotlib import pyplot
import math
from scipy.fftpack import fft, ifft
from scipy import signal
from scipy.signal import butter, lfilter,iirfilter,filtfilt
from scipy.signal import hilbert
from biosppy.signals import tools
import pandas as pd
from matplotlib import rc

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
    nyq  = fs/2.0
    low  = freq - band/2.0
    high = freq + band/2.0
    low  = low/nyq
    high = high/nyq
    b, a = iirfilter(order, [low, high], rp=ripple, btype='bandstop',analog=False, ftype=filter_type)
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
    weights = (np.ones(window_size))/window_size
    a=np.ones(1)
    return lfilter(weights,a,values)



# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/EEG_VIC0758_3h.csv',sep=',',header=None)
# Raw_EEG= csv_reader.values
# Raw_EEG_arr=[]
# for item in Raw_EEG:
#     Raw_EEG_arr.append(float(item))
# print(len(Raw_EEG_arr))
# t0=np.linspace(2.99416,2.99416+0.00000109*(len(Raw_EEG_arr)-1),len(Raw_EEG_arr))
# pyplot.plot(t0[15800000:15900000],Raw_EEG_arr[15800000:15900000],'grey',alpha=0.3)
# pyplot.xlabel('Time(hour)',fontsize=13)
# pyplot.ylabel('Voltage',fontsize=13)
# pyplot.title('EEG recordings in VIC0758',fontsize=13)
# pyplot.show()
# pyplot.plot(t0[15880000:15890000],Raw_EEG_arr[15880000:15890000],'grey',alpha=0.3)
# pyplot.xlabel('Time(hour)',fontsize=13)
# pyplot.ylabel('Voltage',fontsize=13)
# pyplot.title('EEG recordings in VIC0758',fontsize=13)
# pyplot.show()



# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/VIC0758/Rawvariance_VIC0758_163h_Apirl_15s.csv',sep=',',header=None)
# # Raw_variance_EEG= csv_reader.values
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/VIC0758/Rawauto_VIC0758_163h_Apirl_15s.csv',sep=',',header=None)
# # Raw_auto_EEG= csv_reader.values
#
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/EEGvariance_VIC0758_15s_3h.csv',sep=',',header=None)
Raw_variance_EEG= csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/EEGauto_VIC0758_15s_3h.csv',sep=',',header=None)
Raw_auto_EEG= csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/EEGauto_lag1_VIC0758_15s_3h.csv',sep=',',header=None)
Raw_auto1_EEG= csv_reader.values

Raw_variance_EEG_arr=[]
for item in Raw_variance_EEG:
    Raw_variance_EEG_arr.append(float(item))
Raw_auto_EEG_arr=[]
for item in Raw_auto_EEG:
    Raw_auto_EEG_arr.append(float(item))



t=np.linspace(2.99416,2.99416+0.00416667*(len(Raw_variance_EEG_arr)-1),len(Raw_variance_EEG_arr))

# pyplot.plot(t,Raw_variance_EEG_arr,'grey',alpha=0.5)
# pyplot.ylabel('Voltage',fontsize=13)
# pyplot.xlabel('Time(hour)',fontsize=13)
# pyplot.title('EEG variance in VIC0758',fontsize=13)
# pyplot.show()

var_arr=[]
for item in Raw_variance_EEG_arr:
    if item<1e-8:
        var_arr.append(item)
    else:
        var_arr.append(var_arr[-1])
Raw_variance_EEG_arr=var_arr
# pyplot.plot(t,Raw_variance_EEG_arr,'grey',alpha=0.5)
# pyplot.ylabel('Voltage',fontsize=13)
# pyplot.xlabel('Time(hour)',fontsize=13)
# pyplot.title('EEG variance in VIC0758',fontsize=13)
# pyplot.show()




window_time_arr=t[0:17760]
seizure_timing_index=[]
for k in range(len(window_time_arr)):
    if window_time_arr[k]<4.45416 and window_time_arr[k+1]>=4.45416:
        seizure_timing_index.append(k)
    if window_time_arr[k]<18.5086 and window_time_arr[k+1]>=18.5086:
        seizure_timing_index.append(k)
    if window_time_arr[k]<24.548 and window_time_arr[k+1]>=24.548:
        seizure_timing_index.append(k)
    if window_time_arr[k]<32.0750488 and window_time_arr[k+1]>=32.0750488:
        seizure_timing_index.append(k)
    if window_time_arr[k]<54.873 and window_time_arr[k+1]>=54.873:
        seizure_timing_index.append(k)
print(seizure_timing_index)

#
# # # # ### EEG variance
Raw_variance_EEG=Raw_variance_EEG_arr
window_time_arr=t
# Raw_variance_EEG=Raw_variance_EEG_arr[0:17760]
# window_time_arr=t[0:17760]


long_rhythm_var_arr=movingaverage(Raw_variance_EEG,5760)
medium_rhythm_var_arr=movingaverage(Raw_variance_EEG,240)
medium_rhythm_var_arr_2=movingaverage(Raw_variance_EEG,240*3)
medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG,240*6)
medium_rhythm_var_arr_4=movingaverage(Raw_variance_EEG,240*12)
short_rhythm_var_arr_plot=movingaverage(Raw_variance_EEG,20)


fig=pyplot.figure(figsize=(8,6))
# pyplot.plot(window_time_arr,Raw_variance_EEG,'grey',alpha=0.5,label='Raw variance')
pyplot.plot(window_time_arr,short_rhythm_var_arr_plot,'grey',alpha=0.5,label='5min')
pyplot.plot(window_time_arr,medium_rhythm_var_arr,'g',alpha=0.5,label='1 hour')
pyplot.plot(window_time_arr,medium_rhythm_var_arr_2,'k',label='3 hour')
pyplot.plot(window_time_arr,medium_rhythm_var_arr_3,'orange',label='6 hour')
pyplot.plot(window_time_arr,medium_rhythm_var_arr_4,'b',label='12 hour')
pyplot.plot(window_time_arr,long_rhythm_var_arr,'r',alpha=0.7,label='1 day')
pyplot.title('EEG variance in VIC0758',fontsize=15)
pyplot.xlabel('Time(hour)',fontsize=15)
pyplot.ylabel('Voltage ($\mathregular{v^2}$)',fontsize=15)
pyplot.legend(loc='upper left',fontsize=10)
locs, labels = pyplot.xticks(fontsize=15)
locs, labels = pyplot.yticks(fontsize=15)
pyplot.tight_layout()
pyplot.show()


long_rhythm_var_arr=medium_rhythm_var_arr_4

var_trans=hilbert(long_rhythm_var_arr)
var_trans_nomal=[]
for m in var_trans:
    var_trans_nomal.append(m/abs(m))
SIvarlong=sum(var_trans_nomal)/len(var_trans_nomal)
print(SIvarlong)
seizure_phase=[]
for item in seizure_timing_index:
     seizure_phase.append(var_trans_nomal[item])
SIvarlongseizure=sum(seizure_phase)/len(seizure_phase)
print(SIvarlongseizure)
var_phase=np.angle(var_trans)
phase_long_EEGvariance_arr=var_phase
seizure_phase_var_long=[]
for item in seizure_timing_index:
    seizure_phase_var_long.append(phase_long_EEGvariance_arr[item])
print(seizure_phase_var_long)

# df_rolling=pd.DataFrame(np.array(phase_long_EEGvariance_arr))
# rolmean_long_EEGvar = df_rolling.rolling(240*2).mean()
# rolmean_long_EEGvar=rolmean_long_EEGvar.values

from matplotlib import gridspec
fig = pyplot.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
ax1=pyplot.subplot(gs[0])
ax1.plot(window_time_arr,long_rhythm_var_arr,'orange')
ax1.set_title('EEG variance in VIC0758',fontsize=15)
ax1.set_xlabel('Time(hour)',fontsize=15)
ax1.set_ylabel('Voltage ($\mathregular{v^2}$)',fontsize=15)
locs, labels = pyplot.xticks(fontsize=15)
locs, labels = pyplot.yticks(fontsize=15)
ax1.scatter(window_time_arr[350],long_rhythm_var_arr[350],s=40,c='k')
ax1.scatter(window_time_arr[3723],long_rhythm_var_arr[3723],s=40,c='k')
ax1.scatter(window_time_arr[5172],long_rhythm_var_arr[5172],s=40,c='k')
ax1.scatter(window_time_arr[6979],long_rhythm_var_arr[6979],s=40,c='k')
ax1.scatter(window_time_arr[12450],long_rhythm_var_arr[12450],s=40,c='k')
pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# pyplot.axhline(-0.0000000001,0,0.02364865,linewidth=8,c='grey',alpha=0.4)
# pyplot.axhline(-0.0000000001,0.02364865,0.02364865+0.16216216,linewidth=8,c='k')
# pyplot.axhline(-0.0000000001,0.02364865+0.16216216,0.02364865+0.16216216*2,linewidth=8,c='grey',alpha=0.4)
# pyplot.axhline(-0.0000000001,0.02364865+0.16216216*2,0.02364865+0.16216216*3,linewidth=8,c='k')
# pyplot.axhline(-0.0000000001,0.02364865+0.16216216*3,0.02364865+0.16216216*4,linewidth=8,c='grey',alpha=0.4)
# pyplot.axhline(-0.0000000001,0.02364865+0.16216216*4,0.02364865+0.16216216*5,linewidth=8,c='k')
# pyplot.axhline(-0.0000000001,0.02364865+0.16216216*5,0.02364865+0.16216216*6,linewidth=8,c='grey',alpha=0.4)
# pyplot.axhline(-0.0000000001,0.02364865+0.16216216*6,0.02364865+0.16216216*7,linewidth=8,c='k')
# pyplot.axhline(-0.0000000001,0.02364865+0.16216216*7,1,linewidth=8,c='grey',alpha=0.4)
ax2=pyplot.subplot(gs[1])
ax2.set_xlabel('Time(hour)',fontsize=15)
ax2.set_title('Instantaneous Phase',fontsize=15)
pyplot.xlim(window_time_arr[0],window_time_arr[-1])
ax2.plot(window_time_arr,phase_long_EEGvariance_arr,c='k',alpha=0.5,label='instantaneous phase')
pyplot.hlines(0,window_time_arr[0],window_time_arr[-1],'k','dashed')
# ax2.plot(window_time_arr,rolmean_long_EEGvar,'b',alpha=0.7,label='smoothed phase')
ax2.set_xlabel('Time(hour)',fontsize=15)
locs, labels = pyplot.xticks(fontsize=15)
locs, labels = pyplot.yticks(fontsize=15)
ax2.scatter(window_time_arr[350],phase_long_EEGvariance_arr[350],s=40,c='k')
ax2.scatter(window_time_arr[3723],phase_long_EEGvariance_arr[3723],s=40,c='k')
ax2.scatter(window_time_arr[5172],phase_long_EEGvariance_arr[5172],s=40,c='k')
ax2.scatter(window_time_arr[6979],phase_long_EEGvariance_arr[6979],s=40,c='k')
ax2.scatter(window_time_arr[12450],phase_long_EEGvariance_arr[12450],s=40,c='k')
# locs, labels = pyplot.yticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi],['-$\pi$','Rising','0','Falling','$\pi$'],rotation='vertical',fontsize=15)
# pyplot.ylim(-np.pi,np.pi)
locs, labels = pyplot.yticks([-0.5*np.pi,-0.25*np.pi,0,0.25*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=15)
pyplot.tight_layout()
pyplot.show()

bins_number = 18
# bins = np.linspace(0, 2*np.pi, bins_number + 1)
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
nEEGsvar, _, _ = pyplot.hist(phase_long_EEGvariance_arr, bins)
nEEGsvarsei, _, _ = pyplot.hist(seizure_phase_var_long, bins)
# nEEGsvar, _, _ = pyplot.hist(rolmean_long_EEGvar, bins)
# seizure_phase_var_short=[]
# for item in seizure_timing_index:
#     seizure_phase_var_short=seizure_phase_var_short+list(phase_long_EEGvariance_arr[item])
# nEEGsvarsei, _, _ = pyplot.hist(seizure_phase_var_short, bins)
print(nEEGsvar)
print(nEEGsvarsei)
#
# width = 2*np.pi / bins_number
# params = dict(projection='polar')
# fig, ax = pyplot.subplots(subplot_kw=params)
# print(bins[:bins_number])
# ax.bar(bins[:bins_number], nEEGsvar/sum(nEEGsvar),width=width, color='b',alpha=0.6,linewidth=2, fill=True,edgecolor='k')
# pyplot.setp(ax.get_yticklabels(), color='k')
# ax.set_title('Phase histogram in long EEG variance(SA0124)',fontsize=13)
# ax.set_yticks([0.1,0.2,0.3,0.4,0.5])
# ax.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',])
# pyplot.show()
# params = dict(projection='polar')
# fig, ax2 = pyplot.subplots(subplot_kw=params)
# ax2.bar(bins[:bins_number], nEEGsvarsei/sum(nEEGsvarsei),width=width, color='r',alpha=0.7,linewidth=2,edgecolor='k')
# ax2.set_yticks([0.1,0.2,0.3,0.4,0.5])
# ax2.set_title('seizure probability in long EEG variance(SA0124)',fontsize=13)
# # ax2.set_rlim([0,0.002])
# ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',])
# pyplot.show()




Raw_auto_EEG=Raw_auto_EEG_arr
window_time_arr=t

# pyplot.plot(t,Raw_auto_EEG,'grey',alpha=0.5)
# pyplot.xlabel('Time(hour)',fontsize=13)
# pyplot.title('EEG autocorrelation in VIC0758',fontsize=13)
# pyplot.show()
#
value_arr=[]
for item in Raw_auto_EEG_arr:
    if item<500:
        value_arr.append(item)
    else:
        value_arr.append(value_arr[-1])
Raw_auto_EEG=value_arr
# pyplot.plot(t,Raw_auto_EEG,'grey',alpha=0.5)
# pyplot.xlabel('Time(hour)',fontsize=13)
# pyplot.title('EEG autocorrelation in VIC0758',fontsize=13)
# pyplot.show()


# Raw_auto_EEG=Raw_auto_EEG[0:17760]
# window_time_arr=t[0:17760]

long_rhythm_value_arr=movingaverage(Raw_auto_EEG,5760)
medium_rhythm_value_arr=movingaverage(Raw_auto_EEG,240)
medium_rhythm_value_arr_2=movingaverage(Raw_auto_EEG,240*3)
medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG,240*6)
medium_rhythm_value_arr_4=movingaverage(Raw_auto_EEG,240*12)
short_rhythm_value_arr_plot=movingaverage(Raw_auto_EEG,20)


fig=pyplot.figure(figsize=(8,6))
# pyplot.plot(window_time_arr,Raw_auto_EEG,'grey',alpha=0.5,label='Raw autocorrelation')
# pyplot.plot(window_time_arr[20:],short_rhythm_value_arr_plot[20:],'grey',alpha=0.5,label='5min')
# pyplot.plot(window_time_arr[240:],medium_rhythm_value_arr[240:],'g',label='1 hour')
# pyplot.plot(window_time_arr[240*3:],medium_rhythm_value_arr_2[240*3:],'k',label='3 hour')
# pyplot.plot(window_time_arr[240*6:],medium_rhythm_value_arr_3[240*6:],'orange',label='6 hour')
# pyplot.plot(window_time_arr[240*12:],medium_rhythm_value_arr_4[240*12:],'b',label='12 hour')
# pyplot.plot(window_time_arr[5760:],long_rhythm_value_arr[5760:],'r',alpha=0.7,label='1 day')
pyplot.plot(window_time_arr,short_rhythm_value_arr_plot,'grey',alpha=0.5,label='5min')
pyplot.plot(window_time_arr,medium_rhythm_value_arr,'g',label='1 hour')
pyplot.plot(window_time_arr,medium_rhythm_value_arr_2,'k',label='3 hour')
pyplot.plot(window_time_arr,medium_rhythm_value_arr_3,'orange',label='6 hour')
pyplot.plot(window_time_arr,medium_rhythm_value_arr_4,'b',label='12 hour')
pyplot.plot(window_time_arr,long_rhythm_value_arr,'r',alpha=0.7,label='1 day')
pyplot.title('EEG autocorrelation in VIC0758',fontsize=15)
pyplot.xlabel('Time(hour)',fontsize=15)
pyplot.ylabel('Autocorrelation',fontsize=15)
pyplot.legend(loc='upper left',fontsize=10)
locs, labels = pyplot.xticks(fontsize=15)
locs, labels = pyplot.yticks(fontsize=15)
pyplot.tight_layout()
pyplot.annotate('',xy=(4.45416,np.max(short_rhythm_value_arr_plot)),xytext=(4.45416,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.annotate('',xy=(18.5086,np.max(short_rhythm_value_arr_plot)),xytext=(18.5086,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.annotate('',xy=(24.548,np.max(short_rhythm_value_arr_plot)),xytext=(24.548,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.annotate('',xy=(32.0750488,np.max(short_rhythm_value_arr_plot)),xytext=(32.0750488,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.annotate('',xy=(54.873,np.max(short_rhythm_value_arr_plot)),xytext=(54.873,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.show()





long_rhythm_value_arr=medium_rhythm_value_arr_4

value_trans=hilbert(long_rhythm_value_arr)
value_trans_nomal=[]
for m in value_trans:
    value_trans_nomal.append(m/abs(m))
SIvaluelong=sum(value_trans_nomal)/len(value_trans_nomal)
print(SIvaluelong)
seizure_phase=[]
for item in seizure_timing_index:
    seizure_phase.append(value_trans_nomal[item])
SIvaluelongseizure=sum(seizure_phase)/len(seizure_phase)
print(SIvaluelongseizure)
value_phase=np.angle(value_trans)

phase_long_EEGauto_arr=value_phase
seizure_phase_value_long=[]
for item in seizure_timing_index:
    seizure_phase_value_long.append(phase_long_EEGauto_arr[item])
print(seizure_phase_value_long)


from matplotlib import gridspec
fig = pyplot.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
ax2=pyplot.subplot(gs[0])
ax2.plot(window_time_arr,long_rhythm_value_arr,'orange',alpha=0.7)
pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# pyplot.axhline(-9,0,0.02364865,linewidth=8,c='grey',alpha=0.4)
# pyplot.axhline(-9,0.02364865,0.02364865+0.16216216,linewidth=8,c='k')
# pyplot.axhline(-9,0.02364865+0.16216216,0.02364865+0.16216216*2,linewidth=8,c='grey',alpha=0.4)
# pyplot.axhline(-9,0.02364865+0.16216216*2,0.02364865+0.16216216*3,linewidth=8,c='k')
# pyplot.axhline(-9,0.02364865+0.16216216*3,0.02364865+0.16216216*4,linewidth=8,c='grey',alpha=0.4)
# pyplot.axhline(-9,0.02364865+0.16216216*4,0.02364865+0.16216216*5,linewidth=8,c='k')
# pyplot.axhline(-9,0.02364865+0.16216216*5,0.02364865+0.16216216*6,linewidth=8,c='grey',alpha=0.4)
# pyplot.axhline(-9,0.02364865+0.16216216*6,0.02364865+0.16216216*7,linewidth=8,c='k')
# pyplot.axhline(-9,0.02364865+0.16216216*7,1,linewidth=8,c='grey',alpha=0.4)
ax2.set_title('EEG autocorrelation in VIC0758',fontsize=15)
ax2.set_xlabel('Time(hour)',fontsize=15)
locs, labels = pyplot.xticks(fontsize=15)
locs, labels = pyplot.yticks(fontsize=15)
ax2.scatter(window_time_arr[350],long_rhythm_value_arr[350],s=40,c='k')
ax2.scatter(window_time_arr[3723],long_rhythm_value_arr[3723],s=40,c='k')
ax2.scatter(window_time_arr[5172],long_rhythm_value_arr[5172],s=40,c='k')
ax2.scatter(window_time_arr[6979],long_rhythm_value_arr[6979],s=40,c='k')
ax2.scatter(window_time_arr[12450],long_rhythm_value_arr[12450],s=40,c='k')
ax3=pyplot.subplot(gs[1])
ax3.set_xlabel('Time(hour)',fontsize=15)
ax3.set_title('Instantaneous Phase',fontsize=15)
pyplot.xlim(window_time_arr[0],window_time_arr[-1])
locs, labels = pyplot.xticks(fontsize=15)
locs, labels = pyplot.yticks(fontsize=15)
ax3.plot(window_time_arr,phase_long_EEGauto_arr,'k',alpha=0.5,label='instantaneous phase')
pyplot.hlines(0,window_time_arr[0],window_time_arr[-1],'k','dashed')
# ax3.plot(window_time_arr,phase_long_EEGauto_arr,'b',alpha=0.7,label='smoothed phase')
ax3.scatter(window_time_arr[350],phase_long_EEGauto_arr[350],s=50,c='k')
ax3.scatter(window_time_arr[3723],phase_long_EEGauto_arr[3723],s=40,c='k')
ax3.scatter(window_time_arr[5172],phase_long_EEGauto_arr[5172],s=50,c='k')
ax3.scatter(window_time_arr[6979],phase_long_EEGauto_arr[6979],s=50,c='k')
ax3.scatter(window_time_arr[12450],phase_long_EEGauto_arr[12450],s=50,c='k')
ax3.set_xlabel('Time(hour)',fontsize=15)
# locs, labels = pyplot.yticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi],['-$\pi$','Rising','0','Falling','$\pi$'],rotation='vertical',fontsize=15)
locs, labels = pyplot.yticks([-0.5*np.pi,-0.25*np.pi,0,0.25*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=15)
pyplot.tight_layout()
pyplot.show()

bins_number = 18
# bins = np.linspace(0, 2*np.pi, bins_number + 1)
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
nEEGsauto, _, _ = pyplot.hist(phase_long_EEGauto_arr, bins)
nEEGsautosei, _, _ = pyplot.hist(seizure_phase_value_long, bins)
# nEEGsauto, _, _ = pyplot.hist(rolmean_long_EEGauto, bins)
# seizure_phase_auto_long=[]
# for item in seizure_timing_index:
#     seizure_phase_auto_long=seizure_phase_auto_long+list(rolmean_long_EEGauto[item])
# nEEGsautosei, _, _ = pyplot.hist(seizure_phase_auto_long, bins)
print(nEEGsauto)
print(nEEGsautosei)
# width = 2*np.pi / bins_number
# params = dict(projection='polar')
# fig, ax = pyplot.subplots(subplot_kw=params)
# ax.bar(bins[:bins_number], nEEGsauto/sum(nEEGsauto),width=width, color='g',alpha=0.6,linewidth=2, fill=True,edgecolor='k')
# pyplot.setp(ax.get_yticklabels(), color='k')
# ax.set_title('Phase histogram in long EEG autocorrelation(SA0124)',fontsize=13)
# ax.set_yticks([0.1,0.2,0.3,0.4,0.5])
# ax.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',])
# pyplot.show()
# params = dict(projection='polar')
# fig, ax2 = pyplot.subplots(subplot_kw=params)
# ax2.bar(bins[:bins_number], nEEGsautosei/sum(nEEGsautosei),width=width, color='orange',alpha=0.9,linewidth=2,edgecolor='k')
# ax2.set_yticks([0.1,0.2,0.3,0.4,0.5])
# ax2.set_title('seizure probability in long EEG autocorrelation(SA0124)',fontsize=13)
# ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',])
# # ax2.set_rlim([0,0.002])
# pyplot.show()





csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_timewindowarr_VIC0758_15s_3h.csv',sep=',',header=None)
rri_t= csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_rawvariance_VIC0758_15s_3h.csv',sep=',',header=None)
RRI_var= csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_rawauto_VIC0758_15s_3h.csv',sep=',',header=None)
Raw_auto_RRI31= csv_reader.values


rri_t_arr=[]
for item in rri_t:
    rri_t_arr.append(float(item))
Raw_variance_RRI31_arr=[]
for item in RRI_var:
    Raw_variance_RRI31_arr.append(float(item))
Raw_auto_RRI31_arr=[]
for item in Raw_auto_RRI31:
    Raw_auto_RRI31_arr.append(float(item))
#
#
# df_rolling=pd.DataFrame(np.array(Raw_variance_RRI31_arr))
# rolmean_rri = df_rolling.rolling(3600).mean()
# rolmean_rri=rolmean_rri.values
#



window_time_arr=t
Raw_variance_RRI31=Raw_variance_RRI31_arr

pyplot.figure()
pyplot.plot(t,Raw_variance_RRI31_arr,'grey',alpha=0.5)
pyplot.show()

var_arr=[]
for item in Raw_variance_RRI31_arr:
    if item<0.02:
        var_arr.append(item)
    else:
        var_arr.append(var_arr[-1])
Raw_variance_RRI31_arr=var_arr

pyplot.figure()
pyplot.plot(t,Raw_variance_RRI31_arr,'grey',alpha=0.5)
pyplot.show()



# window_time_arr=t[0:17760]
# Raw_variance_RRI31=Raw_variance_RRI31_arr[0:17760]


long_rhythm_var_arr=movingaverage(Raw_variance_RRI31,5760)
medium_rhythm_var_arr=movingaverage(Raw_variance_RRI31,240)
medium_rhythm_var_arr_2=movingaverage(Raw_variance_RRI31,240*3)
medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31,240*6)
medium_rhythm_var_arr_4=movingaverage(Raw_variance_RRI31,240*12)
short_rhythm_var_arr_plot=movingaverage(Raw_variance_RRI31,20)


fig=pyplot.figure(figsize=(8,6))
# pyplot.plot(window_time_arr,Raw_variance_EEG,'grey',alpha=0.5,label='Raw variance')
pyplot.plot(window_time_arr,short_rhythm_var_arr_plot,'grey',alpha=0.5,label='5min')
pyplot.plot(window_time_arr,medium_rhythm_var_arr,'g',label='1 hour')
pyplot.plot(window_time_arr,medium_rhythm_var_arr_2,'k',label='3 hour')
pyplot.plot(window_time_arr,medium_rhythm_var_arr_3,'orange',label='6 hour')
pyplot.plot(window_time_arr,medium_rhythm_var_arr_4,'b',label='12 hour')
pyplot.plot(window_time_arr,long_rhythm_var_arr,'r',alpha=0.7,label='1 day')
pyplot.title('RRI variance in VIC0758',fontsize=15)
pyplot.xlabel('Time(hour)',fontsize=15)
pyplot.ylabel('Second ($\mathregular{s^2}$)',fontsize=15)
pyplot.legend(loc='upper left',fontsize=10)
locs, labels = pyplot.xticks(fontsize=15)
locs, labels = pyplot.yticks(fontsize=15)
pyplot.tight_layout()
pyplot.show()


long_rhythm_var_arr=medium_rhythm_var_arr_4

var_trans=hilbert(long_rhythm_var_arr)
var_trans_nomal=[]
for m in var_trans:
    var_trans_nomal.append(m/abs(m))
SIvarlong=sum(var_trans_nomal)/len(var_trans_nomal)
print(SIvarlong)
seizure_phase=[]
for item in seizure_timing_index:
     seizure_phase.append(var_trans_nomal[item])
SIvarlongseizure=sum(seizure_phase)/len(seizure_phase)
print(SIvarlongseizure)
var_phase=np.angle(var_trans)
phase_whole_long=var_phase
seizure_phase_var_long=[]
for item in seizure_timing_index:
    seizure_phase_var_long.append(phase_whole_long[item])
print(seizure_phase_var_long)

# df_rolling=pd.DataFrame(np.array(phase_whole_long))
# rolmean_long_RRIvar = df_rolling.rolling(240*2).mean()
# rolmean_long_RRIvar=rolmean_long_RRIvar.values


from matplotlib import gridspec
fig = pyplot.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
ax1=pyplot.subplot(gs[0])
ax1.plot(window_time_arr,long_rhythm_var_arr,'orange',alpha=0.7)
pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# pyplot.axhline(-0.007,0,0.02364865,linewidth=8,c='grey',alpha=0.4)
# pyplot.axhline(-0.007,0.02364865,0.02364865+0.16216216,linewidth=8,c='k')
# pyplot.axhline(-0.007,0.02364865+0.16216216,0.02364865+0.16216216*2,linewidth=8,c='grey',alpha=0.4)
# pyplot.axhline(-0.007,0.02364865+0.16216216*2,0.02364865+0.16216216*3,linewidth=8,c='k')
# pyplot.axhline(-0.007,0.02364865+0.16216216*3,0.02364865+0.16216216*4,linewidth=8,c='grey',alpha=0.4)
# pyplot.axhline(-0.007,0.02364865+0.16216216*4,0.02364865+0.16216216*5,linewidth=8,c='k')
# pyplot.axhline(-0.007,0.02364865+0.16216216*5,0.02364865+0.16216216*6,linewidth=8,c='grey',alpha=0.4)
# pyplot.axhline(-0.007,0.02364865+0.16216216*6,0.02364865+0.16216216*7,linewidth=8,c='k')
# pyplot.axhline(-0.007,0.02364865+0.16216216*7,1,linewidth=8,c='grey',alpha=0.4)
ax1.set_title('RRI variance in VIC0758',fontsize=15)
ax1.set_xlabel('Time(hour)',fontsize=15)
ax1.set_ylabel('Second($\mathregular{s^2}$)',fontsize=15)
locs, labels = pyplot.xticks(fontsize=15)
locs, labels = pyplot.yticks(fontsize=15)
ax1.scatter(window_time_arr[350],long_rhythm_var_arr[350],s=40,c='k')
ax1.scatter(window_time_arr[3723],long_rhythm_var_arr[3723],s=40,c='k')
ax1.scatter(window_time_arr[5172],long_rhythm_var_arr[5172],s=40,c='k')
ax1.scatter(window_time_arr[6979],long_rhythm_var_arr[6979],s=40,c='k')
ax1.scatter(window_time_arr[12450],long_rhythm_var_arr[12450],s=40,c='k')
pyplot.xlim(window_time_arr[0],window_time_arr[-1])
ax2=pyplot.subplot(gs[1])
ax2.set_xlabel('Time(hour)',fontsize=15)
ax2.set_title('Instantaneous Phase',fontsize=15)
pyplot.xlim(window_time_arr[0],window_time_arr[-1])
ax2.plot(window_time_arr,phase_whole_long,'k',alpha=0.5)
ax2.scatter(window_time_arr[350],phase_whole_long[350],s=40,c='k')
ax2.scatter(window_time_arr[3723],phase_whole_long[3723],s=40,c='k')
ax2.scatter(window_time_arr[5172],phase_whole_long[5172],s=40,c='k')
ax2.scatter(window_time_arr[6979],phase_whole_long[6979],s=40,c='k')
ax2.scatter(window_time_arr[12450],phase_whole_long[12450],s=40,c='k')
ax2.set_xlabel('Time(hour)',fontsize=15)
locs, labels = pyplot.xticks(fontsize=15)
locs, labels = pyplot.yticks(fontsize=15)
pyplot.hlines(0,window_time_arr[0],window_time_arr[-1],'k','dashed')
# locs, labels = pyplot.yticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi],['-$\pi$','Rising','0','Falling','$\pi$'],rotation='vertical',fontsize=15)
locs, labels = pyplot.yticks([-0.5*np.pi,-0.25*np.pi,0,0.25*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=15)
pyplot.tight_layout()
pyplot.show()
bins_number = 18
# bins = np.linspace(0, 2*np.pi, bins_number + 1)
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
nRRIsvar, _, _ = pyplot.hist(phase_whole_long, bins)
nRRIsvarsei, _, _ = pyplot.hist(seizure_phase_var_long, bins)
# nRRIsvar, _, _ = pyplot.hist(rolmean_long_RRIvar, bins)
# seizure_phase_var_long=[]
# for item in seizure_timing_index:
#     seizure_phase_var_long=seizure_phase_var_long+list(rolmean_long_RRIvar[item])
# nRRIsvarsei, _, _ = pyplot.hist(seizure_phase_var_long, bins)
print(nRRIsvar)
print(nRRIsvarsei)

# width = 2*np.pi / bins_number
# params = dict(projection='polar')
# fig, ax = pyplot.subplots(subplot_kw=params)
# ax.bar(bins[:bins_number], nRRIsvar/sum(nRRIsvar),width=width, color='grey',alpha=0.6,linewidth=2, fill=True,edgecolor='k')
# pyplot.setp(ax.get_yticklabels(), color='k', alpha=0.7)
# ax.set_title('Phase histogram in long RRI variance(SA0124)',fontsize=13)
# ax.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',])
# pyplot.show()
# params = dict(projection='polar')
# fig, ax2 = pyplot.subplots(subplot_kw=params)
# ax2.bar(bins[:bins_number], nRRIsvarsei/sum(nRRIsvarsei),width=width, color='C0',edgecolor='k',linewidth=2)
# ax2.set_title('seizure probability in long RRI variance(SA0124)',fontsize=13)
# ax2.set_yticks([0.1,0.2,0.3,0.4,0.5])
# # ax2.set_rlim([0,0.002])
# ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',])
# pyplot.show()


# #
Raw_auto_RRI31=Raw_auto_RRI31_arr
window_time_arr=t
# Raw_auto_RRI31=Raw_auto_RRI31_arr[0:17760]
# window_time_arr=t[0:17760]

long_rhythm_value_arr=movingaverage(Raw_auto_RRI31,5760)
medium_rhythm_value_arr=movingaverage(Raw_auto_RRI31,240)
medium_rhythm_value_arr_2=movingaverage(Raw_auto_RRI31,240*3)
medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31,240*6)
medium_rhythm_value_arr_4=movingaverage(Raw_auto_RRI31,240*12)
short_rhythm_value_arr_plot=movingaverage(Raw_auto_RRI31,20)


fig=pyplot.figure(figsize=(8,6))
# pyplot.plot(window_time_arr,Raw_auto_EEG,'grey',alpha=0.5,label='Raw autocorrelation')
# pyplot.plot(window_time_arr[20:],short_rhythm_value_arr_plot[20:],'grey',alpha=0.5,label='5min')
# pyplot.plot(window_time_arr[240:],medium_rhythm_value_arr[240:],'g',label='1 hour')
# pyplot.plot(window_time_arr[240*3:],medium_rhythm_value_arr_2[240*3:],'k',label='3 hour')
# pyplot.plot(window_time_arr[240*6:],medium_rhythm_value_arr_3[240*6:],'orange',label='6 hour')
# pyplot.plot(window_time_arr[240*12:],medium_rhythm_value_arr_4[240*12:],'b',label='12 hour')
# pyplot.plot(window_time_arr[5760:],long_rhythm_value_arr[5760:],'r',alpha=0.7,label='1 day')
pyplot.plot(window_time_arr,short_rhythm_value_arr_plot,'grey',alpha=0.5,label='5min')
pyplot.plot(window_time_arr,medium_rhythm_value_arr,'g',label='1 hour')
pyplot.plot(window_time_arr,medium_rhythm_value_arr_2,'k',label='3 hour')
pyplot.plot(window_time_arr,medium_rhythm_value_arr_3,'orange',label='6 hour')
pyplot.plot(window_time_arr,medium_rhythm_value_arr_4,'b',label='12 hour')
pyplot.plot(window_time_arr,long_rhythm_value_arr,'r',alpha=0.7,label='1 day')
pyplot.title('RRI autocorrelation in VIC0758',fontsize=15)
pyplot.xlabel('Time(hour)',fontsize=15)
pyplot.ylabel('Autocorrelation',fontsize=15)
pyplot.legend(loc='upper left',fontsize=10)
locs, labels = pyplot.xticks(fontsize=15)
locs, labels = pyplot.yticks(fontsize=15)
pyplot.tight_layout()
pyplot.show()



long_rhythm_value_arr=medium_rhythm_value_arr_4
value_trans=hilbert(long_rhythm_value_arr)
value_trans_nomal=[]
for m in value_trans:
    value_trans_nomal.append(m/abs(m))
SIvaluelong=sum(value_trans_nomal)/len(value_trans_nomal)
print(SIvaluelong)
seizure_phase=[]
for item in seizure_timing_index:
    seizure_phase.append(value_trans_nomal[item])
SIvaluelongseizure=sum(seizure_phase)/len(seizure_phase)
print(SIvaluelongseizure)
value_phase=np.angle(value_trans)
phase_whole_value_long=value_phase
seizure_phase_value_long=[]
for item in seizure_timing_index:
    seizure_phase_value_long.append(phase_whole_value_long[item])
print(seizure_phase_value_long)

# df_rolling=pd.DataFrame(np.array(long_rhythm_value_arr))
# rolmean_long_RRIauto = df_rolling.rolling(240*3).mean()
# rolmean_long_RRIauto=rolmean_long_RRIauto.values

from matplotlib import gridspec
fig = pyplot.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
ax1=pyplot.subplot(gs[0])
ax1.set_title('RRI autocorrelation in VIC0758',fontsize=15)
ax1.set_xlabel('Time(hour)',fontsize=15)
locs, labels = pyplot.xticks(fontsize=15)
locs, labels = pyplot.yticks(fontsize=15)
ax1.plot(window_time_arr,long_rhythm_value_arr,'orange',alpha=0.7)
pyplot.ylim(0,5)
# ax1.plot(window_time_arr[240*6:],rolmean_long_RRIauto,'C0',alpha=0.7)
# pyplot.xlim(window_time_arr[0],window_time_arr[-1])
ax2.scatter(window_time_arr[350],long_rhythm_value_arr[350],s=40,c='k')
ax2.scatter(window_time_arr[3723],long_rhythm_value_arr[3723],s=40,c='k')
ax2.scatter(window_time_arr[5172],long_rhythm_value_arr[5172],s=40,c='k')
ax2.scatter(window_time_arr[6979],long_rhythm_value_arr[6979],s=40,c='k')
ax2.scatter(window_time_arr[12450],long_rhythm_value_arr[12450],s=40,c='k')
# pyplot.axhline(-1.6,0,0.02364865,linewidth=8,c='grey',alpha=0.4)
# pyplot.axhline(-1.6,0.02364865,0.02364865+0.16216216,linewidth=8,c='k')
# pyplot.axhline(-1.6,0.02364865+0.16216216,0.02364865+0.16216216*2,linewidth=8,c='grey',alpha=0.4)
# pyplot.axhline(-1.6,0.02364865+0.16216216*2,0.02364865+0.16216216*3,linewidth=8,c='k')
# pyplot.axhline(-1.6,0.02364865+0.16216216*3,0.02364865+0.16216216*4,linewidth=8,c='grey',alpha=0.4)
# pyplot.axhline(-1.6,0.02364865+0.16216216*4,0.02364865+0.16216216*5,linewidth=8,c='k')
# pyplot.axhline(-1.6,0.02364865+0.16216216*5,0.02364865+0.16216216*6,linewidth=8,c='grey',alpha=0.4)
# pyplot.axhline(-1.6,0.02364865+0.16216216*6,0.02364865+0.16216216*7,linewidth=8,c='k')
# pyplot.axhline(-1.6,0.02364865+0.16216216*7,1,linewidth=8,c='grey',alpha=0.4)
ax2=pyplot.subplot(gs[1])
ax2.set_title('Instantaneous Phase',fontsize=15)
ax2.set_xlabel('Time(hour)',fontsize=15)
locs, labels = pyplot.xticks(fontsize=15)
locs, labels = pyplot.yticks(fontsize=15)
pyplot.hlines(0,window_time_arr[5760],window_time_arr[-1],'k','dashed')
# pyplot.xlim(window_time_arr[0],window_time_arr[-1])
ax2.plot(window_time_arr,phase_whole_value_long,'k',alpha=0.5)
ax2.scatter(window_time_arr[350],phase_whole_value_long[350],s=40,c='k')
ax2.scatter(window_time_arr[3723],phase_whole_value_long[3723],s=40,c='k')
ax2.scatter(window_time_arr[5172],phase_whole_value_long[5172],s=40,c='k')
ax2.scatter(window_time_arr[6979],phase_whole_value_long[6979],s=40,c='k')
ax2.scatter(window_time_arr[12450],phase_whole_value_long[12450],s=40,c='k')
ax2.set_xlabel('Time(hour)',fontsize=15)
# locs, labels = pyplot.yticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi],['-$\pi$','Rising','0','Falling','$\pi$'],rotation='vertical',fontsize=15)
locs, labels = pyplot.yticks([-0.5*np.pi,-0.25*np.pi,0,0.25*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=15)
pyplot.tight_layout()
pyplot.show()
bins_number = 18
# bins = np.linspace(0, 2*np.pi, bins_number + 1)
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
nRRIsauto, _, _ = pyplot.hist(phase_whole_value_long, bins)
nRRIsautosei, _, _ = pyplot.hist(seizure_phase_value_long, bins)
# nRRIsauto, _, _ = pyplot.hist(rolmean_long_RRIauto, bins)
# seizure_phase_value_long=[]
# for item in seizure_timing_index:
#     seizure_phase_value_long=seizure_phase_value_long+list(rolmean_long_RRIauto[item])
# nRRIsautosei, _, _ = pyplot.hist(seizure_phase_value_long, bins)
print(nRRIsauto)
print(nRRIsautosei)

# # width = 2*np.pi / bins_number
# # params = dict(projection='polar')
# # fig, ax = pyplot.subplots(subplot_kw=params)
# # ax.bar(bins[:bins_number], nRRIsauto/sum(nRRIsauto),width=width, color='grey',alpha=0.6,linewidth=2, fill=True,edgecolor='k')
# # pyplot.setp(ax.get_yticklabels(), color='k')
# # ax.set_title('phase histogram in long RRI autocorrelation(SA0124)',fontsize=13)
# # ax.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',])
# # pyplot.show()
# # params = dict(projection='polar')
# # fig, ax2 = pyplot.subplots(subplot_kw=params)
# # ax2.bar(bins[:bins_number], nRRIsautosei/sum(nRRIsautosei),width=width, color='g',edgecolor='k',linewidth=2)
# # ax2.set_title('seizure probability in long RRI autocorrelation(SA0124)',fontsize=13)
# # # ax2.set_yticks([0.0008,0.0012,0.0016,0.002])
# # ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',])
# # pyplot.show()





