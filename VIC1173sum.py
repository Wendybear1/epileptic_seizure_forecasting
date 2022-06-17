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




csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.Oct/Cz_EEGvariance_VIC1173_15s_3h.csv',sep=',',header=None)
Raw_variance_EEG= csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.Oct/Cz_EEGauto_VIC1173_15s_3h.csv',sep=',',header=None)
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.Oct/Cz_EEGauto_lag1_VIC1173_15s_3h.csv',sep=',',header=None)
Raw_auto_EEG= csv_reader.values
Raw_variance_EEG_arr=[]
for item in Raw_variance_EEG:
    Raw_variance_EEG_arr.append(float(item))
Raw_auto_EEG_arr=[]
for item in Raw_auto_EEG:
    Raw_auto_EEG_arr.append(float(item))
print(len(Raw_variance_EEG_arr));print(len(Raw_auto_EEG_arr));

t=np.linspace(0.00416667,0.00416667+0.00416667*(len(Raw_variance_EEG_arr)-1),len(Raw_variance_EEG_arr))
print(t[-1]);print(t[-1]-t[0]);print(len(t));
print(t[22318]);print(t[22318]-t[0]);print(t[-1]-t[22318]);


# pyplot.plot(t,Raw_variance_EEG_arr,'grey',alpha=0.3)
# pyplot.xlabel('Time(hour)',fontsize=13)
# pyplot.ylabel('Voltage($\mathregular{V}$)',fontsize=13)
# pyplot.title('EEG variance in VIC1137',fontsize=13)
# pyplot.show()
var_arr=[]
for item in Raw_variance_EEG_arr:
    if item<1e-8:
        var_arr.append(item)
    else:
        var_arr.append(var_arr[-1])
Raw_variance_EEG_arr=var_arr
# pyplot.plot(t,Raw_variance_EEG_arr,'grey',alpha=0.3)
# pyplot.xlabel('Time(hour)',fontsize=13)
# pyplot.ylabel('Voltage($\mathregular{V}$)',fontsize=13)
# pyplot.title('EEG variance in VIC1137',fontsize=13)
# pyplot.show()

# pyplot.plot(t,Raw_auto_EEG_arr,'grey',alpha=0.3)
# pyplot.xlabel('Time(hour)',fontsize=13)
# pyplot.title('EEG autocorrelation in VIC1137',fontsize=13)
# pyplot.show()
value_arr=[]
for item in Raw_auto_EEG_arr:
    if item<500:
        value_arr.append(item)
    else:
        value_arr.append(value_arr[-1])
Raw_auto_EEG_arr=value_arr
# pyplot.plot(t,Raw_auto_EEG_arr,'grey',alpha=0.5)
# pyplot.xlabel('Time(hour)',fontsize=13)
# pyplot.title('EEG autocorrelation in VIC1137',fontsize=13)
# pyplot.show()



# # # # ### EEG var EEG var EEG var EEG var
Raw_variance_EEG=Raw_variance_EEG_arr[0:len(t)]
window_time_arr=t
# Raw_variance_EEG=Raw_variance_EEG_arr[0:22318]
# window_time_arr=t[0:22318]

seizure_timing_index = []
for k in range(len(window_time_arr)):
    if window_time_arr[k] < 18.9266667 and window_time_arr[k + 1] >= 18.9266667:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 48.4025 and window_time_arr[k + 1] >= 48.4025:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 92.0886111 and window_time_arr[k + 1] >= 92.0886111:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 121.011667 and window_time_arr[k + 1] >= 121.011667:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 125.851667 and window_time_arr[k + 1] >= 125.851667:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 137.429722 and window_time_arr[k + 1] >= 137.429722:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 145.430278 and window_time_arr[k + 1] >= 145.430278:
        seizure_timing_index.append(k)
print(seizure_timing_index)
index_ictal = seizure_timing_index


seizure_timing_index=[]
for k in range(len(window_time_arr)):
    if window_time_arr[k]<95.018333 and window_time_arr[k+1]>=95.018333:
        seizure_timing_index.append(k)
    if window_time_arr[k]<126.42278 and window_time_arr[k+1]>=126.42278:
        seizure_timing_index.append(k)
    if window_time_arr[k]<126.68611 and window_time_arr[k+1]>=126.68611:
        seizure_timing_index.append(k)
    if window_time_arr[k]<127.90267 and window_time_arr[k+1]>=127.90267:
        seizure_timing_index.append(k)
    if window_time_arr[k]<147.35583 and window_time_arr[k+1]>=147.35583:
        seizure_timing_index.append(k)
    if window_time_arr[k]<147.889167 and window_time_arr[k+1]>=147.889167:
        seizure_timing_index.append(k)
print(seizure_timing_index)
index_cluster=seizure_timing_index


x=[4541, 11615, 22100, 29041, 30203, 32982, 34902]
y=[22803, 30340, 30403, 30695, 35364, 35492]
a=[30203]
b=[30695]
list_arr=[]
for i in range(len(a)):
    c=range(a[i]+1,b[i]+1)
    list_arr=list_arr+list(c)
list_arr.append(22803);list_arr.append(30340);
list_arr.append(30403);list_arr.append(35364);list_arr.append(35492);
index_cluster=list_arr
print(index_cluster)

duration=[2,3,20,80,361,25,20]
index_ictal_sum=[]
for m in range(len(index_ictal)):
    for j in range(duration[m]+1):
        index_ictal_sum.append(index_ictal[m] + j)
index_cluster_sum=[]
for item in index_cluster:
    for j in range(8):
        index_cluster_sum.append(item+j)
index_cluster_sum=pd.unique(index_cluster_sum).tolist()
index_ictal_sum.sort();index_cluster_sum.sort()
print(index_ictal_sum);print(index_cluster_sum);

index_pre_sum=[]
for item in index_ictal:
    for i in range(1,60):
        index_pre_sum.append(item-i)
index_pre_cluster_sum=[]
for item in index_cluster:
    for i in range(1,60):
        index_pre_cluster_sum.append(item-i)
index_pre_sum.sort(); index_pre_cluster_sum.sort()
index_pre_sum=pd.unique(index_pre_sum).tolist()
index_pre_cluster_sum=pd.unique(index_pre_cluster_sum).tolist()
print(index_pre_sum);print(index_pre_cluster_sum);

x=np.ones(len(t))
for k in range(len(x)):
    if k in index_ictal_sum:
        x[k] = 1
    elif k in index_pre_sum:
        x[k] = 2
    elif k in index_cluster_sum:
        x[k] = 11
    elif k in index_pre_cluster_sum:
        x[k] = 22
    else:
        x[k] = 0
np.savetxt("C:/Users/wxiong/Documents/PHD/combine_features/VIC1173_tags.csv", x, delimiter=",", fmt='%s')



# long_rhythm_var_arr=movingaverage(Raw_variance_EEG,240*6)
# medium_rhythm_var_arr=movingaverage(Raw_variance_EEG,240)
# medium_rhythm_var_arr_2=movingaverage(Raw_variance_EEG,240*3)
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG,240*6)
# medium_rhythm_var_arr_4=movingaverage(Raw_variance_EEG,240*12)
# short_rhythm_var_arr_plot=movingaverage(Raw_variance_EEG,240*24)
#
# # fig=pyplot.figure(figsize=(8,6))
# # # pyplot.plot(window_time_arr,Raw_variance_EEG,'grey',alpha=0.5,label='Raw variance')
# # # pyplot.plot(window_time_arr,short_rhythm_var_arr_plot,'grey',alpha=0.5,label='5min')
# # pyplot.plot(window_time_arr,medium_rhythm_var_arr,'g',label='1 hour')
# # pyplot.plot(window_time_arr,medium_rhythm_var_arr_2,'k',label='3 hour')
# # pyplot.plot(window_time_arr,medium_rhythm_var_arr_3,'orange',label='6 hour')
# # pyplot.plot(window_time_arr,medium_rhythm_var_arr_4,'b',label='12 hour')
# # pyplot.plot(window_time_arr,long_rhythm_var_arr,'r',alpha=0.7,label='1 day')
# # pyplot.title('EEG variance in VIC1173',fontsize=15)
# # pyplot.xlabel('Time(hour)',fontsize=15)
# # pyplot.ylabel('Voltage($\mathregular{v^2}$)',fontsize=15)
# # pyplot.legend(loc='upper left',fontsize=10)
# # locs, labels = pyplot.xticks(fontsize=15)
# # locs, labels = pyplot.yticks(fontsize=15)
# # pyplot.tight_layout()
# # pyplot.annotate('',xy=(18.9266667,np.max(short_rhythm_var_arr_plot)),xytext=(18.9266667,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(48.4025,np.max(short_rhythm_var_arr_plot)),xytext=(48.4025,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(92.0886111,np.max(short_rhythm_var_arr_plot)),xytext=(92.0886111,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(121.011667,np.max(short_rhythm_var_arr_plot)),xytext=(121.011667,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(125.851667,np.max(short_rhythm_var_arr_plot)),xytext=(125.851667,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(137.429722,np.max(short_rhythm_var_arr_plot)),xytext=(137.429722,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(145.430278,np.max(short_rhythm_var_arr_plot)),xytext=(145.430278,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.show()
#
#
# long_rhythm_var_arr=long_rhythm_var_arr
# var_trans=hilbert(long_rhythm_var_arr)
# var_trans_nomal=[]
# for m in var_trans:
#     var_trans_nomal.append(m/abs(m))
# SIvarlong=sum(var_trans_nomal)/len(var_trans_nomal)
# print(SIvarlong)
# seizure_phase=[]
# for item in seizure_timing_index:
#      seizure_phase.append(var_trans_nomal[item])
# SIvarlongseizure=sum(seizure_phase)/len(seizure_phase)
# print(SIvarlongseizure)
# var_phase=np.angle(var_trans)
# phase_long_EEGvariance_arr=var_phase
# seizure_phase_var_long=[]
# for item in seizure_timing_index:
#     seizure_phase_var_long.append(phase_long_EEGvariance_arr[item])
# print(seizure_phase_var_long)
# n=0
# for item in seizure_phase_var_long:
#     if item <0:
#         n=n+1
# print(n/len(seizure_phase_var_long))
#
# # from matplotlib import gridspec
# # fig = pyplot.figure(figsize=(12, 10))
# # gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
# # ax1=pyplot.subplot(gs[0])
# # ax1.plot(window_time_arr[240*6:],long_rhythm_var_arr[240*6:],'orange')
# # ax1.set_title('EEG variance in VIC1173',fontsize=15)
# # ax1.set_xlabel('Time(hour)',fontsize=15)
# # ax1.set_ylabel('Voltage ($\mathregular{v^2}$)',fontsize=15)
# # locs, labels = pyplot.xticks(fontsize=15)
# # locs, labels = pyplot.yticks(fontsize=15)
# # ax1.scatter(window_time_arr[4541],long_rhythm_var_arr[4541],s=40,c='k')
# # ax1.scatter(window_time_arr[11615],long_rhythm_var_arr[11615],s=40,c='k')
# # ax1.scatter(window_time_arr[22100],long_rhythm_var_arr[22100],s=40,c='k')
# # # ax1.scatter(window_time_arr[29041],long_rhythm_var_arr[29041],s=40,c='k')
# # # ax1.scatter(window_time_arr[30203],long_rhythm_var_arr[30203],s=40,c='k')
# # # ax1.scatter(window_time_arr[32982],long_rhythm_var_arr[32982],s=40,c='k')
# # # ax1.scatter(window_time_arr[34902],long_rhythm_var_arr[34902],s=40,c='k')
# # ax2=pyplot.subplot(gs[1])
# # ax2.set_xlabel('Time(hour)',fontsize=15)
# # ax2.set_title('Instantaneous Phase',fontsize=15)
# # ax2.plot(window_time_arr[240*6:],phase_long_EEGvariance_arr[240*6:],c='k',alpha=0.5,label='instantaneous phase')
# # pyplot.hlines(0,window_time_arr[240*6],window_time_arr[-1],'k','dashed')
# # ax2.set_xlabel('Time(hour)',fontsize=15)
# # locs, labels = pyplot.xticks(fontsize=15)
# # locs, labels = pyplot.yticks(fontsize=15)
# # ax2.scatter(window_time_arr[4541],phase_long_EEGvariance_arr[4541],s=40,c='k')
# # ax2.scatter(window_time_arr[11615],phase_long_EEGvariance_arr[11615],s=40,c='k')
# # ax2.scatter(window_time_arr[22100],phase_long_EEGvariance_arr[22100],s=40,c='k')
# # # ax2.scatter(window_time_arr[29041],phase_long_EEGvariance_arr[29041],s=40,c='k')
# # # ax2.scatter(window_time_arr[30203],phase_long_EEGvariance_arr[30203],s=40,c='k')
# # # ax2.scatter(window_time_arr[32982],phase_long_EEGvariance_arr[32982],s=40,c='k')
# # # ax2.scatter(window_time_arr[34902],phase_long_EEGvariance_arr[34902],s=40,c='k')
# # # pyplot.ylim(-np.pi,np.pi)
# # locs, labels = pyplot.yticks([-0.5*np.pi,-0.25*np.pi,0,0.25*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=15)
# # pyplot.tight_layout()
# # pyplot.show()
# # bins_number = 18
# # bins = np.linspace(-np.pi, np.pi, bins_number + 1)
# # nEEGsvar, _, _ = pyplot.hist(phase_long_EEGvariance_arr, bins)
# # nEEGsvarsei, _, _ = pyplot.hist(seizure_phase_var_long, bins)
# # print(nEEGsvar)
# # print(nEEGsvarsei)
#
#
#
#
# # # # # ### EEG auto EEG auto EEG auto EEG auto
# # Raw_auto_EEG=Raw_auto_EEG_arr[0:len(t)]
# # window_time_arr=t
# Raw_auto_EEG=Raw_auto_EEG_arr[0:22318]
# window_time_arr=t[0:22318]
#
# long_rhythm_value_arr=movingaverage(Raw_auto_EEG,240*6)
# medium_rhythm_value_arr=movingaverage(Raw_auto_EEG,240)
# medium_rhythm_value_arr_2=movingaverage(Raw_auto_EEG,240*3)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG,240*6)
# medium_rhythm_value_arr_4=movingaverage(Raw_auto_EEG,240*12)
# short_rhythm_value_arr_plot=movingaverage(Raw_auto_EEG,240*24)
#
# # fig=pyplot.figure(figsize=(8,6))
# # # pyplot.plot(window_time_arr,short_rhythm_value_arr_plot,'grey',alpha=0.5,label='5min')
# # pyplot.plot(window_time_arr,medium_rhythm_value_arr,'g',label='1 hour')
# # pyplot.plot(window_time_arr,medium_rhythm_value_arr_2,'k',label='3 hour')
# # pyplot.plot(window_time_arr,medium_rhythm_value_arr_3,'orange',label='6 hour')
# # pyplot.plot(window_time_arr,medium_rhythm_value_arr_4,'b',label='12 hour')
# # pyplot.plot(window_time_arr,long_rhythm_value_arr,'r',alpha=0.7,label='1 day')
# # pyplot.title('EEG autocorrelation in VIC1173',fontsize=15)
# # pyplot.xlabel('Time(hour)',fontsize=15)
# # pyplot.ylabel('Autocorrelation',fontsize=15)
# # pyplot.legend(loc='upper left',fontsize=10)
# # locs, labels = pyplot.xticks(fontsize=15)
# # locs, labels = pyplot.yticks(fontsize=15)
# # pyplot.tight_layout()
# # pyplot.annotate('',xy=(18.9266667,np.max(short_rhythm_value_arr_plot)),xytext=(18.9266667,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(48.4025,np.max(short_rhythm_value_arr_plot)),xytext=(48.4025,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(92.0886111,np.max(short_rhythm_value_arr_plot)),xytext=(92.0886111,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(121.011667,np.max(short_rhythm_value_arr_plot)),xytext=(121.011667,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(125.851667,np.max(short_rhythm_value_arr_plot)),xytext=(125.851667,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(137.429722,np.max(short_rhythm_value_arr_plot)),xytext=(137.429722,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(145.430278,np.max(short_rhythm_value_arr_plot)),xytext=(145.430278,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.show()
#
# long_rhythm_value_arr=long_rhythm_value_arr
# value_trans=hilbert(long_rhythm_value_arr)
# value_trans_nomal=[]
# for m in value_trans:
#     value_trans_nomal.append(m/abs(m))
# SIvaluelong=sum(value_trans_nomal)/len(value_trans_nomal)
# print(SIvaluelong)
# seizure_phase=[]
# for item in seizure_timing_index:
#     seizure_phase.append(value_trans_nomal[item])
# SIvaluelongseizure=sum(seizure_phase)/len(seizure_phase)
# print(SIvaluelongseizure)
# value_phase=np.angle(value_trans)
# phase_long_EEGauto_arr=value_phase
# seizure_phase_value_long=[]
# for item in seizure_timing_index:
#     seizure_phase_value_long.append(phase_long_EEGauto_arr[item])
# print(seizure_phase_value_long)
# n=0
# for item in seizure_phase_value_long:
#     if item <0:
#         n=n+1
# print(n/len(seizure_phase_value_long))
#
# # from matplotlib import gridspec
# # fig = pyplot.figure(figsize=(12, 10))
# # gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
# # ax2=pyplot.subplot(gs[0])
# # ax2.plot(window_time_arr[240*6:],long_rhythm_value_arr[240*6:],'orange',alpha=0.7)
# # ax2.set_title('EEG autocorrelation in VIC1173',fontsize=15)
# # ax2.set_xlabel('Time(hour)',fontsize=15)
# # locs, labels = pyplot.xticks(fontsize=15)
# # locs, labels = pyplot.yticks(fontsize=15)
# # ax2.scatter(window_time_arr[4541],long_rhythm_value_arr[4541],c='k')
# # ax2.scatter(window_time_arr[11615],long_rhythm_value_arr[11615],c='k')
# # ax2.scatter(window_time_arr[22100],long_rhythm_value_arr[22100],c='k')
# # # ax2.scatter(window_time_arr[29041],long_rhythm_value_arr[29041],c='k')
# # # ax2.scatter(window_time_arr[30203],long_rhythm_value_arr[30203],c='k')
# # # ax2.scatter(window_time_arr[32982],long_rhythm_value_arr[32982],c='k')
# # # ax2.scatter(window_time_arr[34902],long_rhythm_value_arr[34902],c='k')
# # ax3=pyplot.subplot(gs[1])
# # ax3.set_xlabel('Time(hour)',fontsize=15)
# # ax3.set_title('Instantaneous Phase',fontsize=15)
# # # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # locs, labels = pyplot.xticks(fontsize=15)
# # locs, labels = pyplot.yticks(fontsize=15)
# # ax3.plot(window_time_arr[240*6:],phase_long_EEGauto_arr[240*6:],'k',alpha=0.5,label='instantaneous phase')
# # pyplot.hlines(0,window_time_arr[240*6],window_time_arr[-1],'k','dashed')
# # ax3.scatter(window_time_arr[4541],phase_long_EEGauto_arr[4541],c='k')
# # ax3.scatter(window_time_arr[11615],phase_long_EEGauto_arr[11615],c='k')
# # ax3.scatter(window_time_arr[22100],phase_long_EEGauto_arr[22100],c='k')
# # # ax3.scatter(window_time_arr[29041],phase_long_EEGauto_arr[29041],c='k')
# # # ax3.scatter(window_time_arr[30203],phase_long_EEGauto_arr[30203],c='k')
# # # ax3.scatter(window_time_arr[32982],phase_long_EEGauto_arr[32982],c='k')
# # # ax3.scatter(window_time_arr[34902],phase_long_EEGauto_arr[34902],c='k')
# # ax3.set_xlabel('Time(hour)',fontsize=15)
# # locs, labels = pyplot.yticks([-0.5*np.pi,-0.25*np.pi,0,0.25*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=15)
# # # locs, labels = pyplot.yticks([-0.25*np.pi,-0.125*np.pi,0,0.125*np.pi,0.25*np.pi],['-0.25$\pi$','Rising','0','Falling','0.25$\pi$'],rotation='vertical',fontsize=15)
# # pyplot.tight_layout()
# # pyplot.show()
# # bins_number = 18
# # bins = np.linspace(-np.pi, np.pi, bins_number + 1)
# # nEEGsauto, _, _ = pyplot.hist(phase_long_EEGauto_arr, bins)
# # nEEGsautosei, _, _ = pyplot.hist(seizure_phase_value_long, bins)
# # print(nEEGsauto)
# # print(nEEGsautosei)
#
#
#
#
#
#
# # # ### ECG
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC1173/RRI_ch31_timewindowarr_VIC1173_15s_3h.csv',sep=',',header=None)
# rri_t= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC1173/RRI_ch31_rawvariance_VIC1173_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.September/VIC1173/RRI_ch31_rawauto_VIC1173_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
# rri_t_arr=[]
# for item in rri_t:
#     rri_t_arr.append(0+float(item))
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
# print(rri_t_arr[0]); print(len(rri_t_arr));print(len(Raw_variance_RRI31_arr));print(len(Raw_auto_RRI31_arr))
#
# # pyplot.plot(rri_t_arr,Raw_variance_RRI31_arr[0:len(rri_t_arr)],'grey',alpha=0.5)
# # pyplot.xlabel('Time(hour)',fontsize=13)
# # pyplot.ylabel('$Second^2$',fontsize=13)
# # pyplot.title('RRI variance in VIC1173',fontsize=13)
# # pyplot.show()
# # pyplot.plot(rri_t_arr,Raw_auto_RRI31_arr[0:len(rri_t_arr)],'grey',alpha=0.5)
# # pyplot.xlabel('Time(hour)',fontsize=13)
# # pyplot.title('RRI autocorrelation in VIC1173',fontsize=13)
# # pyplot.show()
# #
# # seizure_timing_index=[]
# # for k in range(len(rri_t_arr)):
# #     if rri_t_arr[k]<18.9266667 and rri_t_arr[k+1]>=18.9266667:
# #         seizure_timing_index.append(k)
# #     if rri_t_arr[k]<48.4025 and rri_t_arr[k+1]>=48.4025:
# #         seizure_timing_index.append(k)
# #     if rri_t_arr[k]<92.0886111 and rri_t_arr[k+1]>=92.0886111:
# #         seizure_timing_index.append(k)
# #     if rri_t_arr[k]<121.011667 and rri_t_arr[k+1]>=121.011667:
# #         seizure_timing_index.append(k)
# #     if rri_t_arr[k]<125.851667 and rri_t_arr[k+1]>=125.851667:
# #         seizure_timing_index.append(k)
# #     if rri_t_arr[k]<137.429722 and rri_t_arr[k+1]>=137.429722:
# #         seizure_timing_index.append(k)
# #     if rri_t_arr[k]<145.430278 and rri_t_arr[k+1]>=145.430278:
# #         seizure_timing_index.append(k)
# # print(seizure_timing_index)
#
#
# # window_time_arr=t
# # Raw_variance_RRI31=Raw_variance_RRI31_arr[0:len(t)]
# window_time_arr=rri_t_arr[0:22318]
# Raw_variance_RRI31=Raw_variance_RRI31_arr[0:22318]
# long_rhythm_var_arr=movingaverage(Raw_variance_RRI31,240*6)
# medium_rhythm_var_arr=movingaverage(Raw_variance_RRI31,240)
# medium_rhythm_var_arr_2=movingaverage(Raw_variance_RRI31,240*3)
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31,240*6)
# medium_rhythm_var_arr_4=movingaverage(Raw_variance_RRI31,240*12)
# short_rhythm_var_arr_plot=movingaverage(Raw_variance_RRI31,240*24)
#
# # fig=pyplot.figure(figsize=(8,6))
# # # pyplot.plot(window_time_arr,Raw_variance_EEG,'grey',alpha=0.5,label='Raw variance')
# # pyplot.plot(window_time_arr,short_rhythm_var_arr_plot,'grey',alpha=0.5,label='5min')
# # pyplot.plot(window_time_arr,medium_rhythm_var_arr,'g',label='1 hour')
# # pyplot.plot(window_time_arr,medium_rhythm_var_arr_2,'k',label='3 hour')
# # pyplot.plot(window_time_arr,medium_rhythm_var_arr_3,'orange',label='6 hour')
# # pyplot.plot(window_time_arr,medium_rhythm_var_arr_4,'b',label='12 hour')
# # pyplot.plot(window_time_arr,long_rhythm_var_arr,'r',alpha=0.7,label='1 day')
# # pyplot.title('RRI variance in VIC1173',fontsize=15)
# # pyplot.xlabel('Time(hour)',fontsize=15)
# # pyplot.ylabel('Second ($\mathregular{s^2}$)',fontsize=15)
# # pyplot.legend(loc='upper left',fontsize=10)
# # locs, labels = pyplot.xticks(fontsize=15)
# # locs, labels = pyplot.yticks(fontsize=15)
# # pyplot.tight_layout()
# # pyplot.annotate('',xy=(18.9266667,np.max(short_rhythm_var_arr_plot)),xytext=(18.9266667,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(48.4025,np.max(short_rhythm_var_arr_plot)),xytext=(48.4025,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(92.0886111,np.max(short_rhythm_var_arr_plot)),xytext=(92.0886111,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(121.011667,np.max(short_rhythm_var_arr_plot)),xytext=(121.011667,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(125.851667,np.max(short_rhythm_var_arr_plot)),xytext=(125.851667,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(137.429722,np.max(short_rhythm_var_arr_plot)),xytext=(137.429722,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(145.430278,np.max(short_rhythm_var_arr_plot)),xytext=(145.430278,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.show()
#
#
# long_rhythm_var_arr=long_rhythm_var_arr
# var_trans=hilbert(long_rhythm_var_arr)
# var_trans_nomal=[]
# for m in var_trans:
#     var_trans_nomal.append(m/abs(m))
# SIvarlong=sum(var_trans_nomal)/len(var_trans_nomal)
# print(SIvarlong)
# seizure_phase=[]
# for item in seizure_timing_index:
#      seizure_phase.append(var_trans_nomal[item])
# SIvarlongseizure=sum(seizure_phase)/len(seizure_phase)
# print(SIvarlongseizure)
# var_phase=np.angle(var_trans)
# phase_whole_long=var_phase
# seizure_phase_var_long=[]
# for item in seizure_timing_index:
#     seizure_phase_var_long.append(phase_whole_long[item])
# print(seizure_phase_var_long)
# n=0
# for item in seizure_phase_var_long:
#     if item <0:
#         n=n+1
# print(n/len(seizure_phase_var_long))
#
#
# # from matplotlib import gridspec
# # fig = pyplot.figure(figsize=(12, 10))
# # gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
# # ax1=pyplot.subplot(gs[0])
# # ax1.plot(window_time_arr[240*6:],long_rhythm_var_arr[240*6:],'orange',alpha=0.7)
# # # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax1.set_title('RRI variance in VIC1173',fontsize=15)
# # ax1.set_xlabel('Time(hour)',fontsize=15)
# # ax1.set_ylabel('Second($\mathregular{s^2}$)',fontsize=15)
# # locs, labels = pyplot.xticks(fontsize=15)
# # locs, labels = pyplot.yticks(fontsize=15)
# # ax1.scatter(window_time_arr[4541],long_rhythm_var_arr[4541],c='k')
# # ax1.scatter(window_time_arr[11615],long_rhythm_var_arr[11615],c='k')
# # ax1.scatter(window_time_arr[22100],long_rhythm_var_arr[22100],c='k')
# # # ax1.scatter(window_time_arr[29041],long_rhythm_var_arr[29041],c='k')
# # # ax1.scatter(window_time_arr[30203],long_rhythm_var_arr[30203],c='k')
# # # ax1.scatter(window_time_arr[32982],long_rhythm_var_arr[32982],c='k')
# # # ax1.scatter(window_time_arr[34902],long_rhythm_var_arr[34902],c='k')
# # # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax2=pyplot.subplot(gs[1])
# # ax2.set_xlabel('Time(hour)',fontsize=15)
# # ax2.set_title('Instantaneous Phase',fontsize=15)
# # # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax2.plot(window_time_arr[240*6:],phase_whole_long[240*6:],'k',alpha=0.5)
# # ax2.scatter(window_time_arr[4541],phase_whole_long[4541],c='k')
# # ax2.scatter(window_time_arr[11615],phase_whole_long[11615],c='k')
# # ax2.scatter(window_time_arr[22100],phase_whole_long[22100],c='k')
# # # ax2.scatter(window_time_arr[29041],phase_whole_long[29041],c='k')
# # # ax2.scatter(window_time_arr[30203],phase_whole_long[30203],c='k')
# # # ax2.scatter(window_time_arr[32982],phase_whole_long[32982],c='k')
# # # ax2.scatter(window_time_arr[34902],phase_whole_long[34902],c='k')
# # ax2.set_xlabel('Time(hour)',fontsize=15)
# # locs, labels = pyplot.xticks(fontsize=15)
# # locs, labels = pyplot.yticks(fontsize=15)
# # pyplot.hlines(0,window_time_arr[240*6],window_time_arr[-1],'k','dashed')
# # # locs, labels = pyplot.yticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi],['-$\pi$','Rising','0','Falling','$\pi$'],rotation='vertical',fontsize=15)
# # locs, labels = pyplot.yticks([-0.5*np.pi,-0.25*np.pi,0,0.25*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=15)
# # pyplot.tight_layout()
# # pyplot.show()
# # bins_number = 18
# # bins = np.linspace(-np.pi, np.pi, bins_number + 1)
# # nRRIsvar, _, _ = pyplot.hist(phase_whole_long, bins)
# # nRRIsvarsei, _, _ = pyplot.hist(seizure_phase_var_long, bins)
# # print(nRRIsvar)
# # print(nRRIsvarsei)
#
#
#
# # Raw_auto_RRI31=Raw_auto_RRI31_arr[0:len(t)]
# # window_time_arr=t
# Raw_auto_RRI31=Raw_auto_RRI31_arr[0:22318]
# window_time_arr=rri_t_arr[0:22318]
# long_rhythm_value_arr=movingaverage(Raw_auto_RRI31,240*6)
# medium_rhythm_value_arr=movingaverage(Raw_auto_RRI31,240)
# medium_rhythm_value_arr_2=movingaverage(Raw_auto_RRI31,240*3)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31,240*6)
# medium_rhythm_value_arr_4=movingaverage(Raw_auto_RRI31,240*12)
# short_rhythm_value_arr_plot=movingaverage(Raw_auto_RRI31,240*24)
#
# # # fig=pyplot.figure(figsize=(8,6))
# # # pyplot.plot(window_time_arr,short_rhythm_value_arr_plot,'grey',alpha=0.5,label='5min')
# # # pyplot.plot(window_time_arr,medium_rhythm_value_arr,'g',label='1 hour')
# # # pyplot.plot(window_time_arr,medium_rhythm_value_arr_2,'k',label='3 hour')
# # # pyplot.plot(window_time_arr,medium_rhythm_value_arr_3,'orange',label='6 hour')
# # # pyplot.plot(window_time_arr,medium_rhythm_value_arr_4,'b',label='12 hour')
# # # pyplot.plot(window_time_arr,long_rhythm_value_arr,'r',alpha=0.7,label='1 day')
# # # pyplot.title('RRI autocorrelation in VIC1173',fontsize=15)
# # # pyplot.xlabel('Time(hour)',fontsize=15)
# # # pyplot.ylabel('Autocorrelation',fontsize=15)
# # # pyplot.legend(loc='upper left',fontsize=10)
# # # locs, labels = pyplot.xticks(fontsize=15)
# # # locs, labels = pyplot.yticks(fontsize=15)
# # # pyplot.tight_layout()
# # # pyplot.annotate('',xy=(18.9266667,np.max(short_rhythm_value_arr_plot)),xytext=(18.9266667,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(48.4025,np.max(short_rhythm_value_arr_plot)),xytext=(48.4025,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(92.0886111,np.max(short_rhythm_value_arr_plot)),xytext=(92.0886111,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # # pyplot.annotate('',xy=(121.011667,np.max(short_rhythm_value_arr_plot)),xytext=(121.011667,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # # pyplot.annotate('',xy=(125.851667,np.max(short_rhythm_value_arr_plot)),xytext=(125.851667,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # # pyplot.annotate('',xy=(137.429722,np.max(short_rhythm_value_arr_plot)),xytext=(137.429722,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # # pyplot.annotate('',xy=(145.430278,np.max(short_rhythm_value_arr_plot)),xytext=(145.430278,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.show()
#
#
# long_rhythm_value_arr=long_rhythm_value_arr
# value_trans=hilbert(long_rhythm_value_arr)
# value_trans_nomal=[]
# for m in value_trans:
#     value_trans_nomal.append(m/abs(m))
# SIvaluelong=sum(value_trans_nomal)/len(value_trans_nomal)
# print(SIvaluelong)
# seizure_phase=[]
# for item in seizure_timing_index:
#     seizure_phase.append(value_trans_nomal[item])
# SIvaluelongseizure=sum(seizure_phase)/len(seizure_phase)
# print(SIvaluelongseizure)
# value_phase=np.angle(value_trans)
# phase_whole_value_long=value_phase
# seizure_phase_value_long=[]
# for item in seizure_timing_index:
#     seizure_phase_value_long.append(phase_whole_value_long[item])
# print(seizure_phase_value_long)
# n=0
# for item in seizure_phase_value_long:
#     if item <0:
#         n=n+1
# print(n/len(seizure_phase_value_long))
#
# # from matplotlib import gridspec
# # fig = pyplot.figure(figsize=(12, 10))
# # gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
# # ax1=pyplot.subplot(gs[0])
# # ax1.set_title('RRI autocorrelation in VIC1173',fontsize=15)
# # ax1.set_xlabel('Time(hour)',fontsize=15)
# # locs, labels = pyplot.xticks(fontsize=15)
# # locs, labels = pyplot.yticks(fontsize=15)
# # ax1.plot(window_time_arr[240*6:],long_rhythm_value_arr[240*6:],'orange',alpha=0.7)
# # # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax1.scatter(window_time_arr[4541],long_rhythm_value_arr[4541],c='k',s=40)
# # ax1.scatter(window_time_arr[11615],long_rhythm_value_arr[11615],c='k',s=40)
# # ax1.scatter(window_time_arr[22100],long_rhythm_value_arr[22100],c='k',s=40)
# # # ax1.scatter(window_time_arr[29041],long_rhythm_value_arr[29041],c='k',s=40)
# # # ax1.scatter(window_time_arr[30203],long_rhythm_value_arr[30203],c='k',s=40)
# # # ax1.scatter(window_time_arr[32982],long_rhythm_value_arr[32982],c='k',s=40)
# # # ax1.scatter(window_time_arr[34902],long_rhythm_value_arr[34902],c='k',s=40)
# # ax2=pyplot.subplot(gs[1])
# # ax2.set_title('Instantaneous Phase',fontsize=15)
# # ax2.set_xlabel('Time(hour)',fontsize=15)
# # locs, labels = pyplot.xticks(fontsize=15)
# # locs, labels = pyplot.yticks(fontsize=15)
# # pyplot.hlines(0,window_time_arr[240*6],window_time_arr[-1],'k','dashed')
# # # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax2.plot(window_time_arr[240*6:],phase_whole_value_long[240*6:],'k',alpha=0.5)
# # ax2.scatter(window_time_arr[4541],phase_whole_value_long[4541],c='k',s=40)
# # ax2.scatter(window_time_arr[11615],phase_whole_value_long[11615],c='k',s=40)
# # ax2.scatter(window_time_arr[22100],phase_whole_value_long[22100],c='k',s=40)
# # # ax2.scatter(window_time_arr[29041],phase_whole_value_long[29041],c='k',s=40)
# # # ax2.scatter(window_time_arr[30203],phase_whole_value_long[30203],c='k',s=40)
# # # ax2.scatter(window_time_arr[32982],phase_whole_value_long[32982],c='k',s=40)
# # # ax2.scatter(window_time_arr[34902],phase_whole_value_long[34902],c='k',s=40)
# # ax2.set_xlabel('Time(hour)',fontsize=15)
# # # locs, labels = pyplot.yticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi],['-$\pi$','Rising','0','Falling','$\pi$'],rotation='vertical',fontsize=15)
# # locs, labels = pyplot.yticks([-0.5*np.pi,-0.25*np.pi,0,0.25*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=15)
# # pyplot.tight_layout()
# # pyplot.show()
# # bins_number = 18
# # bins = np.linspace(-np.pi, np.pi, bins_number + 1)
# # nRRIsauto, _, _ = pyplot.hist(phase_whole_value_long, bins)
# # nRRIsautosei, _, _ = pyplot.hist(seizure_phase_value_long, bins)
# # print(nRRIsauto)
# # print(nRRIsautosei)
#
#
#
# # #### add circadian
# # a=np.where(t<6.67388889+0)
# # print(a)
# # print(t[1600]);print(t[1601])
# # t[0:1601]=t[0:1601]-0+17.3261111
# # t[1601:]=t[1601:]-6.67388889-0
# # print(t[1600]);print(t[1601]);
# # print(type(t));print(t[0])
# # time_feature_arr=[]
# # for i in range(len(t)):
# #     if t[i]>24:
# #         time_feature_arr.append(t[i] - (t[i] // 24) * 24)
# #     else:
# #         time_feature_arr.append(t[i])
# #
# # seizure_time=[time_feature_arr[4541],time_feature_arr[11615],time_feature_arr[22100],
# #               # time_feature_arr[29041],time_feature_arr[30203],
# #               # time_feature_arr[32982],time_feature_arr[34902]
# #               ]
# # print(seizure_time)
# # # bins_number = 18
# # # bins = np.linspace(0, 24, bins_number + 1)
# # # nEEGsvarsei, _, _ = pyplot.hist(seizure_time, bins)
# # # bins = np.linspace(0, 2*np.pi, bins_number + 1)
# # # width = 2*np.pi / bins_number
# # # params = dict(projection='polar')
# # # fig, ax = pyplot.subplots(subplot_kw=params)
# # # ax.bar(bins[:bins_number], nEEGsvarsei/sum(nEEGsvarsei),width=width, color='k',alpha=0.5)
# # # pyplot.setp(ax.get_yticklabels(), color='k', alpha=0.7)
# # # ax.set_title('seizure timing histogram (VIC1173)',fontsize=15)
# # # ax.set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))
# # # ax.set_xticklabels(range(24))
# # # pyplot.show()
# # bins_number = 18
# # bins = np.linspace(0, 24, bins_number + 1)
# # print(bins)
# # ntimes, _, _ = pyplot.hist(time_feature_arr[0:22318], bins)
# # ntimesei, _, _ = pyplot.hist(seizure_time, bins)
# # print(ntimes)
# # print(ntimesei)
# #
# # bins = np.linspace(0, 2*np.pi, bins_number + 1)
# # print(bins)
# # width = 2*np.pi / bins_number
# # params = dict(projection='polar')
# # fig, ax = pyplot.subplots(subplot_kw=params)
# # ax.bar(bins[:bins_number], ntimesei/sum(ntimesei),width=width, color='grey',alpha=0.7,edgecolor='grey',linewidth=2)
# # pyplot.setp(ax.get_yticklabels(), color='k')
# # ax.set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))
# # ax.set_xticklabels(['0 am','','','Night','','','6 am','','','Morning','','','12 am','','','Afternoon','','','18 pm','','','Evening','','','24 pm'],fontsize=16)
# # locs, labels = pyplot.yticks([0.2,0.6,1],['0.2','0.6','1'],fontsize=16)
# # ax.annotate("", xy=(-2.58095776837751, 0.83177), xytext=(0, 0),arrowprops=dict(arrowstyle="->",color='g',linewidth=2))
# # pyplot.show()
# #
# # bins = np.linspace(0, 2*np.pi, bins_number + 1)
# # width = 2*np.pi / bins_number
# # params = dict(projection='polar')
# # fig, ax = pyplot.subplots(subplot_kw=params)
# # ax.bar(bins[:bins_number], ntimesei/sum(ntimes),width=width, color='grey',alpha=0.7,edgecolor='k',linewidth=2)
# # pyplot.setp(ax.get_yticklabels(), color='k')
# # # ax.set_title('seizure timing histogram (SA0124)',fontsize=23)
# # ax.set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))
# # ax.set_xticklabels(['0 am','','','Night','','','6 am','','','Morning','','','12 am','','','Afternoon','','','18 pm','','','Evening','','','24 pm'],fontsize=16)
# # # locs, labels = pyplot.xticks([0,3,6,9,12,15,18,21,24],['0','Night','6','Morning','12','Afternoon','18','Evening','24'],fontsize=16)
# # locs, labels = pyplot.yticks([0.00004,0.00010,0.00016],['0.00004','0.00010','0.00016'],fontsize=16)
# # pyplot.show()
#
#
#
#
# #### section 2 training training
# medium_rhythm_var_arr_3 = movingaverage(Raw_variance_EEG, 240 * 6)
# long_rhythm_var_arr = medium_rhythm_var_arr_3
# var_trans = hilbert(long_rhythm_var_arr)
# var_phase = np.angle(var_trans)
# phase_long_EEGvariance_arr = var_phase
# print(len(phase_long_EEGvariance_arr));
# medium_rhythm_value_arr_3 = movingaverage(Raw_auto_EEG, 240 * 6)
# long_rhythm_value_arr = medium_rhythm_value_arr_3
# value_trans = hilbert(long_rhythm_value_arr)
# value_phase = np.angle(value_trans)
# phase_long_EEGauto_arr = value_phase
# print(len(phase_long_EEGauto_arr));
# medium_rhythm_RRIvar_arr_3 = movingaverage(Raw_variance_RRI31, 240 * 6)
# long_rhythm_RRIvar_arr = medium_rhythm_RRIvar_arr_3
# var_trans = hilbert(long_rhythm_RRIvar_arr)
# var_phase = np.angle(var_trans)
# phase_long_RRIvariance_arr = var_phase
# print(len(phase_long_RRIvariance_arr));
# medium_rhythm_RRIvalue_arr_3 = movingaverage(Raw_auto_RRI31, 240 * 6)
# long_rhythm_RRIvalue_arr = medium_rhythm_RRIvalue_arr_3
# value_trans = hilbert(long_rhythm_RRIvalue_arr)
# value_phase = np.angle(value_trans)
# phase_long_RRIauto_arr = value_phase
# print(len(phase_long_RRIauto_arr));
#
# #### combined probability calculation
# bins_number = 18
# bins = np.linspace(-np.pi, np.pi, bins_number + 1)
# pro_eegvars_time = []
# pro_eegvars_time_false = []
# for i in range(len(phase_long_EEGvariance_arr)):
#     if phase_long_EEGvariance_arr[i] >= bins[0] and phase_long_EEGvariance_arr[i] < bins[1]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[1] and phase_long_EEGvariance_arr[i] < bins[2]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[2] and phase_long_EEGvariance_arr[i] < bins[3]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[3] and phase_long_EEGvariance_arr[i] < bins[4]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[4] and phase_long_EEGvariance_arr[i] < bins[5]:
#         pro_eegvars_time_false.append(0.001747703)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[5] and phase_long_EEGvariance_arr[i] < bins[6]:
#         pro_eegvars_time_false.append(0.110239749)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[6] and phase_long_EEGvariance_arr[i] <= bins[7]:
#         pro_eegvars_time_false.append(0.105041452)
#         pro_eegvars_time.append(0.333333333)
#     elif phase_long_EEGvariance_arr[i] > bins[7] and phase_long_EEGvariance_arr[i] < bins[8]:
#         pro_eegvars_time_false.append(0.120412279)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[8] and phase_long_EEGvariance_arr[i] <= bins[9]:
#         pro_eegvars_time_false.append(0.171274927)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] > bins[9] and phase_long_EEGvariance_arr[i] < bins[10]:
#         pro_eegvars_time_false.append(0.143804616)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[10] and phase_long_EEGvariance_arr[i] <= bins[11]:
#         pro_eegvars_time_false.append(0.110508626)
#         pro_eegvars_time.append(0.333333333)
#     elif phase_long_EEGvariance_arr[i] > bins[11] and phase_long_EEGvariance_arr[i] < bins[12]:
#         pro_eegvars_time_false.append(0.122115169)
#         pro_eegvars_time.append(0.333333333)
#     elif phase_long_EEGvariance_arr[i] > bins[12] and phase_long_EEGvariance_arr[i] < bins[13]:
#         pro_eegvars_time_false.append(0.101725297)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[13] and phase_long_EEGvariance_arr[i] < bins[14]:
#         pro_eegvars_time_false.append(0.013130181)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[14] and phase_long_EEGvariance_arr[i] < bins[15]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[15] and phase_long_EEGvariance_arr[i] < bins[16]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[16] and phase_long_EEGvariance_arr[i] < bins[17]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif phase_long_EEGvariance_arr[i] >= bins[17]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
# print(pro_eegvars_time[4541]);print(pro_eegvars_time[11615]);print(pro_eegvars_time[22100]);
# pro_eegautos_time = []
# pro_eegautos_time_false = []
# for i in range(len(phase_long_EEGauto_arr)):
#     if phase_long_EEGauto_arr[i] >= bins[0] and phase_long_EEGauto_arr[i] < bins[1]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[1] and phase_long_EEGauto_arr[i] < bins[2]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[2] and phase_long_EEGauto_arr[i] < bins[3]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[3] and phase_long_EEGauto_arr[i] < bins[4]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[4] and phase_long_EEGauto_arr[i] < bins[5]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[5] and phase_long_EEGauto_arr[i] < bins[6]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[6] and phase_long_EEGauto_arr[i] < bins[7]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[7] and phase_long_EEGauto_arr[i] < bins[8]:
#         pro_eegautos_time_false.append(0.075644186)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[8] and phase_long_EEGauto_arr[i] < bins[9]:
#         pro_eegautos_time_false.append(0.462334752)
#         pro_eegautos_time.append(0.333333333)
#     elif phase_long_EEGauto_arr[i] >= bins[9] and phase_long_EEGauto_arr[i] <= bins[10]:
#         pro_eegautos_time_false.append(0.42652924)
#         pro_eegautos_time.append(0.333333333)
#     elif phase_long_EEGauto_arr[i] > bins[10] and phase_long_EEGauto_arr[i] < bins[11]:
#         pro_eegautos_time_false.append(0.02312346)
#         pro_eegautos_time.append(0.333333333)
#     elif phase_long_EEGauto_arr[i] >= bins[11] and phase_long_EEGauto_arr[i] < bins[12]:
#         pro_eegautos_time_false.append(0.005601613)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[12] and phase_long_EEGauto_arr[i] < bins[13]:
#         pro_eegautos_time_false.append(0.003674658)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[13] and phase_long_EEGauto_arr[i] < bins[14]:
#         pro_eegautos_time_false.append(0.003092091)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[14] and phase_long_EEGauto_arr[i] < bins[15]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[15] and phase_long_EEGauto_arr[i] < bins[16]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[16] and phase_long_EEGauto_arr[i] < bins[17]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif phase_long_EEGauto_arr[i] >= bins[17]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
# print(pro_eegautos_time[4541]);print(pro_eegautos_time[11615]);print(pro_eegautos_time[22100]);
# pro_RRIvars_time = []
# pro_RRIvars_time_false = []
# for i in range(len(phase_long_RRIvariance_arr)):
#     if phase_long_RRIvariance_arr[i] >= bins[0] and phase_long_RRIvariance_arr[i] < bins[1]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[1] and phase_long_RRIvariance_arr[i] < bins[2]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[2] and phase_long_RRIvariance_arr[i] < bins[3]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[3] and phase_long_RRIvariance_arr[i] < bins[4]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[4] and phase_long_RRIvariance_arr[i] < bins[5]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[5] and phase_long_RRIvariance_arr[i] < bins[6]:
#         pro_RRIvars_time_false.append(0.001254761)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[6] and phase_long_RRIvariance_arr[i] < bins[7]:
#         pro_RRIvars_time_false.append(0.0552095)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[7] and phase_long_RRIvariance_arr[i] <= bins[8]:
#         pro_RRIvars_time_false.append(0.147927403)
#         pro_RRIvars_time.append(0.333333333)
#     elif phase_long_RRIvariance_arr[i] > bins[8] and phase_long_RRIvariance_arr[i] < bins[9]:
#         pro_RRIvars_time_false.append(0.263141385)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[9] and phase_long_RRIvariance_arr[i] <= bins[10]:
#         pro_RRIvars_time_false.append(0.306430652)
#         pro_RRIvars_time.append(0.333333333)
#     elif phase_long_RRIvariance_arr[i] > bins[10] and phase_long_RRIvariance_arr[i] <= bins[11]:
#         pro_RRIvars_time_false.append(0.216401524)
#         pro_RRIvars_time.append(0.333333333)
#     elif phase_long_RRIvariance_arr[i] > bins[11] and phase_long_RRIvariance_arr[i] < bins[12]:
#         pro_RRIvars_time_false.append(0.009276272)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[12] and phase_long_RRIvariance_arr[i] < bins[13]:
#         pro_RRIvars_time_false.append(0.000134439)
#         pro_RRIvars_time.append(0.076923077)
#     elif phase_long_RRIvariance_arr[i] >= bins[13] and phase_long_RRIvariance_arr[i] < bins[14]:
#         pro_RRIvars_time_false.append(0.000224065)
#         pro_RRIvars_time.append(0.076923077)
#     elif phase_long_RRIvariance_arr[i] >= bins[14] and phase_long_RRIvariance_arr[i] < bins[15]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[15] and phase_long_RRIvariance_arr[i] < bins[16]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[16] and phase_long_RRIvariance_arr[i] < bins[17]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif phase_long_RRIvariance_arr[i] >= bins[17]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
# print(pro_RRIvars_time[4541]);print(pro_RRIvars_time[11615]);print(pro_RRIvars_time[22100]);
# pro_RRIautos_time = []
# pro_RRIautos_time_false = []
# for i in range(len(phase_long_RRIauto_arr)):
#     if phase_long_RRIauto_arr[i] >= bins[0] and phase_long_RRIauto_arr[i] < bins[1]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[1] and phase_long_RRIauto_arr[i] < bins[2]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[2] and phase_long_RRIauto_arr[i] < bins[3]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[3] and phase_long_RRIauto_arr[i] < bins[4]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[4] and phase_long_RRIauto_arr[i] < bins[5]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[5] and phase_long_RRIauto_arr[i] < bins[6]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[6] and phase_long_RRIauto_arr[i] < bins[7]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[7] and phase_long_RRIauto_arr[i] <= bins[8]:
#         pro_RRIautos_time_false.append(0.037822093)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] > bins[8] and phase_long_RRIauto_arr[i] <= bins[9]:
#         pro_RRIautos_time_false.append(0.54156397)
#         pro_RRIautos_time.append(0.666666667)
#     elif phase_long_RRIauto_arr[i] > bins[9] and phase_long_RRIauto_arr[i] <= bins[10]:
#         pro_RRIautos_time_false.append(0.387317948)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] > bins[10] and phase_long_RRIauto_arr[i] < bins[11]:
#         pro_RRIautos_time_false.append(0.015460453)
#         pro_RRIautos_time.append(0.333333333)
#     elif phase_long_RRIauto_arr[i] >= bins[11] and phase_long_RRIauto_arr[i] < bins[12]:
#         pro_RRIautos_time_false.append(0.005332736)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[12] and phase_long_RRIauto_arr[i] < bins[13]:
#         pro_RRIautos_time_false.append(0.003674658)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[13] and phase_long_RRIauto_arr[i] < bins[14]:
#         pro_RRIautos_time_false.append(0.008828143)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[14] and phase_long_RRIauto_arr[i] < bins[15]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[15] and phase_long_RRIauto_arr[i] < bins[16]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[16] and phase_long_RRIauto_arr[i] < bins[17]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif phase_long_RRIauto_arr[i] >= bins[17]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
# print(pro_RRIautos_time[4541]);print(pro_RRIautos_time[11615]);print(pro_RRIautos_time[22100]);
# print(len(pro_RRIautos_time));print(len(pro_RRIvars_time));print(len(pro_eegautos_time));print(len(pro_eegvars_time));
#
# Pseizureeegvar = 0.000134421;
# Pnonseizureeegvar = 0.999865579;
# t=np.linspace(0+0.00416667,0+0.00416667+0.00416667*(len(Raw_variance_EEG)-1),len(Raw_variance_EEG))
# window_time_arr=t
# # Pcombined = []
# # for m in range(len(pro_eegvars_time)):
# #     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIvars_time[m]*pro_RRIautos_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
# # Pcombined = []
# # for m in range(len(pro_eegvars_time)):
# #     P1=Pseizureeegvar*pro_eegvars_time[m]*pro_eegautos_time[m]*pro_RRIautos_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIautos_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
# # Pcombined = []
# # for m in range(len(pro_eegvars_time)):
# #     P1=Pseizureeegvar*pro_RRIvars_time[m]*pro_eegautos_time[m]*pro_RRIautos_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIautos_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
# # Pcombined = []
# # for m in range(len(pro_eegvars_time)):
# #     P1=Pseizureeegvar*pro_eegautos_time[m]*pro_RRIautos_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_eegautos_time_false[m]*pro_RRIautos_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
# # print(len(Pcombined))
#
# Pcombined = []
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))
# print(len(Pcombined))
#
# pyplot.figure(figsize=(12, 5))
# pyplot.plot(window_time_arr, Pcombined)
# pyplot.title('combined probability in VIC1173', fontsize=15)
# pyplot.annotate('', xy=(18.9266667, np.max(Pcombined)), xytext=(18.9266667, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(48.4025, np.max(Pcombined)), xytext=(48.4025, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(92.0886111, np.max(Pcombined)), xytext=(92.0886111, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.tight_layout()
# # pyplot.hlines(1.65974856104031e-06, window_time_arr[0],window_time_arr[-1],'r')
# # pyplot.hlines(4.979396779209983e-06, window_time_arr[0],window_time_arr[-1],'r')
# pyplot.xlabel('Time(h)', fontsize=15)
# pyplot.ylabel('seizure probability', fontsize=15)
# pyplot.show()
# for item in seizure_timing_index:
#     print(Pcombined[item])
#
# t = np.linspace(0 + 0.00416667, 0 + 0.00416667 + 0.00416667 * (len(Raw_variance_RRI31_arr) - 1), len(Raw_variance_RRI31_arr))
# window_time_arr = t
# a=np.where(t<6.67388889+0)
# print(a)
# print(t[1600]);print(t[1601])
# t[0:1601]=t[0:1601]-0+17.3261111
# t[1601:]=t[1601:]-6.67388889-0
# print(t[1600]);print(t[1601]);print(type(t));print(t[0])
# time_feature_arr=[]
# for i in range(len(t)):
#     if t[i]>24:
#         time_feature_arr.append(t[i] - (t[i] // 24) * 24)
#     else:
#         time_feature_arr.append(t[i])
# seizure_time=[time_feature_arr[4541],time_feature_arr[11615],
# time_feature_arr[22100],time_feature_arr[29041],time_feature_arr[30203],time_feature_arr[32982],time_feature_arr[34902]]
# bins_number = 18
# bins = np.linspace(0, 24, bins_number + 1)
# pro_circadian_time=[]
# pro_circadian_time_false=[]
# for i in range(len(time_feature_arr)):
#     if time_feature_arr[i] >= bins[0] and time_feature_arr[i] <= bins[1]:
#         pro_circadian_time_false.append(0.05736052)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] > bins[1] and time_feature_arr[i] < bins[2]:
#         pro_circadian_time_false.append(0.05736052)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[2] and time_feature_arr[i] < bins[3]:
#         pro_circadian_time_false.append(0.05736052)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[3] and time_feature_arr[i] < bins[4]:
#         pro_circadian_time_false.append(0.05736052)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[4] and time_feature_arr[i] < bins[5]:
#         pro_circadian_time_false.append(0.05736052)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[5] and time_feature_arr[i] <= bins[6]:
#         pro_circadian_time_false.append(0.05736052)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] > bins[6] and time_feature_arr[i] < bins[7]:
#         pro_circadian_time_false.append(0.05736052)
#         pro_circadian_time.append(0.076923077)
#     elif time_feature_arr[i] >= bins[7] and time_feature_arr[i] <= bins[8]:
#         pro_circadian_time_false.append(0.05736052)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] > bins[8] and time_feature_arr[i] < bins[9]:
#         pro_circadian_time_false.append(0.05736052)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[9] and time_feature_arr[i] < bins[10]:
#         pro_circadian_time_false.append(0.057315707)
#         pro_circadian_time.append(0.333333333)
#     elif time_feature_arr[i] >= bins[10] and time_feature_arr[i] < bins[11]:
#         pro_circadian_time_false.append(0.053596236)
#         pro_circadian_time.append(0.333333333)
#     elif time_feature_arr[i] >= bins[11] and time_feature_arr[i] < bins[12]:
#         pro_circadian_time_false.append(0.04302039)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[12] and time_feature_arr[i] < bins[13]:
#         pro_circadian_time_false.append(0.043065203)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[13] and time_feature_arr[i] < bins[14]:
#         pro_circadian_time_false.append(0.057315707)
#         pro_circadian_time.append(0.333333333)
#     elif time_feature_arr[i] >= bins[14] and time_feature_arr[i] < bins[15]:
#         pro_circadian_time_false.append(0.05736052)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[15] and time_feature_arr[i] < bins[16]:
#         pro_circadian_time_false.append(0.05736052)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[16] and time_feature_arr[i] < bins[17]:
#         pro_circadian_time_false.append(0.05736052)
#         pro_circadian_time.append(0)
#     elif time_feature_arr[i] >= bins[17] and time_feature_arr[i] <= bins[18]:
#         pro_circadian_time_false.append(0.05736052)
#         pro_circadian_time.append(0)
#
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIvars_time[m]*pro_RRIautos_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=pro_eegvars_time[m]*pro_eegautos_time[m]*Pseizureeegvar*pro_RRIautos_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=pro_RRIvars_time[m]*pro_eegautos_time[m]*Pseizureeegvar*pro_RRIautos_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
# # print(len(Pcombined))
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=pro_eegautos_time[m]*Pseizureeegvar*pro_RRIautos_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_eegautos_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_RRIautos_time[m]*pro_circadian_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
#     Pcombined.append(P1/(P1+P2))
#
# t=np.linspace(0+0.00416667,0+0.00416667+0.00416667*(len(Raw_variance_EEG)-1),len(Raw_variance_EEG))
# window_time_arr=t
# pyplot.figure(figsize=(12, 5))
# pyplot.plot(window_time_arr, Pcombined)
# pyplot.title('combined probability in VIC1173', fontsize=15)
# pyplot.annotate('', xy=(18.9266667, np.max(Pcombined)), xytext=(18.9266667, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(48.402, np.max(Pcombined)), xytext=(48.402, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.annotate('', xy=(92.088611, np.max(Pcombined)), xytext=(92.088611, np.max(Pcombined) + 0.00000000001),arrowprops=dict(facecolor='black', shrink=0.05))
# pyplot.tight_layout()
# pyplot.xlim(window_time_arr[0], window_time_arr[-1])
# # pyplot.hlines(5.532470719903305e-07, window_time_arr[0],window_time_arr[-1],'r')
# # pyplot.hlines(1.6597423756242214e-06, window_time_arr[0],window_time_arr[-1],'r')
# pyplot.xlabel('Time(h)', fontsize=15)
# pyplot.ylabel('seizure probability', fontsize=15)
# pyplot.show()
# for item in seizure_timing_index:
#     print(Pcombined[item])
#
#
#
#
#
#
#
# ### section 3 froecast
# # # # # # ###forecast EEG var
# t=np.linspace(0,0+0.00416667*(len(Raw_variance_EEG_arr)-1),len(Raw_variance_EEG_arr))
# t_window_arr=t
# fore_arr_EEGvars=[]
# for k in range(93,94):
#     variance_arr = Raw_variance_EEG_arr[0:(22318+240*k)]
#     long_rhythm_var_arr=movingaverage(variance_arr,240*6)
#     pyplot.figure(figsize=(6, 3))
#     pyplot.title('EEG variance')
#     pyplot.ylabel('Voltage ($\mathregular{v^2}$)')
#     pyplot.xlabel('Time(h)')
#     pyplot.plot(t_window_arr[240*6:(22318+240*k)], long_rhythm_var_arr[240*6:],'orange')
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.Oct/forecast93hsignal_3hcycle_EEGvar_VIC1173.csv',sep=',',header=None)
# forecast_var_EEG= csv_reader.values
# forecast_var_EEG_arr=[]
# for item in forecast_var_EEG:
#     forecast_var_EEG_arr=forecast_var_EEG_arr+list(item)
# t=np.linspace(t_window_arr[22318],t_window_arr[22318]+0.1666667*(len(forecast_var_EEG_arr)-1),len(forecast_var_EEG_arr))
# pyplot.plot(t, forecast_var_EEG_arr,'k',label='forecast EEG var')
# pyplot.legend()
# pyplot.show()
#
# fore_arr_EEGauto=[]
# for k in range(93,94):
#     auto_arr = Raw_auto_EEG_arr[0:(22318+240*k)]
#     long_rhythm_auto_arr=movingaverage(auto_arr,240*6)
#     pyplot.figure(figsize=(6, 3))
#     pyplot.title('EEG autocorrelation')
#     pyplot.xlabel('time(h)')
#     pyplot.plot(t_window_arr[240*6:(22318+240*k)], long_rhythm_auto_arr[240*6:],'orange')
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.Oct/forecast93hsignal_3hcycle_EEGauto_VIC1173.csv',sep=',',header=None)
# forecast_auto_EEG= csv_reader.values
# forecast_auto_EEG_arr=[]
# for item in forecast_auto_EEG:
#     forecast_auto_EEG_arr=forecast_auto_EEG_arr+list(item)
# t=np.linspace(t_window_arr[22318],t_window_arr[22318]+0.1666667*(len(forecast_auto_EEG_arr)-1),len(forecast_auto_EEG_arr))
# pyplot.plot(t, forecast_auto_EEG_arr,'k',label='forecast EEG auto')
# pyplot.legend()
# pyplot.show()
#
# fore_arr_RRIvars=[]
# for k in range(93, 94):
#     variance_arr = Raw_variance_RRI31_arr[0:(22318+240*k)]
#     long_rhythm_var_arr=movingaverage(variance_arr,240*6)
#     pyplot.figure(figsize=(6, 3))
#     pyplot.title('RRI variance')
#     pyplot.ylabel('Second ($\mathregular{s^2}$)')
#     pyplot.xlabel('Time(h)')
#     pyplot.plot(t_window_arr[240*6:(22318+240*k)], long_rhythm_var_arr[240*6:-2],'orange')
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.Oct/forecast93hsignal_3hcycle_RRIvar_VIC1173.csv', sep=',',header=None)
# forecast_var_RRI31= csv_reader.values
# forecast_var_RRI31_arr=[]
# for item in forecast_var_RRI31:
#     forecast_var_RRI31_arr=forecast_var_RRI31_arr+list(item)
# t=np.linspace(t_window_arr[22318],t_window_arr[22318]+0.1666667*(len(forecast_var_RRI31_arr)-1),len(forecast_var_RRI31_arr))
# pyplot.plot(t, forecast_var_RRI31_arr,'k',label='forecast RRI var')
# pyplot.legend()
# pyplot.show()
#
# fore_arr_RRIautos=[]
# save_data_RRIautos=[]
# for k in range(93,94):
#     auto_arr = Raw_auto_RRI31_arr[0:22318+240*k]
#     long_rhythm_auto_arr=movingaverage(auto_arr,240*6)
#     pyplot.figure(figsize=(6,3))
#     pyplot.title('RRI autocorrelation')
#     pyplot.xlabel('Time(h)')
#     pyplot.plot(t_window_arr[240*6:22318+240*k], long_rhythm_auto_arr[240*6:-2],'orange')
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.Oct/forecast93hsignal_3hcycle_RRIauto_VIC1173.csv',sep=',',header=None)
# forecast_auto_RRI31= csv_reader.values
# forecast_auto_RRI31_arr=[]
# for item in forecast_auto_RRI31:
#     forecast_auto_RRI31_arr=forecast_auto_RRI31_arr+list(item)
# t=np.linspace(t_window_arr[22318],t_window_arr[22318]+0.1666667*(len(forecast_auto_RRI31_arr)-1),len(forecast_auto_RRI31_arr))
# pyplot.plot(t, forecast_auto_RRI31_arr,'k',label='forecast RRI auto')
# pyplot.legend()
# pyplot.show()
# print(len(forecast_var_EEG_arr));print(len(forecast_auto_EEG_arr));print(len(forecast_var_RRI31_arr));print(len(forecast_auto_RRI31_arr));
#
#
# # ### predict, forecast data
# var_trans=hilbert(forecast_var_EEG_arr)
# var_phase=np.angle(var_trans)
# rolmean_short_EEGvar=var_phase
#
# var_trans=hilbert(forecast_auto_EEG_arr)
# var_phase=np.angle(var_trans)
# rolmean_short_EEGauto=var_phase
#
# var_trans=hilbert(forecast_var_RRI31_arr)
# var_phase=np.angle(var_trans)
# rolmean_short_RRIvar=var_phase
#
# var_trans=hilbert(forecast_auto_RRI31_arr)
# var_phase=np.angle(var_trans)
# rolmean_short_RRIauto=var_phase
#
# bins_number = 18
# bins = np.linspace(-np.pi, np.pi, bins_number + 1)
# pro_eegvars_time = []
# pro_eegvars_time_false = []
# for i in range(len(rolmean_short_EEGvar)):
#     if rolmean_short_EEGvar[i] >= bins[0] and rolmean_short_EEGvar[i] < bins[1]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[1] and rolmean_short_EEGvar[i] < bins[2]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[2] and rolmean_short_EEGvar[i] < bins[3]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[3] and rolmean_short_EEGvar[i] < bins[4]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[4] and rolmean_short_EEGvar[i] < bins[5]:
#         pro_eegvars_time_false.append(0.001747703)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[5] and rolmean_short_EEGvar[i] < bins[6]:
#         pro_eegvars_time_false.append(0.110239749)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[6] and rolmean_short_EEGvar[i] <= bins[7]:
#         pro_eegvars_time_false.append(0.105041452)
#         pro_eegvars_time.append(0.333333333)
#     elif rolmean_short_EEGvar[i] > bins[7] and rolmean_short_EEGvar[i] < bins[8]:
#         pro_eegvars_time_false.append(0.120412279)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[8] and rolmean_short_EEGvar[i] <= bins[9]:
#         pro_eegvars_time_false.append(0.171274927)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] > bins[9] and rolmean_short_EEGvar[i] < bins[10]:
#         pro_eegvars_time_false.append(0.143804616)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[10] and rolmean_short_EEGvar[i] <= bins[11]:
#         pro_eegvars_time_false.append(0.110508626)
#         pro_eegvars_time.append(0.333333333)
#     elif rolmean_short_EEGvar[i] > bins[11] and rolmean_short_EEGvar[i] < bins[12]:
#         pro_eegvars_time_false.append(0.122115169)
#         pro_eegvars_time.append(0.333333333)
#     elif rolmean_short_EEGvar[i] > bins[12] and rolmean_short_EEGvar[i] < bins[13]:
#         pro_eegvars_time_false.append(0.101725297)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[13] and rolmean_short_EEGvar[i] < bins[14]:
#         pro_eegvars_time_false.append(0.013130181)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[14] and rolmean_short_EEGvar[i] < bins[15]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[15] and rolmean_short_EEGvar[i] < bins[16]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[16] and rolmean_short_EEGvar[i] < bins[17]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
#     elif rolmean_short_EEGvar[i] >= bins[17]:
#         pro_eegvars_time_false.append(0)
#         pro_eegvars_time.append(0)
# pro_eegautos_time = []
# pro_eegautos_time_false = []
# for i in range(len(rolmean_short_EEGauto)):
#     if rolmean_short_EEGauto[i] >= bins[0] and rolmean_short_EEGauto[i] < bins[1]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[1] and rolmean_short_EEGauto[i] < bins[2]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[2] and rolmean_short_EEGauto[i] < bins[3]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[3] and rolmean_short_EEGauto[i] < bins[4]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[4] and rolmean_short_EEGauto[i] < bins[5]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[5] and rolmean_short_EEGauto[i] < bins[6]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[6] and rolmean_short_EEGauto[i] < bins[7]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[7] and rolmean_short_EEGauto[i] < bins[8]:
#         pro_eegautos_time_false.append(0.075644186)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[8] and rolmean_short_EEGauto[i] < bins[9]:
#         pro_eegautos_time_false.append(0.462334752)
#         pro_eegautos_time.append(0.333333333)
#     elif rolmean_short_EEGauto[i] >= bins[9] and rolmean_short_EEGauto[i] <= bins[10]:
#         pro_eegautos_time_false.append(0.42652924)
#         pro_eegautos_time.append(0.333333333)
#     elif rolmean_short_EEGauto[i] > bins[10] and rolmean_short_EEGauto[i] < bins[11]:
#         pro_eegautos_time_false.append(0.02312346)
#         pro_eegautos_time.append(0.333333333)
#     elif rolmean_short_EEGauto[i] >= bins[11] and rolmean_short_EEGauto[i] < bins[12]:
#         pro_eegautos_time_false.append(0.005601613)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[12] and rolmean_short_EEGauto[i] < bins[13]:
#         pro_eegautos_time_false.append(0.003674658)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[13] and rolmean_short_EEGauto[i] < bins[14]:
#         pro_eegautos_time_false.append(0.003092091)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[14] and rolmean_short_EEGauto[i] < bins[15]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[15] and rolmean_short_EEGauto[i] < bins[16]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[16] and rolmean_short_EEGauto[i] < bins[17]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
#     elif rolmean_short_EEGauto[i] >= bins[17]:
#         pro_eegautos_time_false.append(0)
#         pro_eegautos_time.append(0)
# pro_RRIvars_time = []
# pro_RRIvars_time_false = []
# for i in range(len(rolmean_short_RRIvar)):
#     if rolmean_short_RRIvar[i] >= bins[0] and rolmean_short_RRIvar[i] < bins[1]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[1] and rolmean_short_RRIvar[i] < bins[2]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[2] and rolmean_short_RRIvar[i] < bins[3]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[3] and rolmean_short_RRIvar[i] < bins[4]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[4] and rolmean_short_RRIvar[i] < bins[5]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[5] and rolmean_short_RRIvar[i] < bins[6]:
#         pro_RRIvars_time_false.append(0.001254761)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[6] and rolmean_short_RRIvar[i] < bins[7]:
#         pro_RRIvars_time_false.append(0.0552095)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[7] and rolmean_short_RRIvar[i] <= bins[8]:
#         pro_RRIvars_time_false.append(0.147927403)
#         pro_RRIvars_time.append(0.333333333)
#     elif rolmean_short_RRIvar[i] > bins[8] and rolmean_short_RRIvar[i] < bins[9]:
#         pro_RRIvars_time_false.append(0.263141385)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[9] and rolmean_short_RRIvar[i] <= bins[10]:
#         pro_RRIvars_time_false.append(0.306430652)
#         pro_RRIvars_time.append(0.333333333)
#     elif rolmean_short_RRIvar[i] > bins[10] and rolmean_short_RRIvar[i] <= bins[11]:
#         pro_RRIvars_time_false.append(0.216401524)
#         pro_RRIvars_time.append(0.333333333)
#     elif rolmean_short_RRIvar[i] > bins[11] and rolmean_short_RRIvar[i] < bins[12]:
#         pro_RRIvars_time_false.append(0.009276272)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[12] and rolmean_short_RRIvar[i] < bins[13]:
#         pro_RRIvars_time_false.append(0.000134439)
#         pro_RRIvars_time.append(0.076923077)
#     elif rolmean_short_RRIvar[i] >= bins[13] and rolmean_short_RRIvar[i] < bins[14]:
#         pro_RRIvars_time_false.append(0.000224065)
#         pro_RRIvars_time.append(0.076923077)
#     elif rolmean_short_RRIvar[i] >= bins[14] and rolmean_short_RRIvar[i] < bins[15]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[15] and rolmean_short_RRIvar[i] < bins[16]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[16] and rolmean_short_RRIvar[i] < bins[17]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
#     elif rolmean_short_RRIvar[i] >= bins[17]:
#         pro_RRIvars_time_false.append(0)
#         pro_RRIvars_time.append(0)
# pro_RRIautos_time = []
# pro_RRIautos_time_false = []
# for i in range(len(rolmean_short_RRIauto)):
#     if rolmean_short_RRIauto[i] >= bins[0] and rolmean_short_RRIauto[i] < bins[1]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[1] and rolmean_short_RRIauto[i] < bins[2]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[2] and rolmean_short_RRIauto[i] < bins[3]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[3] and rolmean_short_RRIauto[i] < bins[4]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[4] and rolmean_short_RRIauto[i] < bins[5]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[5] and rolmean_short_RRIauto[i] < bins[6]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[6] and rolmean_short_RRIauto[i] < bins[7]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[7] and rolmean_short_RRIauto[i] <= bins[8]:
#         pro_RRIautos_time_false.append(0.037822093)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] > bins[8] and rolmean_short_RRIauto[i] <= bins[9]:
#         pro_RRIautos_time_false.append(0.54156397)
#         pro_RRIautos_time.append(0.666666667)
#     elif rolmean_short_RRIauto[i] > bins[9] and rolmean_short_RRIauto[i] <= bins[10]:
#         pro_RRIautos_time_false.append(0.387317948)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] > bins[10] and rolmean_short_RRIauto[i] < bins[11]:
#         pro_RRIautos_time_false.append(0.015460453)
#         pro_RRIautos_time.append(0.333333333)
#     elif rolmean_short_RRIauto[i] >= bins[11] and rolmean_short_RRIauto[i] < bins[12]:
#         pro_RRIautos_time_false.append(0.005332736)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[12] and rolmean_short_RRIauto[i] < bins[13]:
#         pro_RRIautos_time_false.append(0.003674658)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[13] and rolmean_short_RRIauto[i] < bins[14]:
#         pro_RRIautos_time_false.append(0.008828143)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[14] and rolmean_short_RRIauto[i] < bins[15]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[15] and rolmean_short_RRIauto[i] < bins[16]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[16] and rolmean_short_RRIauto[i] < bins[17]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#     elif rolmean_short_RRIauto[i] >= bins[17]:
#         pro_RRIautos_time_false.append(0)
#         pro_RRIautos_time.append(0)
#
#
#
# Pseizureeegvar = 0.000134421;
# Pnonseizureeegvar = 0.999865579;
#
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=Pseizureeegvar*pro_RRIvars_time[m]*pro_eegautos_time[m]*pro_RRIautos_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIautos_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
# # print(len(Pcombined));
#
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=Pseizureeegvar*pro_eegautos_time[m]*pro_RRIautos_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_eegautos_time_false[m]*pro_RRIautos_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
# # print(len(Pcombined))
#
# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))
# print(len(Pcombined))
#
# pyplot.figure(figsize=(8,4))
# RRI_timewindow_arr=t
# pyplot.plot(RRI_timewindow_arr,Pcombined)
# pyplot.annotate('',xy=(121.011667,np.max(Pcombined)),xytext=(121.011667,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
# pyplot.annotate('',xy=(125.851667,np.max(Pcombined)),xytext=(125.851667,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='k',shrink=0.05))
# pyplot.annotate('',xy=(137.429722,np.max(Pcombined)),xytext=(137.429722,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='red',shrink=0.05))
# pyplot.annotate('',xy=(145.430278,np.max(Pcombined)),xytext=(145.430278,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='red',shrink=0.05))
# # pyplot.annotate('',xy=(95.018333,np.max(Pcombined)),xytext=(95.018333,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# # pyplot.annotate('',xy=(126.42278,np.max(Pcombined)),xytext=(126.42278,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# # pyplot.annotate('',xy=(126.68611,np.max(Pcombined)),xytext=(126.68611,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# # pyplot.annotate('',xy=(147.35583,np.max(Pcombined)),xytext=(147.35583,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# # pyplot.annotate('',xy=(147.889167,np.max(Pcombined)),xytext=(147.889167,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# pyplot.title('Forecast seizures in VIC1173')
# pyplot.xlabel('Time(h)')
# pyplot.ylabel('Seizure probability')
# # pyplot.hlines(1.65974856104031e-06, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'r')
# # pyplot.hlines(4.979396779209983e-06, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'r')
# pyplot.hlines(4.551466140045025e-05, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'r')
# # pyplot.hlines(0.6*4.551466140045025e-05, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'orange')
# # pyplot.hlines(0.3*4.551466140045025e-05, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'orange')
# # pyplot.hlines(1.2*4.551466140045025e-05, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'orange')
# # pyplot.hlines(2*4.551466140045025e-05, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'orange')
# pyplot.show()
# # Pcombined=split(Pcombined,6)
# # print(len(Pcombined))
# # index=[]
# # for i in range(len(Pcombined)):
# #     for item in Pcombined[i]:
# #         if item >= 4.551466140045025e-05:
# #             index.append(6*i+0)
# # print(RRI_timewindow_arr[index])
# # a=np.unique(RRI_timewindow_arr[index])
# # print(a); print(len(a))
# # index=[]
# # for i in range(len(Pcombined)):
# #     for item in Pcombined[i]:
# #         if item >= 0.3*4.551466140045025e-05:
# #             index.append(6*i+0)
# # print(RRI_timewindow_arr[index])
# # a=np.unique(RRI_timewindow_arr[index])
# # print(a); print(len(a))
# # index=[]
# # for i in range(len(Pcombined)):
# #     for item in Pcombined[i]:
# #         if item >= 0.6*4.551466140045025e-05:
# #             index.append(6*i+0)
# # print(RRI_timewindow_arr[index])
# # a=np.unique(RRI_timewindow_arr[index])
# # print(a); print(len(a))
# # index=[]
# # for i in range(len(Pcombined)):
# #     for item in Pcombined[i]:
# #         if item >= 1.2*4.551466140045025e-05:
# #             index.append(6*i+0)
# # print(RRI_timewindow_arr[index])
# # a=np.unique(RRI_timewindow_arr[index])
# # print(a); print(len(a))
# # index=[]
# # for i in range(len(Pcombined)):
# #     for item in Pcombined[i]:
# #         if item >= 2*4.551466140045025e-05:
# #             index.append(6*i+0)
# # print(RRI_timewindow_arr[index])
# # a=np.unique(RRI_timewindow_arr[index])
# # print(a); print(len(a))
#
# # index=[]
# # for i, j in enumerate(Pcombined):
# #     if j < 4.551466140045025e-05:
# #         index.append(i)
# # print(RRI_timewindow_arr[index])
# # print(len(index))
# # time_arr=[121.011667,125.851667,137.429722,145.430278]
# # k=0
# # n_arr=[]
# # for m in time_arr:
# #     for n in RRI_timewindow_arr[index]:
# #         if m-n<=1 and m-n>=0:
# #             k=k+1
# #             n_arr.append(n)
# # print(k)
# # index=[]
# # for i, j in enumerate(Pcombined):
# #     if j < 0.3*4.551466140045025e-05:
# #         index.append(i)
# # print(RRI_timewindow_arr[index])
# # print(len(index))
# # time_arr=[121.011667,125.851667,137.429722,145.430278]
# # k=0
# # n_arr=[]
# # for m in time_arr:
# #     for n in RRI_timewindow_arr[index]:
# #         if m-n<=1 and m-n>=0:
# #             k=k+1
# #             n_arr.append(n)
# # print(k)
# # index=[]
# # for i, j in enumerate(Pcombined):
# #     if j < 0.6*4.551466140045025e-05:
# #         index.append(i)
# # print(RRI_timewindow_arr[index])
# # print(len(index))
# # time_arr=[121.011667,125.851667,137.429722,145.430278]
# # k=0
# # n_arr=[]
# # for m in time_arr:
# #     for n in RRI_timewindow_arr[index]:
# #         if m-n<=1 and m-n>=0:
# #             k=k+1
# #             n_arr.append(n)
# # print(k)
# # index=[]
# # for i, j in enumerate(Pcombined):
# #     if j < 1.2*4.551466140045025e-05:
# #         index.append(i)
# # print(RRI_timewindow_arr[index])
# # print(len(index))
# # time_arr=[121.011667,125.851667,137.429722,145.430278]
# # k=0
# # n_arr=[]
# # for m in time_arr:
# #     for n in RRI_timewindow_arr[index]:
# #         if m-n<=1 and m-n>=0:
# #             k=k+1
# #             n_arr.append(n)
# # print(k)
# # index=[]
# # for i, j in enumerate(Pcombined):
# #     if j < 2*4.551466140045025e-05:
# #         index.append(i)
# # print(RRI_timewindow_arr[index])
# # print(len(index))
# # time_arr=[121.011667,125.851667,137.429722,145.430278]
# # k=0
# # n_arr=[]
# # for m in time_arr:
# #     for n in RRI_timewindow_arr[index]:
# #         if m-n<=1 and m-n>=0:
# #             k=k+1
# #             n_arr.append(n)
# # print(k)
#
# Th1=4.551466140045025e-05
# Pcombined = split(Pcombined, 6)
# print(len(Pcombined))
# time_arr_arr=[]
# AUC_cs_arr=[]
# for i in range(5000):
#     time_arr = np.random.uniform(low=t_window_arr[22318], high=t_window_arr[-1], size=4)
#     time_arr_arr.append(time_arr)
#     time_arr=np.sort(time_arr)
#
#     index = []
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= Th1:
#                 index.append(6 * i + 0)
#     # print(RRI_timewindow_arr[index])
#     a1 = np.unique(RRI_timewindow_arr[index])
#     # print(a1);
#     print(len(a1))
#     k1 = 0
#     n_arr = []
#     for m in time_arr:
#         for n in a1:
#             if m - n <= 1 and m - n >= 0:
#                 k1 = k1 + 1
#                 n_arr.append(n)
#     print(k1)
#
#     index = []
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= 0.3 * Th1:
#                 index.append(6 * i + 0)
#     # print(RRI_timewindow_arr[index])
#     a2 = np.unique(RRI_timewindow_arr[index])
#     # print(a2);
#     print(len(a2))
#     k2 = 0
#     n_arr = []
#     for m in time_arr:
#         for n in a2:
#             if m - n <= 1 and m - n >= 0:
#                 k2 = k2 + 1
#                 n_arr.append(n)
#     print(k2)
#
#     index = []
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= 0.6 * Th1:
#                 index.append(6 * i + 0)
#     # print(RRI_timewindow_arr[index])
#     a3 = np.unique(RRI_timewindow_arr[index])
#     # print(a3);
#     print(len(a3))
#     k3 = 0
#     n_arr = []
#     for m in time_arr:
#         for n in a3:
#             if m - n <= 1 and m - n >= 0:
#                 k3 = k3 + 1
#                 n_arr.append(n)
#     print(k3)
#
#     index = []
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= 1.2 * Th1:
#                 index.append(6 * i + 0)
#     # print(RRI_timewindow_arr[index])
#     a4 = np.unique(RRI_timewindow_arr[index])
#     # print(a);
#     print(len(a4))
#     k4 = 0
#     n_arr = []
#     for m in time_arr:
#         for n in a4:
#             if m - n <= 1 and m - n >= 0:
#                 k4 = k4 + 1
#                 n_arr.append(n)
#     print(k4)
#
#     index = []
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= 2 * Th1:
#                 index.append(6 * i + 0)
#     # print(RRI_timewindow_arr[index])
#     a5 = np.unique(RRI_timewindow_arr[index])
#     # print(a5);
#     print(len(a5))
#     k5 = 0
#     n_arr = []
#     for m in time_arr:
#         for n in a5:
#             if m - n <= 1 and m - n >= 0:
#                 k5 = k5 + 1
#                 n_arr.append(n)
#     print(k5)
#
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
#     AUC_cs_arr.append(AUC_cs)
#
# print(AUC_cs_arr)
# print(time_arr_arr)
# np.savetxt("C:/Users/wxiong/PycharmProjects/EEGstudy/chance_predictor/AUC_EEG_ECG_6h_VIC1173.csv", AUC_cs_arr, delimiter=",", fmt='%s')
# np.savetxt("C:/Users/wxiong/PycharmProjects/EEGstudy/chance_predictor/seizure_labels_EEG_ECG_6h_VIC1173.csv", time_arr_arr, delimiter=",", fmt='%s')
#
#
#
#
#
#
#
# ### add circadian
# t1=np.linspace(0+0.00416667,0+0.00416667+0.00416667*(len(Raw_variance_EEG_arr)-1),len(Raw_variance_EEG_arr))
# a=np.where(t1<6.67388889+0)
# t1[0:1601]=t1[0:1601]+17.3261111
# t1[1601:]=t1[1601:]-6.67388889-0
# time_feature_arr=[]
# for i in range(len(t1)):
#     if t1[i]>24:
#         time_feature_arr.append(t1[i] - (t1[i] // 24) * 24)
#     else:
#         time_feature_arr.append(t1[i])
# print(len(time_feature_arr))
# time_arr=time_feature_arr[22318:]
# print(len(time_arr))
# new_arr=[]
# for j in range(0,558):
#     new_arr.append(time_arr[40*j])
#
# bins_number = 18
# bins = np.linspace(0, 24, bins_number + 1)
# pro_circadian_time=[]
# pro_circadian_time_false=[]
# for i in range(len(new_arr)):
#     if new_arr[i] >= bins[0] and new_arr[i] <= bins[1]:
#         pro_circadian_time_false.append(0.05736052)
#         pro_circadian_time.append(0)
#     elif new_arr[i] > bins[1] and new_arr[i] < bins[2]:
#         pro_circadian_time_false.append(0.05736052)
#         pro_circadian_time.append(0)
#     elif new_arr[i] >= bins[2] and new_arr[i] < bins[3]:
#         pro_circadian_time_false.append(0.05736052)
#         pro_circadian_time.append(0)
#     elif new_arr[i] >= bins[3] and new_arr[i] < bins[4]:
#         pro_circadian_time_false.append(0.05736052)
#         pro_circadian_time.append(0)
#     elif new_arr[i] >= bins[4] and new_arr[i] < bins[5]:
#         pro_circadian_time_false.append(0.05736052)
#         pro_circadian_time.append(0)
#     elif new_arr[i] >= bins[5] and new_arr[i] <= bins[6]:
#         pro_circadian_time_false.append(0.05736052)
#         pro_circadian_time.append(0)
#     elif new_arr[i] > bins[6] and new_arr[i] < bins[7]:
#         pro_circadian_time_false.append(0.05736052)
#         pro_circadian_time.append(0.076923077)
#     elif new_arr[i] >= bins[7] and new_arr[i] <= bins[8]:
#         pro_circadian_time_false.append(0.05736052)
#         pro_circadian_time.append(0)
#     elif new_arr[i] > bins[8] and new_arr[i] < bins[9]:
#         pro_circadian_time_false.append(0.05736052)
#         pro_circadian_time.append(0)
#     elif new_arr[i] >= bins[9] and new_arr[i] < bins[10]:
#         pro_circadian_time_false.append(0.057315707)
#         pro_circadian_time.append(0.333333333)
#     elif new_arr[i] >= bins[10] and new_arr[i] < bins[11]:
#         pro_circadian_time_false.append(0.053596236)
#         pro_circadian_time.append(0.333333333)
#     elif new_arr[i] >= bins[11] and new_arr[i] < bins[12]:
#         pro_circadian_time_false.append(0.04302039)
#         pro_circadian_time.append(0)
#     elif new_arr[i] >= bins[12] and new_arr[i] < bins[13]:
#         pro_circadian_time_false.append(0.043065203)
#         pro_circadian_time.append(0)
#     elif new_arr[i] >= bins[13] and new_arr[i] < bins[14]:
#         pro_circadian_time_false.append(0.057315707)
#         pro_circadian_time.append(0.333333333)
#     elif new_arr[i] >= bins[14] and new_arr[i] < bins[15]:
#         pro_circadian_time_false.append(0.05736052)
#         pro_circadian_time.append(0)
#     elif new_arr[i] >= bins[15] and new_arr[i] < bins[16]:
#         pro_circadian_time_false.append(0.05736052)
#         pro_circadian_time.append(0)
#     elif new_arr[i] >= bins[16] and new_arr[i] < bins[17]:
#         pro_circadian_time_false.append(0.05736052)
#         pro_circadian_time.append(0)
#     elif new_arr[i] >= bins[17] and new_arr[i] <= bins[18]:
#         pro_circadian_time_false.append(0.05736052)
#         pro_circadian_time.append(0)
# # print(np.sort(pro_circadian_time))
# # RRI_timewindow_arr=t
# # print(RRI_timewindow_arr[-1]-RRI_timewindow_arr[0])
# # pyplot.figure(figsize=(8,4))
# # pyplot.plot(RRI_timewindow_arr,pro_circadian_time)
# # pyplot.annotate('',xy=(121.011667,np.max(pro_circadian_time)),xytext=(121.011667,np.max(pro_circadian_time)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(125.851667,np.max(pro_circadian_time)),xytext=(125.851667,np.max(pro_circadian_time)+0.000000000001),arrowprops=dict(facecolor='red',shrink=0.05))
# # pyplot.annotate('',xy=(137.429722,np.max(pro_circadian_time)),xytext=(137.429722,np.max(pro_circadian_time)+0.00000000001),arrowprops=dict(facecolor='red',shrink=0.05))
# # pyplot.annotate('',xy=(145.430278,np.max(pro_circadian_time)),xytext=(145.430278,np.max(pro_circadian_time)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.hlines(0.33333, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'r')
# # pyplot.hlines(0.11111, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'orange')
# # pyplot.hlines(0.22222, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'orange')
# # pyplot.hlines(0.55555, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'orange')
# # pyplot.hlines(0.5*0.0769, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'orange')
# # pyplot.title('Forecast seizures in VIC1173')
# # pyplot.xlabel('Time(h)')
# # pyplot.ylabel('Seizure probability')
# # pyplot.show()
#
# # pro_circadian_time=split(pro_circadian_time,6)
# # print(len(pro_circadian_time))
# # index=[]
# # for i in range(len(pro_circadian_time)):
# #     for item in pro_circadian_time[i]:
# #          if item >= 0.33333:
# #             index.append(6*i+0)
# # print(RRI_timewindow_arr[index])
# # a=np.unique(RRI_timewindow_arr[index])
# # print(a); print(len(a))
# # index=[]
# # for i in range(len(pro_circadian_time)):
# #     for item in pro_circadian_time[i]:
# #          if item >= 0.3*0.33333:
# #             index.append(6*i+0)
# # print(RRI_timewindow_arr[index])
# # a=np.unique(RRI_timewindow_arr[index])
# # print(a); print(len(a))
# # index=[]
# # for i in range(len(pro_circadian_time)):
# #     for item in pro_circadian_time[i]:
# #          if item >= 0.6*0.33333:
# #             index.append(6*i+0)
# # print(RRI_timewindow_arr[index])
# # a=np.unique(RRI_timewindow_arr[index])
# # print(a); print(len(a))
# # index=[]
# # for i in range(len(pro_circadian_time)):
# #     for item in pro_circadian_time[i]:
# #          if item >= 1.2*0.33333:
# #             index.append(6*i+0)
# # print(RRI_timewindow_arr[index])
# # a=np.unique(RRI_timewindow_arr[index])
# # print(a); print(len(a))
# # index=[]
# # for i in range(len(pro_circadian_time)):
# #     for item in pro_circadian_time[i]:
# #          if item >= 2*0.33333:
# #             index.append(6*i+0)
# # print(RRI_timewindow_arr[index])
# # a=np.unique(RRI_timewindow_arr[index])
# # print(a); print(len(a))
#
# # index=[]
# # for i, j in enumerate(pro_circadian_time):
# #     if j >= 0.33333:
# #         index.append(i)
# # print(RRI_timewindow_arr[index])
# # print(len(index))
# # time_arr=[121.011667,125.851667,137.429722,145.430278]
# # k=0
# # n_arr=[]
# # for m in time_arr:
# #     for n in RRI_timewindow_arr[index]:
# #         if m-n<=1 and m-n>=0:
# #             k=k+1
# #             n_arr.append(n)
# # print(k)
# # index=[]
# # for i, j in enumerate(pro_circadian_time):
# #     if j >= 0.3*0.33333:
# #         index.append(i)
# # print(RRI_timewindow_arr[index])
# # print(len(index))
# # time_arr=[121.011667,125.851667,137.429722,145.430278]
# # k=0
# # n_arr=[]
# # for m in time_arr:
# #     for n in RRI_timewindow_arr[index]:
# #         if m-n<=1 and m-n>=0:
# #             k=k+1
# #             n_arr.append(n)
# # print(k)
# # index=[]
# # for i, j in enumerate(pro_circadian_time):
# #     if j >= 0.6*0.33333:
# #         index.append(i)
# # print(RRI_timewindow_arr[index])
# # print(len(index))
# # time_arr=[121.011667,125.851667,137.429722,145.430278]
# # k=0
# # n_arr=[]
# # for m in time_arr:
# #     for n in RRI_timewindow_arr[index]:
# #         if m-n<=1 and m-n>=0:
# #             k=k+1
# #             n_arr.append(n)
# # print(k)
# # index=[]
# # for i, j in enumerate(pro_circadian_time):
# #     if j >= 0.33333*1.2:
# #         index.append(i)
# # print(RRI_timewindow_arr[index])
# # print(len(index))
# # time_arr=[121.011667,125.851667,137.429722,145.430278]
# # k=0
# # n_arr=[]
# # for m in time_arr:
# #     for n in RRI_timewindow_arr[index]:
# #         if m-n<=1 and m-n>=0:
# #             k=k+1
# #             n_arr.append(n)
# # print(k)
# # index=[]
# # for i, j in enumerate(pro_circadian_time):
# #     if j >= 0.33333*2:
# #         index.append(i)
# # print(RRI_timewindow_arr[index])
# # print(len(index))
# # time_arr=[121.011667,125.851667,137.429722,145.430278]
# # k=0
# # n_arr=[]
# # for m in time_arr:
# #     for n in RRI_timewindow_arr[index]:
# #         if m-n<=1 and m-n>=0:
# #             k=k+1
# #             n_arr.append(n)
# # print(k)
#
# Th2=1/3
# Pcombined=pro_circadian_time
# Pcombined = split(Pcombined, 6)
# print(len(Pcombined))
# time_arr_arr_EEGcirca=[]
# AUC_com_arr_EEGcirca=[]
# for i in range(5000):
#     time_arr = np.random.uniform(low=t_window_arr[22318], high=t_window_arr[-1], size=4)
#     time_arr_arr_EEGcirca.append(time_arr)
#     time_arr=np.sort(time_arr)
#     index=[]
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= Th2:
#                 index.append(6*i+0)
#     # print(RRI_timewindow_arr[index])
#     a6=np.unique(RRI_timewindow_arr[index])
#     # print(a6);
#     print(len(a6))
#     k6=0
#     n_arr=[]
#     for m in time_arr:
#         for n in a6:
#             if m-n<=1 and m-n>=0:
#                 k6=k6+1
#                 n_arr.append(n)
#     print(k6)
#
#     index=[]
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= 0.3*Th2:
#                 index.append(6*i+0)
#     # print(RRI_timewindow_arr[index])
#     a7=np.unique(RRI_timewindow_arr[index])
#     # print(a7);
#     print(len(a7))
#     k7=0
#     n_arr=[]
#     for m in time_arr:
#         for n in a7:
#             if m-n<=1 and m-n>=0:
#                 k7=k7+1
#                 n_arr.append(n)
#     print(k7)
#
#     index=[]
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= 0.6*Th2:
#                 index.append(6*i+0)
#     # print(RRI_timewindow_arr[index])
#     a8=np.unique(RRI_timewindow_arr[index])
#     # print(a8);
#     print(len(a8))
#     k8=0
#     n_arr=[]
#     for m in time_arr:
#         for n in a8:
#             if m-n<=1 and m-n>=0:
#                 k8=k8+1
#                 n_arr.append(n)
#     print(k8)
#
#     index=[]
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= 1.2*Th2:
#                 index.append(6*i+0)
#     # print(RRI_timewindow_arr[index])
#     a9=np.unique(RRI_timewindow_arr[index])
#     # print(a9);
#     print(len(a9))
#     k9=0
#     n_arr=[]
#     for m in time_arr:
#         for n in a9:
#             if m-n<=1 and m-n>=0:
#                 k9=k9+1
#                 n_arr.append(n)
#     print(k9)
#
#     index=[]
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= 2*Th2:
#                 index.append(6*i+0)
#     # print(RRI_timewindow_arr[index])
#     a10=np.unique(RRI_timewindow_arr[index])
#     k10=0
#     n_arr=[]
#     for m in time_arr:
#         for n in a10:
#             if m-n<=1 and m-n>=0:
#                 k10=k10+1
#                 n_arr.append(n)
#     print(k10)
#
#     Sen6=k6/len(time_arr);Sen7=k7/len(time_arr);Sen8=k8/len(time_arr);Sen9=k9/len(time_arr);Sen10=k10/len(time_arr);
#     FPR6=(len(a6)-k6)/len(Pcombined);FPR7=(len(a7)-k7)/len(Pcombined);FPR8=(len(a8)-k8)/len(Pcombined);FPR9=(len(a9)-k9)/len(Pcombined);FPR10=(len(a10)-k10)/len(Pcombined);
#     Sen_arr_COM=[0,Sen6,Sen7,Sen8,Sen9,Sen10,1]
#     FPR_arr_COM=[0,FPR6,FPR7,FPR8,FPR9,FPR10,1]
#     print(Sen_arr_COM);print(FPR_arr_COM);
#
#     from sklearn.metrics import auc
#     AUC_com=auc(np.sort(FPR_arr_COM),np.sort(Sen_arr_COM))
#     AUC_com_arr_EEGcirca.append(AUC_com)
#
# print(AUC_com_arr_EEGcirca)
# print(time_arr_arr_EEGcirca)
# np.savetxt("C:/Users/wxiong/PycharmProjects/EEGstudy/chance_predictor/AUC_circa6h_VIC1173.csv", AUC_com_arr_EEGcirca, delimiter=",", fmt='%s')
# np.savetxt("C:/Users/wxiong/PycharmProjects/EEGstudy/chance_predictor/seizure_labels_circa6h_VIC1173.csv", time_arr_arr_EEGcirca, delimiter=",", fmt='%s')
#
#
#
#
# Pseizureeegvar = 0.000134421;
# Pnonseizureeegvar = 0.999865579;
#
#
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=pro_RRIvars_time[m]*pro_eegautos_time[m]*Pseizureeegvar*pro_RRIautos_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
# # print(len(Pcombined))
#
# # Pcombined=[]
# # for m in range(len(pro_eegvars_time)):
# #     P1=pro_eegautos_time[m]*Pseizureeegvar*pro_RRIautos_time[m]*pro_circadian_time[m]
# #     P2=Pnonseizureeegvar*(1-pro_eegautos_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
# #     Pcombined.append(P1/(P1+P2))
# # print(len(Pcombined))
# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_RRIautos_time[m]*pro_circadian_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
#     Pcombined.append(P1/(P1+P2))
# print(len(Pcombined))
#
# pyplot.figure(figsize=(8,4))
# RRI_timewindow_arr=t
# pyplot.plot(RRI_timewindow_arr,Pcombined)
# pyplot.annotate('',xy=(121.011667,np.max(Pcombined)),xytext=(121.011667,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
# pyplot.annotate('',xy=(125.851667,np.max(Pcombined)),xytext=(125.851667,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='red',shrink=0.05))
# pyplot.annotate('',xy=(137.429722,np.max(Pcombined)),xytext=(137.429722,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='red',shrink=0.05))
# pyplot.annotate('',xy=(145.430278,np.max(Pcombined)),xytext=(145.430278,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='red',shrink=0.05))
# # pyplot.annotate('',xy=(95.018333,np.max(Pcombined)),xytext=(95.018333,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# # pyplot.annotate('',xy=(126.42278,np.max(Pcombined)),xytext=(126.42278,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# # pyplot.annotate('',xy=(126.68611,np.max(Pcombined)),xytext=(126.68611,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# # pyplot.annotate('',xy=(147.35583,np.max(Pcombined)),xytext=(147.35583,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# # pyplot.annotate('',xy=(147.889167,np.max(Pcombined)),xytext=(147.889167,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# pyplot.title('Forecast seizures in VIC1173')
# pyplot.title('Forecast seizures in VIC1173')
# pyplot.xlabel('Time(h)')
# pyplot.ylabel('Seizure probability')
# # pyplot.hlines(5.532470719903305e-07, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'r')
# # pyplot.hlines(1.6597423756242214e-06, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'r')
# pyplot.hlines(1.4949839024941904e-05, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'r')
# # pyplot.hlines(0.3*1.4949839024941904e-05, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'orange')
# # pyplot.hlines(0.6*1.4949839024941904e-05, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'orange')
# # pyplot.hlines(1.2*1.4949839024941904e-05, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'orange')
# # pyplot.hlines(2*1.4949839024941904e-05, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'orange')
# pyplot.show()
# # Pcombined=split(Pcombined,6)
# # print(len(Pcombined))
# # index=[]
# # for i in range(len(Pcombined)):
# #     for item in Pcombined[i]:
# #         if item >= 1.4949839024941904e-05:
# #             index.append(6*i+0)
# # print(RRI_timewindow_arr[index])
# # a=np.unique(RRI_timewindow_arr[index])
# # print(a); print(len(a))
# # index=[]
# # for i in range(len(Pcombined)):
# #     for item in Pcombined[i]:
# #         if item >= 0.3*1.4949839024941904e-05:
# #             index.append(6*i+0)
# # print(RRI_timewindow_arr[index])
# # a=np.unique(RRI_timewindow_arr[index])
# # print(a); print(len(a))
# # index=[]
# # for i in range(len(Pcombined)):
# #     for item in Pcombined[i]:
# #         if item >= 0.6*1.4949839024941904e-05:
# #             index.append(6*i+0)
# # print(RRI_timewindow_arr[index])
# # a=np.unique(RRI_timewindow_arr[index])
# # print(a); print(len(a))
# # index=[]
# # for i in range(len(Pcombined)):
# #     for item in Pcombined[i]:
# #         if item >= 1.2*1.4949839024941904e-05:
# #             index.append(6*i+0)
# # print(RRI_timewindow_arr[index])
# # a=np.unique(RRI_timewindow_arr[index])
# # print(a); print(len(a))
# # index=[]
# # for i in range(len(Pcombined)):
# #     for item in Pcombined[i]:
# #         if item >= 2*1.4949839024941904e-05:
# #             index.append(6*i+0)
# # print(RRI_timewindow_arr[index])
# # a=np.unique(RRI_timewindow_arr[index])
# # print(a); print(len(a))
#
# # index=[]
# # for i, j in enumerate(Pcombined):
# #     if j >= 1.4949839024941904e-05:
# #         index.append(i)
# # print(RRI_timewindow_arr[index])
# # print(len(index))
# # time_arr=[121.011667,125.851667,137.429722,145.430278]
# # k=0
# # n_arr=[]
# # for m in time_arr:
# #     for n in RRI_timewindow_arr[index]:
# #         if m-n<=1 and m-n>=0:
# #             k=k+1
# #             n_arr.append(n)
# # print(k)
# # index=[]
# # for i, j in enumerate(Pcombined):
# #     if j >= 0.3*1.4949839024941904e-05:
# #         index.append(i)
# # print(RRI_timewindow_arr[index])
# # print(len(index))
# # time_arr=[121.011667,125.851667,137.429722,145.430278]
# # k=0
# # n_arr=[]
# # for m in time_arr:
# #     for n in RRI_timewindow_arr[index]:
# #         if m-n<=1 and m-n>=0:
# #             k=k+1
# #             n_arr.append(n)
# # print(k)
# # index=[]
# # for i, j in enumerate(Pcombined):
# #     if j >= 0.6*1.4949839024941904e-05:
# #         index.append(i)
# # print(RRI_timewindow_arr[index])
# # print(len(index))
# # time_arr=[121.011667,125.851667,137.429722,145.430278]
# # k=0
# # n_arr=[]
# # for m in time_arr:
# #     for n in RRI_timewindow_arr[index]:
# #         if m-n<=1 and m-n>=0:
# #             k=k+1
# #             n_arr.append(n)
# # print(k)
# # index=[]
# # for i, j in enumerate(Pcombined):
# #     if j >= 1.2*1.4949839024941904e-05:
# #         index.append(i)
# # print(RRI_timewindow_arr[index])
# # print(len(index))
# # time_arr=[121.011667,125.851667,137.429722,145.430278]
# # k=0
# # n_arr=[]
# # for m in time_arr:
# #     for n in RRI_timewindow_arr[index]:
# #         if m-n<=1 and m-n>=0:
# #             k=k+1
# #             n_arr.append(n)
# # print(k)
# # index=[]
# # for i, j in enumerate(Pcombined):
# #     if j >= 2*1.4949839024941904e-05:
# #         index.append(i)
# # print(RRI_timewindow_arr[index])
# # print(len(index))
# # time_arr=[121.011667,125.851667,137.429722,145.430278]
# # k=0
# # n_arr=[]
# # for m in time_arr:
# #     for n in RRI_timewindow_arr[index]:
# #         if m-n<=1 and m-n>=0:
# #             k=k+1
# #             n_arr.append(n)
# # print(k)
#
# Th3=1.4949839024941904e-05
# Pcombined = split(Pcombined, 6)
# print(len(Pcombined))
# time_arr_arr_EEGcirca=[]
# AUC_com_arr_EEGcirca=[]
# for i in range(5000):
#     time_arr = np.random.uniform(low=t_window_arr[22318], high=t_window_arr[-1], size=4)
#     time_arr_arr_EEGcirca.append(time_arr)
#     time_arr=np.sort(time_arr)
#     index=[]
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= Th3:
#                 index.append(6*i+0)
#     # print(RRI_timewindow_arr[index])
#     a6=np.unique(RRI_timewindow_arr[index])
#     # print(a6);
#     print(len(a6))
#     k6=0
#     n_arr=[]
#     for m in time_arr:
#         for n in a6:
#             if m-n<=1 and m-n>=0:
#                 k6=k6+1
#                 n_arr.append(n)
#     print(k6)
#
#     index=[]
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= 0.3*Th3:
#                 index.append(6*i+0)
#     # print(RRI_timewindow_arr[index])
#     a7=np.unique(RRI_timewindow_arr[index])
#     # print(a7);
#     print(len(a7))
#     k7=0
#     n_arr=[]
#     for m in time_arr:
#         for n in a7:
#             if m-n<=1 and m-n>=0:
#                 k7=k7+1
#                 n_arr.append(n)
#     print(k7)
#
#     index=[]
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= 0.6*Th3:
#                 index.append(6*i+0)
#     # print(RRI_timewindow_arr[index])
#     a8=np.unique(RRI_timewindow_arr[index])
#     # print(a8);
#     print(len(a8))
#     k8=0
#     n_arr=[]
#     for m in time_arr:
#         for n in a8:
#             if m-n<=1 and m-n>=0:
#                 k8=k8+1
#                 n_arr.append(n)
#     print(k8)
#
#     index=[]
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= 1.2*Th3:
#                 index.append(6*i+0)
#     # print(RRI_timewindow_arr[index])
#     a9=np.unique(RRI_timewindow_arr[index])
#     # print(a9);
#     print(len(a9))
#     k9=0
#     n_arr=[]
#     for m in time_arr:
#         for n in a9:
#             if m-n<=1 and m-n>=0:
#                 k9=k9+1
#                 n_arr.append(n)
#     print(k9)
#
#     index=[]
#     for i in range(len(Pcombined)):
#         for item in Pcombined[i]:
#             if item >= 2*Th3:
#                 index.append(6*i+0)
#     # print(RRI_timewindow_arr[index])
#     a10=np.unique(RRI_timewindow_arr[index])
#     k10=0
#     n_arr=[]
#     for m in time_arr:
#         for n in a10:
#             if m-n<=1 and m-n>=0:
#                 k10=k10+1
#                 n_arr.append(n)
#     print(k10)
#
#     Sen6=k6/len(time_arr);Sen7=k7/len(time_arr);Sen8=k8/len(time_arr);Sen9=k9/len(time_arr);Sen10=k10/len(time_arr);
#     FPR6=(len(a6)-k6)/len(Pcombined);FPR7=(len(a7)-k7)/len(Pcombined);FPR8=(len(a8)-k8)/len(Pcombined);FPR9=(len(a9)-k9)/len(Pcombined);FPR10=(len(a10)-k10)/len(Pcombined);
#     Sen_arr_COM=[0,Sen6,Sen7,Sen8,Sen9,Sen10,1]
#     FPR_arr_COM=[0,FPR6,FPR7,FPR8,FPR9,FPR10,1]
#     print(Sen_arr_COM);print(FPR_arr_COM);
#
#     from sklearn.metrics import auc
#     AUC_com=auc(np.sort(FPR_arr_COM),np.sort(Sen_arr_COM))
#     AUC_com_arr_EEGcirca.append(AUC_com)
#
# print(AUC_com_arr_EEGcirca)
# print(time_arr_arr_EEGcirca)
# np.savetxt("C:/Users/wxiong/PycharmProjects/EEGstudy/chance_predictor/AUC_EEG_ECG_circa6h_VIC1173.csv", AUC_com_arr_EEGcirca, delimiter=",", fmt='%s')
# np.savetxt("C:/Users/wxiong/PycharmProjects/EEGstudy/chance_predictor/seizure_labels_EEG_ECGcirca6h_VIC1173.csv", time_arr_arr_EEGcirca, delimiter=",", fmt='%s')
