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


# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/EEG_QLD0227_3h.csv',sep=',',header=None)
# Raw_EEG= csv_reader.values
# Raw_EEG_arr=[]
# for item in Raw_EEG:
#     Raw_EEG_arr.append(float(item))
#
# t0=np.linspace(2.9975,2.9975+0.00000109*(len(Raw_EEG_arr)-1),len(Raw_EEG_arr))
# pyplot.plot(t0,Raw_EEG_arr,'grey',alpha=0.3)
# pyplot.xlabel('Time(hour)',fontsize=13)
# pyplot.ylabel('Voltage',fontsize=13)
# pyplot.title('EEG recordings in QLD0227',fontsize=13)
# pyplot.show()


csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/QLD0227channels/EEGvariance_QLD0227_15s_3h.csv',sep=',',header=None)
Raw_variance_EEG= csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/QLD0227channels/EEGauto_QLD0227_15s_3h.csv',sep=',',header=None)
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/EEGauto_lag1_QLD0227_15s_3h.csv',sep=',',header=None)
Raw_auto_EEG= csv_reader.values
Raw_variance_EEG_arr=[]
for item in Raw_variance_EEG:
    Raw_variance_EEG_arr.append(float(item))
Raw_auto_EEG_arr=[]
for item in Raw_auto_EEG:
    Raw_auto_EEG_arr.append(float(item))

# # ###fft
# # x=Raw_variance_EEG_arr
# # x=np.array(x)
# # fs=1/15
# # f, Sxx = signal.periodogram(x, fs)
# # pyplot.plot(f,Sxx)
# # pyplot.xlabel('Frequency [Hz]',fontsize=13)
# # pyplot.ylabel('Power spectral density ($v^2$)',fontsize=13)
# # pyplot.title('EEG variance in QLD0227')
# # pyplot.show()
# # x=Raw_auto_EEG_arr
# # x=np.array(x)
# # fs=1/15
# # f, Sxx = signal.periodogram(x, fs)
# # pyplot.plot(f,Sxx)
# # pyplot.xlabel('Frequency [Hz]',fontsize=13)
# # pyplot.ylabel('Power spectral density',fontsize=13)
# # pyplot.title('EEG autocorrelation in QLD0227')
# # pyplot.show()
# #
#
t=np.linspace(2.9975,2.9975+0.00416667*(len(Raw_variance_EEG_arr)-1),len(Raw_variance_EEG_arr))
print(t[-1]);print(t[-1]-t[0]);
print(len(t))
print(t[19640]);print(len(t));print(t[0]);
print(t[-1]-t[0]);
print(t[19640]-t[0]);print(t[-1]-t[19640]);

# pyplot.plot(t,Raw_variance_EEG_arr,'grey',alpha=0.3)
# pyplot.xlabel('Time(hour)',fontsize=13)
# pyplot.ylabel('Voltage($\mathregular{V}$)',fontsize=13)
# pyplot.title('EEG variance in QLD0227',fontsize=13)
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
# pyplot.title('EEG variance in QLD0227',fontsize=13)
# pyplot.show()
#
#
# pyplot.plot(t,Raw_auto_EEG_arr,'grey',alpha=0.3)
# pyplot.xlabel('Time(hour)',fontsize=13)
# pyplot.title('EEG autocorrelation in QLD0227',fontsize=13)
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
# pyplot.title('EEG autocorrelation in QLD0227',fontsize=13)
# pyplot.show()


# # # ### EEG var EEG var EEG var EEG var
Raw_variance_EEG=Raw_variance_EEG_arr
window_time_arr=t
# Raw_variance_EEG=Raw_variance_EEG_arr[0:19640]
# window_time_arr=t[0:19640]

seizure_timing_index=[]
for k in range(len(window_time_arr)):
    if window_time_arr[k]<45.6175 and window_time_arr[k+1]>=45.6175:
        seizure_timing_index.append(k)
    if window_time_arr[k]<64.7838889 and window_time_arr[k+1]>=64.7838889:
        seizure_timing_index.append(k)
    if window_time_arr[k]<74.0208333 and window_time_arr[k+1]>=74.0208333:
        seizure_timing_index.append(k)
    if window_time_arr[k]<81.1208333 and window_time_arr[k+1]>=81.1208333:
        seizure_timing_index.append(k)
    if window_time_arr[k]<114.798888 and window_time_arr[k+1]>=114.798888:
        seizure_timing_index.append(k)
print(seizure_timing_index)
index_ictal=seizure_timing_index

seizure_timing_index=[]
for k in range(len(window_time_arr)):
    if window_time_arr[k]<66.62 and window_time_arr[k+1]>=66.62:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 67.3768889 and window_time_arr[k + 1] >= 67.3768889:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 68.32716 and window_time_arr[k + 1] >= 68.32716:
        seizure_timing_index.append(k)
print(seizure_timing_index)
index_cluster=seizure_timing_index

duration=[1,1,1,1,1]

index_ictal_sum=[]
for m in range(len(index_ictal)):
    for j in range(duration[m]+1):
        index_ictal_sum.append(index_ictal[m] + j)
index_cluster_sum=[]
for item in index_cluster:
    for j in range(8):
        index_cluster_sum.append(item+j)
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

np.savetxt("C:/Users/wxiong/Documents/PHD/combine_features/QLD0227_tags.csv", x, delimiter=",", fmt='%s')















# long_rhythm_var_arr=movingaverage(Raw_variance_EEG,240*6)
# medium_rhythm_var_arr=movingaverage(Raw_variance_EEG,240)
# medium_rhythm_var_arr_2=movingaverage(Raw_variance_EEG,240*3)
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG,240*6)
# medium_rhythm_var_arr_4=movingaverage(Raw_variance_EEG,240*12)
# short_rhythm_var_arr_plot=movingaverage(Raw_variance_EEG,240*24)
#
# # fig=pyplot.figure(figsize=(8,6))
# # pyplot.plot(window_time_arr,Raw_variance_EEG,'grey',alpha=0.5,label='Raw variance')
# # # pyplot.plot(window_time_arr,short_rhythm_var_arr_plot,'grey',alpha=0.5,label='5min')
# # # pyplot.plot(window_time_arr,medium_rhythm_var_arr,'g',label='1 hour')
# # # pyplot.plot(window_time_arr,medium_rhythm_var_arr_2,'k',label='3 hour')
# # # pyplot.plot(window_time_arr,medium_rhythm_var_arr_3,'orange',label='6 hour')
# # # pyplot.plot(window_time_arr,medium_rhythm_var_arr_4,'b',label='12 hour')
# # # pyplot.plot(window_time_arr,long_rhythm_var_arr,'r',alpha=0.7,label='1 day')
# # pyplot.title('EEG variance in QLD0227',fontsize=23)
# # pyplot.xlabel('Time(hour)',fontsize=23)
# # pyplot.ylabel('Voltage($\mathregular{v^2}$)',fontsize=23)
# # pyplot.legend(loc='upper left',fontsize=10)
# # locs, labels = pyplot.xticks(fontsize=23)
# # locs, labels = pyplot.yticks(fontsize=23)
# # pyplot.tight_layout()
# # pyplot.annotate('',xy=(45.6175,np.max(short_rhythm_var_arr_plot)),xytext=(45.6175,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(64.7838889,np.max(short_rhythm_var_arr_plot)),xytext=(64.7838889,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(66.62,np.max(short_rhythm_var_arr_plot)),xytext=(66.62,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(74.0208333,np.max(short_rhythm_var_arr_plot)),xytext=(74.0208333,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(81.1208333,np.max(short_rhythm_var_arr_plot)),xytext=(81.1208333,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(114.798888,np.max(short_rhythm_var_arr_plot)),xytext=(114.798888,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.show()
#
#
# long_rhythm_var_arr=long_rhythm_var_arr*(10**12)
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
# # ax1.plot(window_time_arr[240*6:],long_rhythm_var_arr[240*6:],'darkblue',alpha=0.8)
# # # ax1.plot(window_time_arr,rolmean_long_EEGvar,'C0')
# # # ax1.set_title('EEG variance in QLD0227',fontsize=23)
# # ax1.set_title('EEG variance',fontsize=23)
# # ax1.set_xlabel('Time(hour)',fontsize=23)
# # ax1.set_ylabel('$\mathregular{\u03BCV^2}$',fontsize=23)
# # locs, labels = pyplot.xticks(fontsize=23)
# # locs, labels = pyplot.yticks(fontsize=23)
# # ax1.scatter(window_time_arr[10228],long_rhythm_var_arr[10228],s=60,c='orange')
# # ax1.scatter(window_time_arr[14828],long_rhythm_var_arr[14828],s=60,c='orange')
# # ax1.scatter(window_time_arr[15269],long_rhythm_var_arr[15269],s=60,c='r')
# # ax1.scatter(window_time_arr[17045],long_rhythm_var_arr[17045],s=60,c='orange')
# # ax1.scatter(window_time_arr[18749],long_rhythm_var_arr[18749],s=60,c='orange')
# # # ax1.scatter(window_time_arr[26832],long_rhythm_var_arr[26832],s=60,c='orange')
# # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax2=pyplot.subplot(gs[1])
# # ax2.set_xlabel('Time(hour)',fontsize=23)
# # ax2.set_title('Instantaneous Phase',fontsize=23)
# # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax2.plot(window_time_arr[240*6:],phase_long_EEGvariance_arr[240*6:],c='k',alpha=0.5,label='instantaneous phase')
# # pyplot.hlines(0,window_time_arr[240*6],window_time_arr[-1],'k','dashed')
# # ax2.set_xlabel('Time(hour)',fontsize=23)
# # locs, labels = pyplot.xticks(fontsize=23)
# # locs, labels = pyplot.yticks(fontsize=23)
# # ax2.scatter(window_time_arr[10228],phase_long_EEGvariance_arr[10228],s=60,c='orange')
# # ax2.scatter(window_time_arr[14828],phase_long_EEGvariance_arr[14828],s=60,c='orange')
# # ax2.scatter(window_time_arr[15269],phase_long_EEGvariance_arr[15269],s=60,c='r')
# # ax2.scatter(window_time_arr[17045],phase_long_EEGvariance_arr[17045],s=60,c='orange')
# # ax2.scatter(window_time_arr[18749],phase_long_EEGvariance_arr[18749],s=60,c='orange')
# # # ax2.scatter(window_time_arr[26832],phase_long_EEGvariance_arr[26832],s=60,c='orange')
# # # locs, labels = pyplot.yticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi],['-$\pi$','Rising','0','Falling','$\pi$'],rotation='vertical',fontsize=23)
# # # pyplot.ylim(-np.pi,np.pi)
# # locs, labels = pyplot.yticks([-0.5*np.pi,-0.15*np.pi,0,0.3*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=23)
# # pyplot.tight_layout()
# # pyplot.show()
# #
# # bins_number = 18
# # # bins = np.linspace(0, 2*np.pi, bins_number + 1)
# # bins = np.linspace(-np.pi, np.pi, bins_number + 1)
# # nEEGsvar, _, _ = pyplot.hist(phase_long_EEGvariance_arr, bins)
# # nEEGsvarsei, _, _ = pyplot.hist(seizure_phase_var_long, bins)
# # # nEEGsvar, _, _ = pyplot.hist(rolmean_long_EEGvar, bins)
# # # seizure_phase_var_short=[]
# # # for item in seizure_timing_index:
# # #     seizure_phase_var_short=seizure_phase_var_short+list(phase_long_EEGvariance_arr[item])
# # # nEEGsvarsei, _, _ = pyplot.hist(seizure_phase_var_short, bins)
# # print(nEEGsvar)
# # print(nEEGsvarsei)
# # width = 2*np.pi / bins_number
# # params = dict(projection='polar')
# # fig, ax = pyplot.subplots(subplot_kw=params)
# # ax.bar(bins[:bins_number], nEEGsvarsei/nEEGsvar,width=width, color='grey',alpha=0.6,linewidth=2, fill=True,edgecolor='k')
# # pyplot.setp(ax.get_yticklabels(), color='k')
# # # ax.set_title('Seizure probability in EEG variance',fontsize=13)
# # locs, labels = pyplot.yticks([0.0003,0.0006],['0.0003','0.0006'],fontsize=13)
# # ax.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',],fontsize=16)
# # pyplot.show()
#
# # width = 2*np.pi / bins_number
# # # params = dict(projection='polar')
# # # fig, ax = pyplot.subplots(subplot_kw=params)
# # # ax.bar(bins[:bins_number], nEEGsvarsei/nEEGsvar,width=width, color='grey',alpha=0.6,linewidth=2, fill=True,edgecolor='k')
# # # pyplot.setp(ax.get_yticklabels(), color='k')
# # # # ax.set_title('Seizure probability in EEG variance',fontsize=13)
# # # ax.set_yticks([0.0005,0.001,0.0015,0.002])
# # # ax.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',])
# # # pyplot.show()
#
#
#
#
# # # # signal=pd.DataFrame(phase_long_EEGvariance_arr[240*6:])
# # # # from statsmodels.tsa.stattools import adfuller
# # # # result=adfuller(signal)
# # # # print(result[0]);print(result[1])
# # # #
# # # # pyplot.figure()
# # # # pyplot.plot(window_time_arr[240*6:],phase_long_EEGvariance_arr[240*6:])
# # # # pyplot.xlabel('Time(h)')
# # # # pyplot.ylabel('Phases')
# # # # pyplot.show()
# # # #
# # # # signal=phase_long_EEGvariance_arr[240*6:]
# # # # first_signal=pd.DataFrame(phase_long_EEGvariance_arr[240*6:]).diff()
# # # # pyplot.plot(window_time_arr[240*6:],first_signal)
# # # # pyplot.xlabel('Time(h)')
# # # # pyplot.ylabel('Phases diff')
# # # # pyplot.show()
# # # # print(first_signal)
# # # # # pyplot.plot(window_time_arr[240*6:],first_signal.diff())
# # # # # pyplot.xlabel('Time(h)')
# # # # # pyplot.ylabel('Phases 2rd diff')
# # # # # pyplot.show()
# # # # # second_signal=first_signal.diff()
# # # #
# # # # # from pandas.plotting import autocorrelation_plot
# # # # # autocorrelation_plot(signal)
# # # # # pyplot.show()
# # # # # from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
# # # # # plot_acf(signal)
# # # # # pyplot.show()
# # # # # plot_pacf(signal, lags=25)
# # # # # pyplot.show()
# # # # from pandas.plotting import autocorrelation_plot
# # # # autocorrelation_plot(first_signal[1:])
# # # # pyplot.show()
# # # # from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
# # # # plot_acf(first_signal[1:])
# # # # pyplot.show()
# # # # plot_pacf(first_signal[1:], lags=25)
# # # # pyplot.show()
# #
# #
# #
# # # # # ### EEG auto EEG auto EEG auto EEG auto
# Raw_auto_EEG=Raw_auto_EEG_arr
# window_time_arr=t
# # Raw_auto_EEG=Raw_auto_EEG_arr[0:19640]
# # window_time_arr=t[0:19640]
#
# long_rhythm_value_arr=movingaverage(Raw_auto_EEG,240*6)
# medium_rhythm_value_arr=movingaverage(Raw_auto_EEG,240)
# medium_rhythm_value_arr_2=movingaverage(Raw_auto_EEG,240*3)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG,240*6)
# medium_rhythm_value_arr_4=movingaverage(Raw_auto_EEG,240*12)
# short_rhythm_value_arr_plot=movingaverage(Raw_auto_EEG,240*24)
#
# # fig=pyplot.figure(figsize=(8,6))
# # # pyplot.plot(window_time_arr,Raw_auto_EEG,'grey',alpha=0.5,label='Raw autocorrelation')
# # # pyplot.plot(window_time_arr[20:],short_rhythm_value_arr_plot[20:],'grey',alpha=0.5,label='5min')
# # # pyplot.plot(window_time_arr[240:],medium_rhythm_value_arr[240:],'g',label='1 hour')
# # # pyplot.plot(window_time_arr[240*3:],medium_rhythm_value_arr_2[240*3:],'k',label='3 hour')
# # # pyplot.plot(window_time_arr[240*6:],medium_rhythm_value_arr_3[240*6:],'orange',label='6 hour')
# # # pyplot.plot(window_time_arr[240*12:],medium_rhythm_value_arr_4[240*12:],'b',label='12 hour')
# # # pyplot.plot(window_time_arr[5760:],long_rhythm_value_arr[5760:],'r',alpha=0.7,label='1 day')
# # pyplot.plot(window_time_arr,short_rhythm_value_arr_plot,'grey',alpha=0.5,label='5min')
# # pyplot.plot(window_time_arr,medium_rhythm_value_arr,'g',label='1 hour')
# # pyplot.plot(window_time_arr,medium_rhythm_value_arr_2,'k',label='3 hour')
# # pyplot.plot(window_time_arr,medium_rhythm_value_arr_3,'orange',label='6 hour')
# # pyplot.plot(window_time_arr,medium_rhythm_value_arr_4,'b',label='12 hour')
# # pyplot.plot(window_time_arr,long_rhythm_value_arr,'r',alpha=0.7,label='1 day')
# # pyplot.title('EEG autocorrelation in QLD0227',fontsize=23)
# # pyplot.xlabel('Time(hour)',fontsize=23)
# # pyplot.ylabel('Autocorrelation',fontsize=23)
# # pyplot.legend(loc='upper left',fontsize=10)
# # locs, labels = pyplot.xticks(fontsize=23)
# # locs, labels = pyplot.yticks(fontsize=23)
# # pyplot.tight_layout()
# # pyplot.annotate('',xy=(45.6175,np.max(short_rhythm_value_arr_plot)),xytext=(45.6175,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(64.7838889,np.max(short_rhythm_value_arr_plot)),xytext=(64.7838889,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(66.62,np.max(short_rhythm_value_arr_plot)),xytext=(66.62,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(74.0208333,np.max(short_rhythm_value_arr_plot)),xytext=(74.0208333,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(81.1208333,np.max(short_rhythm_value_arr_plot)),xytext=(81.1208333,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(114.798888,np.max(short_rhythm_value_arr_plot)),xytext=(114.798888,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.show()
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
# # # ax2.plot(window_time_arr[5760:],rolmean_long_EEGauto,'C0',alpha=0.7)
# # # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax2.set_title('EEG autocorrelation in QLD0227 (T4)',fontsize=23)
# # ax2.set_xlabel('Time(hour)',fontsize=23)
# # locs, labels = pyplot.xticks(fontsize=23)
# # locs, labels = pyplot.yticks(fontsize=23)
# # ax2.scatter(window_time_arr[10228],long_rhythm_value_arr[10228],c='k')
# # ax2.scatter(window_time_arr[14828],long_rhythm_value_arr[14828],c='k')
# # ax2.scatter(window_time_arr[15269],long_rhythm_value_arr[15269],c='k')
# # ax2.scatter(window_time_arr[17045],long_rhythm_value_arr[17045],c='k')
# # ax2.scatter(window_time_arr[18749],long_rhythm_value_arr[18749],c='k')
# # # ax2.scatter(window_time_arr[26832],long_rhythm_value_arr[26832],c='k')
# # ax3=pyplot.subplot(gs[1])
# # ax3.set_xlabel('Time(hour)',fontsize=23)
# # ax3.set_title('Instantaneous Phase',fontsize=23)
# # # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # locs, labels = pyplot.xticks(fontsize=23)
# # locs, labels = pyplot.yticks(fontsize=23)
# # ax3.plot(window_time_arr[240*6:],phase_long_EEGauto_arr[240*6:],'k',alpha=0.5,label='instantaneous phase')
# # pyplot.hlines(0,window_time_arr[240*6],window_time_arr[-1],'k','dashed')
# # ax3.scatter(window_time_arr[10228],phase_long_EEGauto_arr[10228],c='k')
# # ax3.scatter(window_time_arr[14828],phase_long_EEGauto_arr[14828],c='k')
# # ax3.scatter(window_time_arr[15269],phase_long_EEGauto_arr[15269],c='k')
# # ax3.scatter(window_time_arr[17045],phase_long_EEGauto_arr[17045],c='k')
# # ax3.scatter(window_time_arr[18749],phase_long_EEGauto_arr[18749],c='k')
# # # ax3.scatter(window_time_arr[26832],phase_long_EEGauto_arr[26832],c='k')
# # ax3.set_xlabel('Time(hour)',fontsize=23)
# # # locs, labels = pyplot.yticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi],['-$\pi$','Rising','0','Falling','$\pi$'],rotation='vertical',fontsize=23)
# # locs, labels = pyplot.yticks([-0.5*np.pi,-0.25*np.pi,0,0.25*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=23)
# # pyplot.tight_layout()
# # pyplot.show()
# # bins_number = 18
# # # bins = np.linspace(0, 2*np.pi, bins_number + 1)
# # bins = np.linspace(-np.pi, np.pi, bins_number + 1)
# # nEEGsauto, _, _ = pyplot.hist(phase_long_EEGauto_arr, bins)
# # nEEGsautosei, _, _ = pyplot.hist(seizure_phase_value_long, bins)
# # # nEEGsauto, _, _ = pyplot.hist(rolmean_long_EEGauto, bins)
# # # seizure_phase_auto_long=[]
# # # for item in seizure_timing_index:
# # #     seizure_phase_auto_long=seizure_phase_auto_long+list(rolmean_long_EEGauto[item])
# # # nEEGsautosei, _, _ = pyplot.hist(seizure_phase_auto_long, bins)
# # print(nEEGsauto)
# # print(nEEGsautosei)
# # width = 2*np.pi / bins_number
# # params = dict(projection='polar')
# # fig, ax = pyplot.subplots(subplot_kw=params)
# # ax.bar(bins[:bins_number], nEEGsauto/sum(nEEGsauto),width=width, color='g',alpha=0.6,linewidth=2, fill=True,edgecolor='k')
# # pyplot.setp(ax.get_yticklabels(), color='k')
# # ax.set_title('Phase histogram in EEG autocorrelation(QLD0227)',fontsize=13)
# # # ax.set_yticks([0.1,0.2,0.3,0.4,0.5])
# # ax.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',])
# # pyplot.show()
# # params = dict(projection='polar')
# # fig, ax2 = pyplot.subplots(subplot_kw=params)
# # ax2.bar(bins[:bins_number], nEEGsautosei/sum(nEEGsautosei),width=width, color='orange',alpha=0.9,linewidth=2,edgecolor='k')
# # ax2.set_yticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8])
# # ax2.set_title('seizure probability in EEG autocorrelation(QLD0227)',fontsize=13)
# # ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',])
# # pyplot.show()
#
#
#
#
#
# # # ### ECG
# # # # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/QLD0227/rawRRI_timearr_QLD0227_160h_15s.csv',sep=',',header=None)
# # # # rri_t= csv_reader.values
# # # # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/QLD0227/rawRRI_QLD0227_160h_15s.csv',sep=',',header=None)
# # # # RRI_var= csv_reader.values
# # # # rri_t_arr=[]
# # # # for item in rri_t:
# # # #     rri_t_arr.append(float(item)/3600)
# # # # RRI_var_arr=[]
# # # # for item in RRI_var:
# # # #     RRI_var_arr.append(float(item))
# #
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_ch31_timewindowarr_QLD0227_15s_3h.csv',sep=',',header=None)
# rri_t= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_ch31_rawvariance_QLD0227_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_ch31_rawauto_QLD0227_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/ch2-1/RRI_ch21_rawvariance_QLD0227_15s_3h.csv',sep=',',header=None)
# # RRI_var= csv_reader.values
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/ch2-1/RRI_ch21_rawauto_QLD0227_15s_3h.csv',sep=',',header=None)
# # Raw_auto_RRI31= csv_reader.values
#
#
#
# rri_t_arr=[]
# for item in rri_t:
#     rri_t_arr.append(2.9975+float(item))
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
#
# print(rri_t_arr[0]);
# # df_rolling=pd.DataFrame(np.array(RRI_var_arr))
# # rolmean_rri = df_rolling.rolling(3600).mean()
# # rolmean_rri=rolmean_rri.values
#
# # # # ###fft
# # # # x=Raw_variance_RRI31_arr
# # # # x=np.array(x)
# # # # fs=1/15
# # # # f, Sxx = signal.periodogram(x, fs)
# # # # pyplot.plot(f,Sxx)
# # # # # pyplot.xlim(-0.00001,0.0005)
# # # # pyplot.xlabel('Frequency [Hz]',fontsize=13)
# # # # pyplot.ylabel('Power spectral density ($s^2$)',fontsize=13)
# # # # pyplot.title('RRI variance in QLD0227')
# # # # pyplot.show()
# # # # x=Raw_auto_RRI31_arr
# # # # x=np.array(x)
# # # # fs=1/15
# # # # f, Sxx = signal.periodogram(x, fs)
# # # # pyplot.plot(f,Sxx)
# # # # pyplot.xlim(-0.00001,0.0005)
# # # # pyplot.xlabel('Frequency [Hz]',fontsize=13)
# # # # pyplot.ylabel('Power spectral density',fontsize=13)
# # # # pyplot.title('RRI autocorrelation in QLD0227')
# # # # pyplot.show()
#
# # pyplot.plot(rri_t_arr,Raw_variance_RRI31_arr,'grey')
# # pyplot.xlabel('Time(hour)',fontsize=13)
# # pyplot.ylabel('$Second^2$',fontsize=13)
# # pyplot.title('RRI variance in QLD0227',fontsize=13)
# # pyplot.show()
# #
# # pyplot.plot(rri_t_arr,Raw_auto_RRI31_arr,'grey')
# # pyplot.xlabel('Time(hour)',fontsize=13)
# # pyplot.title('RRI autocorrelation in QLD0227',fontsize=13)
# # pyplot.show()
#
#
# seizure_timing_index=[]
# for k in range(len(rri_t_arr)):
#     if rri_t_arr[k]<45.6175 and rri_t_arr[k+1]>=45.6175:
#         seizure_timing_index.append(k)
#     if rri_t_arr[k]<64.7838889 and rri_t_arr[k+1]>=64.7838889:
#         seizure_timing_index.append(k)
#     # if rri_t_arr[k]<66.62 and rri_t_arr[k+1]>=66.62:
#     #     seizure_timing_index.append(k)
#     if rri_t_arr[k]<74.0208333 and rri_t_arr[k+1]>=74.0208333:
#         seizure_timing_index.append(k)
#     if rri_t_arr[k]<81.1208333 and rri_t_arr[k+1]>=81.1208333:
#         seizure_timing_index.append(k)
#     if rri_t_arr[k]<114.798888 and rri_t_arr[k+1]>=114.798888:
#         seizure_timing_index.append(k)
# print(seizure_timing_index)
#
#
# window_time_arr=rri_t_arr
# Raw_variance_RRI31=Raw_variance_RRI31_arr
# # window_time_arr=rri_t_arr[0:19640]
# # Raw_variance_RRI31=Raw_variance_RRI31_arr[0:19640]
#
# long_rhythm_var_arr=movingaverage(Raw_variance_RRI31,240*6)
# medium_rhythm_var_arr=movingaverage(Raw_variance_RRI31,240)
# medium_rhythm_var_arr_2=movingaverage(Raw_variance_RRI31,240*3)
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31,240*6)
# medium_rhythm_var_arr_4=movingaverage(Raw_variance_RRI31,240*12)
# short_rhythm_var_arr_plot=movingaverage(Raw_variance_RRI31,240*24)
#
# # # fig=pyplot.figure(figsize=(8,6))
# # # # pyplot.plot(window_time_arr,Raw_variance_EEG,'grey',alpha=0.5,label='Raw variance')
# # # pyplot.plot(window_time_arr,short_rhythm_var_arr_plot,'grey',alpha=0.5,label='5min')
# # # pyplot.plot(window_time_arr,medium_rhythm_var_arr,'g',label='1 hour')
# # # pyplot.plot(window_time_arr,medium_rhythm_var_arr_2,'k',label='3 hour')
# # # pyplot.plot(window_time_arr,medium_rhythm_var_arr_3,'orange',label='6 hour')
# # # pyplot.plot(window_time_arr,medium_rhythm_var_arr_4,'b',label='12 hour')
# # # pyplot.plot(window_time_arr,long_rhythm_var_arr,'r',alpha=0.7,label='1 day')
# # # pyplot.title('RRI variance in QLD0227',fontsize=23)
# # # pyplot.xlabel('Time(hour)',fontsize=23)
# # # pyplot.ylabel('Second ($\mathregular{s^2}$)',fontsize=23)
# # # pyplot.legend(loc='upper left',fontsize=10)
# # # locs, labels = pyplot.xticks(fontsize=23)
# # # locs, labels = pyplot.yticks(fontsize=23)
# # # pyplot.tight_layout()
# # # pyplot.annotate('',xy=(45.6175,np.max(short_rhythm_var_arr_plot)),xytext=(45.6175,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(64.7838889,np.max(short_rhythm_var_arr_plot)),xytext=(64.7838889,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # # pyplot.annotate('',xy=(66.62,np.max(short_rhythm_var_arr_plot)),xytext=(66.62,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(74.0208333,np.max(short_rhythm_var_arr_plot)),xytext=(74.0208333,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(81.1208333,np.max(short_rhythm_var_arr_plot)),xytext=(81.1208333,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(114.798888,np.max(short_rhythm_var_arr_plot)),xytext=(114.798888,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.show()
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
# # from matplotlib import gridspec
# # fig = pyplot.figure(figsize=(12, 10))
# # gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
# # ax1=pyplot.subplot(gs[0])
# # ax1.plot(window_time_arr,long_rhythm_var_arr,'orange',alpha=0.7)
# # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax1.set_title('RRI variance in QLD0227',fontsize=23)
# # ax1.set_xlabel('Time(hour)',fontsize=23)
# # ax1.set_ylabel('Second($\mathregular{s^2}$)',fontsize=23)
# # locs, labels = pyplot.xticks(fontsize=23)
# # locs, labels = pyplot.yticks(fontsize=23)
# # ax1.scatter(window_time_arr[10228],long_rhythm_var_arr[10228],c='k')
# # ax1.scatter(window_time_arr[14828],long_rhythm_var_arr[14828],c='k')
# # # ax1.scatter(window_time_arr[15269],long_rhythm_var_arr[15269],c='k')
# # ax1.scatter(window_time_arr[17045],long_rhythm_var_arr[17045],c='k')
# # ax1.scatter(window_time_arr[18749],long_rhythm_var_arr[18749],c='k')
# # # ax1.scatter(window_time_arr[26832],long_rhythm_var_arr[26832],c='k')
# # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax2=pyplot.subplot(gs[1])
# # ax2.set_xlabel('Time(hour)',fontsize=23)
# # ax2.set_title('Instantaneous Phase',fontsize=23)
# # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax2.plot(window_time_arr,phase_whole_long,'k',alpha=0.5)
# # # ax2.plot(window_time_arr,rolmean_long_RRIvar,'b',alpha=0.7)
# # ax2.scatter(window_time_arr[10228],phase_whole_long[10228],c='k')
# # ax2.scatter(window_time_arr[14828],phase_whole_long[14828],c='k')
# # # ax2.scatter(window_time_arr[15269],phase_whole_long[15269],c='k')
# # ax2.scatter(window_time_arr[17045],phase_whole_long[17045],c='k')
# # ax2.scatter(window_time_arr[18749],phase_whole_long[18749],c='k')
# # # ax2.scatter(window_time_arr[26832],phase_whole_long[26832],c='k')
# # ax2.set_xlabel('Time(hour)',fontsize=23)
# # locs, labels = pyplot.xticks(fontsize=23)
# # locs, labels = pyplot.yticks(fontsize=23)
# # pyplot.hlines(0,window_time_arr[0],window_time_arr[-1],'k','dashed')
# # # locs, labels = pyplot.yticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi],['-$\pi$','Rising','0','Falling','$\pi$'],rotation='vertical',fontsize=23)
# # locs, labels = pyplot.yticks([-0.5*np.pi,-0.25*np.pi,0,0.25*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=23)
# # pyplot.tight_layout()
# # pyplot.show()
# #
# # bins_number = 18
# # # bins = np.linspace(0, 2*np.pi, bins_number + 1)
# # bins = np.linspace(-np.pi, np.pi, bins_number + 1)
# # nRRIsvar, _, _ = pyplot.hist(phase_whole_long, bins)
# # nRRIsvarsei, _, _ = pyplot.hist(seizure_phase_var_long, bins)
# # # nRRIsvar, _, _ = pyplot.hist(rolmean_long_RRIvar, bins)
# # # seizure_phase_var_long=[]
# # # for item in seizure_timing_index:
# # #     seizure_phase_var_long=seizure_phase_var_long+list(rolmean_long_RRIvar[item])
# # # nRRIsvarsei, _, _ = pyplot.hist(seizure_phase_var_long, bins)
# # print(nRRIsvar)
# # print(nRRIsvarsei)
#
#
#
#
#
#
# # signal=pd.DataFrame(phase_whole_long[240*6:])
# # from statsmodels.tsa.stattools import adfuller
# # result=adfuller(signal)
# # print(result[0]);print(result[1])
# # pyplot.figure()
# # pyplot.plot(window_time_arr[240*6:],phase_whole_long[240*6:])
# # pyplot.xlabel('Time(h)')
# # pyplot.ylabel('Phases')
# # pyplot.show()
# # signal=phase_whole_long[240*6:]
# # first_signal=pd.DataFrame(phase_whole_long[240*6:]).diff()
# # pyplot.plot(window_time_arr[240*6:],first_signal)
# # pyplot.xlabel('Time(h)')
# # pyplot.ylabel('Phases diff')
# # pyplot.show()
# # print(first_signal)
# # # pyplot.plot(window_time_arr[240*6:],first_signal.diff())
# # # pyplot.xlabel('Time(h)')
# # # pyplot.ylabel('Phases 2rd diff')
# # # pyplot.show()
# # # second_signal=first_signal.diff()
# # from pandas.plotting import autocorrelation_plot
# # autocorrelation_plot(signal)
# # pyplot.show()
# # from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
# # plot_acf(signal)
# # pyplot.show()
# # plot_pacf(signal, lags=25)
# # pyplot.show()
# # from pandas.plotting import autocorrelation_plot
# # autocorrelation_plot(first_signal[1:])
# # pyplot.show()
# # from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
# # plot_acf(first_signal[1:])
# # pyplot.show()
# # plot_pacf(first_signal[1:], lags=25)
# # pyplot.show()
#
#
#
#
#
# Raw_auto_RRI31=Raw_auto_RRI31_arr
# window_time_arr=rri_t_arr
# # Raw_auto_RRI31=Raw_auto_RRI31_arr[0:19640]
# # window_time_arr=rri_t_arr[0:19640]
#
#
# long_rhythm_value_arr=movingaverage(Raw_auto_RRI31,240*6)
# medium_rhythm_value_arr=movingaverage(Raw_auto_RRI31,240)
# medium_rhythm_value_arr_2=movingaverage(Raw_auto_RRI31,240*3)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31,240*6)
# medium_rhythm_value_arr_4=movingaverage(Raw_auto_RRI31,240*12)
# short_rhythm_value_arr_plot=movingaverage(Raw_auto_RRI31,240*24)
#
#
# # fig=pyplot.figure(figsize=(8,6))
# # # pyplot.plot(window_time_arr,Raw_auto_EEG,'grey',alpha=0.5,label='Raw autocorrelation')
# # # pyplot.plot(window_time_arr[20:],short_rhythm_value_arr_plot[20:],'grey',alpha=0.5,label='5min')
# # # pyplot.plot(window_time_arr[240:],medium_rhythm_value_arr[240:],'g',label='1 hour')
# # # pyplot.plot(window_time_arr[240*3:],medium_rhythm_value_arr_2[240*3:],'k',label='3 hour')
# # # pyplot.plot(window_time_arr[240*6:],medium_rhythm_value_arr_3[240*6:],'orange',label='6 hour')
# # # pyplot.plot(window_time_arr[240*12:],medium_rhythm_value_arr_4[240*12:],'b',label='12 hour')
# # # pyplot.plot(window_time_arr[5760:],long_rhythm_value_arr[5760:],'r',alpha=0.7,label='1 day')
# # pyplot.plot(window_time_arr,short_rhythm_value_arr_plot,'grey',alpha=0.5,label='5min')
# # pyplot.plot(window_time_arr,medium_rhythm_value_arr,'g',label='1 hour')
# # pyplot.plot(window_time_arr,medium_rhythm_value_arr_2,'k',label='3 hour')
# # pyplot.plot(window_time_arr,medium_rhythm_value_arr_3,'orange',label='6 hour')
# # pyplot.plot(window_time_arr,medium_rhythm_value_arr_4,'b',label='12 hour')
# # pyplot.plot(window_time_arr,long_rhythm_value_arr,'r',alpha=0.7,label='1 day')
# # pyplot.title('RRI autocorrelation in QLD0227',fontsize=23)
# # pyplot.xlabel('Time(hour)',fontsize=23)
# # pyplot.ylabel('Autocorrelation',fontsize=23)
# # pyplot.legend(loc='upper left',fontsize=10)
# # locs, labels = pyplot.xticks(fontsize=23)
# # locs, labels = pyplot.yticks(fontsize=23)
# # pyplot.tight_layout()
# # pyplot.annotate('',xy=(45.6175,np.max(short_rhythm_value_arr_plot)),xytext=(45.6175,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(64.7838889,np.max(short_rhythm_value_arr_plot)),xytext=(64.7838889,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(66.62,np.max(short_rhythm_value_arr_plot)),xytext=(66.62,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(74.0208333,np.max(short_rhythm_value_arr_plot)),xytext=(74.0208333,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(81.1208333,np.max(short_rhythm_value_arr_plot)),xytext=(81.1208333,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(114.798888,np.max(short_rhythm_value_arr_plot)),xytext=(114.798888,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.show()
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
#
# # from matplotlib import gridspec
# # fig = pyplot.figure(figsize=(12, 10))
# # gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
# # ax1=pyplot.subplot(gs[0])
# # ax1.set_title('RRI autocorrelation in QLD0227',fontsize=23)
# # ax1.set_xlabel('Time(hour)',fontsize=23)
# # locs, labels = pyplot.xticks(fontsize=23)
# # locs, labels = pyplot.yticks(fontsize=23)
# # ax1.plot(window_time_arr[240*6:],long_rhythm_value_arr[240*6:],'orange',alpha=0.7)
# # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax1.scatter(window_time_arr[10228],long_rhythm_value_arr[10228],c='k',s=40)
# # ax1.scatter(window_time_arr[14828],long_rhythm_value_arr[14828],c='k',s=40)
# # # ax1.scatter(window_time_arr[15269],long_rhythm_value_arr[15269],c='k',s=40)
# # ax1.scatter(window_time_arr[17045],long_rhythm_value_arr[17045],c='k',s=40)
# # ax1.scatter(window_time_arr[18749],long_rhythm_value_arr[18749],c='k',s=40)
# # # ax1.scatter(window_time_arr[26832],long_rhythm_value_arr[26832],c='k',s=40)
# # ax2=pyplot.subplot(gs[1])
# # ax2.set_title('Instantaneous Phase',fontsize=23)
# # ax2.set_xlabel('Time(hour)',fontsize=23)
# # locs, labels = pyplot.xticks(fontsize=23)
# # locs, labels = pyplot.yticks(fontsize=23)
# # pyplot.hlines(0,window_time_arr[240*6],window_time_arr[-1],'k','dashed')
# # # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax2.plot(window_time_arr[240*6:],phase_whole_value_long[240*6:],'k',alpha=0.5)
# # ax2.scatter(window_time_arr[10228],phase_whole_value_long[10228],c='k',s=40)
# # ax2.scatter(window_time_arr[14828],phase_whole_value_long[14828],c='k',s=40)
# # # ax2.scatter(window_time_arr[15269],phase_whole_value_long[15269],c='k',s=40)
# # ax2.scatter(window_time_arr[17045],phase_whole_value_long[17045],c='k',s=40)
# # ax2.scatter(window_time_arr[18749],phase_whole_value_long[18749],c='k',s=40)
# # # ax2.scatter(window_time_arr[26832],phase_whole_value_long[26832],c='k',s=40)
# # ax2.set_xlabel('Time(hour)',fontsize=23)
# # # locs, labels = pyplot.yticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi],['-$\pi$','Rising','0','Falling','$\pi$'],rotation='vertical',fontsize=23)
# # locs, labels = pyplot.yticks([-0.5*np.pi,-0.25*np.pi,0,0.25*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=23)
# # pyplot.tight_layout()
# # pyplot.tight_layout()
# # pyplot.show()
# # bins_number = 18
# # # bins = np.linspace(0, 2*np.pi, bins_number + 1)
# # bins = np.linspace(-np.pi, np.pi, bins_number + 1)
# # nRRIsauto, _, _ = pyplot.hist(phase_whole_value_long, bins)
# # nRRIsautosei, _, _ = pyplot.hist(seizure_phase_value_long, bins)
# # # nRRIsauto, _, _ = pyplot.hist(rolmean_long_RRIauto, bins)
# # # seizure_phase_value_long=[]
# # # for item in seizure_timing_index:
# # #     seizure_phase_value_long=seizure_phase_value_long+list(rolmean_long_RRIauto[item])
# # # nRRIsautosei, _, _ = pyplot.hist(seizure_phase_value_long, bins)
# # print(nRRIsauto)
# # print(nRRIsautosei)
# # width = 2*np.pi / bins_number
# # params = dict(projection='polar')
# # fig, ax = pyplot.subplots(subplot_kw=params)
# # ax.bar(bins[:bins_number], nRRIsauto/sum(nRRIsauto),width=width, color='grey',alpha=0.6,linewidth=2, fill=True,edgecolor='k')
# # pyplot.setp(ax.get_yticklabels(), color='k')
# # ax.set_title('phase histogram in RRI autocorrelation(QLD0227)',fontsize=13)
# # ax.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',])
# # pyplot.show()
# # params = dict(projection='polar')
# # fig, ax2 = pyplot.subplots(subplot_kw=params)
# # ax2.bar(bins[:bins_number], nRRIsautosei/sum(nRRIsautosei),width=width, color='g',edgecolor='k',linewidth=2)
# # ax2.set_title('seizure probability in RRI autocorrelation(QLD0227)',fontsize=13)
# # # ax2.set_yticks([0.0008,0.0012,0.0016,0.002])
# # ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',])
# # pyplot.show()
#
#
#
#
# # # # # signal=pd.DataFrame(phase_whole_value_long[240*6:])
# # # # #
# # # # # from statsmodels.tsa.stattools import adfuller
# # # # # result=adfuller(signal)
# # # # # print(result[0]);print(result[1])
# # # # #
# # # # # pyplot.figure()
# # # # # pyplot.plot(window_time_arr[240*6:],phase_whole_value_long[240*6:])
# # # # # pyplot.xlabel('Time(h)')
# # # # # pyplot.ylabel('Phases')
# # # # # pyplot.show()
# # # # #
# # # # # signal=phase_whole_value_long[240*6:]
# # # # # first_signal=pd.DataFrame(phase_whole_value_long[240*6:]).diff()
# # # # # pyplot.plot(window_time_arr[240*6:],first_signal)
# # # # # pyplot.xlabel('Time(h)')
# # # # # pyplot.ylabel('Phases diff')
# # # # # pyplot.show()
# # # # #
# # # # # # pyplot.plot(window_time_arr[240*6:],first_signal.diff())
# # # # # # pyplot.xlabel('Time(h)')
# # # # # # pyplot.ylabel('Phases 2rd diff')
# # # # # # pyplot.show()
# # # # # # second_signal=first_signal.diff()
# # # # #
# # # # # from pandas.plotting import autocorrelation_plot
# # # # # autocorrelation_plot(signal)
# # # # # pyplot.show()
# # # # # from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
# # # # # plot_acf(signal)
# # # # # pyplot.show()
# # # # # plot_pacf(signal, lags=25)
# # # # # pyplot.show()
# # # # # from pandas.plotting import autocorrelation_plot
# # # # # autocorrelation_plot(first_signal[1:])
# # # # # pyplot.show()
# # # # # from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
# # # # # plot_acf(first_signal[1:])
# # # # # pyplot.show()
# # # # # plot_pacf(first_signal[1:], lags=25)
# # # # # pyplot.show()
#
#
#
# # #### add circadian
# # a=np.where(t<7.630556+2.9975)
# # print(a)
# # print(t[1832]);print(t[1833])
# # t[0:1832]=t[0:1832]-2.9975+16.3684
# # t[1832:]=t[1832:]-7.630556-2.9975
# # print(t[1832]);print(t);print(type(t));print(t[0])
# #
# # time_feature_arr=[]
# # for i in range(len(t)):
# #     if t[i]>24:
# #         time_feature_arr.append(t[i] - (t[i] // 24) * 24)
# #     else:
# #         time_feature_arr.append(t[i])
# #
# #
# #
# #
# # seizure_time=[time_feature_arr[10228],time_feature_arr[14828],time_feature_arr[17045],time_feature_arr[18749],
# #               # time_feature_arr[26832]
# #               ]
# # print(seizure_time)
# #
# # bins_number = 18
# # bins = np.linspace(0, 24, bins_number + 1)
# # ntimes, _, _ = pyplot.hist(time_feature_arr[0:19640], bins)
# # ntimesei, _, _ = pyplot.hist(seizure_time, bins)
# # print(ntimes)
# # print(ntimesei)
# #
# #
# #
# #
# #
# # bins = np.linspace(0, 2*np.pi, bins_number + 1)
# # width = 2*np.pi / bins_number
# # params = dict(projection='polar')
# # fig, ax = pyplot.subplots(subplot_kw=params)
# # ax.bar(bins[:bins_number], ntimesei/sum(ntimes),width=width, color='grey',alpha=0.7,edgecolor='k',linewidth=2)
# # pyplot.setp(ax.get_yticklabels(), color='k')
# # ax.set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))
# # ax.set_xticklabels(['0 am','','','Night','','','6 am','','','Morning','','','12 am','','','Afternoon','','','18 pm','','','Evening','','','24 pm'],fontsize=16)
# # # locs, labels = pyplot.xticks([0,3,6,9,12,15,18,21,24],['0','Night','6','Morning','12','Afternoon','18','Evening','24'],fontsize=16)
# # locs, labels = pyplot.yticks([0.00004,0.00010,0.00016],['0.00004','0.00010','0.00016'],fontsize=16)
# # pyplot.show()
# #
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
# # ax.annotate("", xy=(3.08779306260312, 0.191788600875932), xytext=(0, 0),arrowprops=dict(arrowstyle="->",color='g',linewidth=2))
# # pyplot.show()
#
#
# # print(seizure_time)
# # bins_number = 24
# # bins = np.linspace(0, 24, bins_number + 1)
# # nEEGsvar, _, _ = pyplot.hist(time_feature_arr[0:19640], bins)
# # nEEGsvarsei, _, _ = pyplot.hist(seizure_time, bins)
# # width = 24 / bins_number
# # fig, ax = pyplot.subplots()
# # ax.bar(bins[:bins_number], nEEGsvarsei,width=width, color='darkblue',alpha=0.6,edgecolor='k',linewidth=2)
# # pyplot.setp(ax.get_yticklabels(), color='k',fontsize=12)
# # ax.set_xticks(np.linspace(0, 24, 24, endpoint=False))
# # ax.set_xticklabels(range(24))
# # ax.set_xlim([-0.5,24.5])
# # pyplot.axhline(-0.04,0,0.25,linewidth=6,c='k')
# # pyplot.axhline(-0.04,0.25,0.75,linewidth=6,c='k',alpha=0.3)
# # pyplot.axhline(-0.04,0.75,1,linewidth=6,c='k')
# # locs, labels = pyplot.xticks([0,3,6,9,12,15,18,21,24],['0 am','Night','6 am','Morning','12 am','Afternoon','18 pm','Evening','24 pm'],rotation=30,fontsize=12)
# # locs, labels = pyplot.yticks([0,1,2,3],['0','1','2','3'],fontsize=16)
# # ax.spines['right'].set_visible(False)
# # ax.spines['top'].set_visible(False)
# # ax.set_xlabel('Time',fontsize=14)
# # ax.set_ylabel('Number of seizures',fontsize=14)
# # pyplot.show()