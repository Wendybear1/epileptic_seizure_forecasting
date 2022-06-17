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





# ####QLD0098 EEG signals
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/QLD0098channels/EEGauto_QLD0098_15s_3h.csv',sep=',',header=None)
Raw_auto_EEG= csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/QLD0098channels/EEGvariance_QLD0098_15s_3h.csv',sep=',',header=None)
Raw_variance_EEG= csv_reader.values

Raw_variance_EEG_arr=[]
for item in Raw_variance_EEG:
    Raw_variance_EEG_arr.append(float(item))
Raw_auto_EEG_arr=[]
for item in Raw_auto_EEG:
    Raw_auto_EEG_arr.append(float(item))



t=np.linspace(2.9219+0.00416667,2.9219+0.00416667+0.00416667*(len(Raw_variance_EEG_arr)-1),len(Raw_variance_EEG_arr))
window_time_arr=t
print(len(t));print(t[0]);print(t[-1]);
print(t[-1]-t[0]);
print(t[19080]-t[0]);
print(t[-1]-t[19080]);




# pyplot.plot(t,Raw_variance_EEG_arr,'grey', alpha=0.3)
# pyplot.xlabel('Time(hours)',fontsize=10)
# pyplot.ylabel('Voltage ($\mathregular{v^2}$)',fontsize=13)
# pyplot.title('raw EEG variance in QLD0098',fontsize=13)
# pyplot.show()
var_arr=[]
for item in Raw_variance_EEG_arr:
    if item<1e-8:
        var_arr.append(item)
    else:
        var_arr.append(var_arr[-1])
Raw_variance_EEG_arr=var_arr
# pyplot.plot(t,Raw_variance_EEG_arr,'grey', alpha=0.3)
# pyplot.xlabel('Time(hours)',fontsize=10)
# pyplot.ylabel('Voltage ($\mathregular{v^2}$)',fontsize=13)
# pyplot.title('EEG variance in QLD0098',fontsize=13)
# pyplot.show()


# pyplot.plot(t,Raw_auto_EEG_arr,'grey',alpha=0.3)
# pyplot.xlabel('Time(hours)',fontsize=13)
# pyplot.title('raw EEG autocorrelation  in QLD0098',fontsize=13)
# pyplot.show()
value_arr=[]
for item in Raw_auto_EEG_arr:
    if item<500:
        value_arr.append(item)
    else:
        value_arr.append(value_arr[-1])
Raw_auto_EEG_arr=value_arr
# pyplot.plot(t,Raw_auto_EEG_arr,'grey',alpha=0.3)
# pyplot.xlabel('Time(hours)',fontsize=13)
# pyplot.title('EEG autocorrelation  in QLD0098',fontsize=13)
# pyplot.show()


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
    if window_time_arr[k] < 109.51 and window_time_arr[k + 1] >= 109.51:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 127.774122 and window_time_arr[k + 1] >= 127.774122:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 134.0119 and window_time_arr[k + 1] >= 134.0119:
        seizure_timing_index.append(k)
    if window_time_arr[k] < 152.234122 and window_time_arr[k + 1] >= 152.234122:
        seizure_timing_index.append(k)
print(seizure_timing_index)
index_ictal=seizure_timing_index

# seizure_timing_index=[]
# for k in range(len(window_time_arr)):
#     if window_time_arr[k]<16.1496768 and window_time_arr[k+1]>=16.1496768:
#         seizure_timing_index.append(k)
#     if window_time_arr[k]<37.4471778 and window_time_arr[k+1]>=37.4471778:
#         seizure_timing_index.append(k)
#     if window_time_arr[k]<71.4077334 and window_time_arr[k+1]>=71.4077334:
#         seizure_timing_index.append(k)
#     if window_time_arr[k]<80.733 and window_time_arr[k+1]>=80.733:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 127.3594 and window_time_arr[k + 1] >= 127.3594:
#         seizure_timing_index.append(k)
#     if window_time_arr[k]<131.777733 and window_time_arr[k+1]>=131.777733:
#         seizure_timing_index.append(k)
#     if window_time_arr[k]<152.2894 and window_time_arr[k+1]>=152.2894:
#         seizure_timing_index.append(k)
# print(seizure_timing_index)
# index_cluster=seizure_timing_index

x=[966, 6793, 12505, 18409, 25580, 29963, 31460, 35833]
y=[3173, 8285, 16435, 18673, 29863, 30924, 35847]
a=[966, 6793, 12505, 18409,  29963,  35833]
b=[3173, 8285, 16435, 18673,  30924, 35847]
list_arr=[]
for i in range(len(a)):
    c=range(a[i]+1,b[i]+1)
    list_arr=list_arr+list(c)
list_arr.append(25580);
list_arr.append(31460);
list_arr.append(29863);
index_cluster=list_arr
print(index_cluster)

duration=[2,3,2,2,2,2,2,2]

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
np.savetxt("C:/Users/wxiong/Documents/PHD/combine_features/QLD0098_tags.csv", x, delimiter=",", fmt='%s')



# # # # # ### EEG variance
# # Raw_variance_EEG=Raw_variance_EEG_arr[0:19080]
# # window_time_arr=t[0:19080]
# Raw_variance_EEG=Raw_variance_EEG_arr
# window_time_arr=t
# # print(window_time_arr[-1]-window_time_arr[0])
#
#
# long_rhythm_var_arr=movingaverage(Raw_variance_EEG,240*6)
# medium_rhythm_var_arr=movingaverage(Raw_variance_EEG,240)
# medium_rhythm_var_arr_2=movingaverage(Raw_variance_EEG,240*3)
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG,240*6)
# medium_rhythm_var_arr_4=movingaverage(Raw_variance_EEG,240*12)
# short_rhythm_var_arr_plot=movingaverage(Raw_variance_EEG,240*24)
#
#
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
# print(seizure_phase)
# SIvarlongseizure=sum(seizure_phase)/len(seizure_phase)
# print(SIvarlongseizure)
# var_phase=np.angle(var_trans)
# phase_long_EEGvar_arr=var_phase
# seizure_phase_var_long=[]
# for item in seizure_timing_index:
#     seizure_phase_var_long.append(phase_long_EEGvar_arr[item])
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
# # ax1.plot(window_time_arr[240*6:],long_rhythm_var_arr[240*6:],'orange',alpha=0.7)
# # ax1.set_title('EEG variance in QLD0098',fontsize=15)
# # ax1.set_xlabel('Time(hour)',fontsize=15)
# # ax1.set_ylabel('Voltage ($\mathregular{v^2}$)',fontsize=15)
# # locs, labels = pyplot.xticks(fontsize=15)
# # locs, labels = pyplot.yticks(fontsize=15)
# # ax1.scatter(window_time_arr[966],long_rhythm_var_arr[966],c='k')
# # ax1.scatter(window_time_arr[6793],long_rhythm_var_arr[6793],c='k')
# # ax1.scatter(window_time_arr[12505],long_rhythm_var_arr[12505],c='k')
# # ax1.scatter(window_time_arr[18409],long_rhythm_var_arr[18409],c='k')
# # # ax1.scatter(window_time_arr[25580],long_rhythm_var_arr[25580],c='k')
# # # ax1.scatter(window_time_arr[29963],long_rhythm_var_arr[29963],c='k')
# # # ax1.scatter(window_time_arr[31460],long_rhythm_var_arr[31460],c='k')
# # # ax1.scatter(window_time_arr[35833],long_rhythm_var_arr[35833],c='k')
# # pyplot.xlim(window_time_arr[240*6],window_time_arr[-1])
# # ax2=pyplot.subplot(gs[1])
# # ax2.set_xlabel('Time(hour)',fontsize=15)
# # ax2.set_title('Instantaneous Phase',fontsize=15)
# # pyplot.xlim(window_time_arr[240*6],window_time_arr[-1])
# # ax2.plot(window_time_arr[240*6:],phase_long_EEGvar_arr[240*6:],c='k',alpha=0.5,label='instantaneous phase')
# # pyplot.hlines(0,window_time_arr[0],window_time_arr[-1],'k','dashed')
# # ax2.set_xlabel('Time(hour)',fontsize=15)
# # locs, labels = pyplot.xticks(fontsize=15)
# # ax2.scatter(window_time_arr[966],phase_long_EEGvar_arr[966],c='k')
# # ax2.scatter(window_time_arr[6793],phase_long_EEGvar_arr[6793],c='k')
# # ax2.scatter(window_time_arr[12505],phase_long_EEGvar_arr[12505],c='k')
# # ax2.scatter(window_time_arr[18409],phase_long_EEGvar_arr[18409],c='k')
# # # ax2.scatter(window_time_arr[25580],phase_long_EEGvar_arr[25580],c='k')
# # # ax2.scatter(window_time_arr[29963],phase_long_EEGvar_arr[29963],c='k')
# # # ax2.scatter(window_time_arr[31460],phase_long_EEGvar_arr[31460],c='k')
# # # ax2.scatter(window_time_arr[35833],phase_long_EEGvar_arr[35833],c='k')
# # # locs, labels = pyplot.yticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi],['-$\pi$','Rising','0','Falling','$\pi$'],rotation='vertical',fontsize=15)
# # locs, labels = pyplot.yticks([-0.5*np.pi,-0.25*np.pi,0,0.25*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=15)
# # pyplot.tight_layout()
# # pyplot.show()
# # bins_number = 18
# # bins = np.linspace(0, 2*np.pi, bins_number + 1)
# # bins = np.linspace(-np.pi, np.pi, bins_number + 1)
# # nEEGsvar, _, _ = pyplot.hist(phase_long_EEGvar_arr, bins)
# # nEEGsvarsei, _, _ = pyplot.hist(seizure_phase_var_long, bins)
# # # nEEGsvar, _, _ = pyplot.hist(rolmean_long_EEGvar, bins)
# # # seizure_phase_var_short=[]
# # # for item in seizure_timing_index:
# # #     seizure_phase_var_short=seizure_phase_var_short+list(rolmean_long_EEGvar[item])
# # # nEEGsvarsei, _, _ = pyplot.hist(seizure_phase_var_short, bins)
# # print(nEEGsvar)
# # print(nEEGsvarsei)
# # width = 2*np.pi / bins_number
# # # params = dict(projection='polar')
# # # fig, ax = pyplot.subplots(subplot_kw=params)
# # # ax.bar(bins[:bins_number], nEEGsvar/sum(nEEGsvar),width=width, color='grey',alpha=0.6,linewidth=2, fill=True)
# # # pyplot.setp(ax.get_yticklabels(), color='k')
# # # ax.set_title('Phase histogram in long EEG variance(QLD0098)',fontsize=13)
# # # pyplot.show()
# # params = dict(projection='polar')
# # fig, ax2 = pyplot.subplots(subplot_kw=params)
# # ax2.bar(bins[:bins_number], nEEGsvarsei/sum(nEEGsvarsei),width=width, color='r',alpha=0.7,linewidth=2,edgecolor='k')
# # # ax2.set_yticks([0.0005,0.001,0.0015,0.002])
# # ax2.set_title('seizure probability in EEG variance(QLD0098)',fontsize=13)
# # # ax2.set_rlim([0,0.002])
# # pyplot.show()
#
#
#
#
#
# Raw_auto_EEG=Raw_auto_EEG_arr
# window_time_arr=t
# # Raw_auto_EEG=Raw_auto_EEG_arr[0:19080]
# # window_time_arr=t[0:19080]
#
# long_rhythm_value_arr=movingaverage(Raw_auto_EEG,240*6)
# medium_rhythm_value_arr=movingaverage(Raw_auto_EEG,240)
# medium_rhythm_value_arr_2=movingaverage(Raw_auto_EEG,240*3)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG,240*6)
# medium_rhythm_value_arr_4=movingaverage(Raw_auto_EEG,240*12)
# short_rhythm_value_arr_plot=movingaverage(Raw_auto_EEG,240*24)
#
#
# # # fig=pyplot.figure(figsize=(8,6))
# # # # pyplot.plot(window_time_arr,Raw_auto_EEG,'grey',alpha=0.5,label='Raw autocorrelation')
# # # # pyplot.plot(window_time_arr[20:],short_rhythm_value_arr_plot[20:],'grey',alpha=0.5,label='5min')
# # # # pyplot.plot(window_time_arr[240:],medium_rhythm_value_arr[240:],'g',label='1 hour')
# # # # pyplot.plot(window_time_arr[240*3:],medium_rhythm_value_arr_2[240*3:],'k',label='3 hour')
# # # # pyplot.plot(window_time_arr[240*6:],medium_rhythm_value_arr_3[240*6:],'orange',label='6 hour')
# # # # pyplot.plot(window_time_arr[240*12:],medium_rhythm_value_arr_4[240*12:],'b',label='12 hour')
# # # # pyplot.plot(window_time_arr[5760:],long_rhythm_value_arr[5760:],'r',alpha=0.7,label='1 day')
# # # pyplot.plot(window_time_arr,short_rhythm_value_arr_plot,'grey',alpha=0.5,label='5min')
# # # pyplot.plot(window_time_arr,medium_rhythm_value_arr,'g',label='1 hour')
# # # pyplot.plot(window_time_arr,medium_rhythm_value_arr_2,'k',label='3 hour')
# # # pyplot.plot(window_time_arr,medium_rhythm_value_arr_3,'orange',label='6 hour')
# # # pyplot.plot(window_time_arr,medium_rhythm_value_arr_4,'b',label='12 hour')
# # # pyplot.plot(window_time_arr,long_rhythm_value_arr,'r',alpha=0.7,label='1 day')
# # # pyplot.annotate('',xy=(6.9544,np.max(short_rhythm_value_arr_plot)),xytext=(6.9544,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(31.2324556,np.max(short_rhythm_value_arr_plot)),xytext=(31.2324556,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(55.0324556,np.max(short_rhythm_value_arr_plot)),xytext=(55.0324556,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(79.63162,np.max(short_rhythm_value_arr_plot)),xytext=(79.63162,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(109.51,np.max(short_rhythm_value_arr_plot)),xytext=(109.51,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(127.774122,np.max(short_rhythm_value_arr_plot)),xytext=(127.774122,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(134.0119,np.max(short_rhythm_value_arr_plot)),xytext=(134.0119,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(152.234122,np.max(short_rhythm_value_arr_plot)),xytext=(152.234122,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.title('EEG autocorrelation in QLD0098',fontsize=15)
# # # pyplot.xlabel('Time(hour)',fontsize=15)
# # # pyplot.ylabel('Autocorrelation',fontsize=15)
# # # pyplot.legend(loc='upper left',fontsize=10)
# # # locs, labels = pyplot.xticks(fontsize=15)
# # # locs, labels = pyplot.yticks(fontsize=15)
# # # pyplot.tight_layout()
# # # pyplot.show()
#
#
#
#
# long_rhythm_value_arr=long_rhythm_value_arr
# var_trans=hilbert(long_rhythm_value_arr)
# value_trans_nomal=[]
# for m in var_trans:
#     value_trans_nomal.append(m/abs(m))
# SIvaluelong=sum(value_trans_nomal)/len(value_trans_nomal)
# print(SIvaluelong)
# seizure_phase=[]
# for item in seizure_timing_index:
#     seizure_phase.append(value_trans_nomal[item])
# SIvaluelongseizure=sum(seizure_phase)/len(seizure_phase)
# print(SIvaluelongseizure)
# var_phase=np.angle(var_trans)
# phase_long_EEGauto_arr=var_phase
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
# # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax2.set_title('EEG autocorrelation in QLD0098',fontsize=15)
# # ax2.set_xlabel('Time(hour)',fontsize=15)
# # locs, labels = pyplot.xticks(fontsize=15)
# # locs, labels = pyplot.yticks(fontsize=15)
# # ax2.scatter(window_time_arr[966],long_rhythm_value_arr[966],c='k')
# # ax2.scatter(window_time_arr[6793],long_rhythm_value_arr[6793],c='k')
# # ax2.scatter(window_time_arr[12505],long_rhythm_value_arr[12505],c='k')
# # ax2.scatter(window_time_arr[18409],long_rhythm_value_arr[18409],c='k')
# # # ax2.scatter(window_time_arr[25580],long_rhythm_value_arr[25580],c='k')
# # # ax2.scatter(window_time_arr[29963],long_rhythm_value_arr[29963],c='k')
# # # ax2.scatter(window_time_arr[31460],long_rhythm_value_arr[31460],c='k')
# # # ax2.scatter(window_time_arr[35833],long_rhythm_value_arr[35833],c='k')
# # ax3=pyplot.subplot(gs[1])
# # ax3.set_xlabel('Time(hour)',fontsize=15)
# # ax3.set_title('Instantaneous Phase',fontsize=15)
# # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # locs, labels = pyplot.xticks(fontsize=15)
# # locs, labels = pyplot.yticks(fontsize=15)
# # ax3.plot(window_time_arr[240*6:],phase_long_EEGauto_arr[240*6:],'k',alpha=0.5,label='instantaneous phase')
# # pyplot.hlines(0,window_time_arr[0],window_time_arr[-1],'k','dashed')
# # ax3.scatter(window_time_arr[966],phase_long_EEGauto_arr[966],c='k')
# # ax3.scatter(window_time_arr[6793],phase_long_EEGauto_arr[6793],c='k')
# # ax3.scatter(window_time_arr[12505],phase_long_EEGauto_arr[12505],c='k')
# # ax3.scatter(window_time_arr[18409],phase_long_EEGauto_arr[18409],c='k')
# # # ax3.scatter(window_time_arr[25580],phase_long_EEGauto_arr[25580],c='k')
# # # ax3.scatter(window_time_arr[29963],phase_long_EEGauto_arr[29963],c='k')
# # # ax3.scatter(window_time_arr[31460],phase_long_EEGauto_arr[31460],c='k')
# # # ax3.scatter(window_time_arr[35833],phase_long_EEGauto_arr[35833],c='k')
# # ax3.set_xlabel('Time(hour)',fontsize=15)
# # # locs, labels = pyplot.yticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi],['-$\pi$','Rising','0','Falling','$\pi$'],rotation='vertical',fontsize=15)
# # locs, labels = pyplot.yticks([-0.5*np.pi,-0.25*np.pi,0,0.25*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=15)
# # pyplot.tight_layout()
# # pyplot.show()
# # bins_number = 18
# # # bins = np.linspace(0, 2*np.pi, bins_number + 1)
# # bins = np.linspace(-np.pi, np.pi, bins_number + 1)
# # print(bins)
# # nEEGsauto, _, _ = pyplot.hist(phase_long_EEGauto_arr, bins)
# # nEEGsautosei, _, _ = pyplot.hist(seizure_phase_value_long, bins)
# # # nEEGsauto, _, _ = pyplot.hist(rolmean_long_EEGauto, bins)
# # # seizure_phase_auto_long=[]
# # # for item in seizure_timing_index:
# # #     seizure_phase_auto_long=seizure_phase_auto_long+list(rolmean_long_EEGauto[item])
# # # nEEGsautosei, _, _ = pyplot.hist(seizure_phase_auto_long, bins)
# # print(nEEGsauto)
# # print(nEEGsautosei)
# # # width = 2*np.pi / bins_number
# # # params = dict(projection='polar')
# # # fig, ax = pyplot.subplots(subplot_kw=params)
# # # ax.bar(bins[:bins_number], nEEGsauto/sum(nEEGsauto),width=width, color='grey',alpha=0.6,linewidth=2, fill=True)
# # # pyplot.setp(ax.get_yticklabels(), color='k')
# # # ax.set_title('Phase histogram in long EEG autocorrelation(QLD0098)',fontsize=13)
# # # # ax.set_yticks([0.02,0.04,0.06,0.08])
# # # pyplot.show()
# # # params = dict(projection='polar')
# # # fig, ax2 = pyplot.subplots(subplot_kw=params)
# # # ax2.bar(bins[:bins_number], nEEGsautosei/sum(nEEGsautosei),width=width, color='orange',alpha=0.9,linewidth=2,edgecolor='k')
# # # # ax2.set_yticks([0.0005,0.001,0.0015,0.002])
# # # ax2.set_title('seizure probability in long EEG autocorrelation(QLD0098)',fontsize=13)
# # # # ax2.set_rlim([0,0.002])
# # # pyplot.show()
#
#
#
#
#
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_ch31_timewindowarr_QLD0098_15s_3h.csv',sep=',',header=None)
# rri_t= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_ch31_rawvariance_QLD0098_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_ch31_rawauto_QLD0098_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
#
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/ch2-1/RRI_ch21_timewindowarr_QLD0098_15s_3h.csv',sep=',',header=None)
# # rri_t= csv_reader.values
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/ch2-1/RRI_ch21_rawvariance_QLD0098_15s_3h.csv',sep=',',header=None)
# # RRI_var= csv_reader.values
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/ch2-1/RRI_ch21_rawauto_QLD0098_15s_3h.csv',sep=',',header=None)
# # Raw_auto_RRI31= csv_reader.values
#
#
# # t_arr=[]
# # for item in t:
# #     t_arr.append(float(item/3600))
# # Raw_RRI31_arr=[]
# # for item in Raw_RRI31:
# #     Raw_RRI31_arr.append(float(item))
#
# t_window_arr=[]
# for item in rri_t:
#     t_window_arr.append(2.9219+float(item))
# print(t_window_arr[0])
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
#
# # pyplot.plot(t_window_arr,Raw_variance_RRI31_arr,'grey')
# # pyplot.xlabel('Time(hour)',fontsize=13)
# # pyplot.ylabel('$Second^2$',fontsize=13)
# # pyplot.title('RRI variance in QLD0098',fontsize=13)
# # pyplot.show()
# #
# # pyplot.plot(t_window_arr,Raw_auto_RRI31_arr,'grey')
# # pyplot.xlabel('Time(hour)',fontsize=13)
# # pyplot.title('RRI autocorrelation in QLD0098',fontsize=13)
# # pyplot.show()
#
#
# # rri_t_arr=t_window_arr
# # seizure_timing_index=[]
# # for k in range(len(rri_t_arr)):
# #     if rri_t_arr[k]<6.9544 and rri_t_arr[k+1]>=6.9544:
# #         seizure_timing_index.append(k)
# #     if rri_t_arr[k]<31.2324556 and rri_t_arr[k+1]>=31.2324556:
# #         seizure_timing_index.append(k)
# #     if rri_t_arr[k]<55.0324556 and rri_t_arr[k+1]>=55.0324556:
# #         seizure_timing_index.append(k)
# #     if rri_t_arr[k]<79.63162 and rri_t_arr[k+1]>=79.63162:
# #         seizure_timing_index.append(k)
# #     if rri_t_arr[k]<109.51 and rri_t_arr[k+1]>=109.51:
# #         seizure_timing_index.append(k)
# #     if rri_t_arr[k]<127.774122 and rri_t_arr[k+1]>=127.774122:
# #         seizure_timing_index.append(k)
# #     if rri_t_arr[k]<134.0119 and rri_t_arr[k+1]>=134.0119:
# #         seizure_timing_index.append(k)
# #     if rri_t_arr[k]<152.234122 and rri_t_arr[k+1]>=152.234122:
# #         seizure_timing_index.append(k)
# # print(seizure_timing_index)
#
#
# # # from matplotlib import gridspec
# # # fig = pyplot.figure(figsize=(12, 8))
# # # gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
# # # ax1=pyplot.subplot(gs[0])
# # # pyplot.plot(rri_t_arr,Raw_RRI31_arr,'grey',alpha=0.4)
# # # pyplot.plot(rri_t_arr,rolmean_rri,'b',alpha=0.5)
# # # pyplot.title('RR intervals in QLD0098',fontsize=15)
# # # pyplot.scatter(6.9544,rolmean_rri[25553],c='k')
# # # pyplot.scatter(31.2324556,rolmean_rri[118307],c='k')
# # # pyplot.scatter(55.0324556,rolmean_rri[209948],c='k')
# # # pyplot.scatter(79.63162,rolmean_rri[306192],c='k')
# # # pyplot.scatter(109.51,rolmean_rri[418014],c='k')
# # # pyplot.scatter(127.774122,rolmean_rri[494992],c='k')
# # # pyplot.scatter(134.0119,rolmean_rri[514527],c='k')
# # # pyplot.scatter(152.234122,rolmean_rri[591526],c='k')
# # # pyplot.xlabel('Time(hour)',fontsize=15)
# # # pyplot.ylabel('Second',fontsize=15)
# # # pyplot.tight_layout()
# # # locs, labels = pyplot.xticks(fontsize=15)
# # # locs, labels = pyplot.yticks(fontsize=15)
# # # ax2=pyplot.subplot(gs[1])
# # # ax2.set_xlabel('Time(hour)',fontsize=15)
# # # pyplot.plot(rri_t_arr[23000:591600],rolmean_rri[23000:591600],'b',alpha=0.5)
# # # pyplot.title('RR intervals in QLD0098',fontsize=15)
# # # pyplot.scatter(6.9544,rolmean_rri[25553],c='k')
# # # pyplot.scatter(31.2324556,rolmean_rri[118307],c='k')
# # # pyplot.scatter(55.0324556,rolmean_rri[209948],c='k')
# # # pyplot.scatter(79.63162,rolmean_rri[306192],c='k')
# # # pyplot.scatter(109.51,rolmean_rri[418014],c='k')
# # # pyplot.scatter(127.774122,rolmean_rri[494992],c='k')
# # # pyplot.scatter(134.0119,rolmean_rri[514527],c='k')
# # # pyplot.scatter(152.234122,rolmean_rri[591526],c='k')
# # # pyplot.show()
# #
# # # # pyplot.figure(figsize=(8,3))
# # # # pyplot.plot(t_arr[297000:306500],Raw_RRI31_arr[297000:306500],'k')
# # # # pyplot.vlines(79.63162,0.4,1.45,'r')
# # # # pyplot.ylabel('RR intervals (s)',fontsize=15)
# # # # pyplot.xlabel('Time(h)',fontsize=15)
# # # # pyplot.xticks(fontsize=15)
# # # # pyplot.yticks(fontsize=15)
# # # # pyplot.show()
# # # # print(t_arr[297000],t_arr[297013])
# # # # print(t_arr[305100],t_arr[305113])
# # # # ### 297413 or 297013
# # # # pyplot.plot(t_arr[297000:297013],Raw_RRI31_arr[297000:297013],'k')
# # # # pyplot.ylabel('RR intervals (s)',fontsize=15)
# # # # pyplot.xlabel('Time(h)',fontsize=15)
# # # # pyplot.xticks(fontsize=12)
# # # # pyplot.yticks(fontsize=15)
# # # # pyplot.show()
# # # # pyplot.plot(t_arr[305100:305113],Raw_RRI31_arr[305100:305113],'k')
# # # # pyplot.ylabel('RR intervals (s)',fontsize=15)
# # # # pyplot.xlabel('Time(h)',fontsize=15)
# # # # pyplot.xticks(fontsize=12)
# # # # pyplot.yticks(fontsize=15)
# # # # pyplot.show()
# # # #
# # # # x = Raw_RRI31_arr[297000:297013]
# # # # y = x - np.mean(x)
# # # # target_signal_std = np.std(x)
# # # # target_signal_var=target_signal_std**2
# # # # print(target_signal_var)
# # # # y = y / target_signal_std
# # # # R = np.correlate(y, y, mode='full')/len(y)
# # # # pyplot.plot(R,'k')
# # # # pyplot.plot([11.6,12.4],[R[12]/2,R[12]/2],'r',label='width of half max=1')
# # # # pyplot.xticks(fontsize=12)
# # # # pyplot.yticks(fontsize=15)
# # # # pyplot.legend(fontsize=12)
# # # # pyplot.show()
# # # # value_arr=[]
# # # # for k in range(len(R)):
# # # #     if R[k] < 0.5 * R.max():
# # # #         k = k + 1
# # # #     else:
# # # #         indice1 = k
# # # #         indice2 = len(R) - indice1
# # # #         value = indice2 - indice1
# # # #         value_arr.append(value)
# # # #         break
# # # # print(indice1)
# # # # print(value_arr)
# # # # print(np.corrcoef(np.array([y[:-1], y[1:]])))
# # # #
# # # # x = Raw_RRI31_arr[305100:305113]
# # # # y = x - np.mean(x)
# # # # target_signal_std = np.std(x)
# # # # target_signal_var=target_signal_std**2
# # # # print(target_signal_var)
# # # # y = y / target_signal_std
# # # # R = np.correlate(y, y, mode='full')/len(y)
# # # # pyplot.plot(R,'k')
# # # # pyplot.plot([11,13],[R[11],R[11]],'r',label='width of half max=3')
# # # # # pyplot.xlim(53800,54200)
# # # # pyplot.xticks(fontsize=12)
# # # # pyplot.yticks(fontsize=15)
# # # # pyplot.legend(fontsize=12)
# # # # pyplot.show()
# # # # value_arr=[]
# # # # for k in range(len(R)):
# # # #     if R[k] < 0.5 * R.max():
# # # #         k = k + 1
# # # #     else:
# # # #         indice1 = k
# # # #         indice2 = len(R) - indice1
# # # #         value = indice2 - indice1
# # # #         value_arr.append(value)
# # # #         break
# # # # print(indice1)
# # # # print(value_arr)
# # # # print(np.corrcoef(np.array([y[:-1], y[1:]])))
# # # #
# # # # from pandas import DataFrame
# # # # def lag_plot_my(series, lag=1, ax=None, **kwds):
# # # #     # workaround because `c='b'` is hardcoded in matplotlibs scatter method
# # # #     import matplotlib.pyplot as plt
# # # #     kwds.setdefault("c", plt.rcParams["patch.facecolor"])
# # # #     data = series.values
# # # #     y1 = data[:-lag]
# # # #     y2 = data[lag:]
# # # #     if ax is None:
# # # #         ax = plt.gca()
# # # #     ax.set_xlabel("RRI(t)",fontsize=15)
# # # #     ax.set_ylabel("RRI(t + {lag})".format(lag=lag),fontsize=15)
# # # #     # ax.set_xlim(0.91, 1.12)
# # # #     # ax.set_ylim(0.91, 1.12)
# # # #     plt.setp(ax.get_xticklabels(), fontsize=15)
# # # #     plt.setp(ax.get_yticklabels(), fontsize=15)
# # # #     ax.scatter(y1, y2,c='k')
# # # #     return ax
# # # #
# # # # signal_1=Raw_RRI31_arr[297000:297013]
# # # # df=pd.DataFrame(np.array(signal_1))
# # # # series = pd.DataFrame(df.values)
# # # # lag_plot_my(series,1)
# # # # print(lag_plot_my(series))
# # # # pyplot.show()
# # # # signal_2=Raw_RRI31_arr[305100:305113]
# # # # df=pd.DataFrame(np.array(signal_2))
# # # # series = pd.DataFrame(df.values)
# # # # lag_plot_my(series,1)
# # # # print(lag_plot_my(series))
# # # # pyplot.show()
# # # # from pandas import concat
# # # # ch_notch=signal_1
# # # # df=pd.DataFrame(np.array(ch_notch))
# # # # values = pd.DataFrame(df.values)
# # # # dataframe = concat([values.shift(1), values], axis=1)
# # # # dataframe.columns = ['t-1', 't+1']
# # # # result = dataframe.corr()
# # # # print(result)
# # # # result = pd.Series(ch_notch).autocorr()
# # # # print(result)
# # # # ch_notch=signal_2
# # # # df=pd.DataFrame(np.array(ch_notch))
# # # # values = pd.DataFrame(df.values)
# # # # dataframe = concat([values.shift(1), values], axis=1)
# # # # dataframe.columns = ['t-1', 't+1',]
# # # # result = dataframe.corr()
# # # # print(result)
# # # # result = pd.Series(ch_notch).autocorr()
# # # # print(result)
# #
# #
# #
# # # # ### RRI variance
# # window_time_arr=t_window_arr[0:19080]
# # Raw_variance_RRI31=Raw_variance_RRI31_arr[0:19080]
# window_time_arr=t_window_arr
# Raw_variance_RRI31=Raw_variance_RRI31_arr
#
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
# # pyplot.title('RRI variance in QLD0098',fontsize=15)
# # pyplot.xlabel('Time(hour)',fontsize=15)
# # pyplot.ylabel('Second ($\mathregular{s^2}$)',fontsize=15)
# # pyplot.legend(loc='upper left',fontsize=10)
# # locs, labels = pyplot.xticks(fontsize=15)
# # locs, labels = pyplot.yticks(fontsize=15)
# # pyplot.tight_layout()
# # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # pyplot.show()
#
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
#
# # from matplotlib import gridspec
# # fig = pyplot.figure(figsize=(12, 10))
# # gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
# # ax1=pyplot.subplot(gs[0])
# # ax1.plot(window_time_arr,long_rhythm_var_arr,'orange',alpha=0.7)
# # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax1.set_title('RRI variance in QLD0098',fontsize=15)
# # ax1.set_xlabel('Time(hour)',fontsize=15)
# # ax1.set_ylabel('Second($\mathregular{s^2}$)',fontsize=15)
# # locs, labels = pyplot.xticks(fontsize=15)
# # locs, labels = pyplot.yticks(fontsize=15)
# # ax1.scatter(window_time_arr[966],long_rhythm_var_arr[966],c='k',s=40)
# # ax1.scatter(window_time_arr[6793],long_rhythm_var_arr[6793],c='k',s=40)
# # ax1.scatter(window_time_arr[12505],long_rhythm_var_arr[12505],c='k',s=40)
# # ax1.scatter(window_time_arr[18409],long_rhythm_var_arr[18409],c='k',s=40)
# # # ax1.scatter(window_time_arr[25580],long_rhythm_var_arr[25580],c='k',s=40)
# # # ax1.scatter(window_time_arr[29963],long_rhythm_var_arr[29963],c='k',s=40)
# # # ax1.scatter(window_time_arr[31460],long_rhythm_var_arr[31460],c='k',s=40)
# # # ax1.scatter(window_time_arr[35833],long_rhythm_var_arr[35833],c='k',s=40)
# # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax2=pyplot.subplot(gs[1])
# # ax2.set_xlabel('Time(hour)',fontsize=15)
# # ax2.set_title('Instantaneous Phase',fontsize=15)
# # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax2.plot(window_time_arr,phase_whole_long,'k',alpha=0.5)
# # pyplot.hlines(0,window_time_arr[0],window_time_arr[-1],'k','dashed')
# # ax2.scatter(window_time_arr[966],phase_whole_long[966],c='k',s=40)
# # ax2.scatter(window_time_arr[6793],phase_whole_long[6793],c='k',s=40)
# # ax2.scatter(window_time_arr[12505],phase_whole_long[12505],c='k',s=40)
# # ax2.scatter(window_time_arr[18409],phase_whole_long[18409],c='k',s=40)
# # # ax2.scatter(window_time_arr[25580],phase_whole_long[25580],c='k',s=40)
# # # ax2.scatter(window_time_arr[29963],phase_whole_long[29963],c='k',s=40)
# # # ax2.scatter(window_time_arr[31460],phase_whole_long[31460],c='k',s=40)
# # # ax2.scatter(window_time_arr[35833],phase_whole_long[35833],c='k',s=40)
# # ax2.set_xlabel('Time(hour)',fontsize=15)
# # locs, labels = pyplot.xticks(fontsize=15)
# # locs, labels = pyplot.yticks(fontsize=15)
# # # locs, labels = pyplot.yticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi],['-$\pi$','Rising','0','Falling','$\pi$'],rotation='vertical',fontsize=15)
# # locs, labels = pyplot.yticks([-0.5*np.pi,-0.25*np.pi,0,0.25*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=15)
# # pyplot.tight_layout()
# # pyplot.show()
# # bins_number = 18
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
# # # width = 2*np.pi / bins_number
# # # params = dict(projection='polar')
# # # fig, ax = pyplot.subplots(subplot_kw=params)
# # # ax.bar(bins[:bins_number], nRRIsvar/sum(nRRIsvar),width=width, color='grey',alpha=0.6,linewidth=2, fill=True)
# # # pyplot.setp(ax.get_yticklabels(), color='k', alpha=0.7)
# # # ax.set_title('Phase histogram in long RRI variance(QLD0098)',fontsize=13)
# # # pyplot.show()
# # # params = dict(projection='polar')
# # # fig, ax2 = pyplot.subplots(subplot_kw=params)
# # # ax2.bar(bins[:bins_number], nRRIsvarsei/sum(nRRIsvarsei),width=width, color='C0',edgecolor='k',linewidth=2)
# # # ax2.set_title('seizure probability in long RRI variance(QLD0098)',fontsize=13)
# # # # ax2.set_yticks([0.0008,0.0012,0.0016,0.002])
# # # # ax2.set_rlim([0,0.002])
# # # pyplot.show()
#
#
#
#
#
#
#
#
# Raw_auto_RRI31=Raw_auto_RRI31_arr
# # Raw_auto_RRI31=Raw_auto_RRI31_arr[0:19080]
#
# long_rhythm_value_arr=movingaverage(Raw_auto_RRI31,240*6)
# medium_rhythm_value_arr=movingaverage(Raw_auto_RRI31,240)
# medium_rhythm_value_arr_2=movingaverage(Raw_auto_RRI31,240*3)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_RRI31,240*6)
# medium_rhythm_value_arr_4=movingaverage(Raw_auto_RRI31,240*12)
# short_rhythm_value_arr_plot=movingaverage(Raw_auto_RRI31,240*24)
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
# # pyplot.title('RRI autocorrelation in QLD0098',fontsize=15)
# # pyplot.xlabel('Time(hour)',fontsize=15)
# # pyplot.ylabel('Autocorrelation',fontsize=15)
# # pyplot.legend(loc='upper left',fontsize=10)
# # locs, labels = pyplot.xticks(fontsize=15)
# # locs, labels = pyplot.yticks(fontsize=15)
# # pyplot.tight_layout()
# # pyplot.show()
#
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
# value_phase=np.unwrap(np.angle(value_trans))
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
# # ax1.set_title('RRI autocorrelation in QLD0098',fontsize=15)
# # ax1.set_xlabel('Time(hour)',fontsize=15)
# # locs, labels = pyplot.xticks(fontsize=15)
# # locs, labels = pyplot.yticks(fontsize=15)
# # ax1.plot(window_time_arr[240*6:],long_rhythm_value_arr[240*6:],'orange',alpha=0.7)
# # # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax1.scatter(window_time_arr[966],long_rhythm_value_arr[966],c='k',s=40)
# # ax1.scatter(window_time_arr[6793],long_rhythm_value_arr[6793],c='k',s=40)
# # ax1.scatter(window_time_arr[12505],long_rhythm_value_arr[12505],c='k',s=40)
# # ax1.scatter(window_time_arr[18409],long_rhythm_value_arr[18409],c='k',s=40)
# # # ax1.scatter(window_time_arr[25580],long_rhythm_value_arr[25580],c='k',s=40)
# # # ax1.scatter(window_time_arr[29963],long_rhythm_value_arr[29963],c='k',s=40)
# # # ax1.scatter(window_time_arr[31460],long_rhythm_value_arr[31460],c='k',s=40)
# # # ax1.scatter(window_time_arr[35833],long_rhythm_value_arr[35833],c='k',s=40)
# # # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax2=pyplot.subplot(gs[1])
# # ax2.set_title('Instantaneous Phase',fontsize=15)
# # ax2.set_xlabel('Time(hour)',fontsize=15)
# # locs, labels = pyplot.xticks(fontsize=15)
# # locs, labels = pyplot.yticks(fontsize=15)
# # # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax2.plot(window_time_arr[240*6:],phase_whole_value_long[240*6:],'k',alpha=0.5)
# # pyplot.hlines(0,window_time_arr[240*6],window_time_arr[-1],'k','dashed')
# # # ax2.plot(window_time_arr,rolmean_long_RRIauto,'b',alpha=0.7)
# # ax2.scatter(window_time_arr[966],phase_whole_value_long[966],c='k',s=40)
# # ax2.scatter(window_time_arr[6793],phase_whole_value_long[6793],c='k',s=40)
# # ax2.scatter(window_time_arr[12505],phase_whole_value_long[12505],c='k',s=40)
# # ax2.scatter(window_time_arr[18409],phase_whole_value_long[18409],c='k',s=40)
# # # ax2.scatter(window_time_arr[25580],phase_whole_value_long[25580],c='k',s=40)
# # # ax2.scatter(window_time_arr[29963],phase_whole_value_long[29963],c='k',s=40)
# # # ax2.scatter(window_time_arr[31460],phase_whole_value_long[31460],c='k',s=40)
# # # ax2.scatter(window_time_arr[35833],phase_whole_value_long[35833],c='k',s=40)
# # ax2.set_xlabel('Time(hour)',fontsize=15)
# # # locs, labels = pyplot.yticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi],['-$\pi$','Rising','0','Falling','$\pi$'],rotation='vertical',fontsize=15)
# # locs, labels = pyplot.yticks([-0.5*np.pi,-0.25*np.pi,0,0.25*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=15)
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
# # # width = 2*np.pi / bins_number
# # # params = dict(projection='polar')
# # # fig, ax = pyplot.subplots(subplot_kw=params)
# # # ax.bar(bins[:bins_number], nRRIsauto/sum(nRRIsauto),width=width, color='grey',alpha=0.6,linewidth=2, fill=True)
# # # pyplot.setp(ax.get_yticklabels(), color='k')
# # # ax.set_title('phase histogram in long RRI autocorrelation(QLD0098)',fontsize=13)
# # # pyplot.show()
# # # params = dict(projection='polar')
# # # fig, ax2 = pyplot.subplots(subplot_kw=params)
# # # ax2.bar(bins[:bins_number], nRRIsautosei/sum(nRRIsautosei),width=width, color='g',edgecolor='k',linewidth=2)
# # # ax2.set_title('seizure probability in long RRI autocorrelation(QLD0098)',fontsize=13)
# # # # ax2.set_yticks([0.0008,0.0012,0.0016,0.002])
# # # # ax2.set_rlim([0,0.002])
# # # pyplot.show()
#
#
#
#
#
#
# # ## Time features Time features Time features Time features
# # # # # # # ### time features
# # a=np.where(t<5.6497222+2.9219)
# # print(a)
# # print(t[1354]);print(t[1355])
# # t[0:1355]=t[0:1355]-2.9219+18.3502778
# # t[1355:]=t[1355:]-5.6497222-2.9219
# # print(t[1355]);
# # print(t);print(type(t));print(t[0])
# #
# # time_feature_arr=[]
# # for i in range(len(t)):
# #     if t[i]>24:
# #         time_feature_arr.append(t[i] - (t[i] // 24) * 24)
# #     else:
# #         time_feature_arr.append(t[i])
# #
# # seizure_time=[time_feature_arr[966],time_feature_arr[6793],time_feature_arr[12505],time_feature_arr[18409],
# #               # time_feature_arr[25580],time_feature_arr[29963],time_feature_arr[31460],time_feature_arr[35833],
# #               ]
# # print(seizure_time)
# #
# # bins_number = 18
# # bins = np.linspace(0, 24, bins_number + 1)
# # print(bins)
# # ntimes, _, _ = pyplot.hist(time_feature_arr[0:19080], bins)
# # ntimesei, _, _ = pyplot.hist(seizure_time, bins)
# # # # print(ntimes)
# # # # print(ntimesei)
# #
# #
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
# # # locs, labels = pyplot.yticks([0.1,0.3,0.5],['0.1','0.3','0.5'],fontsize=16)
# # ax.annotate("", xy=(-0.6153, 0.989), xytext=(0, 0),arrowprops=dict(arrowstyle="->",color='g',linewidth=2))
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



