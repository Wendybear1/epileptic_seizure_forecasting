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



a=[1,3,4,6,8,9,23]
b=movingaverage(a, 3)
print(b)



# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/SA0124channels/EEG_timewindowarr_SA0124_15s.csv',sep=',',header=None)
# t_window_arr= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/SA0124channels/EEGvariance_SA0124_15s_3h.csv',sep=',',header=None)
# Raw_variance_EEG= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/SA0124channels/EEGauto_SA0124_15s_3h.csv',sep=',',header=None)
# Raw_auto_EEG= csv_reader.values
#
#
# Raw_variance_EEG_arr=[]
# for item in Raw_variance_EEG:
#     Raw_variance_EEG_arr.append(float(item))
# Raw_auto_EEG_arr=[]
# for item in Raw_auto_EEG:
#     Raw_auto_EEG_arr.append(float(item))
# # # Raw_auto1_EEG_arr=[]
# # # for item in Raw_auto1_EEG:
# # #     Raw_auto1_EEG_arr.append(float(item))
# # print(t_window_arr[0]);
# # # print(t_window_arr[19080]-t_window_arr[0]);
# # print(len(t_window_arr));print(t_window_arr[0]);
# # print(t_window_arr[-1]-t_window_arr[0]);
# # print(t_window_arr[19624]);print(t_window_arr[19624]-t_window_arr[0]);
# # print(t_window_arr[-1]-t_window_arr[19624]);
#
# window_time_arr=t_window_arr
# # pyplot.plot(window_time_arr,Raw_variance_EEG_arr,'grey',alpha=0.5)
# # pyplot.ylabel('Voltage',fontsize=13)
# # pyplot.xlabel('Time (hours)',fontsize=13)
# # pyplot.title('raw EEG variance in SA0124',fontsize=13)
# # pyplot.show()
# var_arr=[]
# for item in Raw_variance_EEG_arr:
#     if item<1e-8:
#         var_arr.append(item)
#     else:
#         var_arr.append(var_arr[-1])
# Raw_variance_EEG=var_arr
# pyplot.plot(window_time_arr,Raw_variance_EEG,'grey',alpha=0.5)
# pyplot.ylabel('Voltage',fontsize=13)
# pyplot.xlabel('Time (hours)',fontsize=13)
# pyplot.title('EEG variance in SA0124',fontsize=13)
# pyplot.show()



# seizure_timing_index=[]
# for k in range(len(window_time_arr)):
#     if window_time_arr[k]<9.19205 and window_time_arr[k+1]>=9.19205:
#         seizure_timing_index.append(k)
#     if window_time_arr[k]<18.9488833 and window_time_arr[k+1]>=18.9488833:
#         seizure_timing_index.append(k)
#     if window_time_arr[k]<24.16555 and window_time_arr[k+1]>=24.16555:
#         seizure_timing_index.append(k)
#     if window_time_arr[k]<32.9738833 and window_time_arr[k+1]>=32.9738833:
#         seizure_timing_index.append(k)
#     if window_time_arr[k]<45.149161 and window_time_arr[k+1]>=45.149161:
#         seizure_timing_index.append(k)
#     if window_time_arr[k]<55.0694389 and window_time_arr[k+1]>=55.0694389:
#         seizure_timing_index.append(k)
#     if window_time_arr[k]<67.5319389 and window_time_arr[k+1]>=67.5319389:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 80.90055 and window_time_arr[k + 1] >= 80.90055:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 92.7538833 and window_time_arr[k + 1] >= 92.7538833:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 104.916106 and window_time_arr[k + 1] >= 104.916106:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 115.673883 and window_time_arr[k + 1] >= 115.673883:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 123.834 and window_time_arr[k + 1] >= 123.834:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 127.884278 and window_time_arr[k + 1] >= 127.884278:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 135.1409 and window_time_arr[k + 1] >= 135.1409:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 139.57055 and window_time_arr[k + 1] >= 139.57055:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 152.573328 and window_time_arr[k + 1] >= 152.573328:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 159.654944 and window_time_arr[k + 1] >= 159.654944:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 165.256383 and window_time_arr[k + 1] >= 165.256383:
#         seizure_timing_index.append(k)
# # print(seizure_timing_index)
# index_ictal = seizure_timing_index


# seizure_timing_index=[]
# for k in range(len(window_time_arr)):
#     if window_time_arr[k]<11.1298278 and window_time_arr[k+1]>=11.1298278:
#         seizure_timing_index.append(k)
#     if window_time_arr[k]<11.53538 and window_time_arr[k+1]>=11.53538:
#         seizure_timing_index.append(k)
#     if window_time_arr[k]<12.7337133 and window_time_arr[k+1]>=12.7337133:
#         seizure_timing_index.append(k)
#     if window_time_arr[k]<13.8364911 and window_time_arr[k+1]>=13.8364911:
#         seizure_timing_index.append(k)
#     if window_time_arr[k]<15.54649 and window_time_arr[k+1]>=15.54649:
#         seizure_timing_index.append(k)
#     if window_time_arr[k]<19.798 and window_time_arr[k+1]>=19.798:
#         seizure_timing_index.append(k)
#     if window_time_arr[k]<25.11483 and window_time_arr[k+1]>=25.11483:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 25.729 and window_time_arr[k + 1] >= 25.729:
#         seizure_timing_index.append(k)
#     if window_time_arr[k]<26.2083  and window_time_arr[k+1]>=26.2083 :
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 34.553 and window_time_arr[k + 1] >= 34.553:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 36.109667 and window_time_arr[k + 1] >= 36.109667:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 36.316337 and window_time_arr[k + 1] >= 36.316337:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 37.82555 and window_time_arr[k + 1] >= 37.82555:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 39.3927 and window_time_arr[k + 1] >= 39.3927:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 46.149 and window_time_arr[k + 1] >= 46.149:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 47.39538 and window_time_arr[k + 1] >= 47.39538:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 47.9226 and window_time_arr[k + 1] >= 47.9226:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 48.87888 and window_time_arr[k + 1] >= 48.87888:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 55.2219389 and window_time_arr[k + 1] >= 55.2219389:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 55.7047167 and window_time_arr[k + 1] >= 55.7047167:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 56.0986 and window_time_arr[k + 1] >= 56.0986:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 57.6077667 and window_time_arr[k + 1] >= 57.6077667:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 58.7541556 and window_time_arr[k + 1] >= 58.7541556:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 59.1747112 and window_time_arr[k + 1] >= 59.1747112:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 60.9255445 and window_time_arr[k + 1] >= 60.9255445:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 62.3180445 and window_time_arr[k + 1] >= 62.3180445:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 84.52166 and window_time_arr[k + 1] >= 84.52166:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 84.7647 and window_time_arr[k + 1] >= 84.7647:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 85.90777 and window_time_arr[k + 1] >= 85.90777:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 87.1288811 and window_time_arr[k + 1] >= 87.1288811:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 94.2927722 and window_time_arr[k + 1] >= 94.2927722:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 109.217727 and window_time_arr[k + 1] >= 109.217727:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 111.933838 and window_time_arr[k + 1] >= 111.933838:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 117.456383 and window_time_arr[k + 1] >= 117.456383:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 119.572216 and window_time_arr[k + 1] >= 119.572216:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 120.099716 and window_time_arr[k + 1] >= 120.099716:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 129.419 and window_time_arr[k + 1] >= 129.419:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 129.6137 and window_time_arr[k + 1] >= 129.6137:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 140.553328 and window_time_arr[k + 1] >= 140.553328:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 141.0061 and window_time_arr[k + 1] >= 141.0061:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 142.6405 and window_time_arr[k + 1] >= 142.6405:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 143.225222 and window_time_arr[k + 1] >= 143.225222:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 144.27416 and window_time_arr[k + 1] >= 144.27416:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 154.853 and window_time_arr[k + 1] >= 154.853:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 165.555272 and window_time_arr[k + 1] >= 165.555272:
#         seizure_timing_index.append(k)
#     if window_time_arr[k] < 165.77055 and window_time_arr[k + 1] >= 165.77055:
#         seizure_timing_index.append(k)
# print(seizure_timing_index)
# index_cluster=seizure_timing_index
#
#
# duration=[6,5,2,2,1,1,10,1,5,6,6,1,1,4,2,6,4,1]
#
# index_ictal_sum=[]
# for m in range(len(index_ictal)):
#     for j in range(duration[m]+1):
#         index_ictal_sum.append(index_ictal[m] + j)
# index_cluster_sum=[]
# for item in index_cluster:
#     for j in range(8):
#         index_cluster_sum.append(item+j)
#
# index_ictal_sum.sort();index_cluster_sum.sort()
# print(index_ictal_sum);print(index_cluster_sum);
#
# index_pre_sum=[]
# for item in index_ictal:
#     for i in range(1,60):
#         index_pre_sum.append(item-i)
# index_pre_cluster_sum=[]
# for item in index_cluster:
#     for i in range(1,60):
#         index_pre_cluster_sum.append(item-i)
# index_pre_sum.sort(); index_pre_cluster_sum.sort()
# print(index_pre_sum);print(index_pre_cluster_sum);
#
# x=np.ones(len(t_window_arr))
# for k in range(len(x)):
#     if k in index_ictal_sum:
#         x[k] = 1
#     elif k in index_pre_sum:
#         x[k] = 2
#     elif k in index_cluster_sum:
#         x[k] = 11
#     elif k in index_pre_cluster_sum:
#         x[k] = 22
#     else:
#         x[k] = 0
#
# np.savetxt("C:/Users/wxiong/Documents/PHD/combine_features/SA0124_tags.csv", x, delimiter=",", fmt='%s')





# # # # # # # ### EEG variance
# window_time_arr=t_window_arr[0:19624]
# Raw_variance_EEG=Raw_variance_EEG[0:19624]
# # window_time_arr=t_window_arr
# # Raw_variance_EEG=Raw_variance_EEG
#
# print(len(Raw_variance_EEG))
# long_rhythm_var_arr=movingaverage(Raw_variance_EEG,240*24)
# medium_rhythm_var_arr=movingaverage(Raw_variance_EEG,240)
# medium_rhythm_var_arr_2=movingaverage(Raw_variance_EEG,240*3)
# # medium_rhythm_var_arr_3=movingaverage(Raw_variance_EEG,240*6)
# # medium_rhythm_var_arr_4=movingaverage(Raw_variance_EEG,240*12)
# # short_rhythm_var_arr_plot=movingaverage(Raw_variance_EEG,240*24)
# print(len(long_rhythm_var_arr));print(len(medium_rhythm_var_arr));print(len(medium_rhythm_var_arr_2));

# fig, ax1 = pyplot.subplots()
# # ax1 = figure(figsize=(12, 10))
# ax1.plot(window_time_arr,Raw_variance_EEG,'grey',alpha=0.6)
# ax1.set_title('EEG variance',fontsize=15)
# ax1.set_xlabel('Time (hours)',fontsize=15)
# ax1.set_ylabel('$\mathregular{V^2}$',fontsize=15)
# # locs, labels = pyplot.xticks(fontsize=23)
# # locs, labels = pyplot.yticks([20,  100,  180],fontsize=23)
# ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)
# ax1.scatter(window_time_arr[1487],Raw_variance_EEG[1487],s=30,c='k')
# ax1.scatter(window_time_arr[3829],Raw_variance_EEG[3829],s=30,c='k')
# ax1.scatter(window_time_arr[5081],Raw_variance_EEG[5081],s=30,c='k')
# ax1.scatter(window_time_arr[7195],Raw_variance_EEG[7195],s=30,c='k')
# ax1.scatter(window_time_arr[10117],Raw_variance_EEG[10117],s=30,c='k')
# ax1.scatter(window_time_arr[12498],Raw_variance_EEG[12498],s=30,c='k')
# ax1.scatter(window_time_arr[15489],Raw_variance_EEG[15489],s=30,c='k')
# ax1.scatter(window_time_arr[18697],Raw_variance_EEG[18697],s=30,c='k')
# # ax1.scatter(window_time_arr[21542],Raw_variance_EEG[21542],s=60,c='k')
# # ax1.scatter(window_time_arr[24461],Raw_variance_EEG[24461],s=60,c='k')
# # ax1.scatter(window_time_arr[27043],Raw_variance_EEG[27043],s=60,c='k')
# # ax1.scatter(window_time_arr[29002],Raw_variance_EEG[29002],s=60,c='k')
# # ax1.scatter(window_time_arr[29974],Raw_variance_EEG[29974],s=60,c='k')
# # ax1.scatter(window_time_arr[31715],Raw_variance_EEG[31715],s=60,c='k')
# # ax1.scatter(window_time_arr[32778],Raw_variance_EEG[32778],s=60,c='k')
# # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# pyplot.tight_layout()
# pyplot.show()
#
# fig, ax1 = pyplot.subplots()
# # ax1 = figure(figsize=(12, 10))
# ax1.plot(window_time_arr[0:2000],Raw_variance_EEG[0:2000],'grey',alpha=0.6)
# ax1.scatter(window_time_arr[1487],Raw_variance_EEG[1487],s=30,c='k')
# ax1.set_xlabel('Time (hours)',fontsize=15)
# ax1.set_ylabel('$\mathregular{V^2}$',fontsize=15)
# # locs, labels = pyplot.xticks(fontsize=23)
# # locs, labels = pyplot.yticks([20,  100,  180],fontsize=23)
# ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)
# pyplot.tight_layout()
# pyplot.show()


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
# # print(np.mean(seizure_phase_var_long))
# n=0
# for item in seizure_phase_var_long:
#     if item <0:
#         n=n+1
# print(n/len(seizure_phase_var_long))
#
# from matplotlib import gridspec
# fig = pyplot.figure(figsize=(12, 10))
# gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
# ax1=pyplot.subplot(gs[0])
# # ax1.plot(window_time_arr[240*6:33000],long_rhythm_var_arr[240*6:33000],'orange',alpha=0.7)
# ax1.plot(window_time_arr[240*24:33000],long_rhythm_var_arr[240*24:33000],'darkblue',alpha=0.8)
# ax1.set_title('EEG variance in SA0124',fontsize=23)
# ax1.set_xlabel('Time (hours)',fontsize=23)
# # ax1.set_ylabel('$\mathregular{mV^2}$',fontsize=23)
# ax1.set_ylabel('$\mathregular{\u03BCV^2}$',fontsize=23)
# locs, labels = pyplot.xticks(fontsize=23)
# locs, labels = pyplot.yticks([20,  100,  180],fontsize=23)
# # locs, labels = pyplot.yticks([200,  500,  800],fontsize=23)
# ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)
# # ax1.scatter(window_time_arr[1487],long_rhythm_var_arr[1487],s=60,c='k')
# # ax1.scatter(window_time_arr[3829],long_rhythm_var_arr[3829],s=60,c='k')
# # ax1.scatter(window_time_arr[5081],long_rhythm_var_arr[5081],s=60,c='k')
# ax1.scatter(window_time_arr[7195],long_rhythm_var_arr[7195],s=60,c='k')
# ax1.scatter(window_time_arr[10117],long_rhythm_var_arr[10117],s=60,c='k')
# ax1.scatter(window_time_arr[12498],long_rhythm_var_arr[12498],s=60,c='k')
# ax1.scatter(window_time_arr[15489],long_rhythm_var_arr[15489],s=60,c='k')
# ax1.scatter(window_time_arr[18697],long_rhythm_var_arr[18697],s=60,c='k')
# ax1.scatter(window_time_arr[21542],long_rhythm_var_arr[21542],s=60,c='k')
# ax1.scatter(window_time_arr[24461],long_rhythm_var_arr[24461],s=60,c='k')
# ax1.scatter(window_time_arr[27043],long_rhythm_var_arr[27043],s=60,c='k')
# ax1.scatter(window_time_arr[29002],long_rhythm_var_arr[29002],s=60,c='k')
# ax1.scatter(window_time_arr[29974],long_rhythm_var_arr[29974],s=60,c='k')
# ax1.scatter(window_time_arr[31715],long_rhythm_var_arr[31715],s=60,c='k')
# ax1.scatter(window_time_arr[32778],long_rhythm_var_arr[32778],s=60,c='k')
# # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# ax2=pyplot.subplot(gs[1])
# ax2.set_xlabel('Time (hours)',fontsize=23)
# ax2.set_title('Instantaneous Phase',fontsize=23)
# # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax2.plot(window_time_arr[240*6:33000],phase_long_EEGvariance_arr[240*6:33000],c='k',alpha=0.5,label='instantaneous phase')
# ax2.plot(window_time_arr[240*24:33000],phase_long_EEGvariance_arr[240*24:33000],c='k',alpha=0.7,label='instantaneous phase')
# pyplot.hlines(0,window_time_arr[240*24],window_time_arr[33000],'k','dashed')
# # ax2.plot(window_time_arr,rolmean_long_EEGvar,'b',alpha=0.7,label='smoothed phase')
# ax2.set_xlabel('Time (hours)',fontsize=23)
# locs, labels = pyplot.xticks(fontsize=23)
# locs, labels = pyplot.yticks(fontsize=23)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# # ax2.scatter(window_time_arr[1487],phase_long_EEGvariance_arr[1487],s=60,c='k')
# # ax2.scatter(window_time_arr[3829],phase_long_EEGvariance_arr[3829],s=60,c='k')
# # ax2.scatter(window_time_arr[5081],phase_long_EEGvariance_arr[5081],s=60,c='k')
# ax2.scatter(window_time_arr[7195],phase_long_EEGvariance_arr[7195],s=60,c='k')
# ax2.scatter(window_time_arr[10117],phase_long_EEGvariance_arr[10117],s=60,c='k')
# ax2.scatter(window_time_arr[12498],phase_long_EEGvariance_arr[12498],s=60,c='k')
# ax2.scatter(window_time_arr[15489],phase_long_EEGvariance_arr[15489],s=60,c='k')
# ax2.scatter(window_time_arr[18697],phase_long_EEGvariance_arr[18697],s=60,c='k')
# ax2.scatter(window_time_arr[21542],phase_long_EEGvariance_arr[21542],s=60,c='k')
# ax2.scatter(window_time_arr[24461],phase_long_EEGvariance_arr[24461],s=60,c='k')
# ax2.scatter(window_time_arr[27043],phase_long_EEGvariance_arr[27043],s=60,c='k')
# ax2.scatter(window_time_arr[29002],phase_long_EEGvariance_arr[29002],s=60,c='k')
# ax2.scatter(window_time_arr[29974],phase_long_EEGvariance_arr[29974],s=60,c='k')
# ax2.scatter(window_time_arr[31715],phase_long_EEGvariance_arr[31715],s=60,c='k')
# ax2.scatter(window_time_arr[32778],phase_long_EEGvariance_arr[32778],s=60,c='k')
# # locs, labels = pyplot.yticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi],['-$\pi$','Rising','0','Falling','$\pi$'],rotation='vertical',fontsize=23)
# locs, labels = pyplot.yticks([-0.5*np.pi,-0.15*np.pi,0,0.3*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=23)
# pyplot.tight_layout()
# pyplot.show()

# # bins_number = 18
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
# # ax.bar(bins[:bins_number], nEEGsvarsei/nEEGsvar,width=width, color='grey',alpha=0.7,linewidth=2, fill=True,edgecolor='k')
# # pyplot.setp(ax.get_yticklabels(), color='k')
# # # ax.set_title('Seizure probability in EEG variance',fontsize=13)
# # locs, labels = pyplot.yticks([0.0002,0.0010,0.0018],['0.0002','0.0010','0.0018'],fontsize=16)
# # ax.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',],fontsize=16)
# # pyplot.show()
# # params = dict(projection='polar')
# # fig, ax2 = pyplot.subplots(subplot_kw=params)
# # ax2.bar(bins[:bins_number], nEEGsvarsei/sum(nEEGsvarsei),width=width, color='grey',alpha=0.7,linewidth=2,edgecolor='grey')
# # locs, labels = pyplot.yticks([0.2,0.6,1],['0.2','0.6','1'],fontsize=16)
# # ax2.annotate("", xy=(-0.299418, 0.86), xytext=(0, 0),arrowprops=dict(arrowstyle="->",color='g',linewidth=2))
# # ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',],fontsize=16)
# # pyplot.show()
#
# # long_rhythm_var_arr=short_rhythm_var_arr_plot*(10**12)
# # var_trans=hilbert(long_rhythm_var_arr)
# # var_trans_nomal=[]
# # for m in var_trans:
# #     var_trans_nomal.append(m/abs(m))
# # SIvarlong=sum(var_trans_nomal)/len(var_trans_nomal)
# # print(SIvarlong)
# # seizure_phase=[]
# # for item in seizure_timing_index:
# #      seizure_phase.append(var_trans_nomal[item])
# # SIvarlongseizure=sum(seizure_phase)/len(seizure_phase)
# # print(SIvarlongseizure)
# # var_phase=np.angle(var_trans)
# # phase_long_EEGvariance_arr=var_phase
# # seizure_phase_var_long=[]
# # for item in seizure_timing_index:
# #     seizure_phase_var_long.append(phase_long_EEGvariance_arr[item])
# # print(seizure_phase_var_long)
# # item_arr=[]
# # for item in seizure_phase_var_long:
# #     item_arr.append(item/np.pi)
# # print(item_arr)
# #
# # from matplotlib import gridspec
# # fig = pyplot.figure(figsize=(12, 10))
# # gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
# # ax1=pyplot.subplot(gs[0])
# # ax1.plot(window_time_arr[240*6:33000],long_rhythm_var_arr[240*6:33000],'darkblue',alpha=0.5)
# # # ax1.plot(window_time_arr[240*6:],long_rhythm_var_arr[240*6:],'orange',alpha=0.8)
# # ax1.set_title('EEG variance in SA0124',fontsize=23)
# # ax1.set_xlabel('Time (hours)',fontsize=23)
# # # ax1.set_ylabel('$\mathregular{mV^2}$',fontsize=23)
# # ax1.set_ylabel('$\mathregular{\u03BCV^2}$',fontsize=23)
# # locs, labels = pyplot.xticks(fontsize=23)
# # # locs, labels = pyplot.yticks([20,  100,  180],fontsize=23)
# # locs, labels = pyplot.yticks([200,  500,  800],fontsize=23)
# # ax1.spines['right'].set_visible(False)
# # ax1.spines['top'].set_visible(False)
# # ax1.scatter(window_time_arr[1487],long_rhythm_var_arr[1487],s=60,c='k')
# # ax1.scatter(window_time_arr[3829],long_rhythm_var_arr[3829],s=60,c='k')
# # ax1.scatter(window_time_arr[5081],long_rhythm_var_arr[5081],s=60,c='k')
# # ax1.scatter(window_time_arr[7195],long_rhythm_var_arr[7195],s=60,c='k')
# # ax1.scatter(window_time_arr[10117],long_rhythm_var_arr[10117],s=60,c='k')
# # ax1.scatter(window_time_arr[12498],long_rhythm_var_arr[12498],s=60,c='k')
# # ax1.scatter(window_time_arr[15489],long_rhythm_var_arr[15489],s=60,c='k')
# # ax1.scatter(window_time_arr[18697],long_rhythm_var_arr[18697],s=60,c='k')
# # # ax1.scatter(window_time_arr[21542],long_rhythm_var_arr[21542],s=60,c='k')
# # # ax1.scatter(window_time_arr[24461],long_rhythm_var_arr[24461],s=60,c='k')
# # # ax1.scatter(window_time_arr[27043],long_rhythm_var_arr[27043],s=60,c='k')
# # # ax1.scatter(window_time_arr[29002],long_rhythm_var_arr[29002],s=60,c='k')
# # # ax1.scatter(window_time_arr[29974],long_rhythm_var_arr[29974],s=60,c='k')
# # # ax1.scatter(window_time_arr[31715],long_rhythm_var_arr[31715],s=60,c='k')
# # # ax1.scatter(window_time_arr[32778],long_rhythm_var_arr[32778],s=60,c='k')
# # # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax2=pyplot.subplot(gs[1])
# # ax2.set_xlabel('Time (hours)',fontsize=23)
# # ax2.set_title('Instantaneous Phase',fontsize=23)
# # # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax2.plot(window_time_arr[240*6:33000],phase_long_EEGvariance_arr[240*6:33000],c='k',alpha=0.3,label='instantaneous phase')
# # # ax2.plot(window_time_arr[240*6:],phase_long_EEGvariance_arr[240*6:],c='k',alpha=0.3,label='instantaneous phase')
# # pyplot.hlines(0,window_time_arr[240*6],window_time_arr[-1],'k','dashed')
# # # ax2.plot(window_time_arr,rolmean_long_EEGvar,'b',alpha=0.7,label='smoothed phase')
# # ax2.set_xlabel('Time (hours)',fontsize=23)
# # locs, labels = pyplot.xticks(fontsize=23)
# # locs, labels = pyplot.yticks(fontsize=23)
# # ax2.spines['right'].set_visible(False)
# # ax2.spines['top'].set_visible(False)
# # ax2.scatter(window_time_arr[1487],phase_long_EEGvariance_arr[1487],s=60,c='k')
# # ax2.scatter(window_time_arr[3829],phase_long_EEGvariance_arr[3829],s=60,c='k')
# # ax2.scatter(window_time_arr[5081],phase_long_EEGvariance_arr[5081],s=60,c='k')
# # ax2.scatter(window_time_arr[7195],phase_long_EEGvariance_arr[7195],s=60,c='k')
# # ax2.scatter(window_time_arr[10117],phase_long_EEGvariance_arr[10117],s=60,c='k')
# # ax2.scatter(window_time_arr[12498],phase_long_EEGvariance_arr[12498],s=60,c='k')
# # ax2.scatter(window_time_arr[15489],phase_long_EEGvariance_arr[15489],s=60,c='k')
# # ax2.scatter(window_time_arr[18697],phase_long_EEGvariance_arr[18697],s=60,c='k')
# # # ax2.scatter(window_time_arr[21542],phase_long_EEGvariance_arr[21542],s=60,c='k')
# # # ax2.scatter(window_time_arr[24461],phase_long_EEGvariance_arr[24461],s=60,c='k')
# # # ax2.scatter(window_time_arr[27043],phase_long_EEGvariance_arr[27043],s=60,c='k')
# # # ax2.scatter(window_time_arr[29002],phase_long_EEGvariance_arr[29002],s=60,c='k')
# # # ax2.scatter(window_time_arr[29974],phase_long_EEGvariance_arr[29974],s=60,c='k')
# # # ax2.scatter(window_time_arr[31715],phase_long_EEGvariance_arr[31715],s=60,c='k')
# # # ax2.scatter(window_time_arr[32778],phase_long_EEGvariance_arr[32778],s=60,c='k')
# # # locs, labels = pyplot.yticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi],['-$\pi$','Rising','0','Falling','$\pi$'],rotation='vertical',fontsize=23)
# # locs, labels = pyplot.yticks([-0.5*np.pi,-0.15*np.pi,0,0.3*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=23)
# # pyplot.tight_layout()
# # pyplot.show()
#
#
#
#
#
#
#
# ####writing example
# # from matplotlib import gridspec
# # fig = pyplot.figure(figsize=(8, 5))
# # gs = gridspec.GridSpec(1, 1)
# # ax =pyplot.subplot(gs[0])
# # ax.plot(window_time_arr,long_rhythm_var_arr,'k',alpha=0.6)
# # ax.set_title('Long cycle of EEG variance in SA0124',fontsize=23)
# # ax.set_xlabel('Time (hours)',fontsize=23)
# # ax.set_ylabel('Voltage',fontsize=23)
# # pyplot.annotate('',xy=(9.19205,np.max(long_rhythm_var_arr)),xytext=(9.19205,np.max(long_rhythm_var_arr)+0.0000000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# # pyplot.annotate('',xy=(18.9488833,np.max(long_rhythm_var_arr)),xytext=(18.9488833,np.max(long_rhythm_var_arr)+0.0000000000001),arrowprops=dict(facecolor='purple',shrink=0.05))
# # pyplot.annotate('',xy=(24.16555,np.max(long_rhythm_var_arr)),xytext=(24.16555,np.max(long_rhythm_var_arr)+0.0000000000001),arrowprops=dict(facecolor='orange',shrink=0.05))
# # pyplot.annotate('',xy=(32.9738833,np.max(long_rhythm_var_arr)),xytext=(32.9738833,np.max(long_rhythm_var_arr)+0.0000000000001),arrowprops=dict(facecolor='green',shrink=0.05))
# # pyplot.annotate('',xy=(45.149161,np.max(long_rhythm_var_arr)),xytext=(45.149161,np.max(long_rhythm_var_arr)+0.0000000000001),arrowprops=dict(facecolor='yellow',shrink=0.05))
# # pyplot.annotate('',xy=(55.0694389,np.max(long_rhythm_var_arr)),xytext=(55.0694389,np.max(long_rhythm_var_arr)+0.0000000000001),arrowprops=dict(facecolor='red',shrink=0.05))
# # pyplot.annotate('',xy=(67.5319389,np.max(long_rhythm_var_arr)),xytext=(67.5319389,np.max(long_rhythm_var_arr)+0.0000000000001),arrowprops=dict(facecolor='C0',shrink=0.05))
# # pyplot.show()
# # from matplotlib import gridspec
# # fig = pyplot.figure(figsize=(8, 5))
# # gs = gridspec.GridSpec(1, 1)
# # ax =pyplot.subplot(gs[0])
# # ax.plot(window_time_arr,short_rhythm_var_arr_show,'r',alpha=0.6)
# # ax.set_title('EEG variance in SA0124',fontsize=23)
# # ax.set_xlabel('Time (hours)',fontsize=23)
# # ax.set_ylabel('Voltage',fontsize=23)
# # pyplot.annotate('',xy=(9.19205,np.max(short_rhythm_var_arr_show)),xytext=(9.19205,np.max(short_rhythm_var_arr_show)+0.0000000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# # pyplot.annotate('',xy=(18.9488833,np.max(short_rhythm_var_arr_show)),xytext=(18.9488833,np.max(short_rhythm_var_arr_show)+0.0000000000001),arrowprops=dict(facecolor='purple',shrink=0.05))
# # pyplot.annotate('',xy=(24.16555,np.max(short_rhythm_var_arr_show)),xytext=(24.16555,np.max(short_rhythm_var_arr_show)+0.0000000000001),arrowprops=dict(facecolor='orange',shrink=0.05))
# # pyplot.annotate('',xy=(32.9738833,np.max(short_rhythm_var_arr_show)),xytext=(32.9738833,np.max(short_rhythm_var_arr_show)+0.0000000000001),arrowprops=dict(facecolor='green',shrink=0.05))
# # pyplot.annotate('',xy=(45.149161,np.max(short_rhythm_var_arr_show)),xytext=(45.149161,np.max(short_rhythm_var_arr_show)+0.0000000000001),arrowprops=dict(facecolor='yellow',shrink=0.05))
# # pyplot.annotate('',xy=(55.0694389,np.max(short_rhythm_var_arr_show)),xytext=(55.0694389,np.max(short_rhythm_var_arr_show)+0.0000000000001),arrowprops=dict(facecolor='red',shrink=0.05))
# # pyplot.annotate('',xy=(67.5319389,np.max(short_rhythm_var_arr_show)),xytext=(67.5319389,np.max(short_rhythm_var_arr_show)+0.0000000000001),arrowprops=dict(facecolor='C0',shrink=0.05))
# # pyplot.show()
# # from matplotlib import gridspec
# # fig = pyplot.figure(figsize=(8, 5))
# # gs = gridspec.GridSpec(1, 1)
# # ax1=pyplot.subplot(gs[0])
# # ax1.plot(window_time_arr,phase_short_EEGvar_arr,'grey',alpha=0.7)
# # ax1.set_title('EEG variance phases in SA0124',fontsize=23)
# # ax1.set_xlabel('Time (hours)',fontsize=23)
# # pyplot.annotate('',xy=(9.19205,np.max(phase_short_EEGvar_arr)),xytext=(9.19205,np.max(phase_short_EEGvar_arr)+0.0000000003),arrowprops=dict(facecolor='grey',shrink=0.05))
# # pyplot.annotate('',xy=(18.9488833,np.max(phase_short_EEGvar_arr)),xytext=(18.9488833,np.max(phase_short_EEGvar_arr)+0.0000000003),arrowprops=dict(facecolor='purple',shrink=0.05))
# # pyplot.annotate('',xy=(24.16555,np.max(phase_short_EEGvar_arr)),xytext=(24.16555,np.max(phase_short_EEGvar_arr)+0.0000000003),arrowprops=dict(facecolor='orange',shrink=0.05))
# # pyplot.annotate('',xy=(32.9738833,np.max(phase_short_EEGvar_arr)),xytext=(32.9738833,np.max(phase_short_EEGvar_arr)+0.0000000003),arrowprops=dict(facecolor='green',shrink=0.05))
# # pyplot.annotate('',xy=(45.149161,np.max(phase_short_EEGvar_arr)),xytext=(45.149161,np.max(phase_short_EEGvar_arr)+0.0000000003),arrowprops=dict(facecolor='yellow',shrink=0.05))
# # pyplot.annotate('',xy=(55.0694389,np.max(phase_short_EEGvar_arr)),xytext=(55.0694389,np.max(phase_short_EEGvar_arr)+0.0000000003),arrowprops=dict(facecolor='red',shrink=0.05))
# # pyplot.annotate('',xy=(67.5319389,np.max(phase_short_EEGvar_arr)),xytext=(67.5319389,np.max(phase_short_EEGvar_arr)+0.0000000003),arrowprops=dict(facecolor='C0',shrink=0.05))
# # pyplot.tight_layout()
# # pyplot.show()
# # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.May/forecastshiftraw/forecast72h_1hraw_EEGvar_short_SA0124.csv',sep=',',header=None)
# # forecast_var_EEG= csv_reader.values
# # forecast_var_EEG_arr=[]
# # for item in forecast_var_EEG:
# #     forecast_var_EEG_arr=forecast_var_EEG_arr+list(item)
# # t=np.linspace(window_time_arr[-1]+0.1666667,window_time_arr[-1]+0.1666667+0.1666667*(len(forecast_var_EEG_arr)-1),len(forecast_var_EEG_arr))
# # ax2=pyplot.subplot(gs[1])
# # ax2.set_xlabel('Time (hours)',fontsize=8)
# # ax2.set_title('Instantaneous Phase',fontsize=8)
# # ax2.plot(window_time_arr,phase_short_EEGvar_arr,c='grey',alpha=0.4,label='instantaneous phase')
# # pyplot.hlines(np.pi,window_time_arr[0],window_time_arr[-1],'k','dashed')
# # ax2.plot(window_time_arr,rolmean_short_EEGvar,'b', alpha=0.6,label='smoothed phase')
# # ax2.plot(t, forecast_var_EEG_arr,'red',label='forecast phases')
# # pyplot.legend(fontsize=8)
# # locs, labels = pyplot.yticks([0,0.5*np.pi,np.pi,1.5*np.pi,2*np.pi],['0','Falling', '180''\N{DEGREE SIGN}','Rising','360''\N{DEGREE SIGN}'],rotation='vertical',fontsize=8)
# # pyplot.tight_layout()
# # pyplot.show()
#
#
#
#
# # pyplot.plot(t_window_arr,Raw_auto_EEG_arr,'grey',alpha=0.5)
# # pyplot.xlabel('Time (hours)',fontsize=13)
# # pyplot.title('raw EEG autocorrelation in SA0124',fontsize=13)
# # pyplot.show()
# value_arr=[]
# for item in Raw_auto_EEG_arr:
#     if item<500:
#         value_arr.append(item)
#     else:
#         value_arr.append(value_arr[-1])
# Raw_auto_EEG_arr=value_arr
# # pyplot.plot(t_window_arr,Raw_auto_EEG_arr,'grey',alpha=0.5)
# # pyplot.xlabel('Time (hours)',fontsize=13)
# # pyplot.title('EEG autocorrelation in SA0124',fontsize=13)
# # pyplot.show()
#
#
# #
# # Raw_auto_EEG=Raw_auto_EEG_arr[0:19624]
# # window_time_arr=t_window_arr[0:19624]
# Raw_auto_EEG=Raw_auto_EEG_arr
# window_time_arr=t_window_arr
#
# long_rhythm_value_arr=movingaverage(Raw_auto_EEG,240*24)
# medium_rhythm_value_arr=movingaverage(Raw_auto_EEG,240)
# medium_rhythm_value_arr_2=movingaverage(Raw_auto_EEG,240*3)
# medium_rhythm_value_arr_3=movingaverage(Raw_auto_EEG,240*6)
# medium_rhythm_value_arr_4=movingaverage(Raw_auto_EEG,240*12)
# short_rhythm_value_arr_plot=movingaverage(Raw_auto_EEG,240*24)
#
# # fig=pyplot.figure(figsize=(8,6))
# # # pyplot.plot(window_time_arr,Raw_auto_EEG,'grey',alpha=0.5,label='Raw autocorrelation')
# # # pyplot.plot(window_time_arr[20:],short_rhythm_value_arr_plot[20:],'grey',alpha=0.5,label='5min')
# # # pyplot.plot(window_time_arr[240:],medium_rhythm_value_arr[240:],'g',label='1 hours')
# # # pyplot.plot(window_time_arr[240*3:],medium_rhythm_value_arr_2[240*3:],'k',label='3 hours')
# # # pyplot.plot(window_time_arr[240*6:],medium_rhythm_value_arr_3[240*6:],'orange',label='6 hours')
# # # pyplot.plot(window_time_arr[240*12:],medium_rhythm_value_arr_4[240*12:],'b',label='12 hours')
# # # pyplot.plot(window_time_arr[5760:],long_rhythm_value_arr[5760:],'r',alpha=0.7,label='1 day')
# # pyplot.plot(window_time_arr,short_rhythm_value_arr_plot,'grey',alpha=0.5,label='5 min')
# # pyplot.plot(window_time_arr,medium_rhythm_value_arr,'g',label='1 hours')
# # pyplot.plot(window_time_arr,medium_rhythm_value_arr_2,'k',label='3 hours')
# # pyplot.plot(window_time_arr,medium_rhythm_value_arr_3,'orange',label='6 hours')
# # pyplot.plot(window_time_arr,medium_rhythm_value_arr_4,'b',label='12 hours')
# # pyplot.plot(window_time_arr,long_rhythm_value_arr,'r',alpha=0.7,label='1 day')
# # pyplot.title('EEG autocorrelation in SA0124',fontsize=23)
# # pyplot.xlabel('Time (hours)',fontsize=23)
# # pyplot.ylabel('Autocorrelation',fontsize=23)
# # pyplot.legend(loc='upper left',fontsize=10)
# # locs, labels = pyplot.xticks(fontsize=23)
# # locs, labels = pyplot.yticks(fontsize=23)
# # pyplot.tight_layout()
# # pyplot.annotate('',xy=(9.19205,np.max(short_rhythm_value_arr_plot)),xytext=(9.19205,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(18.9488833,np.max(short_rhythm_value_arr_plot)),xytext=(18.9488833,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(24.16555,np.max(short_rhythm_value_arr_plot)),xytext=(24.16555,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(32.9738833,np.max(short_rhythm_value_arr_plot)),xytext=(32.9738833,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(45.149161,np.max(short_rhythm_value_arr_plot)),xytext=(45.149161,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(55.0694389,np.max(short_rhythm_value_arr_plot)),xytext=(55.0694389,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(67.5319389,np.max(short_rhythm_value_arr_plot)),xytext=(67.5319389,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(80.90055,np.max(short_rhythm_value_arr_plot)),xytext=(80.90055,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(92.7538833,np.max(short_rhythm_value_arr_plot)),xytext=(92.7538833,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # # pyplot.annotate('',xy=(104.916106,np.max(short_rhythm_value_arr_plot)),xytext=(104.916106,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # # pyplot.annotate('',xy=(115.673883,np.max(short_rhythm_value_arr_plot)),xytext=(115.673883,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # # pyplot.annotate('',xy=(123.834,np.max(short_rhythm_value_arr_plot)),xytext=(123.834,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # # pyplot.annotate('',xy=(127.884278,np.max(short_rhythm_value_arr_plot)),xytext=(127.884278,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # # pyplot.annotate('',xy=(135.1409,np.max(short_rhythm_value_arr_plot)),xytext=(135.1409,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # # pyplot.annotate('',xy=(139.57055,np.max(short_rhythm_value_arr_plot)),xytext=(139.57055,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
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
#
# from matplotlib import gridspec
# fig = pyplot.figure(figsize=(12, 10))
# gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
# ax2=pyplot.subplot(gs[0])
# ax2.plot(window_time_arr[240*24:33000],long_rhythm_value_arr[240*24:33000],'darkblue',alpha=0.8)
# # ax2.plot(window_time_arr[240*6:19600],long_rhythm_value_arr[240*6:19600],'darkblue',alpha=0.8)
# # ax2.plot(window_time_arr[240*6:],long_rhythm_value_arr[240*6:],'darkblue',alpha=0.8)
# # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# ax2.set_title('EEG autocorrelation in SA0124',fontsize=23)
# ax2.set_xlabel('Time (hours)',fontsize=23)
# locs, labels = pyplot.xticks(fontsize=23)
# locs, labels = pyplot.yticks([18,24,30],fontsize=23)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# # ax2.scatter(window_time_arr[1487],long_rhythm_value_arr[1487],s=60,c='k')
# # ax2.scatter(window_time_arr[3829],long_rhythm_value_arr[3829],s=60,c='k')
# # ax2.scatter(window_time_arr[5081],long_rhythm_value_arr[5081],s=60,c='k')
# ax2.scatter(window_time_arr[7195],long_rhythm_value_arr[7195],s=60,c='k')
# ax2.scatter(window_time_arr[10117],long_rhythm_value_arr[10117],s=60,c='k')
# ax2.scatter(window_time_arr[12498],long_rhythm_value_arr[12498],s=60,c='k')
# ax2.scatter(window_time_arr[15489],long_rhythm_value_arr[15489],s=60,c='k')
# ax2.scatter(window_time_arr[18697],long_rhythm_value_arr[18697],s=60,c='k')
# ax2.scatter(window_time_arr[21542],long_rhythm_value_arr[21542],s=60,c='k')
# ax2.scatter(window_time_arr[24461],long_rhythm_value_arr[24461],s=60,c='k')
# ax2.scatter(window_time_arr[27043],long_rhythm_value_arr[27043],s=60,c='k')
# ax2.scatter(window_time_arr[29002],long_rhythm_value_arr[29002],s=60,c='k')
# ax2.scatter(window_time_arr[29974],long_rhythm_value_arr[29974],s=60,c='k')
# ax2.scatter(window_time_arr[31715],long_rhythm_value_arr[31715],s=60,c='k')
# ax2.scatter(window_time_arr[32778],long_rhythm_value_arr[32778],s=60,c='k')
# ax3=pyplot.subplot(gs[1])
# ax3.set_xlabel('Time (hours)',fontsize=23)
# ax3.set_title('Instantaneous Phase',fontsize=23)
# # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# locs, labels = pyplot.xticks(fontsize=23)
# locs, labels = pyplot.yticks(fontsize=23)
# ax3.spines['right'].set_visible(False)
# ax3.spines['top'].set_visible(False)
# # ax3.plot(window_time_arr[240*6:],phase_long_EEGauto_arr[240*6:],'k',alpha=0.5,label='instantaneous phase')
# ax3.plot(window_time_arr[240*24:33000],phase_long_EEGauto_arr[240*24:33000],'k',alpha=0.7,label='instantaneous phase')
# # ax3.plot(window_time_arr[240*6:19600],phase_long_EEGauto_arr[240*6:19600],'k',alpha=0.7,label='instantaneous phase')
# # pyplot.hlines(0,window_time_arr[240*6],window_time_arr[33000],'k','dashed')
# pyplot.hlines(0,window_time_arr[240*24],window_time_arr[33000],'k','dashed')
# # ax3.scatter(window_time_arr[1487],phase_long_EEGauto_arr[1487],s=60,c='k')
# # ax3.scatter(window_time_arr[3829],phase_long_EEGauto_arr[3829],s=60,c='k')
# # ax3.scatter(window_time_arr[5081],phase_long_EEGauto_arr[5081],s=60,c='k')
# ax3.scatter(window_time_arr[7195],phase_long_EEGauto_arr[7195],s=60,c='k')
# ax3.scatter(window_time_arr[10117],phase_long_EEGauto_arr[10117],s=60,c='k')
# ax3.scatter(window_time_arr[12498],phase_long_EEGauto_arr[12498],s=60,c='k')
# ax3.scatter(window_time_arr[15489],phase_long_EEGauto_arr[15489],s=60,c='k')
# ax3.scatter(window_time_arr[18697],phase_long_EEGauto_arr[18697],s=60,c='k')
# ax3.scatter(window_time_arr[21542],phase_long_EEGauto_arr[21542],s=60,c='k')
# ax3.scatter(window_time_arr[24461],phase_long_EEGauto_arr[24461],s=60,c='k')
# ax3.scatter(window_time_arr[27043],phase_long_EEGauto_arr[27043],s=60,c='k')
# ax3.scatter(window_time_arr[29002],phase_long_EEGauto_arr[29002],s=60,c='k')
# ax3.scatter(window_time_arr[29974],phase_long_EEGauto_arr[29974],s=60,c='k')
# ax3.scatter(window_time_arr[31715],phase_long_EEGauto_arr[31715],s=60,c='k')
# ax3.scatter(window_time_arr[32778],phase_long_EEGauto_arr[32778],s=60,c='k')
# ax3.set_xlabel('Time (hours)',fontsize=23)
# # locs, labels = pyplot.yticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi],['-$\pi$','Rising','0','Falling','$\pi$'],rotation='vertical',fontsize=23)
# locs, labels = pyplot.yticks([-0.5*np.pi,-0.15*np.pi,0,0.3*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=23)
# # locs, labels = pyplot.yticks([-0.25*np.pi,-0.125*np.pi,0,0.125*np.pi,0.25*np.pi],['-0.25$\pi$','Rising','0','Falling','0.25$\pi$'],rotation='vertical',fontsize=23)
# pyplot.tight_layout()
# pyplot.show()
#
# # bins_number = 18
# # bins = np.linspace(-np.pi, np.pi, bins_number + 1)
# # nEEGsauto, _, _ = pyplot.hist(phase_long_EEGauto_arr, bins)
# # nEEGsautosei, _, _ = pyplot.hist(seizure_phase_value_long, bins)
# # print(nEEGsauto)
# # print(nEEGsautosei)
#
# # fig, ax2 = pyplot.subplots(subplot_kw=params)
# # ax2.bar(bins[:bins_number], nEEGsauto/sum(nEEGsauto),width=width, color='grey',alpha=0.7,linewidth=2,edgecolor='k')
# # locs, labels = pyplot.yticks([0.1,0.3,0.5],['0.1','0.3','0.5'],fontsize=16)
# # # ax2.set_title('EEG autocorrelation',fontsize=13)
# # ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',],fontsize=16)
# # # ax2.set_rlim([0,0.002])
# # pyplot.show()
# # fig, ax2 = pyplot.subplots(subplot_kw=params)
# # ax2.bar(bins[:bins_number], nEEGsautosei,width=width, color='w',alpha=0.7,linewidth=2,edgecolor='b')
# # locs, labels = pyplot.yticks([1,3,5],['1','3','5'],fontsize=16)
# # # ax2.set_title('EEG autocorrelation',fontsize=13)
# # ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',],fontsize=16)
# # # ax2.set_rlim([0,0.002])
# # pyplot.show()
#
# # width = 2*np.pi / bins_number
# # params = dict(projection='polar')
# # fig, ax2 = pyplot.subplots(subplot_kw=params)
# # ax2.bar(bins[:bins_number], nEEGsautosei/nEEGsauto,width=width, color='grey',alpha=0.7,linewidth=2,edgecolor='k')
# # locs, labels = pyplot.yticks([0.0001,0.0004,0.0007],['0.0001','0.0004','0.0007'],fontsize=16)
# # ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',],fontsize=16)
# # # ax2.set_rlim([0,0.002])
# # pyplot.show()
# # width = 2*np.pi / bins_number
# # params = dict(projection='polar')
# # fig, ax2 = pyplot.subplots(subplot_kw=params)
# # ax2.bar(bins[:bins_number], nEEGsautosei/sum(nEEGsautosei),width=width, color='grey',alpha=0.7,linewidth=2,edgecolor='grey')
# # locs, labels = pyplot.yticks([0.2,0.6,1],['0.2','0.6','1'],fontsize=16)
# # ax2.annotate("", xy=( -0.21657, 0.97), xytext=(0, 0),arrowprops=dict(arrowstyle="->",color='g',linewidth=2))
# # ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',],fontsize=16)
# # pyplot.show()
#
#
# # long_rhythm_value_arr=short_rhythm_value_arr_plot
# # value_trans=hilbert(long_rhythm_value_arr)
# # value_trans_nomal=[]
# # for m in value_trans:
# #     value_trans_nomal.append(m/abs(m))
# # SIvaluelong=sum(value_trans_nomal)/len(value_trans_nomal)
# # print(SIvaluelong)
# # seizure_phase=[]
# # for item in seizure_timing_index:
# #     seizure_phase.append(value_trans_nomal[item])
# # SIvaluelongseizure=sum(seizure_phase)/len(seizure_phase)
# # print(SIvaluelongseizure)
# #
# # value_phase=np.angle(value_trans)
# # phase_long_EEGauto_arr=value_phase
# # seizure_phase_value_long=[]
# # for item in seizure_timing_index:
# #     seizure_phase_value_long.append(phase_long_EEGauto_arr[item])
# # print(seizure_phase_value_long)
# # item_arr=[]
# # for item in seizure_phase_value_long:
# #     item_arr.append(item/np.pi)
# # print(item_arr)
# # from matplotlib import gridspec
# # fig = pyplot.figure(figsize=(12, 10))
# # gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
# # ax2=pyplot.subplot(gs[0])
# # ax2.plot(window_time_arr[240*6:33000],long_rhythm_value_arr[240*6:33000],'darkblue',alpha=0.5)
# # # ax2.plot(window_time_arr[240*6:19600],long_rhythm_value_arr[240*6:19600],'darkblue',alpha=0.5)
# # # ax2.plot(window_time_arr[240*6:],long_rhythm_value_arr[240*6:],'darkblue',alpha=0.8)
# # # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax2.set_title('EEG autocorrelation in SA0124',fontsize=23)
# # ax2.set_xlabel('Time (hours)',fontsize=23)
# # locs, labels = pyplot.xticks(fontsize=23)
# # locs, labels = pyplot.yticks([18,24,30],fontsize=23)
# # ax2.spines['right'].set_visible(False)
# # ax2.spines['top'].set_visible(False)
# # ax2.scatter(window_time_arr[1487],long_rhythm_value_arr[1487],s=60,c='k')
# # ax2.scatter(window_time_arr[3829],long_rhythm_value_arr[3829],s=60,c='k')
# # ax2.scatter(window_time_arr[5081],long_rhythm_value_arr[5081],s=60,c='k')
# # ax2.scatter(window_time_arr[7195],long_rhythm_value_arr[7195],s=60,c='k')
# # ax2.scatter(window_time_arr[10117],long_rhythm_value_arr[10117],s=60,c='k')
# # ax2.scatter(window_time_arr[12498],long_rhythm_value_arr[12498],s=60,c='k')
# # ax2.scatter(window_time_arr[15489],long_rhythm_value_arr[15489],s=60,c='k')
# # ax2.scatter(window_time_arr[18697],long_rhythm_value_arr[18697],s=60,c='k')
# # # ax2.scatter(window_time_arr[21542],long_rhythm_value_arr[21542],s=60,c='k')
# # # ax2.scatter(window_time_arr[24461],long_rhythm_value_arr[24461],s=60,c='k')
# # # ax2.scatter(window_time_arr[27043],long_rhythm_value_arr[27043],s=60,c='k')
# # # ax2.scatter(window_time_arr[29002],long_rhythm_value_arr[29002],s=60,c='k')
# # # ax2.scatter(window_time_arr[29974],long_rhythm_value_arr[29974],s=60,c='k')
# # # ax2.scatter(window_time_arr[31715],long_rhythm_value_arr[31715],s=60,c='k')
# # # ax2.scatter(window_time_arr[32778],long_rhythm_value_arr[32778],s=60,c='k')
# # ax3=pyplot.subplot(gs[1])
# # ax3.set_xlabel('Time (hours)',fontsize=23)
# # ax3.set_title('Instantaneous Phase',fontsize=23)
# # # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # locs, labels = pyplot.xticks(fontsize=23)
# # locs, labels = pyplot.yticks(fontsize=23)
# # ax3.spines['right'].set_visible(False)
# # ax3.spines['top'].set_visible(False)
# # # ax3.plot(window_time_arr[240*6:],phase_long_EEGauto_arr[240*6:],'k',alpha=0.3,label='instantaneous phase')
# # ax3.plot(window_time_arr[240*6:33000],phase_long_EEGauto_arr[240*6:33000],'k',alpha=0.3,label='instantaneous phase')
# # # ax3.plot(window_time_arr[240*6:19600],phase_long_EEGauto_arr[240*6:19600],'k',alpha=0.7,label='instantaneous phase')
# # # pyplot.hlines(0,window_time_arr[240*6],window_time_arr[33000],'k','dashed')
# # pyplot.hlines(0,window_time_arr[240*6],window_time_arr[-1],'k','dashed')
# # ax3.scatter(window_time_arr[1487],phase_long_EEGauto_arr[1487],s=60,c='k')
# # ax3.scatter(window_time_arr[3829],phase_long_EEGauto_arr[3829],s=60,c='k')
# # ax3.scatter(window_time_arr[5081],phase_long_EEGauto_arr[5081],s=60,c='k')
# # ax3.scatter(window_time_arr[7195],phase_long_EEGauto_arr[7195],s=60,c='k')
# # ax3.scatter(window_time_arr[10117],phase_long_EEGauto_arr[10117],s=60,c='k')
# # ax3.scatter(window_time_arr[12498],phase_long_EEGauto_arr[12498],s=60,c='k')
# # ax3.scatter(window_time_arr[15489],phase_long_EEGauto_arr[15489],s=60,c='k')
# # ax3.scatter(window_time_arr[18697],phase_long_EEGauto_arr[18697],s=60,c='k')
# # # ax3.scatter(window_time_arr[21542],phase_long_EEGauto_arr[21542],s=60,c='k')
# # # ax3.scatter(window_time_arr[24461],phase_long_EEGauto_arr[24461],s=60,c='k')
# # # ax3.scatter(window_time_arr[27043],phase_long_EEGauto_arr[27043],s=60,c='k')
# # # ax3.scatter(window_time_arr[29002],phase_long_EEGauto_arr[29002],s=60,c='k')
# # # ax3.scatter(window_time_arr[29974],phase_long_EEGauto_arr[29974],s=60,c='k')
# # # ax3.scatter(window_time_arr[31715],phase_long_EEGauto_arr[31715],s=60,c='k')
# # # ax3.scatter(window_time_arr[32778],phase_long_EEGauto_arr[32778],s=60,c='k')
# # ax3.set_xlabel('Time (hours)',fontsize=23)
# # # locs, labels = pyplot.yticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi],['-$\pi$','Rising','0','Falling','$\pi$'],rotation='vertical',fontsize=23)
# # locs, labels = pyplot.yticks([-0.5*np.pi,-0.15*np.pi,0,0.3*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=23)
# # # locs, labels = pyplot.yticks([-0.25*np.pi,-0.125*np.pi,0,0.125*np.pi,0.25*np.pi],['-0.25$\pi$','Rising','0','Falling','0.25$\pi$'],rotation='vertical',fontsize=23)
# # pyplot.tight_layout()
# # pyplot.show()
#
#
# from pandas.plotting import autocorrelation_plot
# from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
# import scipy.stats as stats
#
#
#
# # # ### ECG data
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_ch31_timewindowarr_SA0124_15s_3h.csv',sep=',',header=None)
# rri_t= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_ch31_rawvariance_SA0124_15s_3h.csv',sep=',',header=None)
# RRI_var= csv_reader.values
# csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_ch31_rawauto_SA0124_15s_3h.csv',sep=',',header=None)
# Raw_auto_RRI31= csv_reader.values
# # # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/ch2-1/RRI_ch21_rawvariance_SA0124_15s_3h.csv',sep=',',header=None)
# # # RRI_var= csv_reader.values
# # # csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/ch2-1/RRI_ch21_rawauto_SA0124_15s_3h.csv',sep=',',header=None)
# # # Raw_auto_RRI31= csv_reader.values
#
#
# rri_t_arr=[]
# for item in rri_t:
#     rri_t_arr.append(2.98805+float(item))
#
# Raw_variance_RRI31_arr=[]
# for item in RRI_var:
#     Raw_variance_RRI31_arr.append(float(item))
# Raw_auto_RRI31_arr=[]
# for item in Raw_auto_RRI31:
#     Raw_auto_RRI31_arr.append(float(item))
# print(len(Raw_variance_RRI31_arr))
# #
# # # pyplot.plot(Raw_variance_RRI31_arr,'grey',alpha=0.5)
# # # pyplot.xlabel('Time (hours)',fontsize=13)
# # # pyplot.title('RRI variance in SA0124',fontsize=13)
# # # pyplot.show()
# # #
# # # pyplot.plot(Raw_auto_RRI31_arr,'grey',alpha=0.5)
# # # pyplot.xlabel('Time (hours)',fontsize=13)
# # # pyplot.title('RRI autocorrelation in SA0124',fontsize=13)
# # # pyplot.show()
# #
# #
# # # seizure_timing_index=[]
# # # for k in range(len(rri_t_arr)):
# # #     if rri_t_arr[k]<9.19205 and rri_t_arr[k+1]>=9.19205:
# # #         seizure_timing_index.append(k)
# # #     if rri_t_arr[k]<18.9488833 and rri_t_arr[k+1]>=18.9488833:
# # #         seizure_timing_index.append(k)
# # #     if rri_t_arr[k]<24.16555 and rri_t_arr[k+1]>=24.16555:
# # #         seizure_timing_index.append(k)
# # #     if rri_t_arr[k]<32.9738833 and rri_t_arr[k+1]>=32.9738833:
# # #         seizure_timing_index.append(k)
# # #     if rri_t_arr[k]<45.149161 and rri_t_arr[k+1]>=45.149161:
# # #         seizure_timing_index.append(k)
# # #     if rri_t_arr[k]<55.0694389 and rri_t_arr[k+1]>=55.0694389:
# # #         seizure_timing_index.append(k)
# # #     if rri_t_arr[k]<67.5319389 and rri_t_arr[k+1]>=67.5319389:
# # #         seizure_timing_index.append(k)
# # #     if rri_t_arr[k]<80.90055 and rri_t_arr[k+1]>=80.90055:
# # #         seizure_timing_index.append(k)
# # #     if rri_t_arr[k] < 92.7538833 and rri_t_arr[k + 1] >= 92.7538833:
# # #         seizure_timing_index.append(k)
# # #     if rri_t_arr[k] < 104.916106 and rri_t_arr[k + 1] >= 104.916106:
# # #         seizure_timing_index.append(k)
# # #     if rri_t_arr[k] < 115.673883 and rri_t_arr[k + 1] >= 115.673883:
# # #         seizure_timing_index.append(k)
# # #     if rri_t_arr[k] < 123.834 and rri_t_arr[k + 1] >= 123.834:
# # #         seizure_timing_index.append(k)
# # #     if rri_t_arr[k] < 127.884278 and rri_t_arr[k + 1] >= 127.884278:
# # #         seizure_timing_index.append(k)
# # #     if rri_t_arr[k] < 135.1409 and rri_t_arr[k + 1] >= 135.1409:
# # #         seizure_timing_index.append(k)
# # #     if rri_t_arr[k] < 139.57055 and rri_t_arr[k + 1] >= 139.57055:
# # #         seizure_timing_index.append(k)
# # # print(seizure_timing_index)
# #
# #
# #
# window_time_arr=t_window_arr
# Raw_variance_RRI31=Raw_variance_RRI31_arr
# # # window_time_arr=t_window_arr[0:19080]
# # # Raw_variance_RRI31=Raw_variance_RRI31_arr[0:19080]
# # window_time_arr=t_window_arr[0:19624]
# # Raw_variance_RRI31=Raw_variance_RRI31_arr[0:19624]
#
# long_rhythm_var_arr=movingaverage(Raw_variance_RRI31,240*24)
# medium_rhythm_var_arr=movingaverage(Raw_variance_RRI31,240)
# medium_rhythm_var_arr_2=movingaverage(Raw_variance_RRI31,240*3)
# medium_rhythm_var_arr_3=movingaverage(Raw_variance_RRI31,240*6)
# medium_rhythm_var_arr_4=movingaverage(Raw_variance_RRI31,240*12)
# short_rhythm_var_arr_plot=movingaverage(Raw_variance_RRI31,240*24)
#
# # # fig=pyplot.figure(figsize=(8,6))
# # # # pyplot.plot(window_time_arr,Raw_variance_EEG,'grey',alpha=0.5,label='Raw variance')
# # # pyplot.plot(window_time_arr,short_rhythm_var_arr_plot,'grey',alpha=0.5,label='5min')
# # # pyplot.plot(window_time_arr,medium_rhythm_var_arr,'g',label='1 hours')
# # # pyplot.plot(window_time_arr,medium_rhythm_var_arr_2,'k',label='3 hours')
# # # pyplot.plot(window_time_arr,medium_rhythm_var_arr_3,'orange',label='6 hours')
# # # pyplot.plot(window_time_arr,medium_rhythm_var_arr_4,'b',label='12 hours')
# # # pyplot.plot(window_time_arr,long_rhythm_var_arr,'r',alpha=0.7,label='1 day')
# # # pyplot.title('RRI variance in SA0124',fontsize=23)
# # # pyplot.xlabel('Time (hours)',fontsize=23)
# # # pyplot.ylabel('Second ($\mathregular{s^2}$)',fontsize=23)
# # # pyplot.legend(loc='upper left',fontsize=10)
# # # locs, labels = pyplot.xticks(fontsize=23)
# # # locs, labels = pyplot.yticks(fontsize=23)
# # # pyplot.tight_layout()
# # # pyplot.annotate('',xy=(9.19205,np.max(short_rhythm_var_arr_plot)),xytext=(9.19205,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(18.9488833,np.max(short_rhythm_var_arr_plot)),xytext=(18.9488833,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(24.16555,np.max(short_rhythm_var_arr_plot)),xytext=(24.16555,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(32.9738833,np.max(short_rhythm_var_arr_plot)),xytext=(32.9738833,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(45.149161,np.max(short_rhythm_var_arr_plot)),xytext=(45.149161,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(55.0694389,np.max(short_rhythm_var_arr_plot)),xytext=(55.0694389,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(67.5319389,np.max(short_rhythm_var_arr_plot)),xytext=(67.5319389,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(80.90055,np.max(short_rhythm_var_arr_plot)),xytext=(80.90055,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # # pyplot.annotate('',xy=(92.7538833,np.max(short_rhythm_var_arr_plot)),xytext=(92.7538833,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # # pyplot.annotate('',xy=(104.916106,np.max(short_rhythm_var_arr_plot)),xytext=(104.916106,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # # pyplot.annotate('',xy=(115.673883,np.max(short_rhythm_var_arr_plot)),xytext=(115.673883,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # # pyplot.annotate('',xy=(123.834,np.max(short_rhythm_var_arr_plot)),xytext=(123.834,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # # pyplot.annotate('',xy=(127.884278,np.max(short_rhythm_var_arr_plot)),xytext=(127.884278,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # # pyplot.annotate('',xy=(135.1409,np.max(short_rhythm_var_arr_plot)),xytext=(135.1409,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # # pyplot.annotate('',xy=(139.57055,np.max(short_rhythm_var_arr_plot)),xytext=(139.57055,np.max(short_rhythm_var_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.show()
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
# from matplotlib import gridspec
# fig = pyplot.figure(figsize=(12, 10))
# gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
# ax1=pyplot.subplot(gs[0])
# ax1.plot(window_time_arr[240*24:33000],long_rhythm_var_arr[240*24:33000],'darkblue',alpha=0.8)
# # ax1.plot(window_time_arr[240*6:33000],long_rhythm_var_arr[240*6:33000],'orange',alpha=0.7)
# # ax1.plot(window_time_arr[240*6:],long_rhythm_var_arr[240*6:],'darkblue',alpha=0.8)
# # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# ax1.set_title('RRI variance',fontsize=23)
# ax1.set_xlabel('Time (hours)',fontsize=23)
# ax1.set_ylabel('$\mathregular{S^2}$',fontsize=23)
# locs, labels = pyplot.xticks(fontsize=23)
# locs, labels = pyplot.yticks([0.0025,0.01,0.0175],fontsize=23)
# # locs, labels = pyplot.yticks([0.003,0.006,0.009],fontsize=23)
# ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)
# # ax1.scatter(window_time_arr[1487],long_rhythm_var_arr[1487],c='k',s=60)
# # ax1.scatter(window_time_arr[3829],long_rhythm_var_arr[3829],c='k',s=60)
# # ax1.scatter(window_time_arr[5081],long_rhythm_var_arr[5081],c='k',s=60)
# ax1.scatter(window_time_arr[7195],long_rhythm_var_arr[7195],c='k',s=60)
# ax1.scatter(window_time_arr[10117],long_rhythm_var_arr[10117],c='k',s=60)
# ax1.scatter(window_time_arr[12498],long_rhythm_var_arr[12498],c='k',s=60)
# ax1.scatter(window_time_arr[15489],long_rhythm_var_arr[15489],c='k',s=60)
# ax1.scatter(window_time_arr[18697],long_rhythm_var_arr[18697],s=60,c='k')
# ax1.scatter(window_time_arr[21542],long_rhythm_var_arr[21542],s=60,c='k')
# ax1.scatter(window_time_arr[24461],long_rhythm_var_arr[24461],s=60,c='k')
# ax1.scatter(window_time_arr[27043],long_rhythm_var_arr[27043],s=60,c='k')
# ax1.scatter(window_time_arr[29002],long_rhythm_var_arr[29002],s=60,c='k')
# ax1.scatter(window_time_arr[29974],long_rhythm_var_arr[29974],s=60,c='k')
# ax1.scatter(window_time_arr[31715],long_rhythm_var_arr[31715],s=60,c='k')
# ax1.scatter(window_time_arr[32778],long_rhythm_var_arr[32778],s=60,c='k')
# # pyplot.xlim(window_time_arr[0],window_time_arr[33000])
# ax2=pyplot.subplot(gs[1])
# ax2.set_xlabel('Time (hours)',fontsize=23)
# ax2.set_title('Instantaneous Phase',fontsize=23)
# # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax2.plot(window_time_arr[240*6:33000],phase_whole_long[240*6:33000],'k',alpha=0.5)
# ax2.plot(window_time_arr[240*24:33000],phase_whole_long[240*24:33000],'k',alpha=0.7)
# # ax2.scatter(window_time_arr[1487],phase_whole_long[1487],c='k',s=60)
# # ax2.scatter(window_time_arr[3829],phase_whole_long[3829],c='k',s=60)
# # ax2.scatter(window_time_arr[5081],phase_whole_long[5081],c='k',s=60)
# ax2.scatter(window_time_arr[7195],phase_whole_long[7195],c='k',s=60)
# ax2.scatter(window_time_arr[10117],phase_whole_long[10117],c='k',s=60)
# ax2.scatter(window_time_arr[12498],phase_whole_long[12498],c='k',s=60)
# ax2.scatter(window_time_arr[15489],phase_whole_long[15489],c='k',s=60)
# ax2.scatter(window_time_arr[18697],phase_whole_long[18697],c='k',s=60)
# ax2.scatter(window_time_arr[21542],phase_whole_long[21542],c='k',s=60)
# ax2.scatter(window_time_arr[24461],phase_whole_long[24461],c='k',s=60)
# ax2.scatter(window_time_arr[27043],phase_whole_long[27043],c='k',s=60)
# ax2.scatter(window_time_arr[29002],phase_whole_long[29002],c='k',s=60)
# ax2.scatter(window_time_arr[29974],phase_whole_long[29974],c='k',s=60)
# ax2.scatter(window_time_arr[31715],phase_whole_long[31715],c='k',s=60)
# ax2.scatter(window_time_arr[32778],phase_whole_long[32778],c='k',s=60)
# ax2.set_xlabel('Time (hours)',fontsize=23)
# locs, labels = pyplot.xticks(fontsize=23)
# locs, labels = pyplot.yticks(fontsize=23)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# # pyplot.hlines(0,window_time_arr[240*6],window_time_arr[33000],'k','dashed')
# pyplot.hlines(0,window_time_arr[240*24],window_time_arr[33000],'k','dashed')
# # locs, labels = pyplot.yticks([-0.5*np.pi,-0.25*np.pi,0,0.25*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=23)
# locs, labels = pyplot.yticks([-0.5*np.pi,-0.15*np.pi,0,0.3*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=23)
# pyplot.tight_layout()
# pyplot.show()
# # bins_number = 18
# # # bins = np.linspace(0, 2*np.pi, bins_number + 1)
# # bins = np.linspace(-np.pi, np.pi, bins_number + 1)
# # nRRIsvar, _, _ = pyplot.hist(phase_whole_long, bins)
# # nRRIsvarsei, _, _ = pyplot.hist(seizure_phase_var_long, bins)
# # print(nRRIsvar)
# # print(nRRIsvarsei)
# # width = 2*np.pi / bins_number
# # params = dict(projection='polar')
# # fig, ax = pyplot.subplots(subplot_kw=params)
# # ax.bar(bins[:bins_number], nRRIsvarsei/nRRIsvar,width=width, color='grey',alpha=0.6,linewidth=2, fill=True,edgecolor='k')
# # # pyplot.setp(ax.get_yticklabels(), color='k')
# # locs, labels = pyplot.yticks([0.0001,0.0003,0.0005],['0.0001','0.0003','0.0005'],fontsize=16)
# # # ax.set_yticks([0.00005,0.0002,0.00035,0.0005])
# # # ax.set_title('Phase histogram in RRI variance',fontsize=13)
# # ax.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',],fontsize=16)
# # pyplot.show()
# # params = dict(projection='polar')
# # fig, ax2 = pyplot.subplots(subplot_kw=params)
# # ax2.bar(bins[:bins_number], nRRIsvarsei/sum(nRRIsvarsei),width=width, color='grey',alpha=0.7,edgecolor='grey',linewidth=2)
# # # ax2.set_title('RRI variance',fontsize=13)
# # locs, labels = pyplot.yticks([0.2,0.6,1],['0.2','0.6','1'],fontsize=16)
# # ax2.annotate("", xy=( -0.21789, 0.94), xytext=(0, 0),arrowprops=dict(arrowstyle="->",color='g',linewidth=2))
# # ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',],fontsize=16)
# # pyplot.show()
#
# # long_rhythm_var_arr=short_rhythm_var_arr_plot
# # var_trans=hilbert(long_rhythm_var_arr)
# # var_trans_nomal=[]
# # for m in var_trans:
# #     var_trans_nomal.append(m/abs(m))
# # SIvarlong=sum(var_trans_nomal)/len(var_trans_nomal)
# # print(SIvarlong)
# # seizure_phase=[]
# # for item in seizure_timing_index:
# #      seizure_phase.append(var_trans_nomal[item])
# # SIvarlongseizure=sum(seizure_phase)/len(seizure_phase)
# # print(SIvarlongseizure)
# # var_phase=np.angle(var_trans)
# # phase_whole_long=var_phase
# # seizure_phase_var_long=[]
# # for item in seizure_timing_index:
# #     seizure_phase_var_long.append(phase_whole_long[item])
# # print(seizure_phase_var_long)
# # from matplotlib import gridspec
# # fig = pyplot.figure(figsize=(12, 10))
# # gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
# # ax1=pyplot.subplot(gs[0])
# # ax1.plot(window_time_arr[240*6:],long_rhythm_var_arr[240*6:],'orange',alpha=0.7)
# # # ax1.plot(window_time_arr[240*6:33000],long_rhythm_var_arr[240*6:33000],'orange',alpha=0.7)
# # # ax1.plot(window_time_arr[240*6:],long_rhythm_var_arr[240*6:],'darkblue',alpha=0.8)
# # # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax1.set_title('RRI variance',fontsize=23)
# # ax1.set_xlabel('Time (hours)',fontsize=23)
# # ax1.set_ylabel('$\mathregular{S^2}$',fontsize=23)
# # locs, labels = pyplot.xticks(fontsize=23)
# # locs, labels = pyplot.yticks([0.0025,0.01,0.0175],fontsize=23)
# # # locs, labels = pyplot.yticks([0.003,0.006,0.009],fontsize=23)
# # ax1.spines['right'].set_visible(False)
# # ax1.spines['top'].set_visible(False)
# # ax1.scatter(window_time_arr[1487],long_rhythm_var_arr[1487],c='k',s=60)
# # ax1.scatter(window_time_arr[3829],long_rhythm_var_arr[3829],c='k',s=60)
# # ax1.scatter(window_time_arr[5081],long_rhythm_var_arr[5081],c='k',s=60)
# # ax1.scatter(window_time_arr[7195],long_rhythm_var_arr[7195],c='k',s=60)
# # ax1.scatter(window_time_arr[10117],long_rhythm_var_arr[10117],c='k',s=60)
# # ax1.scatter(window_time_arr[12498],long_rhythm_var_arr[12498],c='k',s=60)
# # ax1.scatter(window_time_arr[15489],long_rhythm_var_arr[15489],c='k',s=60)
# # ax1.scatter(window_time_arr[18697],long_rhythm_var_arr[18697],s=60,c='k')
# # # ax1.scatter(window_time_arr[21542],long_rhythm_var_arr[21542],s=60,c='k')
# # # ax1.scatter(window_time_arr[24461],long_rhythm_var_arr[24461],s=60,c='k')
# # # ax1.scatter(window_time_arr[27043],long_rhythm_var_arr[27043],s=60,c='k')
# # # ax1.scatter(window_time_arr[29002],long_rhythm_var_arr[29002],s=60,c='k')
# # # ax1.scatter(window_time_arr[29974],long_rhythm_var_arr[29974],s=60,c='k')
# # # ax1.scatter(window_time_arr[31715],long_rhythm_var_arr[31715],s=60,c='k')
# # # ax1.scatter(window_time_arr[32778],long_rhythm_var_arr[32778],s=60,c='k')
# # # pyplot.xlim(window_time_arr[0],window_time_arr[33000])
# # ax2=pyplot.subplot(gs[1])
# # ax2.set_xlabel('Time (hours)',fontsize=23)
# # ax2.set_title('Instantaneous Phase',fontsize=23)
# # # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # # ax2.plot(window_time_arr[240*6:33000],phase_whole_long[240*6:33000],'k',alpha=0.5)
# # ax2.plot(window_time_arr[240*6:],phase_whole_long[240*6:],'k',alpha=0.7)
# # ax2.scatter(window_time_arr[1487],phase_whole_long[1487],c='k',s=60)
# # ax2.scatter(window_time_arr[3829],phase_whole_long[3829],c='k',s=60)
# # ax2.scatter(window_time_arr[5081],phase_whole_long[5081],c='k',s=60)
# # ax2.scatter(window_time_arr[7195],phase_whole_long[7195],c='k',s=60)
# # ax2.scatter(window_time_arr[10117],phase_whole_long[10117],c='k',s=60)
# # ax2.scatter(window_time_arr[12498],phase_whole_long[12498],c='k',s=60)
# # ax2.scatter(window_time_arr[15489],phase_whole_long[15489],c='k',s=60)
# # ax2.scatter(window_time_arr[18697],phase_whole_long[18697],c='k',s=60)
# # # ax2.scatter(window_time_arr[21542],phase_whole_long[21542],c='k',s=60)
# # # ax2.scatter(window_time_arr[24461],phase_whole_long[24461],c='k',s=60)
# # # ax2.scatter(window_time_arr[27043],phase_whole_long[27043],c='k',s=60)
# # # ax2.scatter(window_time_arr[29002],phase_whole_long[29002],c='k',s=60)
# # # ax2.scatter(window_time_arr[29974],phase_whole_long[29974],c='k',s=60)
# # # ax2.scatter(window_time_arr[31715],phase_whole_long[31715],c='k',s=60)
# # # ax2.scatter(window_time_arr[32778],phase_whole_long[32778],c='k',s=60)
# # ax2.set_xlabel('Time (hours)',fontsize=23)
# # locs, labels = pyplot.xticks(fontsize=23)
# # locs, labels = pyplot.yticks(fontsize=23)
# # ax2.spines['right'].set_visible(False)
# # ax2.spines['top'].set_visible(False)
# # # pyplot.hlines(0,window_time_arr[240*6],window_time_arr[33000],'k','dashed')
# # pyplot.hlines(0,window_time_arr[240*6],window_time_arr[-1],'k','dashed')
# # # locs, labels = pyplot.yticks([-0.5*np.pi,-0.25*np.pi,0,0.25*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=23)
# # locs, labels = pyplot.yticks([-0.5*np.pi,-0.15*np.pi,0,0.3*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=23)
# # pyplot.tight_layout()
# # pyplot.show()
#
#
#
#
#
# Raw_auto_RRI31=Raw_auto_RRI31_arr
# # # Raw_auto_RRI31=Raw_auto_RRI31_arr[0:19080]
# # Raw_auto_RRI31=Raw_auto_RRI31_arr[0:19624]
#
# long_rhythm_value_arr=movingaverage(Raw_auto_RRI31,240*24)
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
# # # pyplot.plot(window_time_arr[240:],medium_rhythm_value_arr[240:],'g',label='1 hours')
# # # pyplot.plot(window_time_arr[240*3:],medium_rhythm_value_arr_2[240*3:],'k',label='3 hours')
# # # pyplot.plot(window_time_arr[240*6:],medium_rhythm_value_arr_3[240*6:],'orange',label='6 hours')
# # # pyplot.plot(window_time_arr[240*12:],medium_rhythm_value_arr_4[240*12:],'b',label='12 hours')
# # # pyplot.plot(window_time_arr[5760:],long_rhythm_value_arr[5760:],'r',alpha=0.7,label='1 day')
# # pyplot.plot(window_time_arr,short_rhythm_value_arr_plot,'grey',alpha=0.5,label='5min')
# # pyplot.plot(window_time_arr,medium_rhythm_value_arr,'g',label='1 hours')
# # pyplot.plot(window_time_arr,medium_rhythm_value_arr_2,'k',label='3 hours')
# # pyplot.plot(window_time_arr,medium_rhythm_value_arr_3,'orange',label='6 hours')
# # pyplot.plot(window_time_arr,medium_rhythm_value_arr_4,'b',label='12 hours')
# # pyplot.plot(window_time_arr,long_rhythm_value_arr,'r',alpha=0.7,label='1 day')
# # pyplot.title('RRI autocorrelation in SA0124',fontsize=23)
# # pyplot.xlabel('Time (hours)',fontsize=23)
# # pyplot.ylabel('Autocorrelation',fontsize=23)
# # pyplot.legend(loc='upper left',fontsize=10)
# # locs, labels = pyplot.xticks(fontsize=23)
# # locs, labels = pyplot.yticks(fontsize=23)
# # pyplot.tight_layout()
# # pyplot.annotate('',xy=(9.19205,np.max(short_rhythm_value_arr_plot)),xytext=(9.19205,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(18.9488833,np.max(short_rhythm_value_arr_plot)),xytext=(18.9488833,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(24.16555,np.max(short_rhythm_value_arr_plot)),xytext=(24.16555,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(32.9738833,np.max(short_rhythm_value_arr_plot)),xytext=(32.9738833,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(45.149161,np.max(short_rhythm_value_arr_plot)),xytext=(45.149161,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(55.0694389,np.max(short_rhythm_value_arr_plot)),xytext=(55.0694389,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(67.5319389,np.max(short_rhythm_value_arr_plot)),xytext=(67.5319389,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # pyplot.annotate('',xy=(80.90055,np.max(short_rhythm_value_arr_plot)),xytext=(80.90055,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(92.7538833,np.max(short_rhythm_value_arr_plot)),xytext=(92.7538833,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(104.916106,np.max(short_rhythm_value_arr_plot)),xytext=(104.916106,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(115.673883,np.max(short_rhythm_value_arr_plot)),xytext=(115.673883,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(123.834,np.max(short_rhythm_value_arr_plot)),xytext=(123.834,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(127.884278,np.max(short_rhythm_value_arr_plot)),xytext=(127.884278,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(135.1409,np.max(short_rhythm_value_arr_plot)),xytext=(135.1409,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# # # pyplot.annotate('',xy=(139.57055,np.max(short_rhythm_value_arr_plot)),xytext=(139.57055,np.max(short_rhythm_value_arr_plot)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
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
# from matplotlib import gridspec
# fig = pyplot.figure(figsize=(12, 10))
# gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
# ax1=pyplot.subplot(gs[0])
# ax1.set_title('RRI autocorrelation',fontsize=23)
# ax1.set_xlabel('Time (hours)',fontsize=23)
# locs, labels = pyplot.xticks(fontsize=23)
# locs, labels = pyplot.yticks([1.6,2.3,3],fontsize=23)
# ax1.spines['right'].set_visible(False)
# ax1.spines['top'].set_visible(False)
# ax1.plot(window_time_arr[240*24:33000],long_rhythm_value_arr[240*24:33000],'darkblue',alpha=0.8)
# # ax1.plot(window_time_arr[240*6:33000],long_rhythm_value_arr[240*6:33000],'orange',alpha=0.7)
# # ax1.plot(window_time_arr[240*6:19600],long_rhythm_value_arr[240*6:19600],'darkblue',alpha=0.8)
# # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax1.scatter(window_time_arr[1487],long_rhythm_value_arr[1487],c='k',s=60)
# # ax1.scatter(window_time_arr[3829],long_rhythm_value_arr[3829],c='k',s=60)
# # ax1.scatter(window_time_arr[5081],long_rhythm_value_arr[5081],c='k',s=60)
# ax1.scatter(window_time_arr[7195],long_rhythm_value_arr[7195],c='k',s=60)
# ax1.scatter(window_time_arr[10117],long_rhythm_value_arr[10117],c='k',s=60)
# ax1.scatter(window_time_arr[12498],long_rhythm_value_arr[12498],c='k',s=60)
# ax1.scatter(window_time_arr[15489],long_rhythm_value_arr[15489],c='k',s=60)
# ax1.scatter(window_time_arr[18697],long_rhythm_value_arr[18697],s=60,c='k')
# ax1.scatter(window_time_arr[21542],long_rhythm_value_arr[21542],s=60,c='k')
# ax1.scatter(window_time_arr[24461],long_rhythm_value_arr[24461],s=60,c='k')
# ax1.scatter(window_time_arr[27043],long_rhythm_value_arr[27043],s=60,c='k')
# ax1.scatter(window_time_arr[29002],long_rhythm_value_arr[29002],s=60,c='k')
# ax1.scatter(window_time_arr[29974],long_rhythm_value_arr[29974],s=60,c='k')
# ax1.scatter(window_time_arr[31715],long_rhythm_value_arr[31715],s=60,c='k')
# ax1.scatter(window_time_arr[32778],long_rhythm_value_arr[32778],s=60,c='k')
# ax2=pyplot.subplot(gs[1])
# ax2.set_title('Instantaneous Phase',fontsize=23)
# ax2.set_xlabel('Time (hours)',fontsize=23)
# locs, labels = pyplot.xticks(fontsize=23)
# locs, labels = pyplot.yticks(fontsize=23)
# ax2.spines['right'].set_visible(False)
# ax2.spines['top'].set_visible(False)
# # pyplot.hlines(0,window_time_arr[240*6],window_time_arr[33000],'k','dashed')
# pyplot.hlines(0,window_time_arr[240*24],window_time_arr[33000],'k','dashed')
# # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# ax2.plot(window_time_arr[240*24:33000],phase_whole_value_long[240*24:33000],'k',alpha=0.5)
# # ax2.plot(window_time_arr[240*6:33000],phase_whole_value_long[240*6:33000],'k',alpha=0.5)
# # ax2.plot(window_time_arr[240*6:19600],phase_whole_value_long[240*6:19600],'k',alpha=0.7)
# # ax2.scatter(window_time_arr[1487],phase_whole_value_long[1487],c='k',s=60)
# # ax2.scatter(window_time_arr[3829],phase_whole_value_long[3829],c='k',s=60)
# # ax2.scatter(window_time_arr[5081],phase_whole_value_long[5081],c='k',s=60)
# ax2.scatter(window_time_arr[7195],phase_whole_value_long[7195],c='k',s=60)
# ax2.scatter(window_time_arr[10117],phase_whole_value_long[10117],c='k',s=60)
# ax2.scatter(window_time_arr[12498],phase_whole_value_long[12498],c='k',s=60)
# ax2.scatter(window_time_arr[15489],phase_whole_value_long[15489],c='k',s=60)
# ax2.scatter(window_time_arr[18697],phase_whole_value_long[18697],c='k',s=60)
# ax2.scatter(window_time_arr[21542],phase_whole_value_long[21542],c='k',s=60)
# ax2.scatter(window_time_arr[24461],phase_whole_value_long[24461],c='k',s=60)
# ax2.scatter(window_time_arr[27043],phase_whole_value_long[27043],c='k',s=60)
# ax2.scatter(window_time_arr[29002],phase_whole_value_long[29002],c='k',s=60)
# ax2.scatter(window_time_arr[29974],phase_whole_value_long[29974],c='k',s=60)
# ax2.scatter(window_time_arr[31715],phase_whole_value_long[31715],c='k',s=60)
# ax2.scatter(window_time_arr[32778],phase_whole_value_long[32778],c='k',s=60)
# ax2.set_xlabel('Time (hours)',fontsize=23)
# # locs, labels = pyplot.yticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi],['-$\pi$','Rising','0','Falling','$\pi$'],rotation='vertical',fontsize=23)
# # locs, labels = pyplot.yticks([-0.5*np.pi,-0.25*np.pi,0,0.25*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=23)
# locs, labels = pyplot.yticks([-0.5*np.pi,-0.15*np.pi,0,0.3*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=23)
# # locs, labels = pyplot.yticks([-0.25*np.pi,-0.125*np.pi,0,0.125*np.pi,0.25*np.pi],['-0.25$\pi$','Rising','0','Falling','0.25$\pi$'],rotation='vertical',fontsize=23)
# pyplot.tight_layout()
# pyplot.show()
#
# # bins_number = 18
# # bins = np.linspace(-np.pi, np.pi, bins_number + 1)
# # nRRIsauto, _, _ = pyplot.hist(phase_whole_value_long, bins)
# # nRRIsautosei, _, _ = pyplot.hist(seizure_phase_value_long, bins)
# # print(nRRIsauto)
# # print(nRRIsautosei)
# # width = 2*np.pi / bins_number
# # params = dict(projection='polar')
# # fig, ax = pyplot.subplots(subplot_kw=params)
# # ax.bar(bins[:bins_number], nRRIsautosei/nRRIsauto,width=width, color='grey',alpha=0.6,linewidth=2, fill=True,edgecolor='k')
# # pyplot.setp(ax.get_yticklabels(), color='k',fontsize=16)
# # # ax.set_title('phase histogram in RRI autocorrelation',fontsize=13)
# # locs, labels = pyplot.yticks([0.0002,0.0004,0.0006],['0.0002','0.0004','0.0006'],fontsize=16)
# # ax.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',],fontsize=16)
# # pyplot.show()
# #
# # params = dict(projection='polar')
# # fig, ax2 = pyplot.subplots(subplot_kw=params)
# # ax2.bar(bins[:bins_number], nRRIsautosei/sum(nRRIsautosei),width=width, color='grey',alpha=0.7,edgecolor='grey',linewidth=2)
# # ax2.annotate("", xy=( -0.22195, 0.97), xytext=(0, 0),arrowprops=dict(arrowstyle="->",color='g',linewidth=2))
# # locs, labels = pyplot.yticks([0.2,0.6,1],['0.2','0.6','1'],fontsize=16)
# # ax2.set_xticklabels(['0','0.25$\pi$','0.5$\pi$','0.75$\pi$','-$\pi$','-0.75$\pi$','-0.5$\pi$','-0.25$\pi$',],fontsize=16)
# # pyplot.show()
#
#
#
#
# # long_rhythm_value_arr=short_rhythm_value_arr_plot
# # value_trans=hilbert(long_rhythm_value_arr)
# # value_trans_nomal=[]
# # for m in value_trans:
# #     value_trans_nomal.append(m/abs(m))
# # SIvaluelong=sum(value_trans_nomal)/len(value_trans_nomal)
# # print(SIvaluelong)
# # seizure_phase=[]
# # for item in seizure_timing_index:
# #     seizure_phase.append(value_trans_nomal[item])
# # SIvaluelongseizure=sum(seizure_phase)/len(seizure_phase)
# # print(SIvaluelongseizure)
# # value_phase=np.angle(value_trans)
# # phase_whole_value_long=value_phase
# # seizure_phase_value_long=[]
# # for item in seizure_timing_index:
# #     seizure_phase_value_long.append(phase_whole_value_long[item])
# # print(seizure_phase_value_long)
# #
# # from matplotlib import gridspec
# # fig = pyplot.figure(figsize=(12, 10))
# # gs = gridspec.GridSpec(2, 1, height_ratios=[3, 3])
# # ax1=pyplot.subplot(gs[0])
# # ax1.set_title('RRI autocorrelation',fontsize=23)
# # ax1.set_xlabel('Time (hours)',fontsize=23)
# # locs, labels = pyplot.xticks(fontsize=23)
# # locs, labels = pyplot.yticks([1.6,2.3,3],fontsize=23)
# # ax1.spines['right'].set_visible(False)
# # ax1.spines['top'].set_visible(False)
# # ax1.plot(window_time_arr[240*6:],long_rhythm_value_arr[240*6:],'orange',alpha=0.7)
# # # ax1.plot(window_time_arr[240*6:33000],long_rhythm_value_arr[240*6:33000],'orange',alpha=0.7)
# # # ax1.plot(window_time_arr[240*6:19600],long_rhythm_value_arr[240*6:19600],'darkblue',alpha=0.8)
# # # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax1.scatter(window_time_arr[1487],long_rhythm_value_arr[1487],c='k',s=60)
# # ax1.scatter(window_time_arr[3829],long_rhythm_value_arr[3829],c='k',s=60)
# # ax1.scatter(window_time_arr[5081],long_rhythm_value_arr[5081],c='k',s=60)
# # ax1.scatter(window_time_arr[7195],long_rhythm_value_arr[7195],c='k',s=60)
# # ax1.scatter(window_time_arr[10117],long_rhythm_value_arr[10117],c='k',s=60)
# # ax1.scatter(window_time_arr[12498],long_rhythm_value_arr[12498],c='k',s=60)
# # ax1.scatter(window_time_arr[15489],long_rhythm_value_arr[15489],c='k',s=60)
# # ax1.scatter(window_time_arr[18697],long_rhythm_value_arr[18697],s=60,c='k')
# # # ax1.scatter(window_time_arr[21542],long_rhythm_value_arr[21542],s=60,c='k')
# # # ax1.scatter(window_time_arr[24461],long_rhythm_value_arr[24461],s=60,c='k')
# # # ax1.scatter(window_time_arr[27043],long_rhythm_value_arr[27043],s=60,c='k')
# # # ax1.scatter(window_time_arr[29002],long_rhythm_value_arr[29002],s=60,c='k')
# # # ax1.scatter(window_time_arr[29974],long_rhythm_value_arr[29974],s=60,c='k')
# # # ax1.scatter(window_time_arr[31715],long_rhythm_value_arr[31715],s=60,c='k')
# # # ax1.scatter(window_time_arr[32778],long_rhythm_value_arr[32778],s=60,c='k')
# # ax2=pyplot.subplot(gs[1])
# # ax2.set_title('Instantaneous Phase',fontsize=23)
# # ax2.set_xlabel('Time (hours)',fontsize=23)
# # locs, labels = pyplot.xticks(fontsize=23)
# # locs, labels = pyplot.yticks(fontsize=23)
# # ax2.spines['right'].set_visible(False)
# # ax2.spines['top'].set_visible(False)
# # # pyplot.hlines(0,window_time_arr[240*6],window_time_arr[33000],'k','dashed')
# # pyplot.hlines(0,window_time_arr[240*6],window_time_arr[-1],'k','dashed')
# # # pyplot.xlim(window_time_arr[0],window_time_arr[-1])
# # ax2.plot(window_time_arr[240*6:],phase_whole_value_long[240*6:],'k',alpha=0.5)
# # # ax2.plot(window_time_arr[240*6:33000],phase_whole_value_long[240*6:33000],'k',alpha=0.5)
# # # ax2.plot(window_time_arr[240*6:19600],phase_whole_value_long[240*6:19600],'k',alpha=0.7)
# # ax2.scatter(window_time_arr[1487],phase_whole_value_long[1487],c='k',s=60)
# # ax2.scatter(window_time_arr[3829],phase_whole_value_long[3829],c='k',s=60)
# # ax2.scatter(window_time_arr[5081],phase_whole_value_long[5081],c='k',s=60)
# # ax2.scatter(window_time_arr[7195],phase_whole_value_long[7195],c='k',s=60)
# # ax2.scatter(window_time_arr[10117],phase_whole_value_long[10117],c='k',s=60)
# # ax2.scatter(window_time_arr[12498],phase_whole_value_long[12498],c='k',s=60)
# # ax2.scatter(window_time_arr[15489],phase_whole_value_long[15489],c='k',s=60)
# # ax2.scatter(window_time_arr[18697],phase_whole_value_long[18697],c='k',s=60)
# # # ax2.scatter(window_time_arr[21542],phase_whole_value_long[21542],c='k',s=60)
# # # ax2.scatter(window_time_arr[24461],phase_whole_value_long[24461],c='k',s=60)
# # # ax2.scatter(window_time_arr[27043],phase_whole_value_long[27043],c='k',s=60)
# # # ax2.scatter(window_time_arr[29002],phase_whole_value_long[29002],c='k',s=60)
# # # ax2.scatter(window_time_arr[29974],phase_whole_value_long[29974],c='k',s=60)
# # # ax2.scatter(window_time_arr[31715],phase_whole_value_long[31715],c='k',s=60)
# # # ax2.scatter(window_time_arr[32778],phase_whole_value_long[32778],c='k',s=60)
# # ax2.set_xlabel('Time (hours)',fontsize=23)
# # # locs, labels = pyplot.yticks([-np.pi,-0.5*np.pi,0,0.5*np.pi,np.pi],['-$\pi$','Rising','0','Falling','$\pi$'],rotation='vertical',fontsize=23)
# # # locs, labels = pyplot.yticks([-0.5*np.pi,-0.25*np.pi,0,0.25*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=23)
# # locs, labels = pyplot.yticks([-0.5*np.pi,-0.15*np.pi,0,0.3*np.pi,0.5*np.pi],['-0.5$\pi$','Rising','0','Falling','0.5$\pi$'],rotation='vertical',fontsize=23)
# # # locs, labels = pyplot.yticks([-0.25*np.pi,-0.125*np.pi,0,0.125*np.pi,0.25*np.pi],['-0.25$\pi$','Rising','0','Falling','0.25$\pi$'],rotation='vertical',fontsize=23)
# # pyplot.tight_layout()
# # pyplot.show()
#
#
#
#
# # t=np.linspace(2.98805+0.00416667,2.98805+0.00416667+0.00416667*(len(Raw_variance_EEG_arr)-1),len(Raw_variance_EEG_arr))
# # window_time_arr=t
# # # print(len(t));
# # # print(t[0]);print(t[19080]);print(t[-1]);
# # a=np.where(t<7.73806+2.98805)
# # # print(a);print(t[1856]);print(t[1857])
# # t[0:1857]=t[0:1857]-2.98805+16.26194
# # t[1857:]=t[1857:]-7.73806-2.98805
# # # print(t[1857]);print(t);print(type(t));print(t[0])
# #
# # time_feature_arr=[]
# # for i in range(len(t)):
# #     if t[i]>24:
# #         time_feature_arr.append(t[i] - (t[i] // 24) * 24)
# #     else:
# #         time_feature_arr.append(t[i])
# # seizure_time=[time_feature_arr[1487],time_feature_arr[3829],time_feature_arr[5081],time_feature_arr[7195],
# # time_feature_arr[10117],time_feature_arr[12498],time_feature_arr[15489],time_feature_arr[18697],
# # # time_feature_arr[21542],time_feature_arr[24461],time_feature_arr[27043],time_feature_arr[29002],
# # # time_feature_arr[29974],time_feature_arr[31715],time_feature_arr[32778],
# # ]
# # print(seizure_time)
# # bins_number = 18
# # bins = np.linspace(0, 24, bins_number + 1)
# # nEEGsvar, _, _ = pyplot.hist(time_feature_arr[0:19624], bins)
# # nEEGsvarsei, _, _ = pyplot.hist(seizure_time, bins)
# # print(nEEGsvarsei)
# # # bins = np.linspace(0, 2*np.pi, bins_number + 1)
# # # width = 2*np.pi / bins_number
# # # params = dict(projection='polar')
# # # fig, ax = pyplot.subplots(subplot_kw=params)
# # # ax.bar(bins[:bins_number], nEEGsvarsei/nEEGsvar,width=width, color='grey',alpha=0.7,edgecolor='k',linewidth=2)
# # # pyplot.setp(ax.get_yticklabels(), color='k')
# # # # ax.set_title('seizure timing histogram (SA0124)',fontsize=23)
# # # ax.set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))
# # # ax.set_xticklabels(range(24))
# # # # ax.set_yticks([0.05,0.1,0.15,0.2,0.25])
# # # pyplot.show()
# #
# #
# # bins = np.linspace(0, 2*np.pi, bins_number + 1)
# # print(bins)
# # width = 2*np.pi / bins_number
# # params = dict(projection='polar')
# # fig, ax = pyplot.subplots(subplot_kw=params)
# # ax.bar(bins[:bins_number], nEEGsvarsei/sum(nEEGsvarsei),width=width, color='grey',alpha=0.7,edgecolor='grey',linewidth=2)
# # pyplot.setp(ax.get_yticklabels(), color='k')
# # ax.set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))
# # ax.set_xticklabels(['0 am','','','Night','','','6 am','','','Morning','','','12 am','','','Afternoon','','','18 pm','','','Evening','','','24 pm'],fontsize=16)
# # locs, labels = pyplot.yticks([0.2,0.6,1],['0.2','0.6','1'],fontsize=16)
# # ax.annotate("", xy=(-1.45006295853334, 0.0958943008675168), xytext=(0, 0),arrowprops=dict(arrowstyle="->",color='g',linewidth=2))
# # pyplot.show()
# #
# #
# #
# # bins = np.linspace(0, 2*np.pi, bins_number + 1)
# # width = 2*np.pi / bins_number
# # params = dict(projection='polar')
# # fig, ax = pyplot.subplots(subplot_kw=params)
# # ax.bar(bins[:bins_number], nEEGsvarsei/sum(nEEGsvar),width=width, color='grey',alpha=0.7,edgecolor='k',linewidth=2)
# # pyplot.setp(ax.get_yticklabels(), color='k')
# # # ax.set_title('seizure timing histogram (SA0124)',fontsize=23)
# # ax.set_xticks(np.linspace(0, 2*np.pi, 24, endpoint=False))
# # ax.set_xticklabels(['0 am','','','Night','','','6 am','','','Morning','','','12 am','','','Afternoon','','','18 pm','','','Evening','','','24 pm'],fontsize=16)
# # # locs, labels = pyplot.xticks([0,3,6,9,12,15,18,21,24],['0','Night','6','Morning','12','Afternoon','18','Evening','24'],fontsize=16)
# # locs, labels = pyplot.yticks([0.00004,0.00010,0.00016],['0.00004','0.00010','0.00016'],fontsize=16)
# # pyplot.show()
# # # bins_number = 18
# # # bins = np.linspace(0, 24, bins_number + 1)
# # # ntimes, _, _ = pyplot.hist(time_feature_arr[0:19624], bins)
# # # ntimesei, _, _ = pyplot.hist(seizure_time, bins)
# # # print(ntimes)
# # # print(ntimesei)
# # # print(seizure_time)
# #
# # # bins_number = 24
# # # bins = np.linspace(0, 24, bins_number + 1)
# # # nEEGsvar, _, _ = pyplot.hist(time_feature_arr[0:19624], bins)
# # # nEEGsvarsei, _, _ = pyplot.hist(seizure_time, bins)
# # # width = 24 / bins_number
# # # fig, ax = pyplot.subplots()
# # # ax.bar(bins[:bins_number], nEEGsvarsei,width=width, color='darkblue',alpha=0.6,edgecolor='k',linewidth=2)
# # # pyplot.setp(ax.get_yticklabels(), color='k',fontsize=12)
# # # ax.set_xticks(np.linspace(0, 24, 24, endpoint=False))
# # # ax.set_xticklabels(range(24))
# # # ax.set_xlim([-0.5,24.5])
# # # pyplot.axhline(-0.04,0,0.25,linewidth=6,c='k')
# # # pyplot.axhline(-0.04,0.25,0.75,linewidth=6,c='k',alpha=0.3)
# # # pyplot.axhline(-0.04,0.75,1,linewidth=6,c='k')
# # # locs, labels = pyplot.xticks([0,3,6,9,12,15,18,21,24],['0 am','Night','6 am','Morning','12 am','Afternoon','18 pm','Evening','24 pm'],rotation=30,fontsize=12)
# # # locs, labels = pyplot.yticks([0,1,2,3],['0','1','2','3'],fontsize=16)
# # # ax.spines['right'].set_visible(False)
# # # ax.spines['top'].set_visible(False)
# # # ax.set_xlabel('Time',fontsize=14)
# # # ax.set_ylabel('Number of seizures',fontsize=14)
# # # pyplot.show()



