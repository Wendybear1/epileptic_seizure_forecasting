from __future__ import division
import mne
import numpy as np
import scipy.signal
from matplotlib import pyplot
import math
from scipy.fftpack import fft, ifft
from scipy import signal
from scipy.signal import butter, lfilter,iirfilter
from scipy.signal import hilbert
from biosppy.signals import tools
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg, ar_select_order
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.ar_model import AutoReg
import scipy.stats as stats
from scipy.stats import norm


from statsmodels.tsa.api import SARIMAX
# ##### An extension to ARIMA that supports the direct modeling of the seasonal component of the series is called SARIMA.
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.ar_model import AutoReg, ar_select_order


def movingaverage(values, window_size):
    weights = (np.ones(window_size))/window_size
    a=np.ones(1)
    return lfilter(weights,a,values)

def split(arr, size):
    arrs = []
    while len(arr) > size:
        pice = arr[:size]
        arrs.append(pice)
        arr = arr[size:]
    arrs.append(arr)
    return arrs

# # ###SA0124
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/SA0124channels/EEG_timewindowarr_SA0124_15s.csv', sep=',',header=None)
t_window_arr = csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_ch31_rawvariance_SA0124_15s_3h.csv',sep=',',header=None)
Raw_variance_RRI31= csv_reader.values
print(t_window_arr[0]);

Raw_variance_RRI31_arr=[]
for item in Raw_variance_RRI31:
    Raw_variance_RRI31_arr.append(float(item))
fore_arr_RRIvars=[]
# for k in range(71,72):
for k in range(81, 82):
    variance_arr = Raw_variance_RRI31_arr[0:(19624+240*k)]
    long_rhythm_var_arr=movingaverage(variance_arr,240*6)
    pyplot.figure(figsize=(6, 3))
    ax = pyplot.subplot(111)
    pyplot.title('RRI variance',fontsize=14)
    pyplot.ylabel('$\mathregular{s^2}$',fontsize=14)
    pyplot.xlabel('Time (hours)',fontsize=14)
    pyplot.plot(t_window_arr[240*6:(19624+240*k)], long_rhythm_var_arr[240*6:],'darkblue',alpha=0.7,label='record')


csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/forecast/forecast81hsignal_3hcycle_RRIvar_SA0124_1.csv',sep=',',header=None)
# csv_reader = pd.read_csv('C:/Users/wxiong/PycharmProjects/EEGstudy/forecast81hsignal_3hcycle_RRIvar_SA0124_ARIMA.csv',sep=',',header=None)
forecast_var_RRI31= csv_reader.values
forecast_var_RRI31_arr=[]
for item in forecast_var_RRI31:
    forecast_var_RRI31_arr=forecast_var_RRI31_arr+list(item)
t=np.linspace(t_window_arr[19624]+0.1666667,t_window_arr[19624]+0.1666667+0.1666667*(len(forecast_var_RRI31_arr)-1),len(forecast_var_RRI31_arr))
# t=np.linspace(t_window_arr[19624],t_window_arr[19624]+0.1666667*(len(forecast_var_RRI31_arr)-1),len(forecast_var_RRI31_arr))
pyplot.plot(t, forecast_var_RRI31_arr,'k',label='forecast')
locs, labels = pyplot.xticks(fontsize=14)
locs, labels = pyplot.yticks([0.005,0.015,0.025],fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
pyplot.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
# pyplot.legend(fontsize=14)
pyplot.show()



# # # # # # # # ###forecast ECG auto
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/ECG/RRI_ch31_rawauto_SA0124_15s_3h.csv',sep=',',header=None)
Raw_auto_RRI31= csv_reader.values
Raw_auto_RRI31_arr=[]
for item in Raw_auto_RRI31:
    Raw_auto_RRI31_arr.append(float(item))
fore_arr_RRIautos=[]
save_data_RRIautos=[]
for k in range(81,82):
    auto_arr = Raw_auto_RRI31_arr[0:19624+240*k]
    long_rhythm_auto_arr=movingaverage(auto_arr,240*6)
    pyplot.figure(figsize=(6,3))
    ax = pyplot.subplot(111)
    pyplot.title('RRI autocorrelation',fontsize=14)
    pyplot.xlabel('Time (hours)',fontsize=14)
    pyplot.plot(t_window_arr[240*6:19624+240*k], long_rhythm_auto_arr[240*6:],'darkblue',alpha=0.7,label='record')
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/forecast/forecast81hsignal_3hcycle_RRIauto_SA0124_1.csv',sep=',',header=None)
# csv_reader = pd.read_csv('C:/Users/wxiong/PycharmProjects/EEGstudy/forecast81hsignal_3hcycle_RRIauto_SA0124_ARIMA.csv',sep=',',header=None)

forecast_auto_RRI31= csv_reader.values
forecast_auto_RRI31_arr=[]
for item in forecast_auto_RRI31:
    forecast_auto_RRI31_arr=forecast_auto_RRI31_arr+list(item)
t=np.linspace(t_window_arr[19624]+0.1666667,t_window_arr[19624]+0.1666667+0.1666667*(len(forecast_auto_RRI31_arr)-1),len(forecast_auto_RRI31_arr))
# t=np.linspace(t_window_arr[19624],t_window_arr[19624]+0.1666667*(len(forecast_auto_RRI31_arr)-1),len(forecast_auto_RRI31_arr))
pyplot.plot(t, forecast_auto_RRI31_arr,'k',label='forecast')
locs, labels = pyplot.yticks([1.7, 2.4, 3.1],fontsize=14)
locs, labels = pyplot.xticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
pyplot.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
# pyplot.legend(fontsize=14)
pyplot.show()




# # # # # ###forecast EEG var
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/SA0124channels/EEGvariance_SA0124_15s_3h.csv',sep=',',header=None)
Raw_variance_EEG= csv_reader.values
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/SA0124channels/EEGauto_SA0124_15s_3h.csv',sep=',',header=None)
Raw_auto_EEG= csv_reader.values
Raw_variance_EEG_arr=[]
for item in Raw_variance_EEG:
    Raw_variance_EEG_arr.append(float(item))
Raw_auto_EEG_arr=[]
for item in Raw_auto_EEG:
    Raw_auto_EEG_arr.append(float(item))

var_arr=[]
for item in Raw_variance_EEG_arr:
    if item<1e-8:
        var_arr.append(item*(10**12))
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



# t=np.linspace(2.98805,2.98805+0.00416667*(len(Raw_variance_EEG_arr)-1),len(Raw_variance_EEG_arr))
# t_window_arr=t

fore_arr_EEGvars=[]
for k in range(81,82):
    variance_arr = Raw_variance_EEG_arr[0:(19624+240*k)]
    long_rhythm_var_arr=movingaverage(variance_arr,240*6)
    pyplot.figure(figsize=(6, 3))
    ax = pyplot.subplot(111)
    pyplot.title('EEG variance',fontsize=14)
    pyplot.ylabel('$\mathregular{\u03BCV^2}$',fontsize=14)
    pyplot.xlabel('Time (hours)',fontsize=14)
    pyplot.plot(t_window_arr[240*6:(19624+240*k)], long_rhythm_var_arr[240*6:],'darkblue',alpha=0.7,label='record')
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/forecast/forecast81hsignal_3hcycle_EEGvar_SA0124_1.csv',sep=',',header=None)
forecast_var_EEG= csv_reader.values

forecast_var_EEG_arr=[]
for item in forecast_var_EEG:
    forecast_var_EEG_arr=forecast_var_EEG_arr+list(item*10**12)
t=np.linspace(t_window_arr[19624]+0.1666667,t_window_arr[19624]+0.1666667+0.1666667*(len(forecast_var_EEG_arr)-1),len(forecast_var_EEG_arr))
# t=np.linspace(t_window_arr[19624],t_window_arr[19624]+0.1666667*(len(forecast_var_EEG_arr)-1),len(forecast_var_EEG_arr))
pyplot.plot(t, forecast_var_EEG_arr,'k',label='forecast')
pyplot.legend(fontsize=14)
locs, labels = pyplot.yticks([0, 300, 600],fontsize=14)
locs, labels = pyplot.xticks(fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
pyplot.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
pyplot.show()





# # # # # # # ###forecast EEG auto
fore_arr_EEGauto=[]
for k in range(81,82):
    auto_arr = Raw_auto_EEG_arr[0:(19624+240*k)]
    long_rhythm_auto_arr=movingaverage(auto_arr,240*6)
    pyplot.figure(figsize=(6, 3))
    ax = pyplot.subplot(111)
    pyplot.title('EEG autocorrelation',fontsize=14)
    pyplot.xlabel('Time (hours)',fontsize=14)
    pyplot.plot(t_window_arr[240*6:(19624+240*k)], long_rhythm_auto_arr[240*6:],'darkblue',alpha=0.7,label='recorded')
csv_reader = pd.read_csv('C:/Users/wxiong/Documents/PHD/2020.August/forecast/forecast81hsignal_3hcycle_EEGauto_SA0124_1.csv',sep=',',header=None)
# csv_reader = pd.read_csv('C:/Users/wxiong/PycharmProjects/EEGstudy/forecast81hsignal_3hcycle_EEGauto_SA0124_ARIMA.csv',sep=',',header=None)
forecast_auto_EEG= csv_reader.values
forecast_auto_EEG_arr=[]
for item in forecast_auto_EEG:
    forecast_auto_EEG_arr=forecast_auto_EEG_arr+list(item)
t=np.linspace(t_window_arr[19624]+0.1666667,t_window_arr[19624]+0.1666667+0.1666667*(len(forecast_auto_EEG_arr)-1),len(forecast_auto_EEG_arr))
# t=np.linspace(t_window_arr[19624],t_window_arr[19624]+0.1666667*(len(forecast_auto_EEG_arr)-1),len(forecast_auto_EEG_arr))
pyplot.plot(t, forecast_auto_EEG_arr,'k',label='forecast')
# pyplot.legend(fontsize=14)
locs, labels = pyplot.xticks(fontsize=14)
locs, labels = pyplot.yticks([20,27,34],fontsize=14)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
pyplot.show()
print(len(forecast_var_EEG_arr));print(len(forecast_auto_EEG_arr));
print(len(forecast_var_RRI31_arr));print(len(forecast_auto_RRI31_arr))



# ### predict, forecast data
var_trans=hilbert(forecast_var_EEG_arr)
var_phase=np.angle(var_trans)
rolmean_short_EEGvar=var_phase

var_trans=hilbert(forecast_auto_EEG_arr)
var_phase=np.angle(var_trans)
rolmean_short_EEGauto=var_phase

var_trans=hilbert(forecast_var_RRI31_arr)
var_phase=np.angle(var_trans)
rolmean_short_RRIvar=var_phase

var_trans=hilbert(forecast_auto_RRI31_arr)
var_phase=np.angle(var_trans)
rolmean_short_RRIauto=var_phase

# print(rolmean_short_EEGvar);print(rolmean_short_EEGauto)
# print(rolmean_short_RRIvar);print(rolmean_short_RRIauto)


# #### combined probability calculation
bins_number = 18
bins = np.linspace(-np.pi, np.pi, bins_number + 1)
pro_eegvars_time = []
pro_eegvars_time_false = []
for i in range(len(rolmean_short_EEGvar)):
    if rolmean_short_EEGvar[i] >= bins[0] and rolmean_short_EEGvar[i] < bins[1]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[1] and rolmean_short_EEGvar[i] < bins[2]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[2] and rolmean_short_EEGvar[i] < bins[3]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[3] and rolmean_short_EEGvar[i] < bins[4]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[4] and rolmean_short_EEGvar[i] < bins[5]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[5] and rolmean_short_EEGvar[i] < bins[6]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[6] and rolmean_short_EEGvar[i] <= bins[7]:
        pro_eegvars_time_false.append(0.065966558)
        pro_eegvars_time.append(0.25)
    elif rolmean_short_EEGvar[i] > bins[7] and rolmean_short_EEGvar[i] < bins[8]:
        pro_eegvars_time_false.append(0.187805873)
        pro_eegvars_time.append(0.125)
    elif rolmean_short_EEGvar[i] >= bins[8] and rolmean_short_EEGvar[i] <= bins[9]:
        pro_eegvars_time_false.append(0.214620718)
        pro_eegvars_time.append(0.125)
    elif rolmean_short_EEGvar[i] > bins[9] and rolmean_short_EEGvar[i] < bins[10]:
        pro_eegvars_time_false.append(0.318107667)
        pro_eegvars_time.append(0.375)
    elif rolmean_short_EEGvar[i] >= bins[10] and rolmean_short_EEGvar[i] <= bins[11]:
        pro_eegvars_time_false.append(0.176998369)
        pro_eegvars_time.append(0.125)
    elif rolmean_short_EEGvar[i] > bins[11] and rolmean_short_EEGvar[i] < bins[12]:
        pro_eegvars_time_false.append(0.008666395)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] > bins[12] and rolmean_short_EEGvar[i] < bins[13]:
        pro_eegvars_time_false.append(0.010195759)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[13] and rolmean_short_EEGvar[i] < bins[14]:
        pro_eegvars_time_false.append(0.017638662)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[14] and rolmean_short_EEGvar[i] < bins[15]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[15] and rolmean_short_EEGvar[i] < bins[16]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[16] and rolmean_short_EEGvar[i] < bins[17]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
    elif rolmean_short_EEGvar[i] >= bins[17]:
        pro_eegvars_time_false.append(0)
        pro_eegvars_time.append(0)
pro_eegautos_time = []
pro_eegautos_time_false = []
for i in range(len(rolmean_short_EEGauto)):
    if rolmean_short_EEGauto[i] >= bins[0] and rolmean_short_EEGauto[i] < bins[1]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[1] and rolmean_short_EEGauto[i] < bins[2]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[2] and rolmean_short_EEGauto[i] < bins[3]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[3] and rolmean_short_EEGauto[i] < bins[4]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[4] and rolmean_short_EEGauto[i] < bins[5]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[5] and rolmean_short_EEGauto[i] < bins[6]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[6] and rolmean_short_EEGauto[i] < bins[7]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[7] and rolmean_short_EEGauto[i] < bins[8]:
        pro_eegautos_time_false.append(0.073664356)
        pro_eegautos_time.append(0.125)
    elif rolmean_short_EEGauto[i] >= bins[8] and rolmean_short_EEGauto[i] < bins[9]:
        pro_eegautos_time_false.append(0.396156199)
        pro_eegautos_time.append(0.375)
    elif rolmean_short_EEGauto[i] >= bins[9] and rolmean_short_EEGauto[i] <= bins[10]:
        pro_eegautos_time_false.append(0.507850734)
        pro_eegautos_time.append(0.5)
    elif rolmean_short_EEGauto[i] > bins[10] and rolmean_short_EEGauto[i] < bins[11]:
        pro_eegautos_time_false.append(0.007442904)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[11] and rolmean_short_EEGauto[i] < bins[12]:
        pro_eegautos_time_false.append(0.004180261)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[12] and rolmean_short_EEGauto[i] < bins[13]:
        pro_eegautos_time_false.append(0.00423124)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[13] and rolmean_short_EEGauto[i] < bins[14]:
        pro_eegautos_time_false.append(0.006474307)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[14] and rolmean_short_EEGauto[i] < bins[15]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[15] and rolmean_short_EEGauto[i] < bins[16]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[16] and rolmean_short_EEGauto[i] < bins[17]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
    elif rolmean_short_EEGauto[i] >= bins[17]:
        pro_eegautos_time_false.append(0)
        pro_eegautos_time.append(0)
pro_RRIvars_time = []
pro_RRIvars_time_false = []
for i in range(len(rolmean_short_RRIvar)):
    if rolmean_short_RRIvar[i] >= bins[0] and rolmean_short_RRIvar[i] < bins[1]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[1] and rolmean_short_RRIvar[i] < bins[2]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[2] and rolmean_short_RRIvar[i] < bins[3]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[3] and rolmean_short_RRIvar[i] < bins[4]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[4] and rolmean_short_RRIvar[i] < bins[5]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[5] and rolmean_short_RRIvar[i] < bins[6]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[6] and rolmean_short_RRIvar[i] < bins[7]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[7] and rolmean_short_RRIvar[i] <= bins[8]:
        pro_RRIvars_time_false.append(0.245309951)
        pro_RRIvars_time.append(0.25)
    elif rolmean_short_RRIvar[i] > bins[8] and rolmean_short_RRIvar[i] < bins[9]:
        pro_RRIvars_time_false.append(0.246074633)
        pro_RRIvars_time.append(0.25)
    elif rolmean_short_RRIvar[i] >= bins[9] and rolmean_short_RRIvar[i] <= bins[10]:
        pro_RRIvars_time_false.append(0.310766721)
        pro_RRIvars_time.append(0.375)
    elif rolmean_short_RRIvar[i] > bins[10] and rolmean_short_RRIvar[i] <= bins[11]:
        pro_RRIvars_time_false.append(0.145238581)
        pro_RRIvars_time.append(0.125)
    elif rolmean_short_RRIvar[i] > bins[11] and rolmean_short_RRIvar[i] < bins[12]:
        pro_RRIvars_time_false.append(0.037826264)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[12] and rolmean_short_RRIvar[i] < bins[13]:
        pro_RRIvars_time_false.append(0.00632137)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[13] and rolmean_short_RRIvar[i] < bins[14]:
        pro_RRIvars_time_false.append(0.00846248)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[14] and rolmean_short_RRIvar[i] < bins[15]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[15] and rolmean_short_RRIvar[i] < bins[16]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[16] and rolmean_short_RRIvar[i] < bins[17]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
    elif rolmean_short_RRIvar[i] >= bins[17]:
        pro_RRIvars_time_false.append(0)
        pro_RRIvars_time.append(0)
pro_RRIautos_time = []
pro_RRIautos_time_false = []
for i in range(len(rolmean_short_RRIauto)):
    if rolmean_short_RRIauto[i] >= bins[0] and rolmean_short_RRIauto[i] < bins[1]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[1] and rolmean_short_RRIauto[i] < bins[2]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[2] and rolmean_short_RRIauto[i] < bins[3]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[3] and rolmean_short_RRIauto[i] < bins[4]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[4] and rolmean_short_RRIauto[i] < bins[5]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[5] and rolmean_short_RRIauto[i] < bins[6]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[6] and rolmean_short_RRIauto[i] < bins[7]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[7] and rolmean_short_RRIauto[i] <= bins[8]:
        pro_RRIautos_time_false.append(0.038081158)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] > bins[8] and rolmean_short_RRIauto[i] <= bins[9]:
        pro_RRIautos_time_false.append(0.497756933)
        pro_RRIautos_time.append(0.75)
    elif rolmean_short_RRIauto[i] > bins[9] and rolmean_short_RRIauto[i] <= bins[10]:
        pro_RRIautos_time_false.append(0.422359299)
        pro_RRIautos_time.append(0.25)
    elif rolmean_short_RRIauto[i] > bins[10] and rolmean_short_RRIauto[i] < bins[11]:
        pro_RRIautos_time_false.append(0.02406199)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[11] and rolmean_short_RRIauto[i] < bins[12]:
        pro_RRIautos_time_false.append(0.006576264)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[12] and rolmean_short_RRIauto[i] < bins[13]:
        pro_RRIautos_time_false.append(0.004486134)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[13] and rolmean_short_RRIauto[i] < bins[14]:
        pro_RRIautos_time_false.append(0.006678222)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[14] and rolmean_short_RRIauto[i] < bins[15]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[15] and rolmean_short_RRIauto[i] < bins[16]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[16] and rolmean_short_RRIauto[i] < bins[17]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)
    elif rolmean_short_RRIauto[i] >= bins[17]:
        pro_RRIautos_time_false.append(0)
        pro_RRIautos_time.append(0)




Pseizureeegvar = 0.000407664;
Pnonseizureeegvar = 0.999592336;

Pcombined=[]
for m in range(len(pro_eegvars_time)):
    P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIvars_time[m]*pro_RRIautos_time[m]
    P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m])
    Pcombined.append(P1/(P1+P2))
print(len(Pcombined))

# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))
# print(len(Pcombined))

# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_RRIvars_time[m]*pro_RRIautos_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m])
#     Pcombined.append(P1/(P1+P2))
# print(len(Pcombined))

pyplot.figure(figsize=(8,4))
RRI_timewindow_arr=t
# pyplot.plot(RRI_timewindow_arr,Pcombined,'darkblue',label='critical slowing predictor')
ax = pyplot.subplot(111)
pyplot.plot(RRI_timewindow_arr,Pcombined,'darkblue')
pyplot.annotate('',xy=(92.7538833,np.max(Pcombined)),xytext=(92.7538833,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.annotate('',xy=(104.916106,np.max(Pcombined)),xytext=(104.916106,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='r',shrink=0.05))
pyplot.annotate('',xy=(115.673883,np.max(Pcombined)),xytext=(115.673883,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='r',shrink=0.05))
pyplot.annotate('',xy=(123.834,np.max(Pcombined)),xytext=(123.834,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.annotate('',xy=(127.884278,np.max(Pcombined)),xytext=(127.884278,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.annotate('',xy=(135.1409,np.max(Pcombined)),xytext=(135.1409,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.annotate('',xy=(139.57055,np.max(Pcombined)),xytext=(139.57055,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.annotate('',xy=(152.573328,np.max(Pcombined)),xytext=(152.573328,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.annotate('',xy=(159.654944,np.max(Pcombined)),xytext=(159.654944,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
# pyplot.annotate('',xy=(94.2927722,np.max(Pcombined)),xytext=(94.2927722,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# pyplot.annotate('',xy=(109.217727,np.max(Pcombined)),xytext=(109.217727,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# pyplot.annotate('',xy=(111.933838,np.max(Pcombined)),xytext=(111.933838,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# pyplot.annotate('',xy=(117.456383,np.max(Pcombined)),xytext=(117.456383,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# pyplot.annotate('',xy=(119.572216,np.max(Pcombined)),xytext=(119.572216,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# pyplot.annotate('',xy=(129.6137,np.max(Pcombined)),xytext=(129.6137,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# pyplot.annotate('',xy=(144.27416,np.max(Pcombined)),xytext=(144.27416,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# pyplot.annotate('',xy=(154.853,np.max(Pcombined)),xytext=(154.853,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
pyplot.xlabel('Time (hours)',fontsize=16)
pyplot.ylabel('Probability',fontsize=16)
pyplot.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
locs, labels = pyplot.yticks([0,1E-5,2E-5],fontsize=16)
locs, labels = pyplot.xticks(fontsize=16)
pyplot.title('Critical slowing forecasting model',fontsize=16)
pyplot.hlines(2.3910434372e-06, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'r')
# pyplot.hlines(1.2806765325921033e-05, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'r')
# pyplot.hlines(1.357739907582539e-05, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'r')
pyplot.hlines(0.3*2.3910434372e-06, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'darkred',alpha=0.8)
pyplot.hlines(0.6*2.3910434372e-06, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'darkred',alpha=0.8)
pyplot.hlines(1.2*2.3910434372e-06, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'darkred',alpha=0.8)
pyplot.hlines(2*2.3910434372e-06, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'darkred',alpha=0.8)
pyplot.show()

# Pcombined=split(Pcombined,6)
# print(len(Pcombined))
# index=[]
# for i in range(len(Pcombined)):
#     for item in Pcombined[i]:
#         if item >= 2.3910434372e-06:
#             index.append(6*i+0)
# print(RRI_timewindow_arr[index])
# a=np.unique(RRI_timewindow_arr[index])
# print(a); print(len(a))
# index=[]
# for i in range(len(Pcombined)):
#     for item in Pcombined[i]:
#         if item >= 0.3*2.3910434372e-06:
#             index.append(6*i+0)
# print(RRI_timewindow_arr[index])
# a=np.unique(RRI_timewindow_arr[index])
# print(a); print(len(a))
# index=[]
# for i in range(len(Pcombined)):
#     for item in Pcombined[i]:
#         if item >= 0.6*2.3910434372e-06:
#             index.append(6*i+0)
# print(RRI_timewindow_arr[index])
# a=np.unique(RRI_timewindow_arr[index])
# print(a); print(len(a))
# index=[]
# for i in range(len(Pcombined)):
#     for item in Pcombined[i]:
#         if item >= 1.2*2.3910434372e-06:
#             index.append(6*i+0)
# print(RRI_timewindow_arr[index])
# a=np.unique(RRI_timewindow_arr[index])
# print(a); print(len(a))
# index=[]
# for i in range(len(Pcombined)):
#     for item in Pcombined[i]:
#         if item >= 2*2.3910434372e-06:
#             index.append(6*i+0)
# print(RRI_timewindow_arr[index])
# a=np.unique(RRI_timewindow_arr[index])
# print(a); print(len(a))

# index=[]
# for i, j in enumerate(Pcombined):
#     if j >= 2.3910434372e-06:
#         index.append(i)
# print(RRI_timewindow_arr[index])
# print(len(index))
# # time_arr=[92.7538833,94.2927722,104.916106,109.217727,111.933838,115.673883,117.456383,119.572216,123.834,127.884278,129.6137,135.1409,139.57055,144.27416,152.573328,154.853,159.654944]
# time_arr=[92.7538833,104.916106,115.673883,123.834,127.884278,135.1409,139.57055,152.573328,159.654944]
# k=0
# n_arr=[]
# for m in time_arr:
#     for n in RRI_timewindow_arr[index]:
#         if m-n<=1 and m-n>=0:
#             k=k+1
#             n_arr.append(n)
# print(k)
# index=[]
# for i, j in enumerate(Pcombined):
#     if j >= 0.3*2.3910434372e-06:
#         index.append(i)
# print(RRI_timewindow_arr[index])
# print(len(index))
# # time_arr=[92.7538833,94.2927722,104.916106,109.217727,111.933838,115.673883,117.456383,119.572216,123.834,127.884278,129.6137,135.1409,139.57055,144.27416,152.573328,154.853,159.654944]
# time_arr=[92.7538833,104.916106,115.673883,123.834,127.884278,135.1409,139.57055,152.573328,159.654944]
# k=0
# n_arr=[]
# for m in time_arr:
#     for n in RRI_timewindow_arr[index]:
#         if m-n<=1 and m-n>=0:
#             k=k+1
#             n_arr.append(n)
# print(k)
# index=[]
# for i, j in enumerate(Pcombined):
#     if j >= 0.6*2.3910434372e-06:
#         index.append(i)
# print(RRI_timewindow_arr[index])
# print(len(index))
# # time_arr=[92.7538833,94.2927722,104.916106,109.217727,111.933838,115.673883,117.456383,119.572216,123.834,127.884278,129.6137,135.1409,139.57055,144.27416,152.573328,154.853,159.654944]
# time_arr=[92.7538833,104.916106,115.673883,123.834,127.884278,135.1409,139.57055,152.573328,159.654944]
# k=0
# n_arr=[]
# for m in time_arr:
#     for n in RRI_timewindow_arr[index]:
#         if m-n<=1 and m-n>=0:
#             k=k+1
#             n_arr.append(n)
# print(k)
# index=[]
# for i, j in enumerate(Pcombined):
#     if j >= 1.2*2.3910434372e-06:
#         index.append(i)
# print(RRI_timewindow_arr[index])
# print(len(index))
# # time_arr=[92.7538833,94.2927722,104.916106,109.217727,111.933838,115.673883,117.456383,119.572216,123.834,127.884278,129.6137,135.1409,139.57055,144.27416,152.573328,154.853,159.654944]
# time_arr=[92.7538833,104.916106,115.673883,123.834,127.884278,135.1409,139.57055,152.573328,159.654944]
# k=0
# n_arr=[]
# for m in time_arr:
#     for n in RRI_timewindow_arr[index]:
#         if m-n<=1 and m-n>=0:
#             k=k+1
#             n_arr.append(n)
# print(k)
# index=[]
# for i, j in enumerate(Pcombined):
#     if j >= 2*2.3910434372e-06:
#         index.append(i)
# print(RRI_timewindow_arr[index])
# print(len(index))
# # time_arr=[92.7538833,94.2927722,104.916106,109.217727,111.933838,115.673883,117.456383,119.572216,123.834,127.884278,129.6137,135.1409,139.57055,144.27416,152.573328,154.853,159.654944]
# time_arr=[92.7538833,104.916106,115.673883,123.834,127.884278,135.1409,139.57055,152.573328,159.654944]
# k=0
# n_arr=[]
# for m in time_arr:
#     for n in RRI_timewindow_arr[index]:
#         if m-n<=1 and m-n>=0:
#             k=k+1
#             n_arr.append(n)
# print(k)

Th1=2.3910434372e-06
Pcombined = split(Pcombined, 6)
print(len(Pcombined))
time_arr_arr=[]
AUC_cs_arr=[]
for i in range(5000):
    time_arr = np.random.uniform(low=t_window_arr[19624], high=t_window_arr[-1], size=4)
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
    print(len(a1))
    k1 = 0
    n_arr = []
    for m in time_arr:
        for n in a1:
            if m - n <= 1 and m - n >= 0:
                k1 = k1 + 1
                n_arr.append(n)
    print(k1)

    index = []
    for i in range(len(Pcombined)):
        for item in Pcombined[i]:
            if item >= 0.3 * Th1:
                index.append(6 * i + 0)
    # print(RRI_timewindow_arr[index])
    a2 = np.unique(RRI_timewindow_arr[index])
    # print(a2);
    print(len(a2))
    k2 = 0
    n_arr = []
    for m in time_arr:
        for n in a2:
            if m - n <= 1 and m - n >= 0:
                k2 = k2 + 1
                n_arr.append(n)
    print(k2)

    index = []
    for i in range(len(Pcombined)):
        for item in Pcombined[i]:
            if item >= 0.6 * Th1:
                index.append(6 * i + 0)
    # print(RRI_timewindow_arr[index])
    a3 = np.unique(RRI_timewindow_arr[index])
    # print(a3);
    print(len(a3))
    k3 = 0
    n_arr = []
    for m in time_arr:
        for n in a3:
            if m - n <= 1 and m - n >= 0:
                k3 = k3 + 1
                n_arr.append(n)
    print(k3)

    index = []
    for i in range(len(Pcombined)):
        for item in Pcombined[i]:
            if item >= 1.2 * Th1:
                index.append(6 * i + 0)
    # print(RRI_timewindow_arr[index])
    a4 = np.unique(RRI_timewindow_arr[index])
    # print(a);
    print(len(a4))
    k4 = 0
    n_arr = []
    for m in time_arr:
        for n in a4:
            if m - n <= 1 and m - n >= 0:
                k4 = k4 + 1
                n_arr.append(n)
    print(k4)

    index = []
    for i in range(len(Pcombined)):
        for item in Pcombined[i]:
            if item >= 2 * Th1:
                index.append(6 * i + 0)
    # print(RRI_timewindow_arr[index])
    a5 = np.unique(RRI_timewindow_arr[index])
    # print(a5);
    print(len(a5))
    k5 = 0
    n_arr = []
    for m in time_arr:
        for n in a5:
            if m - n <= 1 and m - n >= 0:
                k5 = k5 + 1
                n_arr.append(n)
    print(k5)

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
    AUC_cs_arr.append(AUC_cs)

print(AUC_cs_arr)
print(time_arr_arr)
np.savetxt("C:/Users/wxiong/PycharmProjects/EEGstudy/chance_predictor/AUC_EEG_var_auto_6h_SA0124.csv", AUC_cs_arr, delimiter=",", fmt='%s')
np.savetxt("C:/Users/wxiong/PycharmProjects/EEGstudy/chance_predictor/seizure_labels_EEG_var_auto_6h_SA0124.csv", time_arr_arr, delimiter=",", fmt='%s')



## add circadian
t1=np.linspace(2.98805+0.00416667,2.98805+0.00416667+0.00416667*(len(Raw_variance_EEG_arr)-1),len(Raw_variance_EEG_arr))
a=np.where(t1<7.73806+2.98805)
t1[0:1857]=t1[0:1857]-2.98805+16.26194
t1[1857:]=t1[1857:]-7.73806-2.98805
time_feature_arr=[]
for i in range(len(t1)):
    if t1[i]>24:
        time_feature_arr.append(t1[i] - (t1[i] // 24) * 24)
    else:
        time_feature_arr.append(t1[i])
print(len(time_feature_arr))

time_arr=time_feature_arr[19624:]
print(len(time_arr))
new_arr=[]
# for j in range(384):
for j in range(1,487):
    new_arr.append(time_arr[40*j])
print(new_arr)

bins_number = 18
bins = np.linspace(0, 24, bins_number + 1)
pro_circadian_time=[]
pro_circadian_time_false=[]
for i in range(len(new_arr)):
    if new_arr[i] >= bins[0] and new_arr[i] <= bins[1]:
        pro_circadian_time_false.append(0.067114094)
        pro_circadian_time.append(0)
    elif new_arr[i] > bins[1] and new_arr[i] < bins[2]:
        pro_circadian_time_false.append(0.059091862)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[2] and new_arr[i] < bins[3]:
        pro_circadian_time_false.append(0.05033557)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[3] and new_arr[i] < bins[4]:
        pro_circadian_time_false.append(0.05033557)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[4] and new_arr[i] < bins[5]:
        pro_circadian_time_false.append(0.05033557)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[5] and new_arr[i] <= bins[6]:
        pro_circadian_time_false.append(0.05033557)
        pro_circadian_time.append(0)
    elif new_arr[i] > bins[6] and new_arr[i] < bins[7]:
        pro_circadian_time_false.append(0.050230705)
        pro_circadian_time.append(0.25)
    elif new_arr[i] >= bins[7] and new_arr[i] <= bins[8]:
        pro_circadian_time_false.append(0.050283138)
        pro_circadian_time.append(0.125)
    elif new_arr[i] > bins[8] and new_arr[i] < bins[9]:
        pro_circadian_time_false.append(0.05033557)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[9] and new_arr[i] < bins[10]:
        pro_circadian_time_false.append(0.05033557)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[10] and new_arr[i] < bins[11]:
        pro_circadian_time_false.append(0.050283138)
        pro_circadian_time.append(0.125)
    elif new_arr[i] >= bins[11] and new_arr[i] < bins[12]:
        pro_circadian_time_false.append(0.05033557)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[12] and new_arr[i] < bins[13]:
        pro_circadian_time_false.append(0.063810822)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[13] and new_arr[i] < bins[14]:
        pro_circadian_time_false.append(0.067114094)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[14] and new_arr[i] < bins[15]:
        pro_circadian_time_false.append(0.067114094)
        pro_circadian_time.append(0)
    elif new_arr[i] >= bins[15] and new_arr[i] < bins[16]:
        pro_circadian_time_false.append(0.067061661)
        pro_circadian_time.append(0.125)
    elif new_arr[i] >= bins[16] and new_arr[i] < bins[17]:
        pro_circadian_time_false.append(0.066956795)
        pro_circadian_time.append(0.375)
    elif new_arr[i] >= bins[17] and new_arr[i] <= bins[18]:
        pro_circadian_time_false.append(0.067114094)
        pro_circadian_time.append(0)
print(len(pro_circadian_time))
print(len(pro_circadian_time_false))
pyplot.figure(figsize=(8,4))
ax = pyplot.subplot(111)
RRI_timewindow_arr=t
# pyplot.plot(RRI_timewindow_arr,pro_circadian_time,'darkblue',label='critical slowing predictor')
pyplot.plot(RRI_timewindow_arr,pro_circadian_time,'darkblue')
pyplot.annotate('',xy=(92.7538833,np.max(pro_circadian_time)),xytext=(92.7538833,np.max(pro_circadian_time)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.annotate('',xy=(104.916106,np.max(pro_circadian_time)),xytext=(104.916106,np.max(pro_circadian_time)+0.000000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.annotate('',xy=(115.673883,np.max(pro_circadian_time)),xytext=(115.673883,np.max(pro_circadian_time)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.annotate('',xy=(123.834,np.max(pro_circadian_time)),xytext=(123.834,np.max(pro_circadian_time)+0.00000000001),arrowprops=dict(facecolor='red',shrink=0.05))
pyplot.annotate('',xy=(127.884278,np.max(pro_circadian_time)),xytext=(127.884278,np.max(pro_circadian_time)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.annotate('',xy=(135.1409,np.max(pro_circadian_time)),xytext=(135.1409,np.max(pro_circadian_time)+0.000000000001),arrowprops=dict(facecolor='red',shrink=0.05))
pyplot.annotate('',xy=(139.57055,np.max(pro_circadian_time)),xytext=(139.57055,np.max(pro_circadian_time)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.annotate('',xy=(152.573328,np.max(pro_circadian_time)),xytext=(152.573328,np.max(pro_circadian_time)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.annotate('',xy=(159.654944,np.max(pro_circadian_time)),xytext=(159.654944,np.max(pro_circadian_time)+0.00000000001),arrowprops=dict(facecolor='r',shrink=0.05))
pyplot.ylabel('Probability',fontsize=16)
# # pyplot.annotate('',xy=(94.2927722,np.max(Pcombined)),xytext=(94.2927722,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# # pyplot.annotate('',xy=(109.217727,np.max(Pcombined)),xytext=(109.217727,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# # pyplot.annotate('',xy=(111.933838,np.max(Pcombined)),xytext=(111.933838,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# # pyplot.annotate('',xy=(117.456383,np.max(Pcombined)),xytext=(117.456383,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# # pyplot.annotate('',xy=(119.572216,np.max(Pcombined)),xytext=(119.572216,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# # pyplot.annotate('',xy=(129.6137,np.max(Pcombined)),xytext=(129.6137,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# # pyplot.annotate('',xy=(144.27416,np.max(Pcombined)),xytext=(144.27416,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# # pyplot.annotate('',xy=(154.853,np.max(Pcombined)),xytext=(154.853,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
pyplot.hlines(0.125, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'r')
# pyplot.hlines(0.125*0.3, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'orange',alpha=0.8)
# pyplot.hlines(0.125*0.6, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'orange',alpha=0.8)
# pyplot.hlines(0.125*1.2, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'orange',alpha=0.8)
# pyplot.hlines(0.125*2, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'orange',alpha=0.8)
# pyplot.title('Forecast seizures in SA0124')
pyplot.xlabel('Time (hours)',fontsize=16)
pyplot.ylabel('Probability',fontsize=16)
locs, labels = pyplot.yticks([0,0.2,0.4],fontsize=16)
locs, labels = pyplot.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
pyplot.title('Circadian forecasting model',fontsize=16)
pyplot.show()

# pro_circadian_time=split(pro_circadian_time,6)
# print(len(pro_circadian_time))
# index=[]
# for i in range(len(pro_circadian_time)):
#     for item in pro_circadian_time[i]:
#          if item >= 0.125:
#             index.append(6*i+0)
# print(RRI_timewindow_arr[index])
# a=np.unique(RRI_timewindow_arr[index])
# print(a); print(len(a))
# index=[]
# for i in range(len(pro_circadian_time)):
#     for item in pro_circadian_time[i]:
#          if item >= 0.3*0.125:
#             index.append(6*i+0)
# print(RRI_timewindow_arr[index])
# a=np.unique(RRI_timewindow_arr[index])
# print(a); print(len(a))
# index=[]
# for i in range(len(pro_circadian_time)):
#     for item in pro_circadian_time[i]:
#          if item >= 0.6*0.125:
#             index.append(6*i+0)
# print(RRI_timewindow_arr[index])
# a=np.unique(RRI_timewindow_arr[index])
# print(a); print(len(a))
# index=[]
# for i in range(len(pro_circadian_time)):
#     for item in pro_circadian_time[i]:
#          if item >= 1.2*0.125:
#             index.append(6*i+0)
# print(RRI_timewindow_arr[index])
# a=np.unique(RRI_timewindow_arr[index])
# print(a); print(len(a))
# index=[]
# for i in range(len(pro_circadian_time)):
#     for item in pro_circadian_time[i]:
#          if item >= 2*0.125:
#             index.append(6*i+0)
# print(RRI_timewindow_arr[index])
# a=np.unique(RRI_timewindow_arr[index])
# print(a); print(len(a))

# index=[]
# for i, j in enumerate(pro_circadian_time):
#     if j < 0.125:
#         index.append(i)
# print(RRI_timewindow_arr[index])
# print(len(index))
# # time_arr=[92.7538833,94.2927722,104.916106,109.217727,111.933838,115.673883,117.456383,119.572216,123.834,127.884278,129.6137,135.1409,139.57055,144.27416,152.573328,154.853,159.654944]
# time_arr=[92.7538833,104.916106,115.673883,123.834,127.884278,135.1409,139.57055,152.573328,159.654944]
# k=0
# n_arr=[]
# for m in time_arr:
#     for n in RRI_timewindow_arr[index]:
#         if m-n<=1 and m-n>=0:
#             k=k+1
#             n_arr.append(n)
# print(k)
# index=[]
# for i, j in enumerate(pro_circadian_time):
#     if j <  0.125*0.3:
#         index.append(i)
# print(RRI_timewindow_arr[index])
# print(len(index))
# # time_arr=[92.7538833,94.2927722,104.916106,109.217727,111.933838,115.673883,117.456383,119.572216,123.834,127.884278,129.6137,135.1409,139.57055,144.27416,152.573328,154.853,159.654944]
# time_arr=[92.7538833,104.916106,115.673883,123.834,127.884278,135.1409,139.57055,152.573328,159.654944]
# k=0
# n_arr=[]
# for m in time_arr:
#     for n in RRI_timewindow_arr[index]:
#         if m-n<=1 and m-n>=0:
#             k=k+1
#             n_arr.append(n)
# print(k)
# index=[]
# for i, j in enumerate(pro_circadian_time):
#     if j < 0.125*0.6:
#         index.append(i)
# print(RRI_timewindow_arr[index])
# print(len(index))
# # time_arr=[92.7538833,94.2927722,104.916106,109.217727,111.933838,115.673883,117.456383,119.572216,123.834,127.884278,129.6137,135.1409,139.57055,144.27416,152.573328,154.853,159.654944]
# time_arr=[92.7538833,104.916106,115.673883,123.834,127.884278,135.1409,139.57055,152.573328,159.654944]
# k=0
# n_arr=[]
# for m in time_arr:
#     for n in RRI_timewindow_arr[index]:
#         if m-n<=1 and m-n>=0:
#             k=k+1
#             n_arr.append(n)
# print(k)
# index=[]
# for i, j in enumerate(pro_circadian_time):
#     if j < 0.125*1.2:
#         index.append(i)
# print(RRI_timewindow_arr[index])
# print(len(index))
# # time_arr=[92.7538833,94.2927722,104.916106,109.217727,111.933838,115.673883,117.456383,119.572216,123.834,127.884278,129.6137,135.1409,139.57055,144.27416,152.573328,154.853,159.654944]
# time_arr=[92.7538833,104.916106,115.673883,123.834,127.884278,135.1409,139.57055,152.573328,159.654944]
# k=0
# n_arr=[]
# for m in time_arr:
#     for n in RRI_timewindow_arr[index]:
#         if m-n<=1 and m-n>=0:
#             k=k+1
#             n_arr.append(n)
# print(k)
# index=[]
# for i, j in enumerate(pro_circadian_time):
#     if j < 0.125*2:
#         index.append(i)
# print(RRI_timewindow_arr[index])
# print(len(index))
# # time_arr=[92.7538833,94.2927722,104.916106,109.217727,111.933838,115.673883,117.456383,119.572216,123.834,127.884278,129.6137,135.1409,139.57055,144.27416,152.573328,154.853,159.654944]
# time_arr=[92.7538833,104.916106,115.673883,123.834,127.884278,135.1409,139.57055,152.573328,159.654944]
# k=0
# n_arr=[]
# for m in time_arr:
#     for n in RRI_timewindow_arr[index]:
#         if m-n<=1 and m-n>=0:
#             k=k+1
#             n_arr.append(n)
# print(k)

Th2=0.125
Pcombined=pro_circadian_time
Pcombined = split(Pcombined, 6)
print(len(Pcombined))
time_arr_arr_EEGcirca=[]
AUC_com_arr_EEGcirca=[]
for i in range(5000):
    time_arr = np.random.uniform(low=t_window_arr[19624], high=t_window_arr[-1], size=4)
    time_arr_arr_EEGcirca.append(time_arr)
    time_arr=np.sort(time_arr)
    index=[]
    for i in range(len(Pcombined)):
        for item in Pcombined[i]:
            if item >= Th2:
                index.append(6*i+0)
    # print(RRI_timewindow_arr[index])
    a6=np.unique(RRI_timewindow_arr[index])
    # print(a6);
    print(len(a6))
    k6=0
    n_arr=[]
    for m in time_arr:
        for n in a6:
            if m-n<=1 and m-n>=0:
                k6=k6+1
                n_arr.append(n)
    print(k6)

    index=[]
    for i in range(len(Pcombined)):
        for item in Pcombined[i]:
            if item >= 0.3*Th2:
                index.append(6*i+0)
    # print(RRI_timewindow_arr[index])
    a7=np.unique(RRI_timewindow_arr[index])
    # print(a7);
    print(len(a7))
    k7=0
    n_arr=[]
    for m in time_arr:
        for n in a7:
            if m-n<=1 and m-n>=0:
                k7=k7+1
                n_arr.append(n)
    print(k7)

    index=[]
    for i in range(len(Pcombined)):
        for item in Pcombined[i]:
            if item >= 0.6*Th2:
                index.append(6*i+0)
    # print(RRI_timewindow_arr[index])
    a8=np.unique(RRI_timewindow_arr[index])
    # print(a8);
    print(len(a8))
    k8=0
    n_arr=[]
    for m in time_arr:
        for n in a8:
            if m-n<=1 and m-n>=0:
                k8=k8+1
                n_arr.append(n)
    print(k8)

    index=[]
    for i in range(len(Pcombined)):
        for item in Pcombined[i]:
            if item >= 1.2*Th2:
                index.append(6*i+0)
    # print(RRI_timewindow_arr[index])
    a9=np.unique(RRI_timewindow_arr[index])
    # print(a9);
    print(len(a9))
    k9=0
    n_arr=[]
    for m in time_arr:
        for n in a9:
            if m-n<=1 and m-n>=0:
                k9=k9+1
                n_arr.append(n)
    print(k9)

    index=[]
    for i in range(len(Pcombined)):
        for item in Pcombined[i]:
            if item >= 2*Th2:
                index.append(6*i+0)
    # print(RRI_timewindow_arr[index])
    a10=np.unique(RRI_timewindow_arr[index])
    k10=0
    n_arr=[]
    for m in time_arr:
        for n in a10:
            if m-n<=1 and m-n>=0:
                k10=k10+1
                n_arr.append(n)
    print(k10)

    Sen6=k6/len(time_arr);Sen7=k7/len(time_arr);Sen8=k8/len(time_arr);Sen9=k9/len(time_arr);Sen10=k10/len(time_arr);
    FPR6=(len(a6)-k6)/len(Pcombined);FPR7=(len(a7)-k7)/len(Pcombined);FPR8=(len(a8)-k8)/len(Pcombined);FPR9=(len(a9)-k9)/len(Pcombined);FPR10=(len(a10)-k10)/len(Pcombined);
    Sen_arr_COM=[0,Sen6,Sen7,Sen8,Sen9,Sen10,1]
    FPR_arr_COM=[0,FPR6,FPR7,FPR8,FPR9,FPR10,1]
    print(Sen_arr_COM);print(FPR_arr_COM);

    from sklearn.metrics import auc
    AUC_com=auc(np.sort(FPR_arr_COM),np.sort(Sen_arr_COM))
    AUC_com_arr_EEGcirca.append(AUC_com)

print(AUC_com_arr_EEGcirca)
print(time_arr_arr_EEGcirca)
np.savetxt("C:/Users/wxiong/PycharmProjects/EEGstudy/chance_predictor/AUC_circa6h_SA0124.csv", AUC_com_arr_EEGcirca, delimiter=",", fmt='%s')
np.savetxt("C:/Users/wxiong/PycharmProjects/EEGstudy/chance_predictor/seizure_labels_circa6h_SA0124.csv", time_arr_arr_EEGcirca, delimiter=",", fmt='%s')





Pseizureeegvar = 0.000407664;
Pnonseizureeegvar = 0.999592336;

Pcombined=[]
for m in range(len(pro_eegvars_time)):
    P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_RRIvars_time[m]*pro_RRIautos_time[m]*pro_circadian_time[m]
    P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
    Pcombined.append(P1/(P1+P2))
print(len(Pcombined))
# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=pro_eegvars_time[m]*Pseizureeegvar*pro_eegautos_time[m]*pro_circadian_time[m]
#     P2=Pnonseizureeegvar*(1-pro_eegvars_time_false[m]*pro_eegautos_time_false[m]*pro_circadian_time_false[m])
#     Pcombined.append(P1/(P1+P2))
# print(len(Pcombined))
# Pcombined=[]
# for m in range(len(pro_eegvars_time)):
#     P1=Pseizureeegvar*pro_RRIvars_time[m]*pro_RRIautos_time[m]*pro_circadian_time[m]
#     P2=Pnonseizureeegvar*(1-pro_RRIvars_time_false[m]*pro_RRIautos_time_false[m]*pro_circadian_time_false[m])
#     Pcombined.append(P1/(P1+P2))
# print(len(Pcombined))


pyplot.figure(figsize=(8,4))
ax = pyplot.subplot(111)
RRI_timewindow_arr=t
# pyplot.plot(RRI_timewindow_arr,Pcombined,'darkblue',label='critical slowing predictor')
pyplot.plot(RRI_timewindow_arr,Pcombined,'darkblue')
pyplot.annotate('',xy=(92.7538833,np.max(Pcombined)),xytext=(92.7538833,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.annotate('',xy=(104.916106,np.max(Pcombined)),xytext=(104.916106,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.annotate('',xy=(115.673883,np.max(Pcombined)),xytext=(115.673883,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='r',shrink=0.05))
pyplot.annotate('',xy=(123.834,np.max(Pcombined)),xytext=(123.834,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='red',shrink=0.05))
pyplot.annotate('',xy=(127.884278,np.max(Pcombined)),xytext=(127.884278,np.max(Pcombined)+0.000000000001),arrowprops=dict(facecolor='k',shrink=0.05))
pyplot.annotate('',xy=(135.1409,np.max(Pcombined)),xytext=(135.1409,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='red',shrink=0.05))
pyplot.annotate('',xy=(139.57055,np.max(Pcombined)),xytext=(139.57055,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.annotate('',xy=(152.573328,np.max(Pcombined)),xytext=(152.573328,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='black',shrink=0.05))
pyplot.annotate('',xy=(159.654944,np.max(Pcombined)),xytext=(159.654944,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='r',shrink=0.05))
# pyplot.annotate('',xy=(94.2927722,np.max(Pcombined)),xytext=(94.2927722,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# pyplot.annotate('',xy=(109.217727,np.max(Pcombined)),xytext=(109.217727,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# pyplot.annotate('',xy=(111.933838,np.max(Pcombined)),xytext=(111.933838,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# pyplot.annotate('',xy=(117.456383,np.max(Pcombined)),xytext=(117.456383,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# pyplot.annotate('',xy=(119.572216,np.max(Pcombined)),xytext=(119.572216,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# pyplot.annotate('',xy=(129.6137,np.max(Pcombined)),xytext=(129.6137,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# pyplot.annotate('',xy=(144.27416,np.max(Pcombined)),xytext=(144.27416,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
# pyplot.annotate('',xy=(154.853,np.max(Pcombined)),xytext=(154.853,np.max(Pcombined)+0.00000000001),arrowprops=dict(facecolor='grey',shrink=0.05))
pyplot.ylabel('Probability',fontsize=16)
pyplot.xlabel('Time (hours)',fontsize=16)
pyplot.ticklabel_format(axis="y", style="sci",scilimits=(0,0))
locs, labels = pyplot.yticks([0,1.1E-6,2.2E-6],fontsize=16)
locs, labels = pyplot.xticks(fontsize=16)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
pyplot.title('Combine forecasting model',fontsize=16)
pyplot.hlines(5.978085536184582e-07, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'r')
# pyplot.hlines(2.4016073011910382e-06, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'r')
# pyplot.hlines(4.798948723166783e-06, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'r')
pyplot.hlines(0.3*5.978085536184582e-07, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'darkred',alpha=0.8)
pyplot.hlines(0.6*5.978085536184582e-07, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'darkred',alpha=0.8)
pyplot.hlines(1.2*5.978085536184582e-07, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'darkred',alpha=0.8)
pyplot.hlines(2*5.978085536184582e-07, RRI_timewindow_arr[0],RRI_timewindow_arr[-1],'darkred',alpha=0.8)
pyplot.show()

# Pcombined=split(Pcombined,6)
# print(len(Pcombined))
# index=[]
# for i in range(len(Pcombined)):
#     for item in Pcombined[i]:
#         if item >= 5.978085536184582e-07:
#             index.append(6*i+0)
# print(RRI_timewindow_arr[index])
# a=np.unique(RRI_timewindow_arr[index])
# print(a); print(len(a))
# index=[]
# for i in range(len(Pcombined)):
#     for item in Pcombined[i]:
#         if item >= 0.3*5.978085536184582e-07:
#             index.append(6*i+0)
# print(RRI_timewindow_arr[index])
# a=np.unique(RRI_timewindow_arr[index])
# print(a); print(len(a))
# index=[]
# for i in range(len(Pcombined)):
#     for item in Pcombined[i]:
#         if item >= 0.6*5.978085536184582e-07:
#             index.append(6*i+0)
# print(RRI_timewindow_arr[index])
# a=np.unique(RRI_timewindow_arr[index])
# print(a); print(len(a))
# index=[]
# for i in range(len(Pcombined)):
#     for item in Pcombined[i]:
#         if item >= 1.2*5.978085536184582e-07:
#             index.append(6*i+0)
# print(RRI_timewindow_arr[index])
# a=np.unique(RRI_timewindow_arr[index])
# print(a); print(len(a))
# index=[]
# for i in range(len(Pcombined)):
#     for item in Pcombined[i]:
#         if item >= 2*5.978085536184582e-07:
#             index.append(6*i+0)
# print(RRI_timewindow_arr[index])
# a=np.unique(RRI_timewindow_arr[index])
# print(a); print(len(a))

# index=[]
# for i, j in enumerate(Pcombined):
#     if j >= 5.978085536184582e-07:
#         index.append(i)
# print(RRI_timewindow_arr[index])
# print(len(index))
# # time_arr=[92.7538833,94.2927722,104.916106,109.217727,111.933838,115.673883,117.456383,119.572216,123.834,127.884278,129.6137,135.1409,139.57055,144.27416,152.573328,154.853,159.654944]
# time_arr=[92.7538833,104.916106,115.673883,123.834,127.884278,135.1409,139.57055,152.573328,159.654944]
# k=0
# n_arr=[]
# for m in time_arr:
#     for n in RRI_timewindow_arr[index]:
#         if m-n<=1 and m-n>=0:
#             k=k+1
#             n_arr.append(n)
# print(k)
# index=[]
# for i, j in enumerate(Pcombined):
#     if j >= 0.3*5.978085536184582e-07:
#         index.append(i)
# print(RRI_timewindow_arr[index])
# print(len(index))
# # time_arr=[92.7538833,94.2927722,104.916106,109.217727,111.933838,115.673883,117.456383,119.572216,123.834,127.884278,129.6137,135.1409,139.57055,144.27416,152.573328,154.853,159.654944]
# time_arr=[92.7538833,104.916106,115.673883,123.834,127.884278,135.1409,139.57055,152.573328,159.654944]
# k=0
# n_arr=[]
# for m in time_arr:
#     for n in RRI_timewindow_arr[index]:
#         if m-n<=1 and m-n>=0:
#             k=k+1
#             n_arr.append(n)
# print(k)
# index=[]
# for i, j in enumerate(Pcombined):
#     if j >= 0.6*5.978085536184582e-07:
#         index.append(i)
# print(RRI_timewindow_arr[index])
# print(len(index))
# # time_arr=[92.7538833,94.2927722,104.916106,109.217727,111.933838,115.673883,117.456383,119.572216,123.834,127.884278,129.6137,135.1409,139.57055,144.27416,152.573328,154.853,159.654944]
# time_arr=[92.7538833,104.916106,115.673883,123.834,127.884278,135.1409,139.57055,152.573328,159.654944]
# k=0
# n_arr=[]
# for m in time_arr:
#     for n in RRI_timewindow_arr[index]:
#         if m-n<=1 and m-n>=0:
#             k=k+1
#             n_arr.append(n)
# print(k)
# index=[]
# for i, j in enumerate(Pcombined):
#     if j >= 1.2*5.978085536184582e-07:
#         index.append(i)
# print(RRI_timewindow_arr[index])
# print(len(index))
# # time_arr=[92.7538833,94.2927722,104.916106,109.217727,111.933838,115.673883,117.456383,119.572216,123.834,127.884278,129.6137,135.1409,139.57055,144.27416,152.573328,154.853,159.654944]
# time_arr=[92.7538833,104.916106,115.673883,123.834,127.884278,135.1409,139.57055,152.573328,159.654944]
# k=0
# n_arr=[]
# for m in time_arr:
#     for n in RRI_timewindow_arr[index]:
#         if m-n<=1 and m-n>=0:
#             k=k+1
#             n_arr.append(n)
# print(k)
# index=[]
# for i, j in enumerate(Pcombined):
#     if j >= 2*5.978085536184582e-07:
#         index.append(i)
# print(RRI_timewindow_arr[index])
# print(len(index))
# # time_arr=[92.7538833,94.2927722,104.916106,109.217727,111.933838,115.673883,117.456383,119.572216,123.834,127.884278,129.6137,135.1409,139.57055,144.27416,152.573328,154.853,159.654944]
# time_arr=[92.7538833,104.916106,115.673883,123.834,127.884278,135.1409,139.57055,152.573328,159.654944]
# k=0
# n_arr=[]
# for m in time_arr:
#     for n in RRI_timewindow_arr[index]:
#         if m-n<=1 and m-n>=0:
#             k=k+1
#             n_arr.append(n)
# print(k)



Th3=5.978085536184582e-07
Pcombined = split(Pcombined, 6)
print(len(Pcombined))
time_arr_arr_EEGcirca=[]
AUC_com_arr_EEGcirca=[]
for i in range(5000):
    time_arr = np.random.uniform(low=t_window_arr[19624], high=t_window_arr[-1], size=4)
    time_arr_arr_EEGcirca.append(time_arr)
    time_arr=np.sort(time_arr)
    index=[]
    for i in range(len(Pcombined)):
        for item in Pcombined[i]:
            if item >= Th3:
                index.append(6*i+0)
    # print(RRI_timewindow_arr[index])
    a6=np.unique(RRI_timewindow_arr[index])
    # print(a6);
    print(len(a6))
    k6=0
    n_arr=[]
    for m in time_arr:
        for n in a6:
            if m-n<=1 and m-n>=0:
                k6=k6+1
                n_arr.append(n)
    print(k6)

    index=[]
    for i in range(len(Pcombined)):
        for item in Pcombined[i]:
            if item >= 0.3*Th3:
                index.append(6*i+0)
    # print(RRI_timewindow_arr[index])
    a7=np.unique(RRI_timewindow_arr[index])
    # print(a7);
    print(len(a7))
    k7=0
    n_arr=[]
    for m in time_arr:
        for n in a7:
            if m-n<=1 and m-n>=0:
                k7=k7+1
                n_arr.append(n)
    print(k7)

    index=[]
    for i in range(len(Pcombined)):
        for item in Pcombined[i]:
            if item >= 0.6*Th3:
                index.append(6*i+0)
    # print(RRI_timewindow_arr[index])
    a8=np.unique(RRI_timewindow_arr[index])
    # print(a8);
    print(len(a8))
    k8=0
    n_arr=[]
    for m in time_arr:
        for n in a8:
            if m-n<=1 and m-n>=0:
                k8=k8+1
                n_arr.append(n)
    print(k8)

    index=[]
    for i in range(len(Pcombined)):
        for item in Pcombined[i]:
            if item >= 1.2*Th3:
                index.append(6*i+0)
    # print(RRI_timewindow_arr[index])
    a9=np.unique(RRI_timewindow_arr[index])
    # print(a9);
    print(len(a9))
    k9=0
    n_arr=[]
    for m in time_arr:
        for n in a9:
            if m-n<=1 and m-n>=0:
                k9=k9+1
                n_arr.append(n)
    print(k9)

    index=[]
    for i in range(len(Pcombined)):
        for item in Pcombined[i]:
            if item >= 2*Th3:
                index.append(6*i+0)
    # print(RRI_timewindow_arr[index])
    a10=np.unique(RRI_timewindow_arr[index])
    k10=0
    n_arr=[]
    for m in time_arr:
        for n in a10:
            if m-n<=1 and m-n>=0:
                k10=k10+1
                n_arr.append(n)
    print(k10)

    Sen6=k6/len(time_arr);Sen7=k7/len(time_arr);Sen8=k8/len(time_arr);Sen9=k9/len(time_arr);Sen10=k10/len(time_arr);
    FPR6=(len(a6)-k6)/len(Pcombined);FPR7=(len(a7)-k7)/len(Pcombined);FPR8=(len(a8)-k8)/len(Pcombined);FPR9=(len(a9)-k9)/len(Pcombined);FPR10=(len(a10)-k10)/len(Pcombined);
    Sen_arr_COM=[0,Sen6,Sen7,Sen8,Sen9,Sen10,1]
    FPR_arr_COM=[0,FPR6,FPR7,FPR8,FPR9,FPR10,1]
    print(Sen_arr_COM);print(FPR_arr_COM);

    from sklearn.metrics import auc
    AUC_com=auc(np.sort(FPR_arr_COM),np.sort(Sen_arr_COM))
    AUC_com_arr_EEGcirca.append(AUC_com)

print(AUC_com_arr_EEGcirca)
print(time_arr_arr_EEGcirca)
np.savetxt("C:/Users/wxiong/PycharmProjects/EEGstudy/chance_predictor/AUC_EEG_var_auto_circa6h_SA0124.csv", AUC_com_arr_EEGcirca, delimiter=",", fmt='%s')
np.savetxt("C:/Users/wxiong/PycharmProjects/EEGstudy/chance_predictor/seizure_labels_EEG_var_auto_circa6h_SA0124.csv", time_arr_arr_EEGcirca, delimiter=",", fmt='%s')
