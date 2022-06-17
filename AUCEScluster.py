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


# # # ## SA0124
# # x1=[7/9,8/9,8/9,5/9,4/9]; ### time block-based
# # y1=[0.358,0.457,0.519,0.309,0.21];
# # x3=[4/9,5/9,4/9,3/9,2/9];
# # y3=[0.111,0.173,0.160,0.086,0.037];
# # x5=[6/9,6/9,6/9,5/9,5/9];
# # y5=[0.272,0.272,0.272,0.123,0.123];
# # # x1=[0.5,0.818,0.746,0.348,0.250]; ### point-bases
# # # y1=[0.29424,0.54115,0.4465,0.2469,0.18519];
# # # x3=[0.3333,0.517,0.35,0.233,0.155];
# # # y3=[0.07613,0.11317,0.08642,0.06173,0.02675];
# # # x5=[0.708,0.431];
# # # y5=[0.1852,0.072];
# # x6=[0,1];
# # y6=[0,1];
# # pyplot.scatter(y1[0],x1[0],facecolors='none', edgecolors='darkblue',label='EEG var,EEG auto,RRI var, RRI auto')
# # pyplot.scatter(y1[1],x1[1],facecolors='none', edgecolors='darkblue')
# # pyplot.scatter(y1[2],x1[2],facecolors='none', edgecolors='darkblue')
# # pyplot.scatter(y1[3],x1[3],facecolors='none', edgecolors='darkblue')
# # pyplot.scatter(y1[4],x1[4],facecolors='none', edgecolors='darkblue')
# # pyplot.scatter(y3[0],x3[0],marker="^",facecolors='none', edgecolors='g',label='EEG var,EEG auto,RRI var, RRI auto, circadian')
# # pyplot.scatter(y3[1],x3[1],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y3[2],x3[2],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y3[3],x3[3],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y3[4],x3[4],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y5[0],x5[0],marker="3",c='k',label='circadian')
# # pyplot.scatter(y5[1],x5[1],marker="3",c='k')
# # pyplot.scatter(y5[2],x5[2],marker="3",c='k')
# # pyplot.scatter(y5[3],x5[3],marker="3",c='k')
# # pyplot.scatter(y5[4],x5[4],marker="3",c='k')
# # pyplot.plot(x6,y6,'k--')
# # pyplot.xlabel('FPR')
# # pyplot.ylabel('Sensitivity')
# # pyplot.title('Performance in SA0124')
# # pyplot.legend(loc='lower right',fontsize=8)
# # pyplot.show()
# x1=[0,4/9,5/9,7/9,8/9,8/9,1];
# y1=[0,0.21,0.309,0.358,0.457,0.519,1];
# x3=[0,2/9,3/9,4/9,4/9,5/9,1];
# y3=[0,0.037,0.086,0.111,0.160,0.173,1];
# x5=[0,5/9,5/9,6/9,6/9,6/9,1];
# y5=[0,0.123,0.123,0.272,0.272,0.272,1];
# # x1=[0,0.250,0.348,0.5,0.746,0.818,1];
# # y1=[0,0.18519,0.2469,0.29424,0.4465,0.54115,1];
# # x3=[0,0.155,0.233,0.3333,0.35,0.517,1];
# # y3=[0,0.02675,0.06173,0.07613,0.08642,0.11317,1];
# # x5=[0,0.431,0.708,1];
# # y5=[0,0.072,0.1852,1];
# x6=[0,1];
# y6=[0,1];
# pyplot.plot(y1,x1,'darkblue',alpha=0.9,label='critical slowing forecaster')
# pyplot.plot(y3,x3,'g',alpha=0.9,label='combine forecaster')
# pyplot.plot(y5,x5,'k',alpha=0.8,label='circadian forecaster')
# pyplot.plot(x6,y6,'k--')
# pyplot.xlabel('FPR',fontsize=18)
# pyplot.ylabel('Sensitivity',fontsize=18)
# pyplot.text(0.01,0.95,'$P_1$',fontsize=30)
# locs, labels = pyplot.xticks(fontsize=18)
# locs, labels = pyplot.yticks(fontsize=18)
# pyplot.legend(loc='lower right',fontsize=12)
# pyplot.show()
# from sklearn.metrics import auc
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y1,x1)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y3,x3)))
#
#
#
#
# # # # QLD0098 QLD0098
# # x2=[0,1/4,1/4,0,0];
# # y2=[0.097,0.167,0.125,0.069,0.069]
# # x4=[1/4,1/4,1/4,1/4,1/4];
# # y4=[0.042,0.042,0.042,0.042,0.042]
# # x5=[1/2,1/2,1/2,1/4,1/4,];
# # y5=[0.069,0.069,0.069,0.069,0.069]
# # # x2=[0,0.2,0.148,0];
# # # y2=[0.03240741,0.0462963,0.08187135,0.10526];
# # # x4=[0.167,0.167];
# # # y4=[0.03472222,0.03703704];
# # # x5=[0.500,0.042];
# # # y5=[0.05324,0.0787];
# # x6=[0,1];
# # y6=[0,1];
# # pyplot.scatter(y2[0],x2[0],facecolors='none', edgecolors='r',label='EEG var,EEG auto,RRI var, RRI auto')
# # pyplot.scatter(y2[1],x2[1],facecolors='none', edgecolors='r')
# # pyplot.scatter(y2[2],x2[2],facecolors='none', edgecolors='r')
# # pyplot.scatter(y2[3],x2[3],facecolors='none', edgecolors='r')
# # pyplot.scatter(y2[4],x2[4],facecolors='none', edgecolors='r')
# # pyplot.scatter(y4[0],x4[0],marker="^",facecolors='none', edgecolors='g',label='EEG var,EEG auto,RRI var, RRI auto, circadian')
# # pyplot.scatter(y4[1],x4[1],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y4[2],x4[2],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y4[3],x4[3],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y4[4],x4[4],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y5[0],x5[0],marker="3",c='k',label='circadian')
# # pyplot.scatter(y5[1],x5[1],marker="3",c='k')
# # pyplot.scatter(y5[2],x5[2],marker="3",c='k')
# # pyplot.scatter(y5[3],x5[3],marker="3",c='k')
# # pyplot.scatter(y5[4],x5[4],marker="3",c='k')
# # pyplot.plot(x6,y6,'k--')
# # pyplot.xlabel('FPR')
# # pyplot.ylabel('Sensitivity')
# # pyplot.title('performance in QLD0098')
# # pyplot.legend(loc='lower right',fontsize=8)
# # pyplot.show()
#
# x2=[0,0,0,0,1/4,1/4,1];
# y2=[0,0.069,0.069,0.097,0.125,0.167,1]
# x4=[0,1/4,1/4,1/4,1/4,1/4,1];
# y4=[0,0.042,0.042,0.042,0.042,0.042,1]
# x5=[0,1/4,1/4,1/2,1/2,1/2,1];
# y5=[0,0.069,0.069,0.069,0.069,0.069,1]
# # x2=[0,0,0,0.148,0.2,1];
# # y2=[0,0.03240741,0.0462963,0.08187135,0.10526,1];
# # x4=[0,0.167,0.167,1];
# # y4=[0,0.03472222,0.03703704,1];
# # x5=[0,0.042,0.500,1];
# # y5=[0,0.05324,0.0787,1];
# x6=[0,1];
# y6=[0,1];
# pyplot.plot(y2,x2,'darkblue',alpha=0.9)
# pyplot.plot(y4,x4,'g',alpha=0.9)
# pyplot.plot(y5,x5,'k',alpha=0.8)
# pyplot.plot(x6,y6,'k--')
# pyplot.xlabel('FPR',fontsize=18)
# pyplot.ylabel('Sensitivity',fontsize=18)
# pyplot.text(0.01,0.95,'$P_2$',fontsize=30)
# locs, labels = pyplot.xticks(fontsize=18)
# locs, labels = pyplot.yticks(fontsize=18)
# pyplot.show()
# from sklearn.metrics import auc
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y2,x2)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y4,x4)))
#
#
#
# #
# # # # #### QLD0227 QLD0227
# # x2=[1,1,1,1,1];
# # y2=[0.84,0.84,0.84,0.63,0.58];
# # x4=[0,0,0,0,0];
# # y4=[0.3,0.3,0.3,0.21,0.20];
# # x5=[0,0,0,];
# # y5=[0.33,0.33,0.33];
# # # x2=[0.5,0.5,0.5,0.5];
# # # y2=[0.39506173,0.4382716,0.6646,0.68930041];
# # # x4=[0,0,0,0];
# # # y4=[0.1090535,0.12345679,0.1563786,0.1821946];
# # # x5=[0];
# # # y5=[0.22222222];
# # x6=[0,1];
# # y6=[0,1];
# # pyplot.scatter(y2[0],x2[0],facecolors='none', edgecolors='r',label='EEG var,EEG auto,RRI var, RRI auto')
# # pyplot.scatter(y2[1],x2[1],facecolors='none', edgecolors='r')
# # pyplot.scatter(y2[2],x2[2],facecolors='none', edgecolors='r')
# # pyplot.scatter(y2[3],x2[3],facecolors='none', edgecolors='r')
# # pyplot.scatter(y2[4],x2[4],facecolors='none', edgecolors='r')
# # pyplot.scatter(y4[0],x4[0],marker="^",facecolors='none', edgecolors='g',label='EEG var,EEG auto,RRI var, RRI auto, circadian')
# # pyplot.scatter(y4[1],x4[1],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y4[2],x4[2],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y4[3],x4[3],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y4[4],x4[4],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y5[0],x5[0],marker="3",c='k',label='circadian')
# # pyplot.scatter(y5[1],x5[1],marker="3",c='k')
# # pyplot.scatter(y5[2],x5[2],marker="3",c='k')
# # pyplot.plot(x6,y6,'k--')
# # pyplot.xlabel('FPR')
# # pyplot.ylabel('Sensitivity')
# # pyplot.title('performance in QLD0227')
# # pyplot.legend(loc='lower right',fontsize=8)
# # pyplot.show()
# x2=[0,1,1,1,1,1,1];
# y2=[0,0.58,0.63,0.84,0.84,0.84,1];
# x4=[0,0,0,0,0,0,1];
# y4=[0,0.20,0.21,0.3,0.3,0.3,1];
# x5=[0,0,0,0,1];
# y5=[0,0.33,0.33,0.33,1];
# # x2=[0,0.5,0.5,0.5,0.5,1];
# # y2=[0,0.39506173,0.4382716,0.6646,0.68930041,1];
# # x4=[0,0,0,0,0,1];
# # y4=[0,0.1090535,0.12345679,0.1563786,0.1821946,1];
# # x5=[0,0,1];
# # y5=[0,0.22222222,1];
# x6=[0,1];
# y6=[0,1];
# pyplot.plot(y2,x2,'darkblue',alpha=0.9)
# pyplot.plot(y4,x4,'g',alpha=0.9)
# pyplot.plot(y5,x5,'k',alpha=0.8)
# pyplot.plot(x6,y6,'k--')
# pyplot.xlabel('FPR',fontsize=18)
# pyplot.ylabel('Sensitivity',fontsize=18)
# pyplot.text(0.01,0.95,'$P_3$',fontsize=30)
# locs, labels = pyplot.xticks(fontsize=18)
# locs, labels = pyplot.yticks(fontsize=18)
# pyplot.show()
# from sklearn.metrics import auc
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y2,x2)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y4,x4)))
#
#
#
# # ##VIC1202
# # x2=[4/7,4/7,4/7,4/7,0/7];
# # y2=[0.405,0.486,0.450,0.351,0.054]
# # x4=[3/7,3/7,3/7,3/7,1/7];
# # y4=[0.135,0.153,0.153,0.108,0.036]
# # x5=[4/7,4/7,4/7,2/7,2/7];
# # y5=[0.279,0.279,0.279,0.07,0.07];
# # # x2=[0.571,0.571,0.571,0.571,0];
# # # y2=[0.289789,0.40991,0.46096,0.345345,0.02402402,];
# # # x4=[0.143,0.405,0.405,0.405,];
# # # y4=[0.01651652,0.07057,0.09,0.1081];
# # # x5=[0.214,0.476];
# # # y5=[0.04654655,0.222222];
# # x6=[0,1];
# # y6=[0,1];
# # pyplot.scatter(y2[0],x2[0],facecolors='none', edgecolors='r',label='EEG var,EEG auto,RRI var, RRI auto')
# # pyplot.scatter(y2[1],x2[1],facecolors='none', edgecolors='r')
# # pyplot.scatter(y2[2],x2[2],facecolors='none', edgecolors='r')
# # pyplot.scatter(y2[3],x2[3],facecolors='none', edgecolors='r')
# # pyplot.scatter(y2[4],x2[4],facecolors='none', edgecolors='r')
# # pyplot.scatter(y4[0],x4[0],marker="^",facecolors='none', edgecolors='g',label='EEG var,EEG auto,RRI var, RRI auto, circadian')
# # pyplot.scatter(y4[1],x4[1],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y4[2],x4[2],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y4[3],x4[3],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y4[4],x4[4],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y5[0],x5[0],marker="3",c='k',label='circadian')
# # pyplot.scatter(y5[1],x5[1],marker="3",c='k')
# # pyplot.scatter(y5[2],x5[2],marker="3",c='k')
# # pyplot.scatter(y5[3],x5[3],marker="3",c='k')
# # pyplot.scatter(y5[4],x5[4],marker="3",c='k')
# # pyplot.plot(x6,y6,'k--')
# # pyplot.xlabel('FPR')
# # pyplot.ylabel('Sensitivity')
# # pyplot.title('performance in VIC1202')
# # pyplot.legend(loc='lower right',fontsize=8)
# # pyplot.show()
# x2=[0,0/7,4/7,4/7,4/7,4/7,1];
# y2=[0,0.054,0.351,0.405,0.450,0.486,1]
# x4=[0,1/7,3/7,3/7,3/7,3/7,1];
# y4=[0,0.036,0.108,0.135,0.153,0.153,1]
# x5=[0,2/7,5/7,5/7,5/7,1];
# y5=[0,0.07,0.279,0.279,0.279,1];
# # x2=[0,0,0.571,0.571,0.571,0.571,1];
# # y2=[0,0.02402402,0.289789,0.345345,0.40991,0.46096,1];
# # x4=[0,0.143,0.405,0.405,0.405,1];
# # y4=[0,0.01651652,0.07057,0.09,0.1081,1];
# # x5=[0,0.214,0.476,1];
# # y5=[0,0.04654655,0.222222,1];
# x6=[0,1];
# y6=[0,1];
# pyplot.plot(y2,x2,'darkblue',alpha=0.9)
# pyplot.plot(y4,x4,'g',alpha=0.9)
# pyplot.plot(y5,x5,'k',alpha=0.8)
# pyplot.plot(x6,y6,'k--')
# pyplot.xlabel('FPR',fontsize=18)
# pyplot.ylabel('Sensitivity',fontsize=18)
# pyplot.text(0.01,0.95,'$P_4$',fontsize=30)
# locs, labels = pyplot.xticks(fontsize=18)
# locs, labels = pyplot.yticks(fontsize=18)
# pyplot.show()
# from sklearn.metrics import auc
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y2,x2)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y4,x4)))
#
#
#
#
# #
# # ##### VIC1173
# # x2=[2/4,2/4,2/4,2/4,2/4];
# # y2=[0.52,0.52,0.52,0.39,0.39]
# # x4=[1/4,1/4,1/4,1/4,1/4];
# # y4=[0.1,0.1,0.1,0.09,0.08]
# # x5=[1/2,1/2,1/2]
# # y5=[0.2,0.2,0.2]
# # # x2=[0.586,0.586];
# # # y2=[0.3673835,0.43727599];
# # # x4=[0.25,0.182];
# # # y4=[0.04301,0.057347];
# # # x5=[2/4];
# # # y5=[0.12724];
# # x6=[0,1];
# # y6=[0,1];
# # pyplot.scatter(y2[0],x2[0],facecolors='none', edgecolors='r',label='EEG var,EEG auto,RRI var, RRI auto')
# # pyplot.scatter(y2[1],x2[1],facecolors='none', edgecolors='r')
# # pyplot.scatter(y2[2],x2[2],facecolors='none', edgecolors='r')
# # pyplot.scatter(y2[3],x2[3],facecolors='none', edgecolors='r')
# # pyplot.scatter(y2[4],x2[4],facecolors='none', edgecolors='r')
# # pyplot.scatter(y4[0],x4[0],marker="^",facecolors='none', edgecolors='g',label='EEG var,EEG auto,RRI var, RRI auto, circadian')
# # pyplot.scatter(y4[1],x4[1],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y4[2],x4[2],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y4[3],x4[3],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y4[4],x4[4],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y5[0],x5[0],marker="3",c='k',label='circadian')
# # pyplot.scatter(y5[1],x5[1],marker="3",c='k',label='circadian')
# # pyplot.scatter(y5[2],x5[2],marker="3",c='k',label='circadian')
# # pyplot.plot(x6,y6,'k--')
# # pyplot.xlabel('FPR')
# # pyplot.ylabel('Sensitivity')
# # pyplot.title('performance in VIC1173')
# # pyplot.legend(loc='lower right',fontsize=8)
# # pyplot.show()
# x2=[0,2/4,2/4,2/4,2/4,2/4,1];
# y2=[0,0.39,0.39,0.52,0.52,0.52,1]
# x4=[0,1/4,1/4,1/4,1/4,1/4,1];
# y4=[0,0.08,0.09,0.1,0.1,0.1,1]
# x5=[0,1/2,1/2,1/2,1]
# y5=[0,0.2,0.2,0.2,1]
# # x2=[0,0.586,0.586,1];
# # y2=[0,0.3673835,0.43727599,1];
# # x4=[0,0.182,0.25,1];
# # y4=[0,0.04301,0.057347,1];
# # x5=[0,2/4,1];
# # y5=[0,0.12724,1];
# x6=[0,1];
# y6=[0,1];
# pyplot.plot(y2,x2,'darkblue',alpha=0.9)
# pyplot.plot(y4,x4,'g',alpha=0.9)
# pyplot.plot(y5,x5,'k',alpha=0.8)
# pyplot.plot(x6,y6,'k--')
# pyplot.xlabel('FPR',fontsize=18)
# pyplot.ylabel('Sensitivity',fontsize=18)
# pyplot.text(0.01,0.95,'$P_5$',fontsize=30)
# locs, labels = pyplot.xticks(fontsize=18)
# locs, labels = pyplot.yticks(fontsize=18)
# pyplot.show()
# from sklearn.metrics import auc
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y2,x2)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y4,x4)))
#
#
#
#
#
#
# # # ##### VIC1757
# # # x1=[1/4,1/2,1/4,1/4,1/4];
# # # y1=[0.23,0.35,0.23,0.23,0.21]
# # # x3=[1/4,1/4,1/4,1/4,1/4];
# # # y3=[0.11,0.19,0.16,0.11,0.09]
# # # x5=[3/4,3/4,3/4,1/4,1/4];
# # # y5=[0.51,0.51,0.51,0.12,0.12]
# # # # x1=[1/4,1/4,1/4,];
# # # # y1=[0.1608,0.1696,0.2807];
# # # # x3=[0.167,0.167,0.167,0.167];
# # # # y3=[0.0497076,0.05847953,0.08187135,0.10526316];
# # # # x5=[0.417,0.167];
# # # # y5=[0.08187135,0.3128655];
# # # x6=[0,1];
# # # y6=[0,1];
# # # pyplot.scatter(y1[0],x1[0],facecolors='none', edgecolors='r',label='EEG var,EEG auto,RRI var, RRI auto')
# # # pyplot.scatter(y1[1],x1[1],facecolors='none', edgecolors='r')
# # # pyplot.scatter(y1[2],x1[2],facecolors='none', edgecolors='r')
# # # pyplot.scatter(y1[3],x1[3],facecolors='none', edgecolors='r')
# # # pyplot.scatter(y1[4],x1[4],facecolors='none', edgecolors='r')
# # # pyplot.scatter(y3[0],x3[0],marker="^",facecolors='none', edgecolors='g',label='EEG var,EEG auto,RRI var, RRI auto, circadian')
# # # pyplot.scatter(y3[1],x3[1],marker="^",facecolors='none', edgecolors='g')
# # # pyplot.scatter(y3[2],x3[2],marker="^",facecolors='none', edgecolors='g')
# # # pyplot.scatter(y3[3],x3[3],marker="^",facecolors='none', edgecolors='g')
# # # pyplot.scatter(y3[4],x3[4],marker="^",facecolors='none', edgecolors='g')
# # # pyplot.scatter(y5[0],x5[0],marker="3",c='k',label='circadian')
# # # pyplot.scatter(y5[1],x5[1],marker="3",c='k',label='circadian')
# # # pyplot.scatter(y5[2],x5[2],marker="3",c='k',label='circadian')
# # # pyplot.scatter(y5[3],x5[3],marker="3",c='k',label='circadian')
# # # pyplot.scatter(y5[4],x5[4],marker="3",c='k',label='circadian')
# # # pyplot.plot(x6,y6,'k--')
# # # pyplot.xlabel('FPR')
# # # pyplot.ylabel('Sensitivity')
# # # pyplot.title('performance in VIC1757')
# # # pyplot.legend(loc='lower right',fontsize=8)
# # # pyplot.show()
# x1=[0,1/4,1/4,1/4,1/4,1/2,1];
# y1=[0,0.21,0.23,0.23,0.23,0.35,1]
# x3=[0,1/4,1/4,1/4,1/4,1/4,1];
# y3=[0,0.09,0.11,0.11,0.16,0.19,1]
# x5=[0,1/4,1/4,3/4,3/4,3/4,1];
# y5=[0,0.12,0.12,0.482,0.482,0.482,1]
# # x1=[0,1/4,1/4,1/4,1];
# # y1=[0,0.1608,0.1696,0.2807,1];
# # x3=[0,0.167,0.167,0.167,0.167,1];
# # y3=[0,0.0497076,0.05847953,0.08187135,0.10526316,1];
# # x5=[0,0.167,0.417,1];
# # y5=[0,0.08187135,0.3128655,1];
# x6=[0,1];
# y6=[0,1];
# pyplot.plot(y1,x1,'darkblue',alpha=0.9)
# pyplot.plot(y3,x3,'g',alpha=0.9)
# pyplot.plot(y5,x5,'k',alpha=0.8)
# pyplot.plot(x6,y6,'k--')
# pyplot.xlabel('FPR',fontsize=18)
# pyplot.ylabel('Sensitivity',fontsize=18)
# pyplot.text(0.01,0.95,'$P_6$',fontsize=30)
# locs, labels = pyplot.xticks(fontsize=18)
# locs, labels = pyplot.yticks(fontsize=18)
# pyplot.show()
# from sklearn.metrics import auc
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y1,x1)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y3,x3)))
#
#
#
#
#
#
# # # # # ##### VIC2248
# # x2=[1/4,1/4,1/4,1/4,1/4];
# # y2=[0.09,0.09,0.09,0.08,0.08]
# # x4=[0,0,0,0,0]
# # y4=[0,0,0,0,0]
# # x5=[1/4,1/4,1/4,1/4,1/4]
# # y5=[0.187,0.187,0.187,0.107,0.107]
# # # x2=[0.042,0.042];
# # # y2=[0.04222,0.0444];
# # # x4=[0,0];
# # # y4=[0,0];
# # # x5=[0.083,0.083];
# # # y5=[0.0488889,0.1022222];
# # x6=[0,1];
# # y6=[0,1];
# # pyplot.scatter(y2[0],x2[0],facecolors='none', edgecolors='r',label='EEG var,EEG auto,RRI var, RRI auto')
# # pyplot.scatter(y2[1],x2[1],facecolors='none', edgecolors='r')
# # pyplot.scatter(y2[2],x2[2],facecolors='none', edgecolors='r')
# # pyplot.scatter(y2[3],x2[3],facecolors='none', edgecolors='r')
# # pyplot.scatter(y2[4],x2[4],facecolors='none', edgecolors='r')
# # pyplot.scatter(y4[0],x4[0],marker="^",facecolors='none', edgecolors='g',label='EEG var,EEG auto,RRI var, RRI auto, circadian')
# # pyplot.scatter(y4[1],x4[1],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y4[2],x4[2],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y4[3],x4[3],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y4[4],x4[4],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y5[0],x5[0],marker="3",c='k',label='circadian')
# # pyplot.scatter(y5[1],x5[1],marker="3",c='k')
# # pyplot.scatter(y5[2],x5[2],marker="3",c='k')
# # pyplot.scatter(y5[3],x5[3],marker="3",c='k')
# # pyplot.scatter(y5[4],x5[4],marker="3",c='k')
# # pyplot.plot(x6,y6,'k--')
# # pyplot.xlabel('FPR')
# # pyplot.ylabel('Sensitivity')
# # pyplot.title('performance in VIC2284')
# # pyplot.legend(loc='lower right',fontsize=8)
# # pyplot.show()
# x2=[0,1/4,1/4,1/4,1/4,1/4,1];
# y2=[0,0.08,0.08,0.09,0.09,0.09,1]
# x4=[0,0,0,0,0,0,1]
# y4=[0,0,0,0,0,0,1]
# x5=[0,1/4,1/4,1/4,1/4,1/4,1]
# y5=[0,0.107,0.107,0.187,0.187,0.187,1]
# # x2=[0,0.042,0.042,1];
# # y2=[0,0.04222,0.0444,1];
# # x4=[0,0,0,1];
# # y4=[0,0,0,1];
# # x5=[0,0.083,0.083,1];
# # y5=[0,0.0488889,0.1022222,1];
# x6=[0,1];
# y6=[0,1];
# pyplot.plot(y2,x2,'darkblue',alpha=0.9)
# pyplot.plot(y4,x4,'g',alpha=0.9)
# pyplot.plot(y5,x5,'k',alpha=0.8)
# pyplot.plot(x6,y6,'k--')
# pyplot.xlabel('FPR',fontsize=18)
# pyplot.ylabel('Sensitivity',fontsize=18)
# locs, labels = pyplot.xticks(fontsize=18)
# locs, labels = pyplot.yticks(fontsize=18)
# pyplot.text(0.01,0.95,'$P_7$',fontsize=30)
# pyplot.show()
# from sklearn.metrics import auc
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y2,x2)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y4,x4)))
#
#
#
#
#
#
#
# # # # # # ##### QLD0481
# # x1=[5/7,5/7,5/7,5/7,5/7]
# # y1=[0.324,0.324,0.324,0.305,0.305]
# # x3=[1/7,1/7,1/7,1/7,1/7]
# # y3=[0.067,0.067,0.067,0.057,0.038]
# # x5=[2/7,2/7,2/7,0,0]
# # y5=[0.2,0.2,0.2,0.143,0.143]
# # # x1=[0.143,0.143,0]
# # # y1=[0.05873,0.06349,0.014285]
# # # x3=[0/7,0/7,0/7]
# # # y3=[0.012698,0.01746,0.00158]
# # # x5=[0.238,0];
# # # y5=[0.11746,0.095238,];
# # x6=[0,1];
# # y6=[0,1];
# # pyplot.scatter(y1[0],x1[0],facecolors='none', edgecolors='r',label='RRI var, RRI auto')
# # pyplot.scatter(y1[1],x1[1],facecolors='none', edgecolors='r')
# # pyplot.scatter(y1[2],x1[2],facecolors='none', edgecolors='r')
# # pyplot.scatter(y1[3],x1[3],facecolors='none', edgecolors='r')
# # pyplot.scatter(y1[4],x1[4],facecolors='none', edgecolors='r')
# # pyplot.scatter(y3[0],x3[0],marker="^",facecolors='none', edgecolors='g',label='RRI var, RRI auto, circadian')
# # pyplot.scatter(y3[1],x3[1],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y3[2],x3[2],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y3[3],x3[3],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y3[4],x3[4],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y5[0],x5[0],marker="3",c='k',label='circadian')
# # pyplot.scatter(y5[1],x5[1],marker="3",c='k',label='circadian')
# # pyplot.scatter(y5[2],x5[2],marker="3",c='k',label='circadian')
# # pyplot.scatter(y5[3],x5[3],marker="3",c='k',label='circadian')
# # pyplot.scatter(y5[4],x5[4],marker="3",c='k',label='circadian')
# # pyplot.plot(x6,y6,'k--')
# # pyplot.xlabel('FPR')
# # pyplot.ylabel('Sensitivity')
# # pyplot.title('performance in QLD0481')
# # pyplot.legend(loc='lower right',fontsize=8)
# # pyplot.show()
# x1=[0,5/7,5/7,5/7,5/7,5/7,1]
# y1=[0,0.305,0.305,0.324,0.324,0.324,1]
# x3=[0,1/7,1/7,1/7,1/7,1/7,1]
# y3=[0,0.038,0.057,0.067,0.067,0.067,1]
# x5=[0,0,0,2/7,2/7,2/7,1]
# y5=[0,0.143,0.143,0.2,0.2,0.2,1]
# # x1=[0,0,0.143,0.143,1]
# # y1=[0,0.014285,0.05873,0.06349,1]
# # x3=[0,0/7,0/7,0/7,1]
# # y3=[0,0.00158,0.012698,0.01746,1]
# # x5=[0,0,0.238,1];
# # y5=[0,0.095238,0.11746,1];
# x6=[0,1];
# y6=[0,1];
# pyplot.plot(y1,x1,'darkblue',alpha=0.9)
# pyplot.plot(y3,x3,'g',alpha=0.9)
# pyplot.plot(y5,x5,'k',alpha=0.8)
# pyplot.plot(x6,y6,'k--')
# pyplot.xlabel('FPR',fontsize=18)
# pyplot.ylabel('Sensitivity',fontsize=18)
# pyplot.text(0.01,0.95,'$P_8$',fontsize=30)
# locs, labels = pyplot.xticks(fontsize=18)
# locs, labels = pyplot.yticks(fontsize=18)
# pyplot.show()
# from sklearn.metrics import auc
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y1,x1)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y3,x3)))


# # ###VIC1795
# x2=[1,1,1,1,1];
# y2=[0.24,0.24,0.24,0.14,0.14];
# x4=[1,1,1,1,1];
# y4=[0.06,0.06,0.06,0.04,0.02];
# x5=[1,1,1,1/2,1/2];
# y5=[0.27,0.27,0.27,0.14,0.06];
# # x2=[0.167,0.167,0.167];
# # y2=[0.67647,0.444444,0.19608];
# # x4=[0.167,0.167,0.167];
# # y4=[0.10784,0.088235,0.026143];
# # x5=[0.167,0.667];
# # y5=[0.09803922,0.183];
# x6=[0,1];
# y6=[0,1];
# pyplot.scatter(y2[0],x2[0],facecolors='none', edgecolors='r',label='RRI var, RRI auto')
# pyplot.scatter(y2[1],x2[1],facecolors='none', edgecolors='r')
# pyplot.scatter(y2[2],x2[2],facecolors='none', edgecolors='r')
# pyplot.scatter(y2[3],x2[3],facecolors='none', edgecolors='r')
# pyplot.scatter(y2[4],x2[4],facecolors='none', edgecolors='r')
# pyplot.scatter(y4[0],x4[0],marker="^",facecolors='none', edgecolors='g',label='RRI var, RRI auto, circadian')
# pyplot.scatter(y4[1],x4[1],marker="^",facecolors='none', edgecolors='g')
# pyplot.scatter(y4[2],x4[2],marker="^",facecolors='none', edgecolors='g')
# pyplot.scatter(y4[3],x4[3],marker="^",facecolors='none', edgecolors='g')
# pyplot.scatter(y4[4],x4[4],marker="^",facecolors='none', edgecolors='g')
# pyplot.scatter(y5[0],x5[0],marker="3",c='k',label='circadian')
# pyplot.scatter(y5[1],x5[1],marker="3",c='k')
# pyplot.scatter(y5[2],x5[2],marker="3",c='k')
# pyplot.scatter(y5[3],x5[3],marker="3",c='k')
# pyplot.scatter(y5[4],x5[4],marker="3",c='k')
# pyplot.plot(x6,y6,'k--')
# pyplot.xlabel('FPR')
# pyplot.ylabel('Sensitivity')
# pyplot.title('performance in VIC1795')
# pyplot.legend(loc='lower right',fontsize=8)
# pyplot.show()
# x2=[0,1,1,1,1,1,1];
# y2=[0,0.14,0.14,0.24,0.24,0.24,1];
# x4=[0,1,1,1,1,1,1];
# y4=[0,0.02,0.04,0.06,0.06,0.06,1];
# x5=[0,1/2,1/2,1,1,1,1];
# y5=[0,0.06,0.14,0.27,0.27,0.27,1];
# x2=[0,0.167,0.167,0.167,1];
# y2=[0,0.19608,0.444444,0.67647,1];
# x4=[0,0.167,0.167,0.167,1];
# y4=[0,0.026143,0.088235,0.10784,1];
# x5=[0,0.167,0.667,1];
# y5=[0,0.09803922,0.183,1];
x2=[0,0,1/2,1];
y2=[0,0.173,0.255,1];
x4=[0,0,0,1/2,1];
y4=[0,0.058,0.077,0.078,1];
x5=[0,0,0,1/2,1/2,1];
y5=[0,0.077,0.154,0.288,0.288,1];
x6=[0,1];
y6=[0,1];
pyplot.plot(y2,x2,'darkblue',alpha=0.9)
pyplot.plot(y4,x4,'g',alpha=0.9)
pyplot.plot(y5,x5,'k',alpha=0.8)
pyplot.plot(x6,y6,'k--')
pyplot.xlabel('FPR',fontsize=18)
pyplot.ylabel('Sensitivity',fontsize=18)
pyplot.text(0.03,0.90,'$P_9$',fontsize=30)
locs, labels = pyplot.xticks(fontsize=18)
locs, labels = pyplot.yticks(fontsize=18)
pyplot.show()
from sklearn.metrics import auc
print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y2,x2)))
print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y4,x4)))






# # # #### NSW0352
# # x2=[1/2,1/2,1/2,1/2,0];
# # y2=[0.6,0.617,0.617,0.383,0.148];
# # x4=[1/4,1/4,1/4,1/4,1/4];
# # y4=[0.185,0.185,0.179,0.148,0.086];
# # x5=[3/4,3/4,3/4,3/4,3/4];
# # y5=[0.247,0.247,0.247,0.173,0.173];
# # # x2=[0,0.375,0.375,0.375];
# # # y2=[0.1091,0.29629,0.388,0.4979];
# # # x4=[0.208,0.208,0.208];
# # # y4=[0.04115,0.0658,0.09465];
# # # x5=[0.542,0.542];
# # # y5=[0.08847737,0.13786008];
# # x6=[0,1];
# # y6=[0,1];
# # pyplot.scatter(y2[0],x2[0],facecolors='none', edgecolors='r',label='RRI var, RRI auto')
# # pyplot.scatter(y2[1],x2[1],facecolors='none', edgecolors='r')
# # pyplot.scatter(y2[2],x2[2],facecolors='none', edgecolors='r')
# # pyplot.scatter(y2[3],x2[3],facecolors='none', edgecolors='r')
# # pyplot.scatter(y2[4],x2[4],facecolors='none', edgecolors='r')
# # pyplot.scatter(y4[0],x4[0],marker="^",facecolors='none', edgecolors='g',label='RRI var, RRI auto, circadian')
# # pyplot.scatter(y4[1],x4[1],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y4[2],x4[2],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y4[3],x4[3],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y4[4],x4[4],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y5[0],x5[0],marker="3",c='k',label='circadian')
# # pyplot.scatter(y5[1],x5[1],marker="3",c='k')
# # pyplot.scatter(y5[2],x5[2],marker="3",c='k')
# # pyplot.scatter(y5[3],x5[3],marker="3",c='k')
# # pyplot.scatter(y5[4],x5[4],marker="3",c='k')
# # pyplot.plot(x6,y6,'k--')
# # pyplot.xlabel('FPR')
# # pyplot.ylabel('Sensitivity')
# # pyplot.title('performance in NSW0352')
# # pyplot.legend(loc='lower right',fontsize=8)
# # pyplot.show()
#
# x2=[0,0,1/2,1/2,1/2,1/2,1];
# y2=[0,0.148,0.383,0.605,0.617,0.617,1];
# x4=[0,1/4,1/4,1/4,1/4,1/4,1];
# y4=[0,0.086,0.148,0.179,0.185,0.185,1];
# x5=[0,3/4,3/4,3/4,3/4,3/4,1];
# y5=[0,0.173,0.173,0.247,0.247,0.247,1];
# # x2=[0,0,0.375,0.375,0.375,1];
# # y2=[0,0.1091,0.29629,0.388,0.4979,1];
# # x4=[0,0.208,0.208,0.208,1];
# # y4=[0,0.04115,0.0658,0.09465,1];
# # x5=[0,0.542,0.542,1];
# # y5=[0,0.08847737,0.13786008,1];
# x6=[0,1];
# y6=[0,1];
# pyplot.plot(y5,x5,'k',alpha=0.8)
# pyplot.plot(x6,y6,'k--')
# pyplot.plot(y2,x2,'darkblue',alpha=0.9)
# pyplot.plot(y4,x4,'g',alpha=0.9)
# pyplot.xlabel('FPR',fontsize=18)
# pyplot.ylabel('Sensitivity',fontsize=18)
# pyplot.text(0.05,0.95,'$P_{10}$',fontsize=30)
# locs, labels = pyplot.xticks(fontsize=18)
# locs, labels = pyplot.yticks(fontsize=18)
# pyplot.show()
# from sklearn.metrics import auc
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y2,x2)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y4,x4)))
#
#
#
#
#
# # ## VIC0251
# # x2=[4/6,4/6,4/6,0,0];
# # y2=[0.444,0.58,0.58,0.259,0.037]
# # x4=[2/6,2/6,2/6,0,0];
# # y4=[0.148,0.173,0.173,0.12,0.074]
# # x5=[2/6,2/6,2/6,0,0]
# # y5=[0.247,0.247,0.247,0.074,0.074]
# # # x2=[0.583,0.583,0,0];
# # # y2=[0.29218,0.36214,0.17695,0.02675];
# # # x4=[0.056,0.083,0,0]
# # # y4=[0.098765,0.11728,0.08436,0.045267];
# # # x5=[0.139,0];
# # # y5=[0.18312,0.04938];
# # x6=[0,1];
# # y6=[0,1];
# # pyplot.scatter(y2[0],x2[0],facecolors='none', edgecolors='r',label='EEG var, EEG auto, RRI auto')
# # pyplot.scatter(y2[1],x2[1],facecolors='none', edgecolors='r')
# # pyplot.scatter(y2[2],x2[2],facecolors='none', edgecolors='r')
# # pyplot.scatter(y2[3],x2[3],facecolors='none', edgecolors='r')
# # pyplot.scatter(y2[4],x2[4],facecolors='none', edgecolors='r')
# # pyplot.scatter(y4[0],x4[0],marker="^",facecolors='none', edgecolors='g',label='EEG var, EEG auto, RRI auto, circadian')
# # pyplot.scatter(y4[1],x4[1],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y4[2],x4[2],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y4[3],x4[3],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y4[4],x4[4],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y5[0],x5[0],marker="3",c='k',label='circadian')
# # pyplot.scatter(y5[1],x5[1],marker="3",c='k')
# # pyplot.scatter(y5[2],x5[2],marker="3",c='k')
# # pyplot.scatter(y5[3],x5[3],marker="3",c='k')
# # pyplot.scatter(y5[4],x5[4],marker="3",c='k')
# # pyplot.plot(x6,y6,'k--')
# # pyplot.xlabel('FPR')
# # pyplot.ylabel('Sensitivity')
# # pyplot.title('performance in VIC0251')
# # pyplot.legend(loc='lower right',fontsize=8)
# # pyplot.show()
# x2=[0,0,0,4/6,4/6,4/6,1];
# y2=[0,0.037,0.259,0.444,0.556,0.556,1]
# x4=[0,0,0,2/6,2/6,2/6,1];
# y4=[0,0.074,0.123,0.148,0.173,0.173,1]
# x5=[0,0,2/6,2/6,2/6,1]
# y5=[0,0.074,0.247,0.247,0.247,1]
# # x2=[0,0,0,0.583,0.583,1];
# # y2=[0,0.02675,0.17695,0.29218,0.36214,1];
# # x4=[0,0,0,0.056,0.083,1]
# # y4=[0,0.045267,0.08436,0.098765,0.11728,1];
# # x5=[0,0,0.139,1];
# # y5=[0,0.04938,0.18312,1];
# x6=[0,1];
# y6=[0,1];
# pyplot.plot(y2,x2,'darkblue',alpha=0.9)
# pyplot.plot(y4,x4,'g',alpha=0.9)
# pyplot.plot(y5,x5,'k',alpha=0.6,label='circadian classifier')
# pyplot.plot(x6,y6,'k--')
# pyplot.xlabel('FPR',fontsize=18)
# pyplot.ylabel('Sensitivity',fontsize=18)
# pyplot.text(0.05,0.95,'$P_{11}$',fontsize=30)
# locs, labels = pyplot.xticks(fontsize=18)
# locs, labels = pyplot.yticks(fontsize=18)
# pyplot.show()
# from sklearn.metrics import auc
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y2,x2)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y4,x4)))
#
#
#
#
#
# # # #### SA0174
# # x1=[1/5,1/5,1/5,1/5,1/5,];
# # y1=[0.35,0.4,0.4,0.35,0.35];
# # x3=[1/5,1/5,1/5,1/5,1/5,];
# # y3=[0.28,0.28,0.28,0.26,0.26];
# # x5=[1,1,1];
# # y5=[0.76,0.76,0.76];
# # # x1=[0.24,0.24,0.12];
# # # y1=[0.24631579,0.27789474,0.04421053];
# # # x3=[0.24,0.24,0.12];
# # # y3=[0.12,0.12631579,0.02315789];
# # # x5=[0.68];
# # # y5=[0.46526316];
# # x6=[0,1];
# # y6=[0,1];
# # pyplot.scatter(y1[0],x1[0],facecolors='none', edgecolors='r',label='EEG var, EEG auto, RRI auto')
# # pyplot.scatter(y1[1],x1[1],facecolors='none', edgecolors='r')
# # pyplot.scatter(y1[2],x1[2],facecolors='none', edgecolors='r')
# # pyplot.scatter(y1[3],x1[3],facecolors='none', edgecolors='r')
# # pyplot.scatter(y1[4],x1[4],facecolors='none', edgecolors='r')
# # pyplot.scatter(y3[0],x3[0],marker="^",facecolors='none', edgecolors='g',label='EEG var, EEG auto, RRI auto, circadian')
# # pyplot.scatter(y3[1],x3[1],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y3[2],x3[2],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y3[3],x3[3],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y3[4],x3[4],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y5[0],x5[0],marker="3",c='k',label='circadian')
# # pyplot.scatter(y5[1],x5[1],marker="3",c='k',)
# # pyplot.scatter(y5[2],x5[2],marker="3",c='k',)
# # pyplot.plot(x6,y6,'k--')
# # pyplot.xlabel('FPR')
# # pyplot.ylabel('Sensitivity')
# # pyplot.title('performance in SA0174')
# # pyplot.legend(loc='lower right',fontsize=8)
# # pyplot.show()
# x1=[0,1/5,1/5,1/5,1/5,1/5,1/5,1];
# y1=[0,0.113,0.338,0.338,0.388,0.388,0.388,1];
# x3=[0,1/5,1/5,1/5,1/5,1/5,1/5,1];
# y3=[0,0.063,0.26,0.26,0.275,0.275,0.275,1];
# x5=[0,1,1,1,1];
# y5=[0,0.738,0.738,0.738,1];
# # x1=[0,0.12,0.24,0.24,1];
# # y1=[0,0.04421053,0.24631579,0.27789474,1];
# # x3=[0,0.12,0.24,0.24,1];
# # y3=[0,0.02315789,0.12,0.12631579,1];
# # x5=[0,0.68,1];
# # y5=[0,0.46526316,1];
# x6=[0,1];
# y6=[0,1];
# pyplot.plot(y1,x1,'darkblue',alpha=0.9)
# pyplot.plot(y3,x3,'g',alpha=0.9)
# pyplot.plot(y5,x5,'k',alpha=0.8)
# pyplot.plot(x6,y6,'k--')
# pyplot.xlabel('FPR',fontsize=18)
# pyplot.ylabel('Sensitivity',fontsize=18)
# pyplot.text(0.05,0.95,'$P_{12}$',fontsize=30)
# locs, labels = pyplot.xticks(fontsize=18)
# locs, labels = pyplot.yticks(fontsize=18)
# pyplot.show()
# from sklearn.metrics import auc
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y1,x1)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y3,x3)))
#
#
#
#
#
# # # #### VIC1027
# # x1=[0,0,0,0,0];
# # y1=[0.52,0.61,0.61,0.39,0.39];
# # x3=[0,0,0,0,0];
# # y3=[0.19,0.19,0.19,0.19,0.19];
# # x5=[0,0,0,0,0];
# # y5=[0.28,0.28,0.28,0.28,0.28];
# # x6=[0,1];
# # y6=[0,1];
# # pyplot.scatter(y1[0],x1[0],facecolors='none', edgecolors='r',label='EEG var, RRI var')
# # pyplot.scatter(y1[1],x1[1],facecolors='none', edgecolors='r')
# # pyplot.scatter(y1[2],x1[2],facecolors='none', edgecolors='r')
# # pyplot.scatter(y1[3],x1[3],facecolors='none', edgecolors='r')
# # pyplot.scatter(y1[4],x1[4],facecolors='none', edgecolors='r')
# # pyplot.scatter(y3[0],x3[0],marker="^",facecolors='none', edgecolors='g',label='EEG var, RRI var, circadian')
# # pyplot.scatter(y3[1],x3[1],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y3[2],x3[2],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y3[3],x3[3],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y3[4],x3[4],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y5[0],x5[0],marker="3",c='k',label='circadian')
# # pyplot.scatter(y5[1],x5[1],marker="3",c='k',)
# # pyplot.scatter(y5[2],x5[2],marker="3",c='k',)
# # pyplot.plot(x6,y6,'k--')
# # pyplot.xlabel('FPR')
# # pyplot.ylabel('Sensitivity')
# # pyplot.title('performance in VIC1027')
# # pyplot.legend(loc='lower right',fontsize=8)
# # pyplot.show()
# x1=[0,0,0,0,0,0,1];
# y1=[0,0.328,0.328,0.52,0.61,0.61,1];
# x3=[0,0,0,0,0,0,1];
# y3=[0,0.16,0.16,0.19,0.19,0.19,1];
# x5=[0,0,0,0,1];
# y5=[0,0.28,0.28,0.28,1];
# # x1=[0,0,0,0,1];
# # y1=[0,0.21588089,0.41191067,0.4764268,1];
# # x3=[0,0,0,1];
# # y3=[0,0.10421836,0.12406948,1];
# # x5=[0,0,1];
# # y5=[0,0.21836228,1];
# x6=[0,1];
# y6=[0,1];
# pyplot.plot(y1,x1,'darkblue',alpha=0.9)
# pyplot.plot(y3,x3,'g',alpha=0.9)
# pyplot.plot(y5,x5,'k',alpha=0.8)
# pyplot.plot(x6,y6,'k--')
# pyplot.xlabel('FPR',fontsize=18)
# pyplot.ylabel('Sensitivity',fontsize=18)
# pyplot.text(0.05,0.95,'$P_{13}$',fontsize=30)
# locs, labels = pyplot.xticks(fontsize=18)
# locs, labels = pyplot.yticks(fontsize=18)
# pyplot.show()
# from sklearn.metrics import auc
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y1,x1)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y3,x3)))
#
#
#
#
# # # #### VIC0685
# # x1=[0,0,0];
# # y1=[0.31,0.31,0.31];
# # x3=[0,0,0,0,0];
# # y3=[0.09,0.09,0.09,0.09,0.09];
# # x5=[0,0,0];
# # y5=[0.33,0.33,0.33];
# # x6=[0,1];
# # y6=[0,1];
# # pyplot.scatter(y1[0],x1[0],facecolors='none', edgecolors='r',label='EEG auto, RRI var, RRI auto')
# # pyplot.scatter(y1[1],x1[1],facecolors='none', edgecolors='r')
# # pyplot.scatter(y1[2],x1[2],facecolors='none', edgecolors='r')
# # pyplot.scatter(y3[0],x3[0],marker="^",facecolors='none', edgecolors='g',label='EEG auto, RRI var, RRI auto, circadian')
# # pyplot.scatter(y3[1],x3[1],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y3[2],x3[2],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y3[3],x3[3],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y3[4],x3[4],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y5[0],x5[0],marker="3",c='k',label='circadian')
# # pyplot.scatter(y5[1],x5[1],marker="3",c='k',)
# # pyplot.scatter(y5[2],x5[2],marker="3",c='k',)
# # pyplot.plot(x6,y6,'k--')
# # pyplot.xlabel('FPR')
# # pyplot.ylabel('Sensitivity')
# # pyplot.title('performance in VIC0685')
# # pyplot.legend(loc='lower right',fontsize=8)
# # pyplot.show()
# x2=[0,0,0,0,1];
# y2=[0,0.309,0.309,0.309,1];
# x4=[0,0,0,0,1];
# y4=[0,0.086,0.086,0.086,1];
# x5=[0,0,0,0,1];
# y5=[0,0.309,0.309,0.309,1];
# # x2=[0,0,1];
# # y2=[0,0.17695473,1];
# # x4=[0,0,1];
# # y4=[0,0.05144,1];
# # x5=[0,0/2,1];
# # y5=[0,0.20987,1];
# x6=[0,1];
# y6=[0,1];
# pyplot.plot(y2,x2,'darkblue',alpha=0.9)
# pyplot.plot(y4,x4,'g',alpha=0.9)
# pyplot.plot(y5,x5,'k',alpha=0.8)
# pyplot.plot(x6,y6,'k--')
# pyplot.xlabel('FPR',fontsize=18)
# pyplot.ylabel('Sensitivity',fontsize=18)
# pyplot.text(0.05,0.95,'$P_{14}$',fontsize=30)
# locs, labels = pyplot.xticks(fontsize=18)
# locs, labels = pyplot.yticks(fontsize=18)
# pyplot.show()
# from sklearn.metrics import auc
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y2,x2)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y4,x4)))
#
#
#
#
# # #### TAS0102
# # x2=[2/3,2/3,2/3];
# # y2=[0.575,0.575,0.575];
# # x4=[1/3,1/3,1/3];
# # y4=[0.125,0.125,0.125];
# # x5=[1/3,1/3];
# # y5=[0.188,0.188];
# # # x2=[0.2,0.4];
# # # y2=[0.18580376,0.29853862];
# # # x4=[0.2];
# # # y4=[0.05845511];
# # # x5=[0.2];
# # # y5=[0.1565762];
# # x6=[0,1];
# # y6=[0,1];
# #
# # pyplot.scatter(y2[0],x2[0],facecolors='none', edgecolors='r',label='RRI var')
# # pyplot.scatter(y2[1],x2[1],facecolors='none', edgecolors='r')
# # pyplot.scatter(y2[2],x2[2],facecolors='none', edgecolors='r')
# # pyplot.scatter(y4[0],x4[0],marker="^",facecolors='none', edgecolors='g',label='RRI var, circadian')
# # pyplot.scatter(y4[1],x4[1],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y4[2],x4[2],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y5[0],x5[0],marker="3",c='k',label='circadian')
# # pyplot.scatter(y5[1],x5[1],marker="3",c='k')
# # pyplot.plot(x6,y6,'k--')
# # pyplot.xlabel('FPR')
# # pyplot.ylabel('Sensitivity')
# # pyplot.title('performance in TAS0102')
# # pyplot.legend(loc='lower right',fontsize=8)
# # pyplot.show()
# x2=[0,2/3,2/3,2/3,1];
# y2=[0,0.563,0.563,0.563,1];
# x4=[0,1/3,1/3,1/3,1];
# y4=[0,0.113,0.113,0.113,1];
# x5=[0,1/3,1/3,1];
# y5=[0,0.175,0.175,1];
# # x2=[0,0.2,0.4,1];
# # y2=[0,0.18580376,0.29853862,1];
# # x4=[0,0.2,1];
# # y4=[0,0.05845511,1];
# # x5=[0,0.2,1];
# # y5=[0,0.1565762,1];
# x6=[0,1];
# y6=[0,1];
# pyplot.plot(y2,x2,'darkblue',alpha=0.9)
# pyplot.plot(y4,x4,'g',alpha=0.9)
# pyplot.plot(y5,x5,'k',alpha=0.8)
# pyplot.plot(x6,y6,'k--')
# pyplot.xlabel('FPR',fontsize=18)
# pyplot.ylabel('Sensitivity',fontsize=18)
# pyplot.text(0.05,0.95,'$P_{15}$',fontsize=30)
# locs, labels = pyplot.xticks(fontsize=18)
# locs, labels = pyplot.yticks(fontsize=18)
# pyplot.show()
# from sklearn.metrics import auc
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y2,x2)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y4,x4)))
#
#
#
#
# # # #### VIC1006
# # x2=[2/3,2/3,2/3,2/3,1/3];
# # y2=[0.73,0.73,0.73,0.70,0.43];
# # x4=[2/3,2/3,2/3,1/3,1/3];
# # y4=[0.12,0.15,0.12,0.02,0.02];
# # x5=[5/6,5/6,5/6,1/3,1/3];
# # y5=[0.2,0.2,0.2,0.05,0.05];
# # # x2=[0.278,0.472,0.472];
# # # y2=[0.30864,0.463,0.61728];
# # # x4=[0.194,0.417,0.417];
# # # y4=[0.012345,0.0761,0.1049];
# # # x5=[0.222,0.583];
# # # y5=[0.03292,0.15432];
# # x6=[0,1];
# # y6=[0,1];
# # pyplot.scatter(y2[0],x2[0],facecolors='none', edgecolors='r',label='EEG var, EEG auto, RRI auto')
# # pyplot.scatter(y2[1],x2[1],facecolors='none', edgecolors='r')
# # pyplot.scatter(y2[2],x2[2],facecolors='none', edgecolors='r')
# # pyplot.scatter(y2[3],x2[3],facecolors='none', edgecolors='r')
# # pyplot.scatter(y2[4],x2[4],facecolors='none', edgecolors='r')
# # pyplot.scatter(y4[0],x4[0],marker="^",facecolors='none', edgecolors='g',label='EEG var, EEG auto, RRI auto, circadian')
# # pyplot.scatter(y4[1],x4[1],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y4[2],x4[2],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y4[3],x4[3],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y4[4],x4[4],marker="^",facecolors='none', edgecolors='g')
# # pyplot.scatter(y5[0],x5[0],marker="3",c='k',label='circadian')
# # pyplot.scatter(y5[1],x5[1],marker="3",c='k')
# # pyplot.scatter(y5[2],x5[2],marker="3",c='k')
# # pyplot.scatter(y5[3],x5[3],marker="3",c='k')
# # pyplot.scatter(y5[4],x5[4],marker="3",c='k')
# # pyplot.plot(x6,y6,'k--')
# # pyplot.xlabel('FPR')
# # pyplot.ylabel('Sensitivity')
# # pyplot.title('performance in VIC1006')
# # pyplot.legend(loc='lower right',fontsize=8)
# # pyplot.show()
# x2=[0,1/3,2/3,2/3,2/3,2/3,1];
# y2=[0,0.43,0.699,0.728,0.728,0.728,1];
# x4=[0,1/3,1/3,2/3,2/3,2/3,1];
# y4=[0,0.02,0.02,0.12,0.12,0.148,1];
# x5=[0,1/3,1/3,5/6,5/6,5/6,1];
# y5=[0,0.049,0.049,0.188,0.188,0.188,1];
# # x2=[0,0.278,0.472,0.472,1];
# # y2=[0,0.30864,0.463,0.61728,1];
# # x4=[0,0.194,0.417,0.417,1];
# # y4=[0,0.012345,0.0761,0.1049,1];
# # x5=[0,0.222,0.583,1];
# # y5=[0,0.03292,0.15432,1];
# x6=[0,1];
# y6=[0,1];
# pyplot.plot(y2,x2,'darkblue',alpha=0.9)
# pyplot.plot(y4,x4,'g',alpha=0.9)
# pyplot.plot(y5,x5,'k',alpha=0.8)
# pyplot.plot(x6,y6,'k--')
# pyplot.xlabel('FPR',fontsize=18)
# pyplot.ylabel('Sensitivity',fontsize=18)
# pyplot.text(0.05,0.95,'$P_{16}$',fontsize=30)
# locs, labels = pyplot.xticks(fontsize=18)
# locs, labels = pyplot.yticks(fontsize=18)
# pyplot.show()
# from sklearn.metrics import auc
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y5,x5)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y2,x2)))
# print('computed AUC using sklearn.metrics.auc: {}'.format(auc(y4,x4)))
