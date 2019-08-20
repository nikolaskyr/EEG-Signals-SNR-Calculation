# -*- coding: utf-8 -*-
"""
@author: Nikolas Kyriacou
MSc Project on the Investigation of the SNR in EEG Signals
"""
from IIR2Filter import IIR2Filter # This library is obtained from https://github.com/poganyg/IIR-filter/blob/master/src/IIR2Filter.py
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
import math as math
from scipy.interpolate import interp1d
import scipy.stats as stats


def Noise_var_all(start_task,end_task,start_noise,end_noise):
    task_noiseVar=np.zeros(end_task) 
    for taskN in range(start_task,end_task):
        eeg_raw=np.loadtxt('Experiments/Task'+str(taskN+1)+'.csv')
        FilterMains = IIR2Filter(10,[45,55],'bandstop',design='cheby1',rp=2,fs = fsampl)
        eeg_cl_raw = np.zeros(len(eeg_raw)) #Initialize array to store the filtered samples
        for i in range(len(eeg_raw)):
            eeg_cl_raw[i] = FilterMains.filter(eeg_raw[i])
        eeg_cl_alpha = np.zeros(len(eeg_cl_raw))  #Initialize array to store the filtered samples
        FilterAlpha = IIR2Filter(10,[7,13],'bandpass',design='butter',rp=2,fs = fsampl)
        for i in range(len(eeg_cl_alpha)):
            eeg_cl_alpha[i] = FilterAlpha.filter(eeg_cl_raw[i])
        eeg_noise=eeg_cl_alpha[start_noise:end_noise]
        Nvar=np.var(eeg_noise)
        task_noiseVar[taskN]=Nvar
        print("Task", taskN+1, "Noise Variance = ", Nvar,"\n")
    return task_noiseVar

def Noise_var_min(Var_all):
    Nvar_min = Var_all[0]
    for taskN in range(len(Var_all)):
        if Var_all[taskN] < Nvar_min:
            Nvar_min = Var_all[taskN]
    print("Min Variance: ",Nvar_min,"\n")
    return Nvar_min

def Noise_var_max(Var_all):
    Nvar_max = Var_all[0]
    for taskN in range(len(Var_all)):
        if Var_all[taskN] > Nvar_max:
            Nvar_max = Var_all[taskN]
    print("Max Variance: ",Nvar_max,"\n")
    return Nvar_max

def Calc_SNR(noise_var,start_task,end_task,start_signal,end_signal,segment_size,segment_shift):
    Seg_nums = int((end_signal - start_signal-segment_size)/segment_shift) +1
    task_snr=np.empty((end_task,Seg_nums+1)) 
    for taskN in range(start_task,end_task):
        eeg_raw=np.loadtxt('Experiments/Task'+str(taskN+1)+'.csv')
        FilterMains = IIR2Filter(10,[45,55],'bandstop',design='cheby1',rp=2,fs = fsampl)
        eeg_cl_raw = np.zeros(len(eeg_raw)) #Initialize array to store the filtered samples
        for i in range(len(eeg_raw)):
            eeg_cl_raw[i] = FilterMains.filter(eeg_raw[i])
        eeg_cl_alpha = np.zeros(len(eeg_cl_raw))  #Initialize array to store the filtered samples
        FilterAlpha = IIR2Filter(10,[7,13],'bandpass',design='butter',rp=2,fs = fsampl)
        for i in range(len(eeg_cl_alpha)):
            eeg_cl_alpha[i] = FilterAlpha.filter(eeg_cl_raw[i])
        # The following 3 lines are used when the SNR is computed using the noise for each experiment    
        #eeg_noise= eeg_cl_alpha[1500:2500]
        #noise_var=np.var(eeg_noise)
        #print("SNR var: ",noise_var,"\n")
        eeg_signal = eeg_cl_alpha[start_signal:end_signal] 
        all_var = np.var(eeg_signal)
        SNR = 10 * math.log10(all_var/noise_var)
        task_snr[taskN,0]=SNR
        for segmnt in range(0,Seg_nums):
            seg_start = start_signal + segmnt*segment_size
            seg_end = seg_start + segment_size
            eeg_signal = eeg_cl_alpha[seg_start:seg_end] 
            seg_var = np.var(eeg_signal)   #eeg_cl_alpha[seg_start:seg_end])
            SNR = 10 * math.log10(seg_var/noise_var)
            task_snr[taskN,segmnt+1]= SNR 
    return task_snr

fsampl = 1000  #Sampling frequency
Nvar_all =  Noise_var_all(0,7,1500,2500)    
Nvar_max = Noise_var_max(Nvar_all) 
Nvar_min = Noise_var_min(Nvar_all)       
rho = np.sqrt(Nvar_max /Nvar_min)
print("rho: ",rho,"\n")
SNRwall = 10 * math.log10(rho - 1/rho)
print("SNR Wall: ",SNRwall,"\n")
NoiseVariance = Nvar_min * rho 
print("Noise Variance: ",NoiseVariance,"\n")
NM = np.mean(Nvar_all)
print("Noise Mean: ",NM,"\n")
NMn = np.median(Nvar_all)
print("Noise Median: ",NMn,"\n")
noise_var = NoiseVariance
start_task = 0
end_task = 7
start_signal = 4000
end_signal = 8000
segment_size = 1000
segment_shift= 500
SNR_All = Calc_SNR(noise_var,start_task,end_task,start_signal,end_signal,segment_size,segment_shift)
print("SNR: ",SNR_All,"\n")

