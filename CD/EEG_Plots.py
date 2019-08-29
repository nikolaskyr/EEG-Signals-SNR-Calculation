# -*- coding: utf-8 -*-
"""
@author: Nikolas Kyriacou
MSc Project on the Investigation of the SNR in EEG Signals
"""
from IIR2Filter import IIR2Filter # This library is obtained from https://github.com/poganyg/IIR-filter/blob/master/src/IIR2Filter.py
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
#import scipy.io.wavfile as wavfile
import math as math
from scipy.interpolate import interp1d
import scipy.stats as stats


fsampl = 1000 # Sampling frequency
t_step =4000 # time step or segment size in ms 
t_start = 4000 # start time of segment in ms
A_gain = 50 # Gain of the EEG Amplifier
mVolts=1000 #1V = 1000mV
microV=1000000 #1V = 1000000uV

"""
Specify EEG file name
"""
data_in=np.loadtxt('Experiments/Task7.csv')
data=data_in*microV/A_gain  #*mVolts/A_gain
s_size=len(data)   # number of signal samples

"""
Plot the unfiltered EEG signal and its spectrum
"""
ks = np.arange(s_size)*1000/fsampl #Array for signal x-axis 
plt.figure(1)
plt.title('Raw EEG Signal')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude (μV)')
plt.grid(which='both', axis='both')
plt.plot(ks,data)
plt.show()

"""
Plot the spectrum of the unfiltered EEG signal and its spectrum
"""

mask2=np.fft.fft(data)
plt.figure(2)
plt.title('Raw EEG Signal Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Signal Power')
plt.grid(which='both', axis='both')
kf = np.arange(s_size)*fsampl/s_size  #Array for spectrum x-axis 
plt.plot(kf,abs(mask2)/s_size) 
plt.show()  

'''
Filter the signal with a 50Hz notch filter
'''
f_signal = np.zeros(s_size) #Initialize array to store the filtered samples
FilterMains = IIR2Filter(10,[45,55],'bandstop',design='cheby1',rp=2,fs = fsampl)      
for i in range(len(data)):
    f_signal[i] = FilterMains.filter(data[i])

"""
Plot the 50Hz filtered EEG signal and its spectrum
"""
plt.figure(3)
plt.title('50Hz Filtered EEG Signal')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude  (μV)')
plt.grid(which='both', axis='both')
plt.plot(ks,f_signal)
plt.show()  
mask2=np.fft.fft(f_signal)
plt.figure(4)
plt.title('50Hz Filtered EEG Signal Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Signal Power')
plt.grid(which='both', axis='both')
kf = np.arange(s_size)*fsampl/s_size
plt.plot(kf,abs(mask2)/s_size)   
plt.show()  
"""
Plot part of the filtered EEG signal and its spectrum
"""
psize = int(t_step*fsampl/1000)  #/fsampl   # size of partial buffer
pstart = int(t_start*fsampl/1000) #/fsampl  # begining of partial buffer
kp = np.arange(t_start,t_start+t_step,1000/fsampl)   #pstart
seg=np.zeros(psize)
seg[0:psize]=f_signal[pstart:psize+pstart]
plt.figure(5)
plt.title('50Hz Filtered Partial EEG Signal (from ' + str(t_start)+'ms to '+str(t_start+t_step) +'ms)')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude  (μV)')
plt.grid(which='both', axis='both')
plt.plot(kp,seg)
plt.show()  

kfp = np.arange(0,fsampl,fsampl/psize)
mask2=np.fft.fft(seg)
plt.figure(6)
plt.title('50Hz Filtered Partial EEG Signal Spectrum (from ' + str(t_start)+'ms to '+str(t_start+t_step) +'ms)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Signal Power')
plt.grid(which='both', axis='both')
mask2[0:int(5*psize/fsampl)]=0 #Set 0Hz and 1Hz to 0
plot_fmax=40  #maximum plot frequency
#ppsize=psize-300
plt.plot(kfp[0:int(plot_fmax*psize/fsampl)],abs((mask2[0:int(plot_fmax*psize/fsampl)]/psize)))   #display up to 40Hz
plt.show()

"""
Bandpass filter the EEG signal 8Hz to 12Hz
"""
f_signalbp = np.zeros(s_size) #Initialize array to store the filtered samples
#FilterAlpha = IIR2Filter(10,[7,13],'bandpass',design='cheby1',rp=2,fs = fsampl)     
FilterAlpha = IIR2Filter(10,[8,12],'bandpass',design='butter',rp=2,fs = fsampl)      
for i in range(len(data)):
    f_signalbp[i] = FilterAlpha.filter(f_signal[i])


"""
Plot the Bandpass filtered EEG signal and its spectrum
"""
plt.figure(7)
plt.title('Bandpass Filtered EEG Signal')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude  (μV)')
plt.grid(which='both', axis='both')
plt.plot(ks,f_signalbp)
plt.show()  
mask2=np.fft.fft(f_signalbp)
plt.figure(8)
plt.title('Bandpass Filtered EEG Signal Spectrum')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Signal Power')
plt.grid(which='both', axis='both')
kf = np.arange(s_size)*fsampl/s_size
plt.plot(kf,abs(mask2))   
plt.show()  

"""
Plot a segment of the bandpass filtered EEG signal and its spectrum
To set the plot range change t_step and t_start at the beggining of the program
"""

psize = int(t_step*fsampl/1000)  #/fsampl   # size of partial buffer
pstart = int(t_start*fsampl/1000) #/fsampl  # begining of partial buffer
kp = np.arange(t_start,t_start+t_step,1000/fsampl)   #pstart
bpseg=np.zeros(psize)
bpseg[0:psize]=f_signalbp[pstart:psize+pstart]
plt.figure(9)
plt.title('Bandpass Filtered Partial EEG Signal (from ' + str(t_start)+'ms to '+str(t_start+t_step) +'ms)')
plt.xlabel('Time (ms)')
plt.ylabel('Amplitude  (μV)')
plt.grid(which='both', axis='both')
plt.plot(kp,bpseg)
plt.show()  

kfp = np.arange(0,fsampl,fsampl/psize)
mask3=np.fft.fft(bpseg)
plt.figure(10)
plt.title('Bandpass Filtered Partial EEG Signal Spectrum (from ' + str(t_start)+'ms to '+str(t_start+t_step) +'ms)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Signal Power')
plt.grid(which='both', axis='both')
mask3[0:int(5*psize/fsampl)]=0 #Set 0Hz and 1Hz to 0
plot_fmax=40  #maximum plot frequency
plt.plot(kfp[0:int(plot_fmax*psize/fsampl)],abs((mask3[0:int(plot_fmax*psize/fsampl)]/psize)))   #display up to 40Hz
plt.show()


