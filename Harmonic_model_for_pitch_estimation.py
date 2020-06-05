#!/usr/bin/env python
# coding: utf-8

### Imports ###
import sys
import math
import time
import soundfile as sf
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sg

### For reading the wav.-file ###
def read_wav_file(file_path):
    
    #Using soundfile library
    wavdata, Fs = sf.read(file_path)
    return wavdata, Fs

### For generating test wave ###
def generate_sinwave(f0):
    
    #Define the sampling rate as 4.8 kHz
    Fs = 48000
    wavdata = []
    for n in np.arange(Fs * 10):
        s = np.sin(2.0 * np.pi * f0 * n / Fs)
        wavdata.append(s)
    wavdata = np.array(wavdata)
    return wavdata, Fs

### For plotting the raw-wave data and harmonic model ###
def plot_wavs(wavdata, S, Fs, xmin, xmax):
    
    #Define the time axis
    time = np.arange(0, len(wavdata))/Fs
    
    #Plot the raw-wave data
    plt.rcParams["font.size"] = 16
    plt.figure(figsize=(16, 5))
    p1 = plt.plot(time, wavdata)
    p2 = plt.plot(time, S)
    plt.legend((p1[0], p2[0]), ("Wave data in a segment", "Harmonic model"),
               bbox_to_anchor=(1, 1), loc='upper right', borderaxespad=0)
    plt.title('Wave data vs Harmonic summation model')
    plt.xlabel('Time [sec]')
    plt.ylabel('Amplitude')
    plt.xlim(xmin, xmax)
    plt.savefig("./result/segmented_wave.png", dpi=200)

### For plotting STFT ###
def plot_STFT(wavdata, Fs, time_vector, f0_vector, f0_min, f0_max, win_size, win_overlap):
    
    #The window-size and the overlap must be integer
    win_size = round(win_size)
    win_overlap = round(win_overlap)
    
    #Use the library(scipy.signal)
    freq, time, A = sg.stft(wavdata, fs=Fs, window='hann', nperseg=win_size, noverlap=win_overlap)
    P = np.abs(A)[0 : round(win_size/2)]
    P = 20*np.log10(P)
    
    #Set up the range of frequency
    f0_min = round(f0_min * win_size / Fs)
    f0_max = round(f0_max * win_size / Fs)
    
    #Plot the STFT result
    plt.rcParams["font.size"] = 16
    plt.figure(figsize=(18, 5))
    plt.title('Spectrogram and F0 estimates')
    plt.xlabel('Time [sec]')
    plt.ylabel('Frequency [Hz]')
    plt.pcolormesh(time, freq[f0_min : f0_max], P[f0_min : f0_max, :], cmap = 'hot')
    plt.colorbar(orientation='vertical').set_label('Power [dB]')
    plt.scatter(time_vector, f0_vector, c='b', s=20) #Add the F0 plot to spectrogram
    plt.savefig("./result/spectrogram.png", dpi=200)

### For estimating the pitch based on harmonic summation model ###
def get_fundamental_freq(wavdata, Fs, f0_min, f0_max, resolution, L_max):
    
    #Parameter check
    N = len(wavdata)
    if f0_min < Fs / N:
        print('The f_min should be more than the frequency resolution(Fs / N).')
        sys.exit()
    if f0_max > Fs:
        print('The f_max should be less than sampling rate Fs.')
        sys.exit()
    
    #Define the range of frequency
    f_list = np.arange(f0_min, f0_max, resolution)
    
    #Generate the z-vector and harmonic summation list
    z = np.arange(0, N)
    Harmonic_sum = []
    Z_list = []
    
    #Repeat for f one by one
    for i in range(len(f_list)):
        
        #Construct the zc and zs matrix
        zc = []
        zs = []
        for L in range(1, L_max + 1):
            zc.append(np.cos(2*np.pi*f_list[i]*L*z/Fs))
            zs.append(np.sin(2*np.pi*f_list[i]*L*z/Fs))
        
        #Stack the zc and zs to construct Z-matrix
        zc = np.array(zc).T
        zs = np.array(zs).T
        Z = np.concatenate([zc, zs], axis=1) #row=N,column=2L
        
        #Compute the harmonic summation
        Zx = Z.T @ wavdata
        Harmonic_sum.append(Zx.T @ Zx)
        Z_list.append(Z)
    
    #Search for argument max (maximize the harmonic summation value)
    Harmonic_sum = np.array(Harmonic_sum)
    arg_i = np.argmax(Harmonic_sum)
    
    #Fundamental frequency
    arg_f = f_list[arg_i]
    #Maximum value
    Harmonic_max = Harmonic_sum[arg_i]
    #Z(0) N x 2L matrix
    Z0 = Z_list[arg_i]
    
    return arg_f, Harmonic_max, Z0

### Main ###
if __name__ == "__main__":
    
    #Set up
    segTime = 0.064             #The temporal length of a segment (seconds) [Default]0.064
    overlap_ratio = 0.5         #The overlap ratio of the segments (between 0 and 1) [Default]0.5
    f0_min, f0_max = 100, 800   #The range of fundamental frequency (Hz) [Default]100, 800
    f_delta = 1                 #The resolution of frequency (Hz) [Default]1
    L_max = 10                  #The degree of harmonic summation (integer) [Default]10
    
    #Read a sound data
    #[wavdata, Fs] = generate_sinwave(f0=440) #for a quick test
    [wavdata, Fs] = read_wav_file('./data/speech.wav')
    
    #Crop the signal for dropping its edge
    #wavdata = wavdata[round(0.5*Fs) : len(wavdata)-round(0.5*Fs)]
    
    #Calculate the parameter
    wavLen = len(wavdata)
    segLen = round(segTime * Fs)
    N_seg = math.floor((wavLen - overlap_ratio * segLen) / ((1-overlap_ratio) * segLen))
    stride = math.floor((1-overlap_ratio) * segLen)
    time_vector = (segLen/2 + stride * np.arange(0, N_seg)) / Fs
    
    #Do the analysis
    f0_vector = np.zeros((N_seg)) #F0 is defined as a vector
    for i in range(N_seg):
        #Measure time
        start = time.time()
        
        #Trim the wave-data into each segment
        crop_wav = wavdata[round(i*stride) : round(i*stride + segLen)] #Crop the wave-data for each segment
        [arg_f, Harmonic_max, Z0] = get_fundamental_freq(crop_wav, Fs, f0_min, f0_max, f_delta, L_max) #Call my function
        f0_vector[i] = arg_f #Divide the sampling rate (Fs) by the tau to get fundamental frequency
        
        #Report time
        print('Time={:.2f}sec: F0 estimates={:.1f}Hz, Process_time={:.1f}sec'.format((i*stride+segLen/2)/Fs, f0_vector[i], time.time() - start))
        
        #Preserve the second segment for plotting
        if i == 1:
            X = crop_wav
            Z = Z0
    
    #Construct the Harmonic model S from Z
    S1 = Z.T @ X
    S2 = np.linalg.inv(Z.T @ Z)
    S = Z @ S2 @ S1
    
    #Plot the results
    plot_wavs(X, S, Fs, 0, segTime)
    plot_STFT(wavdata, Fs, time_vector, f0_vector, f0_min, f0_max, segLen, segLen*overlap_ratio)
    print('Averaged F0 estimates: {:.1f}Hz'.format(np.average(f0_vector)))