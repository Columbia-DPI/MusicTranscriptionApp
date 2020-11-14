# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 22:14:01 2020

@author: Owner
"""
from __future__ import division

from collections import defaultdict
import sys, os
import argparse

import madmom
import numpy as np
import pandas as pd
import pretty_midi
import librosa
import h5py
import math

from load_data import load_config

import numpy as np




def wav2inputnp(audio_fn,spec_type='cqt',bin_multiple=3):
    print("wav2inputnp")
    bins_per_octave = 12 * bin_multiple #should be a multiple of 12
    n_bins = (max_midi - min_midi + 1) * bin_multiple

    #down-sample,mono-channel
    y,_ = librosa.load(audio_fn,sr)
    S = librosa.cqt(y,fmin=librosa.midi_to_hz(min_midi), sr=sr, hop_length=hop_length,
                      bins_per_octave=bins_per_octave, n_bins=n_bins)
    S = S.T

    #TODO: LogScaleSpectrogram?
    '''
    if spec_type == 'cqt':
        #down-sample,mono-channel
        y,_ = librosa.load(audio_fn,sr)
        S = librosa.cqt(y,fmin=librosa.midi_to_hz(min_midi), sr=sr, hop_length=hop_length,
                          bins_per_octave=bins_per_octave, n_bins=n_bins)
        S = S.T
    else:
        #down-sample,mono-channel
        y = madmom.audio.signal.Signal(audio_fn, sample_rate=sr, num_channels=1)
        S = madmom.audio.spectrogram.LogarithmicFilteredSpectrogram(y,fmin=librosa.midi_to_hz(min_midi),
                                            hop_size=hop_length, num_bands=bins_per_octave, fft_size=4096)'''

    #S = librosa.amplitude_to_db(S)
    S = np.abs(S)

    minDB = np.min(S)

    print (np.min(S), np.max(S), np.mean(S))

    S = np.pad(S, ((window_size//2,window_size//2),(0,0)), 'constant', constant_values=minDB)



    windows = []

    # IMPORTANT NOTE:
    # Since we pad the the spectrogram frame,
    # the onset frames are actually `offset` frames.
    # To obtain a window of the center frame at each true index, we take a slice from i to i+window_size
    # starting at frame 0 of the padded spectrogram
    for i in range(S.shape[0]-window_size+1):
        w = S[i:i+window_size,:]
        windows.append(w)


    #print inputs
    x = np.array(windows)
    return x