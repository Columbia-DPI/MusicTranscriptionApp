import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd

import librosa
import librosa.display

def get_spectogram(file_path, hop_length=512, sr=22050, n_fft=20148):

  #load audiofile
  signal, sr = librosa.load(file_path, sr=sr)

  #----waveform----
  librosa.display.waveplot(signal, sr=sr)
  plt.xlabel("Time (s)")
  plt.ylabel("Amplitude (m)")
  plt.title("Waveform")
  plt.show()

  #-----spectrum (fft)-----
  fourier_transform = np.abs(librosa.stft(signal, hop_length=n_fft+1))
  plt.plot(fourier_transform)
  plt.title("FFT: Spectrum")
  plt.xlabel("Frequency (hz)")
  plt.ylabel("Amplitude (m)")
  plt.show()
  
  #-----spectogram (stft)-------

  #compute linear stft
  stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)

  #remove complex values
  spectogram = np.abs(stft)

  #plot linear spectogram
  librosa.display.specshow(spectogram, sr=sr, hop_length=hop_length)
  plt.colorbar() #amplitude indicated with color
  plt.xlabel("Time (s)")
  plt.ylabel("Frequency (hz)")
  plt.title("STFT: Spectogram linear")
  plt.show()


  #compute log amplitude spectogram
  log_spectogram = librosa.amplitude_to_db(spectogram)

  #plot log amplitude spectogram
  librosa.display.specshow(log_spectogram, sr=sr, hop_length=hop_length)
  plt.colorbar() #amplitude indicated with color
  plt.xlabel("Time (s)")
  plt.ylabel("Frequency (hz)")
  plt.title("STFT: Spectogram (log amplitude)")
  plt.show()

  #plot log amplitude and log frequency spectogram
  librosa.display.specshow(log_spectogram, sr=sr, x_axis='time', y_axis="log")
  plt.colorbar(format='%+2.0f dB')
  plt.xlabel("Time (s)")
  plt.ylabel("Frequency (hz)")
  plt.title('STFT: Spectogram (log amplitude, frequency)')
  plt.show()

  #compute a mel spectogram
  n_mels = 128
  mel = librosa.feature.melspectrogram(signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
  mel_DB = librosa.power_to_db(mel, ref=np.max)
  librosa.display.specshow(mel_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
  plt.colorbar(format='%+2.0f dB')
  plt.title("Mel Spectogram")
  plt.show()


  # compute constant-Q transform (CQT)

  # CQT = alternative to FFT that's better suited for musical data
  #how do we do STFT with CQT??
  CQT = np.abs(librosa.cqt(signal, sr=sr, hop_length=hop_length))

  fig, ax = plt.subplots()
  img = librosa.display.specshow(librosa.amplitude_to_db(CQT, ref=np.max),
                               sr=sr, x_axis='time', y_axis='cqt_note', ax=ax)
  ax.set_title('Constant-Q power spectrum')
  #fig.colorbar(img, ax=ax, format="%+2.0f dB")
  plt.show()


  #compute MFCCs

  # n_ffc = number of MFCC coifficents 
  MFCCs = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

  #plot it
  librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
  plt.colorbar() #amplitude indicated with color
  plt.xlabel("Time (s)")
  plt.ylabel("MFCC")
  plt.title("MFCCs")
  plt.show()

file_path = librosa.util.example_audio_file()

#play audio
ipd.Audio(file_path)

#create visuals
get_spectogram(file_path)

def f(file_path, sampling_rate=512, bins=36):
  """preprocess according to http://cs229.stanford.edu/proj2017/final-reports/5242716.pdf"""

  #compute signal
  signal, sr = librosa.load(file_path, sr=sampling_rate)

  #compute CQT
  CQT = np.abs(librosa.cqt(signal, sr=sr))

  #remove complex values
  CQT_dB = librosa.amplitude_to_db(CQT, ref=np.max)

  #plot CQT
  fig, ax = plt.subplots()
  img = librosa.display.specshow(CQT_dB,
                               sr=sr, 
                               n_bins=bins,
                               x_axis='time', 
                               y_axis='cqt_note', 
                               ax=ax)
  ax.set_title('Constant-Q power spectrum')
  #fig.colorbar(img, ax=ax, format="%+2.0f dB")
  plt.show()

f(file_path)
