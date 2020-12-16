import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

import librosa
import librosa.display

def getb64(plt):
  buf = BytesIO()
  plt.savefig(buf, format="png")
  data = base64.b64encode(buf.getbuffer()).decode("ascii")
  return f"<img src='data:image/png;base64,{data}'/>"

def get_spectogram(file_path, hop_length=512, sr=22050, n_fft=20148):
  res = {}

  #load audiofile
  signal, sr = librosa.load(file_path, sr=sr)

  #----waveform----
  librosa.display.waveplot(signal, sr=sr)
  plt.xlabel("Time (s)")
  plt.ylabel("Amplitude (m)")
  plt.title("Waveform")
  res['waveform'] = getb64(plt)
  plt.clf()

  #-----spectrum (fft)-----
  fourier_transform = np.abs(librosa.stft(signal, hop_length=n_fft+1))
  plt.plot(fourier_transform)
  plt.title("FFT: Spectrum")
  plt.xlabel("Frequency (hz)")
  plt.ylabel("Amplitude (m)")
  res['spectrum'] = getb64(plt)
  plt.clf()

  #-----spectogram (stft)-------

  #compute linear stft
  stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)

  #remove complex values
  spectogram = np.abs(stft)
  #compute log amplitude spectogram
  log_spectogram = librosa.amplitude_to_db(spectogram)
  #plot log amplitude and log frequency spectogram
  librosa.display.specshow(log_spectogram, sr=sr, x_axis='time', y_axis="log")
  plt.colorbar(format='%+2.0f dB')
  plt.xlabel("Time (s)")
  plt.ylabel("Frequency (hz)")
  plt.title('STFT: Spectogram (log amplitude, frequency)')
  res['spectogram'] = getb64(plt)
  plt.clf()

  # compute constant-Q transform (CQT)

  # CQT = alternative to FFT that's better suited for musical data
  #how do we do STFT with CQT??
  CQT = np.abs(librosa.cqt(signal, sr=sr, hop_length=hop_length))

  fig, ax = plt.subplots()
  img = librosa.display.specshow(librosa.amplitude_to_db(CQT, ref=np.max),
                               sr=sr, x_axis='time', y_axis='cqt_note', ax=ax)
  ax.set_title('Constant-Q power spectrum')
  #fig.colorbar(img, ax=ax, format="%+2.0f dB")
  res['cqt'] = getb64(plt)
  plt.clf()


  #compute MFCCs

  # n_ffc = number of MFCC coifficents 
  MFCCs = librosa.feature.mfcc(signal, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

  #plot it
  librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
  plt.colorbar() #amplitude indicated with color
  plt.xlabel("Time (s)")
  plt.ylabel("MFCC")
  plt.title("MFCCs")
  res['mfcc'] = getb64(plt)
  plt.clf()

  return res
