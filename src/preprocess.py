import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd

import librosa
import librosa.display
import pretty_midi
import math

def get_CQT(file_path, sr=22050, hop_length=512, n_bins=84, bins_per_octave=12,
            verbose=False, plot_spectogram_=True):
    '''
    ----Parameter----
    file_path (string): path to audio file
    verbose (boolean): determine wordiness

    ----Return-----
    CQT np.ndarray[shape=(n_bins, t)]: matrix with the size of the number of windows
  that can fit in the data provided (signal), and the number of bins specified.

    '''

    # print length of audio file in seconds
    if verbose:
        print("-" * 10, file_path, "-" * 10)
        print("{:.3f} seconds long".format(librosa.get_duration(filename=file_path)))

    # load audiofile; sr=None preserves native sampling rate
    signal, sr = librosa.load(file_path, sr=None)
    if verbose:
        print("sr={} samples/sec".format(sr))
        print("n_samples: ", signal.size)

    # compute constant-Q transform (CQT)
    CQT = librosa.cqt(signal,
                      sr=sr,
                      hop_length=hop_length,
                      n_bins=n_bins,
                      bins_per_octave=bins_per_octave
                      )

    # get magnitude of CQT; CQT=a+bi so CQT_mag = (a^2 + b^2)^.5
    CQT_mag = np.abs(CQT)  # CQT_mag, phase = librosa.magphase( CQT )
    if verbose:
        print("n_bins={}, t={}".format(CQT.shape[0], CQT.shape[1]))

    # convert amplitude to db, a logarithmic scale
    # https://stackoverflow.com/questions/63347977/what-is-the-conceptual-purpose-of-librosa-amplitude-to-db
    CQT_mag_db = librosa.amplitude_to_db(CQT_mag, ref=np.max)

    CQT = CQT_mag_db

    if plot_spectogram_:
        plot_spectogram(CQT, sr=sr, hop_length=hop_length, bins_per_octave=bins_per_octave)

    return CQT

def plot_spectogram(CQT, sr=22050, hop_length=512, bins_per_octave=12):

  librosa.display.specshow(CQT,
                           sr=sr,
                           hop_length=hop_length,
                           bins_per_octave=bins_per_octave,
                           x_axis='time',
                           y_axis='cqt_note')

  plt.title('Constant-Q power spectrum')
  plt.colorbar(format="%+2.f dB")
  plt.show()

#sr (sampling rate), stride = # of samples between each window
def one_hot_encode(midi_file, sr=22050, stride=1):
    # MIDI files represent a note's pitch with an integer value between 0 and 127
    # (https://www.inspiredacoustics.com/en/MIDI_note_numbers_and_center_frequencies)
    # But on an 88 key piano these pitches only range from 21-108 inclusive
    # (http://newt.phys.unsw.edu.au/jw/notes.html)
    # so at every sample we want a vector with 88 elements. Moreover, if note.pitch==21
    # we want this to be the 0-th element in our vector 88 so we subtract 21 from every
    # pitch
    # Reference: https://raphaellederman.github.io/articles/musicgeneration/#collecting-data

    offset = 21
    num_keys = 88  # number of keys on the piano

    # get list of note objects from MIDI file
    midi_data = pretty_midi.PrettyMIDI(midi_file)
    notes = [note for note in midi_data.instruments[0].notes]

    # time in seconds where this MIDI file ends
    song_length = midi_data.get_end_time()

    # number of samples in the entire song
    n_samples = math.ceil(song_length * sr / stride)

    # initialize one-hot-encoded array
    X_ohe = np.zeros((num_keys, n_samples))

    start_windows = []
    end_windows = []
    norm_pitches = []
    totNotes = 0  # what is this?

    for note in notes:
        norm_pitches.append(note.pitch - 20)  # should this be note.pitch-offset?
        start_windows.append(int(round(note.start * sr / stride)))  # why are we rounding?
        end_windows.append(int(round(note.end * sr / stride)))
        totNotes = totNotes + 1

    for i in range(totNotes):  # 0 to nn-1
        for col in range(start_windows[i] - 1, end_windows[i]):
            X_ohe[norm_pitches[i], col] = 1

    return X_ohe

def plot_piano_roll(pm, start_pitch, end_pitch, fs=100):
    # Use librosa's specshow function for displaying the piano roll
    librosa.display.specshow(pm.get_piano_roll(fs)[start_pitch:end_pitch],
                             hop_length=1, sr=fs, x_axis='time', y_axis='cqt_note',
                             fmin=pretty_midi.note_number_to_hz(start_pitch))

if __name__ == '__main__':
    # file_path = librosa.ex("trumpet")
    # get_CQT(file_path)

    test_midi = '../data/maestro-v2.0.0/2004/MIDI-Unprocessed_SMF_02_R1_2004_01-05_ORIG_MID--AUDIO_02_R1_2004_05_Track05_wav.midi'
    # print(sum(one_hot_encode(test_midi)))
    x = one_hot_encode(test_midi, 4, 1)
    plt.figure(figsize=(16, 6))
    plt.imshow(x, aspect='auto', origin='lower')
    plt.set_cmap('gray_r')
    plt.grid(True)

    # plt.figure(figsize=(16, 6))
    # plot_piano_roll(midi_d, 21, 108)