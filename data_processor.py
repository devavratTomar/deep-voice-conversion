# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 13:26:59 2019

@author: devav
"""

import os
import random
import numpy as np
import librosa
import math

TIMIT_PHONE_DICTIONARY = {
        'iy':0, 'ch':21, 'en':42,
        'ih':1,	'b':22,	'eng':43,
        'eh':2,	'd':23, 'l':44,
        'ey':3,	'g':24, 'r':45,
        'ae':4,	'p':25, 'w':46,
        'aa':5,	't':26, 'y':47,
        'aw':6,	'k':27, 'hh':48,
        'ay':7,	'dx':28, 'hv':49,
        'ah':8,	's':29,	 'el':50,
        'ao':9,	'sh':30, 'bcl':51,
        'oy':10, 'z':31, 'dcl':52,
        'ow':11, 'zh':32, 'gcl':53,	
        'uh':12, 'f':33, 'pcl':54,
        'uw':13, 'th':34, 'tcl':55,
        'ux':14, 'v':35, 'kcl':56,
        'er':15, 'dh':36, 'q':57,
        'ax':16, 'm':37, 'pau':58,
        'ix':17, 'n':38, 'epi':59,
        'axr':18, 'ng':39, 'h#':60,
        'ax-h':19, 'em':40,
        'jh':20, 'nx':41
        }

class DataProcessor_TIMIT(object):
    """
    This class processes the TIMIT DATASET for Phone recognition. There are 61 phonemes.
    https://www.intechopen.com/books/speech-technologies/phoneme-recognition-on-the-timit-database.
    """

    def __init__(self, path='./Dataset/TIMIT/TRAIN/', sampling_rate=16000):
        """
        Initialize the TIMIT folder path
        """
        if not os.path.isdir(path):
            raise Exception("The directory name provided '{}' is incorrect.".format(path))
        
        self.directory = os.path.abspath(path)
        self.dialects = [dialect_name for dialect_name in os.listdir(self.directory)]
        self.samping_rate = sampling_rate
    
    
    def __read_phones(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            
        data = []        
        for line in lines:
            line = line.rstrip('\n').split()
            data.append(line)
        return np.array(data)
    
    def __read_audio(self, filename):
        aud, sr = librosa.load(filename, sr=self.samping_rate)
        return aud
    
    def create_feature_label_ts(self, label_seq, audio_seq, window_size=20):
        """
        label_seq should be 2D array where 1st column represent the start of phone, 2nd column represent end of phone
        and 3rd column represent the phone label.
        
        :param label_seq: The ground truth of phone sequence
        :param audio_seq: The speech wave form
        :param window_size: size of the window on which features should be extracted (milliseconds)
        """
        
        label_seq_start = label_seq[:, 0].astype(int)
        label_codes = label_seq[:, 2]
        
        #clip audio to start and end mark
        audio_seq = audio_seq[label_seq_start[0]: label_seq[:,1].astype(int)[-1]]
        label_seq_start = label_seq_start - label_seq_start[0]
        
        window_len = self.samping_rate*window_size//1000
        hop_len = window_len//4
        
        stft_features = librosa.core.stft(audio_seq, n_fft=window_len, win_length=window_len)
        
        #stft_all = np.concatenate([np.real(stft_features), np.imag(stft_features)], axis=0)
        stft_all = np.log(np.abs(stft_features))
        
        labels = np.zeros(stft_all.shape[1])
        
        for t in range(stft_all.shape[1]):
            arg_label = np.where(label_seq_start <= t*hop_len)[0][-1]
            labels[t] = TIMIT_PHONE_DICTIONARY[label_codes[arg_label]]
            
        return stft_all, labels
     
    def all_speech_getter(self, n_epochs=10, max_time_steps=32):
        """
        Iterable over all speech sequences.
        """
        random.shuffle(self.dialects)
        for epoch in range(n_epochs):
            for dialect in self.dialects:
                speaker_path_base = os.path.join(self.directory, dialect)
                speakers = [speaker for speaker in os.listdir(speaker_path_base)]
                random.shuffle(speakers)
                
                for speaker in speakers:
                    speaker_path = os.path.join(speaker_path_base, speaker)
                    speech_samples = [f[:-4] for f in os.listdir(speaker_path) if f.endswith('WAV')]
                    random.shuffle(speech_samples)
                    
                    for speech_sample in speech_samples:
                        speech_sample_path = os.path.join(speaker_path, speech_sample)
                        speech_features, labels = self.create_feature_label_ts(self.__read_phones(speech_sample_path + '.PHN'),
                                                                               self.__read_audio(speech_sample_path + '.WAV'))
                        
                        n_buffers = math.ceil(speech_features.shape[1]/max_time_steps)
                        buffers_container = []
                        
                        for buffer in range(n_buffers):
                            buffer_start = buffer*max_time_steps
                            buffer_end = min((buffer+1)*max_time_steps, speech_features.shape[1])
                            
                            buffers_container.append([speech_features[:, buffer_start:buffer_end].T, labels[buffer_start:buffer_end], buffer_end - buffer_start])
                        
                        yield buffers_container
