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
import lpc

from config import CONFIG, CONFIG_EMBED

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

    def __init__(self, path='./Dataset/TIMIT/TRAIN/', path_lpc='./Dataset/TIMIT_FEATURES/TRAIN/', sampling_rate=16000):
        """
        Initialize the TIMIT folder path
        """
        if not os.path.isdir(path):
            raise Exception("The directory name provided '{}' is incorrect.".format(path))
        
        self.directory = os.path.abspath(path)
        self.directory_features = os.path.abspath(path_lpc)
        
        self.dialects = [dialect_name for dialect_name in os.listdir(self.directory)]
        self.samping_rate = sampling_rate
        self.all_speakers = self.__get_all_speakers()
    
    ######################################## Helpers for LCP features ##############################################
    
    def create_save_plc_features(self, window_size=320, lpc_order=40, save_path='./Dataset/TIMIT_FEATURES/TRAIN/'):
        """
        Run this only once to create LPC features from raw audio.
        """
        
        for speaker in self.all_speakers:
            audio_names = [f for f in os.listdir(os.path.join(self.directory, speaker)) if f.endswith('WAV')]
            
            for name in audio_names:
                audio_sample = self.__read_audio(os.path.join(self.directory, speaker, name))
                
                # phone processing
                audio_phones = self.__read_phones(os.path.join(self.directory, speaker, name[:-4] + '.PHN'))
                phone_start = audio_phones[:, 0].astype(int)
                phone_codes = audio_phones[:, 2]
                
                #clip the audio to start and end mark
                audio_sample = audio_sample[phone_start[0]: audio_phones[:,1].astype(int)[-1]]
                phone_start = phone_start - phone_start[0]
                
                # get seq of lpc coefficients
                lpc_features, _ = lpc.speech2lpc(audio_sample, window_size, lpc_order)
                
                # create np array of labels
                hop_len = window_size//4
                labels = np.zeros(lpc_features.shape[1])
                
                for t in range(lpc_features.shape[1]):
                    arg_label = np.where(phone_start <= t*hop_len)[0][-1]
                    labels[t] = TIMIT_PHONE_DICTIONARY[phone_codes[arg_label]]
                
                np.save(os.path.join(save_path, speaker, name[:-4]), lpc_features)
                np.save(os.path.join(save_path, speaker, name[:-4] + '_PHN'), labels)
    
    def get_lpc_label_sequence(self, epochs=1, max_time_steps=16):
        """
        This function is an iterable over all speech data in training set and returns sequence of lpc features along with labels
        """

        for epoch in range(epochs):
            for speaker in self.all_speakers:
                audio_names = [f[:-8] for f in os.listdir(os.path.join(self.directory_features, speaker)) if f.endswith('_PHN.npy')]
                for audio in audio_names:
                    audio_features = np.load(os.path.join(self.directory_features, speaker, audio + '.npy'))
                    audio_labels = np.load(os.path.join(self.directory_features, speaker, audio + '_PHN.npy'))
                    
                    n_buffers = math.ceil(audio_features.shape[1]/max_time_steps)
                    buffers_container = []
                    
                    for buffer in range(n_buffers):
                        buffer_start = buffer*max_time_steps
                        buffer_end = min((buffer+1)*max_time_steps, audio_features.shape[1])
                        
                        buffers_container.append([audio_features[:, buffer_start:buffer_end].T, audio_labels[buffer_start:buffer_end], buffer_end - buffer_start])
                    yield buffers_container
    
    def __get_all_speakers(self):
        speakers = []
        for dialect in self.dialects:
            speaker_path_base = os.path.join(self.directory, dialect)
            speakers = speakers + [os.path.join(dialect, speaker) for speaker in os.listdir(speaker_path_base)]
        return speakers
    
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
    
    def create_features_mfcc(self, audio_seq):
        mfccs = librosa.feature.mfcc(y=audio_seq, sr=self.samping_rate, n_mfcc=40)
        return mfccs.T

    def create_features(self, audio_seq, window_size=20):
        window_len = self.samping_rate*window_size//1000
        
        stft_features = librosa.core.stft(audio_seq, n_fft=window_len, win_length=window_len)
        stft_all = np.concatenate([np.real(stft_features), np.imag(stft_features)], axis=0)
        stft_all = np.sign(stft_all)*np.log(np.abs(stft_all) + 1.0)
        return stft_all.T
        
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
        
        stft_all = np.concatenate([np.real(stft_features), np.imag(stft_features)], axis=0)
        
        labels = np.zeros(stft_all.shape[1])
        
        for t in range(stft_all.shape[1]):
            arg_label = np.where(label_seq_start <= t*hop_len)[0][-1]
            labels[t] = TIMIT_PHONE_DICTIONARY[label_codes[arg_label]]
            
        return stft_all, labels
    
    def speaker_embedding_getter(self, n_epochs=1, N=30, M=10, max_time_steps=250):
        """
        Iterable to get batch of M samples from N speakers
        """
        random.shuffle(self.all_speakers)
        num_batches = math.ceil(len(self.all_speakers)/N)
        
        #append starting speakers if num_batches*N is not equal to number of speakers
        all_speakers = self.all_speakers + self.all_speakers[0:(num_batches*N - len(self.all_speakers))]
        
        for epoch in range(n_epochs):
            for batch_iter in range(num_batches):
                speaker_batch = all_speakers[batch_iter*N : (batch_iter+1)*N]
                speakers_sample_data = np.zeros((N*M, max_time_steps, CONFIG_EMBED.num_features), dtype=float)
                seq_length = np.zeros(N*M, dtype=int)
                
                for j, speaker in enumerate(speaker_batch):
                    samples = [f for f in os.listdir(os.path.join(self.directory, speaker)) if f.endswith('WAV')]
                    random.shuffle(samples)
                    for i, sample in enumerate(samples):
                        features = self.create_features_mfcc(self.__read_audio(os.path.join(self.directory, speaker, sample)))
                        seq_length[j*M + i] = min(features.shape[0], max_time_steps)
                        speakers_sample_data[j*M + i, 0:seq_length[j*M + i], :] = features[0:seq_length[j*M + i], :]
                
                yield speakers_sample_data, seq_length
    
    
    def all_speech_getter(self, n_epochs=1, max_time_steps=16):
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
