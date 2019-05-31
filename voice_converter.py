# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:24:04 2019

@author: devav
"""

#TODO: create speaker-embedder model in model generator and load it here
from model_generator import create_model_rnn, create_speaker_embedder_model
from config import CONFIG 

import utils
import tensorflow as tf
import logging
import librosa
import numpy as np
import os
import lpc

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def save_audio(original, converted, name, path='./test_cases/'):
    if not os.path.exists(path):
        os.mkdir(path)
    
    librosa.output.write_wav(os.path.join(path, 'original_' + name + '.WAV'), original, 16000)
    librosa.output.write_wav(os.path.join(path, 'converted_' + name + '.WAV'), converted, 16000)

###################################### Generate speech from lpc and error #######################################
def generate_speech_lpc(feature_matrix, error_matrix, window_size=320, lpc_order=40):
    return lpc.lpc2speech(feature_matrix, error_matrix, window_size, lpc_order)

def lpc_features_from_speech(audio, window_size=320, lpc_order=40):
    features, error = lpc.speech2lpc(audio, window_size, lpc_order)
    return features, error
    
#################################################################################################################
def generate_speech_from_features(audio_features, window_size=20, sampling_rate=16000):
    window_len = sampling_rate*window_size//1000
    hop_len = window_len//4
    num_features = audio_features.shape[0]//2
    
    #first axis is dummy
    audio_features_real = audio_features[:num_features, :]
    audio_features_img = audio_features[num_features:, :]
    print('max val = {}, min val = {}'.format(np.max(audio_features), np.min(audio_features)))
    
    audio_features_cmplx = audio_features_real + 1j*audio_features_img
    print(audio_features_cmplx.shape)
    audio = librosa.istft(audio_features_cmplx, hop_length=hop_len, win_length=window_len)
    
    return audio

def features_from_audio(audio, window_size=20, sampling_rate=16000):
     window_len = sampling_rate*window_size//1000
     stft_features = librosa.core.stft(audio, n_fft=window_len, win_length=window_len)     
     return np.concatenate([np.real(stft_features), np.imag(stft_features)], axis=0).T
     
class VoiceConverter(object):
    """
    This class performs voice conversion using two trained models - 1. Speech recognition (e.g. phoneme classifier) 2. Speaker-embedder
    
    Given two speech sentences : content_speech and style_speech, we generate a new speech that has same content as content_speech and the speech
    style is same as style_speech.
    
    Initial implementation: we reconstruct speech only from phoneme classifier.
    """
    
    def __init__(self, phoneme_recognizer_model_path, speaker_embedder_model_path):
        self.pr_model_path = phoneme_recognizer_model_path
        self.se_model_path = speaker_embedder_model_path        
        
    def create_graphs(self, max_time_step):
        tf.reset_default_graph()
        
        self.content_speech = tf.placeholder(tf.float32, [1, max_time_step, CONFIG.num_features], name='content_speech')
        self.content_seq_length = tf.placeholder(tf.int32, [1], name='content_seq_length')
        
        self.style_speech = tf.placeholder(tf.float32, [1, None, CONFIG.num_features])
        self.style_seq_length = tf.placeholder(tf.int32, [1], name='style_seq_length')
        
        self.speech_gen = utils.get_variable('converted_speech', [1, max_time_step, CONFIG.num_features],
                                             initializer=tf.contrib.layers.xavier_initializer())
        
        # Content generation
        initial_state = tf.contrib.rnn.LSTMStateTuple(tf.zeros([1, CONFIG.num_cell_dim]), tf.zeros([1, CONFIG.num_cell_dim]))
        
        logits_content, self.layers_content = create_model_rnn(self.content_speech,
                                                               self.content_seq_length,
                                                               keep_prob=1.0,
                                                               previous_state=initial_state,
                                                               reuse=False)
        
        logits_gen_speech, self.layers_gen_speech = create_model_rnn(self.speech_gen,
                                                                     self.content_seq_length,
                                                                     keep_prob=1.0,
                                                                     previous_state=initial_state,
                                                                     reuse=True)
        
        
        # Style generation
        embedding_style, _ = create_speaker_embedder_model(self.style_speech,
                                                           self.style_seq_length,
                                                           keep_prob=1.0)
        
        embedding_gen_speech, _ = create_speaker_embedder_model(self.speech_gen,
                                                                self.content_seq_length,
                                                                keep_prob=1.0 , reuse=True)
        
        self.cost = self.__get_cost(self.layers_content['rnn_output'], self.layers_gen_speech['rnn_output'],
                                    embedding_style, embedding_gen_speech)

        
    def __get_cost(self, features_content, features_gen, embedding_style, embedding_gen):
        alpha = 1.0
        print(embedding_style.shape)
        return tf.losses.mean_squared_error(features_content, features_gen)+\
               alpha*tf.losses.cosine_distance(embedding_style, embedding_gen, axis=1)
    
    
    def __get_optimizer(self, global_step):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8)\
                            .minimize(self.cost, global_step=global_step, var_list=[self.speech_gen])
                            
        return optimizer
    
    def restore(self, session):
        """
        Restores a session from a checkpoint
        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """
        ckpt = tf.train.get_checkpoint_state(self.pr_model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rnn_model'),
                                   restore_sequentially=True)
            saver.restore(session, ckpt.model_checkpoint_path )
            logging.info("Model restored from file: {}".format(self.pr_model_path))
        
        else:
            raise Exception("Cannot load the model")
        
        ckpt = tf.train.get_checkpoint_state(self.se_model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='embedding_model'),
                                   restore_sequentially=True)
            saver.restore(session, ckpt.model_checkpoint_path )
            logging.info("Model restored from file: {}".format(self.pr_model_path))
        
        else:
            raise Exception("Cannot load the model")
    
    def output_logs(self, mse, step):
        print("Step: {}, MSE: {}".format(step, mse))
        
    def convert(self, content_speech, style_speech, max_iter=10):
        """
        content_speech should be of the shape [1, max_time, num_features]
        """
        
        max_time_step = content_speech.shape[1]
        style_max_time_step = style_speech.shape[1]
        
        self.create_graphs(max_time_step)
        
        global_step = tf.Variable(0, name='global_step')
        optimizer = self.__get_optimizer(global_step)
        
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init)
            sess.run(tf.assign(self.speech_gen, content_speech + np.random.normal(scale=1.0, size=content_speech.shape)))
            self.restore(sess)
            for it in range(max_iter):
                _, loss, speech_gen = sess.run((optimizer, self.cost, self.speech_gen),
                                               feed_dict= {
                                                       self.content_speech:content_speech,
                                                       self.content_seq_length:[max_time_step],
                                                       self.style_speech:style_speech,
                                                       self.style_seq_length: [style_max_time_step]
                                                       })
                self.output_logs(loss, it)
            
            return speech_gen
        
def convert_save_audio(content='./Dataset/TIMIT/TRAIN/DR6/FAPB0/SA1.WAV',
                       style='./Dataset/TIMIT/TRAIN/DR1/MCPM0/SA1.WAV',
                       speech_to_text_model_path='./output_model',
                       speaker_embedder_path='./output_model_embedder'):
    
    content_audio, _ = librosa.load(content, sr=16000)
    features_content, error_content = lpc_features_from_speech(content_audio)
    
    style_audio, _ = librosa.load(style, sr=16000)
    features_style, _ = lpc_features_from_speech(style_audio)
    
    vc = VoiceConverter(speech_to_text_model_path,
                        speaker_embedder_path)
    
    converted_speech_features = vc.convert(features_content.T[np.newaxis, :, :],
                                           features_style.T[np.newaxis, :, :])

    print("converted_speech_features shape", converted_speech_features.shape)
    converted_audio = generate_speech_lpc(converted_speech_features[0].T, error_content)

    name = 'test_' + content[(content.rfind('/') + 1):]
    save_audio(content_audio, converted_audio, name)
    
    np.save('test_' + content[(content.rfind('/')+1):] + 'spectrogram', converted_speech_features)
    
    return converted_speech_features

converted_features = convert_save_audio()
