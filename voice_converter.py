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

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def save_audio(original, converted, name, path='./test_cases/'):
    if not os.path.exists(path):
        os.mkdir(path)
    
    librosa.output.write_wav(os.path.join(path, 'original_' + name + '.WAV'), original, 16000)
    librosa.output.write_wav(os.path.join(path, 'converted_' + name + '.WAV'), converted, 16000)

def generate_speech_from_features(audio_features, audio_phase, window_size=20, sampling_rate=16000):
    window_len = sampling_rate*window_size//1000
    hop_len = window_len//4
    
    #first axis is dummy
    audio_features = np.exp(audio_features)
    print('max val = {}, min val = {}'.format(np.max(audio_features), np.min(audio_features)))
    audio_features_real = audio_features*np.cos(audio_phase)
    audio_features_img = audio_features*np.sin(audio_phase)
    
    audio_features_cmplx = audio_features_real + 1j*audio_features_img
    print(audio_features_cmplx.shape)
    audio = librosa.istft(audio_features_cmplx, hop_length=hop_len, win_length=window_len)
    
    return audio

def features_from_audio(audio, window_size=20, sampling_rate=16000):
     window_len = sampling_rate*window_size//1000
     stft_features = librosa.core.stft(audio, n_fft=window_len, win_length=window_len)
     stft_log_mag = np.log(np.abs(stft_features))
     stft_phase = np.angle(stft_features)
     
     return stft_log_mag.T, stft_phase.T
     
class VoiceConverter(object):
    """
    This class performs voice conversion using two trained models - 1. Speech recognition (e.g. phoneme classifier) 2. Speaker-embedder
    
    Given two speech sentences : content_speech and style_speech, we generate a new speech that has same content as content_speech and the speech
    style is same as style_speech.
    
    Initial implementation: we reconstruct speech only from phoneme classifier.
    """
    
    def __init__(self, phoneme_recognizer_model_path, speaker_embedder_model_path=None):
        self.pr_model_path = phoneme_recognizer_model_path
        self.se_model_path = speaker_embedder_model_path        
        
    def create_graphs(self, max_time_step):
        #TODO: create the two graphs on separate gpus
        
        tf.reset_default_graph()
        
        self.content_speech = tf.placeholder(tf.float32, [1, max_time_step, CONFIG.num_features], name='content_speech')
        self.content_seq_length = tf.placeholder(tf.int32, [1], name='seq_length')
        
        initial_state = tf.contrib.rnn.LSTMStateTuple(tf.zeros([1, CONFIG.num_cell_dim]), tf.zeros([1, CONFIG.num_cell_dim]))
        
        logits_content, self.layers_content = create_model_rnn(self.content_speech,
                                                               self.content_seq_length,
                                                               keep_prob=1.0,
                                                               previous_state=initial_state,
                                                               reuse=False)
        
        #self.style_speech = tf.placeholder(tf.float32, [1, None, CONFIG.num_features])
        self.speech_gen = utils.get_variable('converted_speech', [1, max_time_step, CONFIG.num_features], initializer=tf.contrib.layers.xavier_initializer())
        
        logits_gen_speech, self.layers_gen_speech = create_model_rnn(self.speech_gen,
                                                                     self.content_seq_length,
                                                                     keep_prob=1.0,
                                                                     previous_state=initial_state,
                                                                     reuse=True)
        
        
        self.cost = self.__get_cost(self.layers_content['rnn_output'], self.layers_gen_speech['rnn_output'])
        
        
    def __get_cost(self, features_content, features_gen, embedding_gen=None, embedding_style=None):
        return tf.losses.mean_squared_error(features_content, features_gen)
    
    
    def __get_optimizer(self, global_step):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)\
                            .minimize(self.cost, global_step=global_step, var_list=[self.speech_gen])
                            
        return optimizer
    
    def restore(self, session):
        """
        Restores a session from a checkpoint
        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """
        #TODO: restore other model as well
        ckpt = tf.train.get_checkpoint_state(self.pr_model_path)
        if ckpt and ckpt.model_checkpoint_path:
            saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rnn_model'), restore_sequentially=True)
            saver.restore(session, ckpt.model_checkpoint_path )
            logging.info("Model restored from file: {}".format(self.pr_model_path))
        
        else:
            raise Exception("Cannot load the model")
    
    def output_logs(self, mse, step):
        print("Step: {}, MSE: {}".format(step, mse))
        
    def convert(self, content_speech, style_speech=None, max_iter=100):
        """
        content_speech should be of the shape [1, max_time, num_features]
        """
        
        max_time_step = content_speech.shape[1]
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
                                                       })
                self.output_logs(loss, it)
            
            return speech_gen
        
def convert_save_audio(filename = './Dataset/TIMIT/TEST/DR6/FDRW0/SI653.WAV', model_path='./output_model'):
    test_audio, _ = librosa.load(filename, sr=16000)
    features_mg, features_angle = features_from_audio(test_audio)
    vc = VoiceConverter(model_path)
    converted_speech_features = vc.convert(features_mg[np.newaxis, :, :])
    converted_audio = generate_speech_from_features(converted_speech_features[0].T, features_angle.T)
    name = 'test_' + filename[(filename.rfind('/') + 1):]
    save_audio(test_audio, converted_audio, name)

convert_save_audio()
