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
        
        logits_content, self.layers_content = create_model_rnn(self.content_speech,
                                                               self.content_seq_length,
                                                               keep_prob=1.0,
                                                               previous_state= tf.zeros([1, CONFIG.num_cell_dim], "float"),
                                                               reuse=False)
        
        #self.style_speech = tf.placeholder(tf.float32, [1, None, CONFIG.num_features])
        self.speech_gen = utils.get_variable('converted_speech', [1, max_time_step, CONFIG.num_features], initializer=tf.contrib.layers.xavier_initializer())
        
        logits_gen_speech, self.layers_gen_speech = create_model_rnn(self.speech_gen,
                                                                     self.content_seq_length,
                                                                     keep_prob=1.0,
                                                                     previous_state= tf.zeros([1, CONFIG.num_cell_dim]),
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
        saver = tf.train.Saver()
        saver.restore(session, self.pr_model_path)
        logging.info("Model restored from file: {}".format(self.pr_model_path))
        
        
    def convert(self, content_speech, style_speech=None, max_iter = 1000):
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
            self.restore(sess)
            
            for it in range(max_iter):
                _, loss, speech_gen = sess.run((optimizer, self.cost, self.speech_gen))
                logging.info("MSE: {}".format(loss))
                
            
            return speech_gen