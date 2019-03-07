# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 10:22:10 2019

@author: devav
"""

"""
This model is inspired by Baidus speech to text paper.

First 2 layers are non recurrent which extract features from the spectrum of speech.

"""

import tensorflow as tf
import utils
from config import CONFIG

def create_speaker_embedder_model():
    pass

def create_model_rnn(input_speech, seq_length, keep_prob, previous_state=None, reuse=False):
    """
    create rnn model with given hyperparameters
    
    :param input_speech: tf.placeholder of input_speech spectrogram of shape [batch_size, max_time_step, num_features]
    :param seq_length: tf.placeholder that represents the length of sequence. shape should be [batch_size]
    :param previous_state: initial hidden state of RNN.
    :param keep_prob: tf.placeholder for dropouts
    """
    
    layers = {}
    
    batch_size = tf.shape(input_speech)[0]
    
    with tf.variable_scope('rnn_model', reuse=reuse):
        #placeholder for input_speech
        
        with tf.name_scope('input_speech'):
            # Permute the time_step and batch axis
            input_data = tf.transpose(input_speech, [1, 0, 2])
            
            # Reshape input_data for 1st layer that is not recurrent
            input_data = tf.reshape(input_data, [-1, CONFIG.num_features])
            layers['input_reshaped'] = input_data
            
            # layer 1:
            b1 = utils.get_variable('b1', [CONFIG.num_hidden_1], tf.zeros_initializer())
            h1 = utils.get_variable('h1', [CONFIG.num_features, CONFIG.num_hidden_1], tf.contrib.layers.xavier_initializer())
            layer_1 = tf.minimum(tf.nn.leaky_relu(tf.add(tf.matmul(input_data, h1), b1)), CONFIG.relu_clip)
            
            layer_1 = tf.nn.dropout(layer_1, keep_prob=keep_prob)
            layers['layer_1'] = layer_1
            
            #layer 2:
            b2 = utils.get_variable('b2', [CONFIG.num_hidden_2], tf.zeros_initializer())
            h2 = utils.get_variable('h2', [CONFIG.num_hidden_1, CONFIG.num_hidden_2], tf.contrib.layers.xavier_initializer())
            layer_2 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_1, h2), b2)), CONFIG.relu_clip)
            
            layer_2 = tf.nn.dropout(layer_2, keep_prob=keep_prob)
            layers['layer_2'] = layer_2
            
            #layer 3:
            b3 = utils.get_variable('b3', [CONFIG.num_hidden_3], tf.zeros_initializer())
            h3 = utils.get_variable('h3', [CONFIG.num_hidden_2, CONFIG.num_hidden_3], tf.contrib.layers.xavier_initializer())
            layer_3 = tf.minimum(tf.nn.relu(tf.add(tf.matmul(layer_2, h3), b3)), CONFIG.relu_clip)
            
            layer_3 = tf.nn.dropout(layer_3, keep_prob=keep_prob)
            layers['layer_3'] = layer_3
            
            
            fw_cell = tf.contrib.rnn.LSTMBlockFusedCell(CONFIG.num_cell_dim, reuse=reuse)
            layers['fw_cell'] = fw_cell
            
            # LSTM RNN expects its input to be of shape [max_time, batch_size, input_size]
            layer_3 = tf.reshape(layer_3, [-1, batch_size, CONFIG.num_hidden_3])
            
            output, output_state = fw_cell(inputs=layer_3, dtype=tf.float32, sequence_length=seq_length, initial_state=previous_state)
            
            # Reshape output from a tensor of shape [n_steps, batch_size, n_cell_dim]
            # to a tensor of shape [n_steps*batch_size, n_cell_dim]
            
            output = tf.reshape(output, [-1, CONFIG.num_cell_dim])
            
            layers['rnn_output'] = output
            layers['rnn_output_states'] = output_state
            
            # Feed output to the layer 5 with dropouts
            
            b5 = utils.get_variable('b5', CONFIG.num_hidden_5, tf.zeros_initializer())
            h5 = utils.get_variable('h5', [CONFIG.num_cell_dim, CONFIG.num_hidden_5],  tf.contrib.layers.xavier_initializer())
            
            layer_5 = tf.minimum(tf.nn.leaky_relu(tf.add(tf.matmul(output, h5), b5)), CONFIG.relu_clip)
            layer_5 = tf.nn.dropout(layer_5, keep_prob=keep_prob)
            
            # Now we muliply layer 5 with matrix h6 and add bias b6 to get phoneme class logits
            
            b6 = utils.get_variable('b6', [CONFIG.num_classes], tf.zeros_initializer())
            h6 = utils.get_variable('h6', [CONFIG.num_hidden_5, CONFIG.num_classes], tf.contrib.layers.xavier_initializer())
            
            layer_6 = tf.add(tf.matmul(layer_5, h6), b6)
            
            layers['layer_6'] = layer_6
            # Finally we reshape layer_6 from a tensor of shape [n_steps*batch_size, n_classes]
            # to tensor [batch_size, n_step, n_classes]
            
            layer_6 = tf.reshape(layer_6, [batch_size, -1, CONFIG.num_classes])
            layers['raw_logits'] = layer_6
            
            return layer_6, layers
