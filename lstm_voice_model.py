# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 17:11:12 2019

@author: devav
"""

import tensorflow as tf
import logging
import shutil
import os
import numpy as np

import utils

from config import CONFIG

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


"""
This model is inspired by Baidus speech to text paper.

First 2 layers are non recurrent which extract features from the spectrum of speech.

"""

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
        
        
class DeepPhonemeModel(object):
    """
    Implementation of Deep phoneme classifier using RNN.
    """
    
    def __init__(self, batch_size=None, max_time_step=16):
        
        tf.reset_default_graph()
        
        self.input_speech = tf.placeholder(tf.float32, [batch_size, max_time_step if max_time_step> 0 else None , CONFIG.num_features], name='input_speech')
        self.seq_length = tf.placeholder(tf.int32, [batch_size], name='seq_length')
        
        self.labels_gt = tf.placeholder(tf.int32, [batch_size, None], name='labels_ground_truth')
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_prob")
        
        previous_state_c = utils.get_variable('previous_state_c', [batch_size, CONFIG.num_cell_dim], initializer=None)
        previous_state_h = utils.get_variable('previous_state_h', [batch_size, CONFIG.num_cell_dim], initializer=None)
        
        previous_state = tf.contrib.rnn.LSTMStateTuple(previous_state_c, previous_state_h)
        
        logits, self.layers = create_model_rnn(self.input_speech,
                                               self.seq_length,
                                               keep_prob=self.keep_prob,
                                               previous_state=previous_state)
        
        new_state_c, new_state_h = self.layers['rnn_output_states']
        
        # We initialize the states to be zero and update them after every iteration
        zero_state = tf.zeros([batch_size, CONFIG.num_cell_dim], "float")
        initialize_c = tf.assign(previous_state_c, zero_state)
        initialize_h = tf.assign(previous_state_h, zero_state)
        
        self.initialize_state = tf.group(initialize_c, initialize_h)
        
        with tf.control_dependencies([tf.assign(previous_state_c, new_state_c), tf.assign(previous_state_h, new_state_h)]):
            logits = tf.identity(logits, name='logits')
        
        self.phoneme_prob = tf.nn.softmax(logits)
        
        self.cost = self.__get_cost(logits)
        
    def __get_cost(self, logits):
        with tf.name_scope("cost"):
            loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels_gt, logits=logits))
            return loss
            
    def predict(self, model_path, test_speech):
        """
        Performs phoneme prediction for the test audio speech using the given model
        
        :param model_path: The path of the trained model
        :param test_speech: Spectrogram of the speech. Shape should be [num_time_steps, num_features]
        """
        
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run((init, self.initialize_state))
            
            self.restore(sess, model_path)
            dummy = np.empty((1, test_speech.shape[0]), dtype="int32")
            seq_len = test_speech.shape[0]
            test_speech = test_speech[np.newaxis, :, :]
            
            predictions = sess.run(self.phoneme_prob, feed_dict={self.input_speech:test_speech,
                                                                 self.labels_gt:dummy,
                                                                 self.keep_prob:1.0,
                                                                 self.seq_length: seq_len})
            
            return np.argmax(predictions, axis=2)
        
        
    def save(self, sess, model_path):
        """
        Saves the current session to a checkpoint
        :param sess: current session
        :param model_path: path to file system location
        """

        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path
    
    def restore(self, sess, model_path):
        """
        Restores a session from a checkpoint
        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """

        saver = tf.train.Saver()
        saver.restore(sess, model_path)
        logging.info("Model restored from file: {}".format(model_path))
        
        
        
class DeepPhoenemeModelTrainer(object):
    """
    This class trains a given model.
    TODO: Monitor validation loss as well
    """
    
    def __init__(self, model):
        self.model = model
    
    def __get_optimizer(self, global_step):
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-8)\
                            .minimize(self.model.cost, global_step=global_step, var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='rnn_model'))
                            
        return optimizer
    
    def __initialize(self, output_path, restore):
        """
        Performs initialization of computation graph and tensorboard summary.
        
        :param output_path: The path where trained model will be saved at every checkpoint
        :param restore: If False, delete old path and create new model (should be used when training from scratch).
                        If True, we resotre the model. So don't delete the model at output_path
                        
        :param prediction_path: Path where prediction on test data will be saved after every epoch.
        """
        
        global_step = tf.Variable(0, name="global_step")
        
        self.optimizer = self.__get_optimizer(global_step)
        
        # add more summary as necessary
        tf.summary.scalar("loss", self.model.cost)
        self.summary_all = tf.summary.merge_all()

        init = tf.global_variables_initializer()
        output_path_abs = os.path.abspath(output_path)
        
        if not restore:
            if os.path.exists(output_path_abs):
                logging.info("Removing '{:}'".format(output_path_abs))
                shutil.rmtree(output_path_abs, ignore_errors=True)
            
        if not os.path.exists(output_path_abs):
            logging.info("Creating '{:}'".format(output_path_abs))
            os.mkdir(output_path_abs)
            
        return init
    
    
    def output_mini_batch_stats(self, session, summary_writer, step, batch_speech, batch_speech_phonemes, batch_speech_seq_length):
        summary_str, prediction_prob, cost = session.run((self.summary_all, self.model.phoneme_prob, self.model.cost),
                                                         feed_dict= {self.model.input_speech: batch_speech,
                                                                     self.model.labels_gt: batch_speech_phonemes,
                                                                     self.model.seq_length: batch_speech_seq_length,
                                                                     self.model.keep_prob: 1.0})
        
        predictions = np.argmax(prediction_prob, axis=2)
        accuracy = np.sum(predictions == batch_speech_phonemes)/batch_speech_phonemes.size
        
        logging.info("step: {}, loss = {:.8f} accuracy = {:.2f},\n predictions = {}, ground_truth = {}".format(step//1000, cost, accuracy*100, predictions, batch_speech_phonemes))
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        
    def train(self, data_provider_train,
              output_path,
              keep_prob,
              max_time_step,
              epochs=10,
              display_step=1,
              model_save_step=1000,
              restore=False,
              write_graph=True):
        """
        Start training the network
        
        :param data_provider_train: callable returning training data
        :param output_path: path where to store checkpoints
        :param epochs: number of epochs
        :param display_step: number of steps till outputting stats
        :param restore: Flag if previous model should be restored
        :param write_graph: Flag if the computation graph should be written as protobuf file to the output path
        """
        
        save_path = os.path.join(output_path, "model.ckpt")
        init = self.__initialize(output_path, restore)
        
        with tf.Session() as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, output_path, "graph.pb")
            
            sess.run(init)
            
            if restore:
                ckpt = tf.train.get_checkpoint_state(output_path)
                if ckpt and ckpt.model_checkpoint_path:
                    logging.info("Model Restored!")
                    self.model.restore(sess, ckpt.model_checkpoint_path)
                    
            
            summary_writer = tf.summary.FileWriter(output_path, graph=sess.graph)
            
            logging.info("Training Started")
            
            step_counter = 0
            for epoch in range(epochs):
                logging.info("Epoch : {}".format(epoch))
                
                for speech in data_provider_train():
                    sess.run(self.model.initialize_state)
                    
                    for speech_seg in speech:
                        features = speech_seg[0]
                        labels = speech_seg[1]
                        seq_len = speech_seg[2]
                        
                        if seq_len != max_time_step:
                            features = utils.append_zeros(features, max_time_step, 0)
                            labels = utils.append_zeros(labels, max_time_step, 0)
                        
                        features = features[np.newaxis, :, :]
                        labels = labels[np.newaxis, :]
                        seq_len = np.array([seq_len])
                        
                        _ = sess.run(self.optimizer, feed_dict={self.model.input_speech: features,
                                                                     self.model.labels_gt: labels,
                                                                     self.model.seq_length: seq_len,
                                                                     self.model.keep_prob: keep_prob})
                        if step_counter % display_step == 0:
                            self.output_mini_batch_stats(sess, summary_writer, step_counter, features, labels, seq_len)
                        
                        if step_counter != 0 and step_counter % model_save_step == 0:
                            save_path = self.model.save(sess, save_path)
                        
                        step_counter += 1

            summary_writer.close()
        
        logging.info("Training Finished")
