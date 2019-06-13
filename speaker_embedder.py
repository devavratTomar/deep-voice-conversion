# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 17:50:41 2019

@author: devav
"""

import tensorflow as tf
import logging
import shutil
import os
import numpy as np

import utils
from model_generator import create_speaker_embedder_model
from config import CONFIG_EMBED


logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class DeepSpeakerEmbedder:
    #we keep batch size as number of speakers*numberofsamples
    def __init__(self, num_speakers=20, num_samples=10, max_time_step=600):
        tf.reset_default_graph()
        batch_size = num_speakers*num_samples
        
        self.speech_data = tf.placeholder(tf.float32, [batch_size,  max_time_step, 2*CONFIG_EMBED.num_features], name='speech_data_batch')
        self.seq_length = tf.placeholder(tf.int32, [batch_size], name='seq_length')
        self.keep_prob = tf.placeholder(tf.float32, name="dropout_prob")
        
        self.embeddings, self.attentions = create_speaker_embedder_model(self.speech_data, self.seq_length, self.keep_prob)
        self.cost = self.__get_cost(self.embeddings, num_speakers, num_samples)
    
    def __get_cost(self, embeddings, num_speakers, num_samples):
        embeddings = tf.reshape(embeddings, [num_speakers, num_samples, -1])
        
        center = utils.normalize(tf.reduce_mean(embeddings, axis=1), axis=1)
        center_except = utils.normalize(tf.reshape(tf.reduce_sum(embeddings, axis=1, keep_dims=True) - embeddings, shape=[num_speakers*num_samples, -1]), axis=1)
        
        S = tf.concat(
            [tf.concat([tf.reduce_sum(center_except[i*num_samples:(i+1)*num_samples,:]*embeddings[j,:,:], axis=1, keep_dims=True) if i==j\
                        else tf.reduce_sum(center[i:(i+1),:]*embeddings[j,:,:], axis=1, keep_dims=True) for i in range(num_speakers)],
                       axis=1) for j in range(num_speakers)], axis=0)
    
        with tf.variable_scope('cost_var'):
            w = tf.get_variable('similarity_scale', initializer= np.array([10], dtype=np.float32))
            b = tf.get_variable('similarity_bias', initializer= np.array([-5], dtype=np.float32))
        
        S = tf.abs(w)*S + b
        S_correct = tf.concat([S[i*num_samples:(i+1)*num_samples, i:(i+1)] for i in range(num_speakers)], axis=0)
        
        return -tf.reduce_mean(S_correct-tf.log(tf.reduce_sum(tf.exp(S), axis=1, keep_dims=True) + 1e-6))
    
    def predict_embeddings(self, model_path, speaker_data, seq_len):
        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init)
            self.restore(sess, model_path)
            predictions = sess.run(self.embeddings, feed_dict={self.speech_data: speaker_data,
                                                               self.seq_length: seq_len,
                                                               self.keep_prob: 1.0})
        return predictions
    
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
        
    
class DeepSpeakerModelTrainer():
    def __init__(self, model):
        self.model = model
        
    def __get_optimizer(self, global_step, lr=1e-3):
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
        return optimizer
    
    def __initialize(self, output_path, restore):
        """
        Performs initialization of computation graph and tensorboard summary.
        
        :param output_path: The path where trained model will be saved at every checkpoint
        :param restore: If False, delete old path and create new model (should be used when training from scratch).
                        If True, we resotre the model. So don't delete the model at output_path
                        
        :param prediction_path: Path where prediction on test data will be saved after every epoch.
        """
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = self.__get_optimizer(global_step)
        
        vars_model = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='embedding_model')
        grads_model, vars_model = zip(*optimizer.compute_gradients(self.model.cost, var_list=vars_model))
        grads_model = tf.clip_by_global_norm(grads_model, 3.0)[0]
        
        self.grads_model = grads_model
        
        vars_cost = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cost_var')
        grads_cost, vars_cost = zip(*optimizer.compute_gradients(self.model.cost, var_list=vars_cost))
        
        grads_cost = tf.clip_by_global_norm(grads_cost, 3.0)[0]
        grads_cost = [0.01*grad for grad in grads_cost]

        vars_all = vars_model + vars_cost
        grads_all = grads_model + grads_cost
        
        self.train_op = optimizer.apply_gradients(zip(grads_all, vars_all), global_step=global_step)
        
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
    
    def output_mini_batch_stats(self, session, summary_writer, step, batch_speech, batch_speech_seq_length):
        summary_str, sh, cost = session.run((self.summary_all, self.model.embeddings, self.model.cost),
                                        feed_dict= {self.model.speech_data: batch_speech,
                                                    self.model.seq_length: batch_speech_seq_length,
                                                    self.model.keep_prob: 1.0})
        
        logging.info("step: {}, loss = {:.8f}, sh: {}".format(step, cost, sh.shape))
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()
        
    def train(self, data_provider_train,
              output_path,
              keep_prob,
              epochs=5,
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
                
                for speech_data, seq_len in data_provider_train():
                    if step_counter % display_step == 0:
                        self.output_mini_batch_stats(sess, summary_writer, step_counter, speech_data, seq_len)
                    
                    _ = sess.run(self.train_op, feed_dict={self.model.speech_data: speech_data,
                                                           self.model.seq_length: seq_len,
                                                           self.model.keep_prob: keep_prob})
                    
                    if step_counter != 0 and step_counter % model_save_step == 0:
                        save_path = self.model.save(sess, save_path)
                    
                    step_counter += 1

            summary_writer.close()
        
        logging.info("Training Finished")
