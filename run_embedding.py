# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 11:53:01 2019

@author: devav
"""

from data_processor import DataProcessor_TIMIT
from speaker_embedder import DeepSpeakerEmbedder, DeepSpeakerModelTrainer
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

dp = DataProcessor_TIMIT()
    

rnn_model = DeepSpeakerEmbedder()

trainer = DeepSpeakerModelTrainer(rnn_model)
trainer.train(dp.speaker_embedding_getter, './output_model_embedder', 0.95, 16, 5, 100, model_save_step=1000, restore=False)
