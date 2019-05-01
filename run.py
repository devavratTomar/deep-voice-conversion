# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:45:47 2019

@author: devav
"""

from data_processor import DataProcessor_TIMIT
from lstm_voice_model import DeepPhonemeModel
from lstm_voice_model import DeepPhoenemeModelTrainer
import os


os.environ['CUDA_VISIBLE_DEVICES'] = '0'

dp = DataProcessor_TIMIT()
    

rnn_model = DeepPhonemeModel(batch_size=1)

trainer = DeepPhoenemeModelTrainer(rnn_model)
trainer.train(dp.get_lpc_label_sequence, './output_model', 0.70, 16, 5, 100, model_save_step=1000, restore=False)
