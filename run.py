# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 08:45:47 2019

@author: devav
"""

from data_processor import DataProcessor_TIMIT
from lstm_voice_model import DeepPhonemeModel
from lstm_voice_model import DeepPhoenemeModelTrainer

dp = DataProcessor_TIMIT()
    

rnn_model = DeepPhonemeModel(batch_size=1)

trainer = DeepPhoenemeModelTrainer(rnn_model)
trainer.train(dp.all_speech_getter, './output_model', 0.95, 32, 1)