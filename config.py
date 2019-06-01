# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:02:39 2019

@author: devav
"""

class Config:
    def __init__(self):
        self._config = None
    
    def __getattr__(self, name):
        if not self._config:
            raise RuntimeError("Global configuration not yet initialized.")
        if not name in self._config:
            raise RuntimeError("Configuration option {} not found in config.".format(name))
        return self._config[name]

CONFIG = Config()
CONFIG._config = {
        'relu_clip':20,
        'num_features':161,
        'num_classes':61,
        'num_rnn_layers':1,
        'num_hidden_1':2048,
        'num_hidden_2':2048,
        'num_hidden_3':2048,
        'num_cell_dim':2048,
        'num_hidden_5':1024
        }

CONFIG_EMBED = Config()
CONFIG_EMBED._config = {
        'relu_clip':20,
        'num_features':161,
        'num_hidden_1':128,
        'num_rnn_hidden':786,
        'num_proj':256
        }
