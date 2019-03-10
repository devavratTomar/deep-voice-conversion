# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:02:39 2019

@author: devav
"""

class ConfigSingleton:
    _config = None

    def __getattr__(self, name):
        if not ConfigSingleton._config:
            raise RuntimeError("Global configuration not yet initialized.")
        if not name in ConfigSingleton._config:
            raise RuntimeError("Configuration option {} not found in config.".format(name))
        return ConfigSingleton._config[name]

CONFIG = ConfigSingleton()

ConfigSingleton._config = {
        'relu_clip':20,
        'num_features':322,
        'num_classes':61,
        'num_rnn_layers':1,
        'num_hidden_1':2048,
        'num_hidden_2':2048,
        'num_hidden_3':2048,
        'num_cell_dim':2048,
        'num_hidden_5':1024
        }
