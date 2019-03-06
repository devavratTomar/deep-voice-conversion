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
        'num_features':322,
        'num_hidden_units':512,
        'num_classes':61,
        'num_rnn_layers':1,
        'num_hidden_1':1024,
        'num_hidden_2':512,
        'num_hidden_3':512,
        'num_cell_dim':1024,
        'num_hidden_5': 1024
        }