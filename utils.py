# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:40:32 2019

@author: devav
"""

import tensorflow as tf
import numpy as np

# can be used to create variables on different devices.

def get_variable(name, shape, initializer):
    return tf.get_variable(name=name, shape=shape, initializer=initializer)


def append_zeros(data, new_size, axis):
    shape = list(data.shape)
    shape[axis] = new_size - shape[axis]
    
    append_data = np.zeros(shape=shape, dtype="float32")
    
    return np.concatenate((data, append_data), axis=axis)


def normalize(data, axis):
    return tf.div(data, tf.norm(data, axis=axis, keep_dims=True) + 1e-6)