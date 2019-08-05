#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 20:35:16 2019

@author: rah4927
"""

import keras
import numpy as np
from keras.layers import Dense, Dropout, Embedding, Flatten, Input, Reshape, Concatenate, Activation
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf 
import argparse 
from tensorflow.python.lib.io import file_io
import pickle 

def main(job_dir, input_dir):
    
    ##Setting up the path for saving logs
    logs_path = job_dir + 'logs/tensorboard'
    
    with tf.device('/device:GPU:0'):
        with open(input_dir, 'rb') as f: 
            data = pickle.load(f)
        


##Running the app
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Input Arguments
    parser.add_argument(
      '--job-dir',
      help='GCS location to write checkpoints and export models',
      required=True
    )
    
    parser.add_argument(
      '--train-data',
      help = 'training data',
      required = True
    )
    
    
    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)