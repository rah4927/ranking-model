#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 20:35:16 2019

@author: rah4927
"""


import tensorflow as tf 
import argparse 
import pickle 
from tensorflow.python.lib.io import file_io

def main(job_dir, train_data, write_data):
    
    ##Setting up the path for saving logs
    logs_path = job_dir + 'logs/tensorboard'
    
    with tf.device('/device:GPU:0'):
        with file_io.FileIO(train_data,'rb') as f:
             data = pickle.load(f)
        with file_io.FileIO(write_data,'wb+') as f:
             pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
             
             
        


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
    
    parser.add_argument(
      '--write-data',
      help = 'write data',
      required = True
    )
    
    
    args = parser.parse_args()
    arguments = args.__dict__

    main(**arguments)
