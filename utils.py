import csv
import numpy as np
import os
import pandas as pd
import scipy.io
import skvideo.io
import tensorflow as tf

from models.echo import *
from models.ecg import *


# data loaders

def load_data_echonet():

    """
    EchoNet-Dynamic data.
    Returns a dictionary with a mapping from subject ids to .npz-file paths which contain a frames and a times array.
    """

    video_cache_folder = './cache/EchoNet-Dynamic/Videos'
    if not os.path.exists(video_cache_folder):
        os.makedirs(video_cache_folder)

    data_info = pd.read_csv('./data/EchoNet-Dynamic/FileList.csv')
    data_info['globalID'] = data_info['FileName'].apply(lambda s: s[:-4]).astype('string')
    data_info.set_index('globalID', inplace=True)

    files = dict()
    for index, row in data_info.iterrows():
        
        filepath = './data/EchoNet-Dynamic/Videos/' + index + '.avi'
        filepath_cached = video_cache_folder + '/' + index + '.npz'

        # cache frames and times, if not existing
        if not os.path.exists(filepath_cached):

            # load from dataset
            frames = skvideo.io.vread(filepath)
            frames = [frame[:, :, 0] for frame in frames]
            
            # times
            time_base = 1/data_info.loc[index]['FPS']
            times = [i*time_base for i in range(len(frames))]

            # cache data
            np.savez(filepath_cached, frames=frames, times=times)

        files[index] = filepath_cached

    return data_info, files

def get_physionet_data(split='training'):

    # load label file
    data_path = f'data/physionet.org/files/challenge-2017/1.0.0/{split}/'
    if split == 'training':
        label_file_path = 'data/physionet.org/files/challenge-2017/1.0.0/training/REFERENCE-v3.csv'
    elif split == 'validation':
        label_file_path = 'data/physionet.org/files/challenge-2017/1.0.0/validation/REFERENCE.csv'
    
    with open(label_file_path, newline='') as label_csv_file:
        csv_reader = csv.reader(label_csv_file, delimiter=',')
        labels = {record_id: label for record_id, label in csv_reader}

    # load measurements
    data = dict()
    if split == 'training':
        for i in range(9):
            for filename in os.listdir(data_path + 'A0' + str(i)):
                if filename.endswith('.mat'):
                    mat_data = scipy.io.loadmat(data_path + 'A0' + str(i) + '/' + filename)
                    data[filename[:-4]] = {
                        'measurements': mat_data['val'][0],
                        'frequency': 300
                    }
    elif split == 'validation':
        for filename in os.listdir(data_path):
            if filename.endswith('.mat'):
                mat_data = scipy.io.loadmat(data_path + filename)
                data[filename[:-4]] = {
                    'measurements': mat_data['val'][0],
                    'frequency': 300
                }
    return labels, data


# TensorFlow model loaders

def load_echonet_dynamic_model(split):

    # initialise model
    weights_path = './trained_models/echonet_dynamic_' + str(split)
    model = EchocardioModel(latent_space_dim=128, batch_size=32, hidden_dim=128)
    model((tf.ragged.constant([[0.0]], dtype='float32'), tf.ragged.constant([[np.full((112, 112), 0.5)]], inner_shape=(112, 112), dtype='float32')))

    # load weights
    model.load_weights(weights_path).expect_partial()

    return model

def load_physionet_ecg_model():

    weights_path = './trained_models/physionet'
    model = ECGModel(latent_space_dim=8, batch_size=64, hidden_dim=128, learning_rate=2e-4)
    model((tf.ragged.constant([[0.0]], dtype='float32'), tf.ragged.constant([[0.0]], dtype='float32')))

    # load weights
    model.load_weights(weights_path).expect_partial()

    return model
