from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tensorflow as tf
from shutil import rmtree
from librosa.feature import mfcc
import numpy as np
from tensorflow.io import gfile
import uuid

from constants import *


def read_dir():
    if not os.path.isdir(SOUNDS_DIR):
        raise Exception('Sound directory with name \'' + SOUNDS_DIR + '\' not found!')

    data = []

    for word in WANTED_WORDS:
        word_dir = SOUNDS_DIR + word
        if not os.path.isdir(word_dir):
            raise Exception('Sounds directory for \'' + word + '\' not found at ' + word_dir + '!')
git add 
        search_path = os.path.join(word_dir, '*.wav')
        for wav_path in gfile.glob(search_path):
            data.append({'word': word, 'file': wav_path})

    return data


def get_features():
    features = []

    print('Extracting MFCC features from WAV files')
    for data in read_dir():
        mfcc_feat = get_MFCC(data['file'])

        features.append({'data': mfcc_feat, 'label': data['word']})

    save_features(features)


def get_MFCC(wav_path):
    wav_loader = tf.io.read_file(wav_path)
    wav_decoded = tf.audio.decode_wav(wav_loader, desired_channels=1).audio[:DESIRED_SAMPLES]

    padding = tf.constant([[DESIRED_SAMPLES - len(wav_decoded), 0], [0, 0]])
    audio_data = tf.pad(wav_decoded, padding)
    reshaped_data = np.array(tf.reshape(audio_data, (SAMPLE_RATE,)))

    feature = mfcc(reshaped_data, SAMPLE_RATE, n_mfcc=FEATURES_COUNT)

    return tf.expand_dims(feature, -1)


def save_features(features):
    if os.path.isdir(MFCCS_DIR):
        rmtree(MFCCS_DIR)

    print('Saving MFCC features as tensor files')
    for feature in features:
        filename = uuid.uuid4().hex + '.mfcc'
        file_path = MFCCS_DIR + feature['label'] + '/' + filename

        tensor = tf.dtypes.cast(feature['data'], dtype=tf.float32)

        tf.io.write_file(file_path, tf.io.serialize_tensor(tensor))
