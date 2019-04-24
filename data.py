from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import tensorflow as tf
from tensorflow.io import gfile
from sklearn.model_selection import train_test_split
import tarfile
import zipfile

from constants import *
import wav2MFCC


def extract_dataset(path):
    print('Extracting...')
    if path.endswith('.zip'):
        zipfile.ZipFile(path, 'r').extractall(SOUNDS_DIR)
    elif path.endswith('.tar.gz') or path.endswith('.tgz'):
        tarfile.open(path, 'r:gz').extractall(SOUNDS_DIR)
    elif path.endswith('.tar.bz2') or path.endswith('.tbz'):
        tarfile.open(path, 'r:bz2').extractall(SOUNDS_DIR)
    else:
        raise Exception('Could not extract \'' + path + '\' as no appropriate extractor is found')
    print('Extract done!')


def download_dataset():
    if not os.path.exists(SOUNDS_DIR):
        os.makedirs(SOUNDS_DIR)

    filename = DATASET_URL.split('/')[-1]
    filepath = os.path.join(SOUNDS_DIR, filename)
    absolute_path = os.path.abspath(filepath)

    if not os.path.exists(filepath) and not os.listdir(SOUNDS_DIR):
        tf.keras.utils.get_file(absolute_path, origin=DATASET_URL, extract=False)
        extract_dataset(absolute_path)
        tf.io.gfile.remove(absolute_path)


def get_words_list():
    return WANTED_WORDS


def get_words_index():
    words_index = {}
    for index, wanted_word in enumerate(get_words_list()):
        words_index[wanted_word] = index

    return words_index


def check_input_availability(all_words):
    if not all_words:
        raise Exception('No .wavs found!')
    for index, wanted_word in enumerate(WANTED_WORDS):
        if wanted_word not in all_words:
            raise Exception(
                'Expected to find ' + wanted_word + ' in labels but only found ' + ', '.join(all_words.keys()))


def get_data_set(extract):
    if extract:
        download_dataset()
        wav2MFCC.get_features()
    else:
        if not os.path.isdir(MFCCS_DIR):
            raise Exception('MFFC directory with name \'' + MFCCS_DIR + '\' not found! Please use --force_extract=True')

    print('Loading MFCC features into memory')
    all_words = {}
    x = []
    y = []
    indexes = get_words_index()

    search_path = os.path.join(MFCCS_DIR, '*', '*.mfcc')
    for wav_path in gfile.glob(search_path):
        _, word = os.path.split(os.path.dirname(wav_path))

        word = word.lower()
        all_words[word] = True

        tensor = tf.io.parse_tensor(tf.io.read_file(wav_path), tf.float32)

        x.append(tensor)
        y.append(indexes[word])

    check_input_availability(all_words)

    return split_data(x, y)


def split_data(x, y):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=TESING_PERCENTAGE / 100)

    return tf.convert_to_tensor(x_train), tf.convert_to_tensor(x_test), tf.convert_to_tensor(
        y_train), tf.convert_to_tensor(y_test),
