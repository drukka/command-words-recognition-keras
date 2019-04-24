from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
from datetime import date
import tensorflow as tf

import data
from constants import *
from model import get_model


def save_model_callback():
    today = date.today().strftime("%Y_%m_%d")
    model_path = MODELS_DIR + today + '.h5'

    return tf.keras.callbacks.ModelCheckpoint(model_path, period=SAVE_PERIOD, save_best_only=True, verbose=VERBOSITY)


def get_validation_percentage(x_train, x_test):
    x_total = len(x_train) + len(x_test)

    validation_data_length = int((x_total * VALIDATION_PERCENTAGE) / 100)

    validation_percentage = (validation_data_length * 100) / len(x_train)

    return int(validation_percentage * 100) / (100 * 100)


def create_label_file():
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    with open(LABELS_PATH, 'w') as f:
        labels = data.get_words_index()
        for label in list(labels.keys()):
            f.write("%s\n" % label)


def main(args):
    x_train, x_test, y_train, y_test = data.get_data_set(args.force_extract)

    model = get_model(args.load_model)

    create_label_file()

    model.fit(x_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              verbose=VERBOSITY,
              callbacks=[save_model_callback()],
              validation_split=get_validation_percentage(x_train, x_test),
              shuffle=True,
              validation_freq=VALIDATION_FREQUENCY)

    if len(x_test):
        model.evaluate(x_test, y_test, verbose=VERBOSITY)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--force_extract',
        type=bool,
        default=False,
        help='If True, MFCCs will get extracted from WAV files, else it uses pre extracted features.')
    parser.add_argument(
        '--load_model',
        type=str,
        default='',
        help='Pre trained model path. If not specified, then a new model is created.')

    parsed, unparsed = parser.parse_known_args()

    main(parsed)
