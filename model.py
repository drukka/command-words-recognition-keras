from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from constants import *
import data
from tensorflow.keras import layers, models


def get_model(model_path):
    if model_path:
        if not os.path.isdir(MODELS_DIR):
            raise Exception('Saved models directory with name \'' + MODELS_DIR + '\' not found!')

        model = models.load_model(MODELS_DIR + model_path, compile=False)
    else:
        model = create_model()

    build_model(model)

    return model


def build_model(model):
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])


def create_model(input_layer=None):
    model = models.Sequential()

    if input_layer:
        model.add(input_layer)

    model.add(layers.Conv2D(filters=16,
                            kernel_size=5,
                            activation='relu',
                            strides=[1, 1],
                            padding='SAME', use_bias=True))
    model.add(layers.MaxPool2D())

    model.add(layers.Conv2D(filters=32,
                            kernel_size=3,
                            activation='relu',
                            strides=[1, 1],
                            padding='SAME', use_bias=True))
    model.add(layers.MaxPool2D())
    model.add(layers.Dropout(DROPOUT_RATE))
    model.add(layers.BatchNormalization())

    model.add(layers.Conv2D(filters=64,
                            kernel_size=3,
                            activation='relu',
                            strides=[2, 2],
                            padding='SAME', use_bias=True))

    model.add(layers.Conv2D(filters=128,
                            kernel_size=3,
                            activation='relu',
                            strides=[1, 1],
                            padding='SAME', use_bias=True))
    model.add(layers.MaxPool2D())
    model.add(layers.Dropout(DROPOUT_RATE))

    model.add(layers.Flatten())
    model.add(layers.Dropout(DROPOUT_RATE))
    model.add(layers.Dense(256, activation='relu'))

    num_of_labels = len(data.get_words_list())
    model.add(layers.Dense(num_of_labels, activation='softmax'))

    return model


def print_model_architecture():
    input_layer = layers.InputLayer(input_shape=[FEATURES_COUNT, 32, 1], batch_size=BATCH_SIZE)

    model = create_model(input_layer)

    print(model.summary())
