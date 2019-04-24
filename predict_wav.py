from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
import tensorflow as tf

from model import get_model
from wav2MFCC import get_MFCC

from constants import *


def get_labels(path):
    if not os.path.exists(path):
        raise Exception('Labels file not exists at \'--labels_path=' + path + '\'.')

    with open(path) as f:
        labels = list(filter(None, f.read().splitlines()))

    return labels


def main(args):
    model = get_model(args.load_model)

    labels = get_labels(args.labels_path)

    if not args.wav_path:
        raise Exception('Please specify the \'--wav_path=./file.wav\' of your WAV sound file')

    wav_mfcc = get_MFCC(args.wav_path)

    d = tf.expand_dims(wav_mfcc, 0)
    predictions = model.predict(d)[0]

    top_k = tf.math.top_k(predictions, args.num_of_predictions)
    for (index, value) in zip(top_k.indices, top_k.values):
        human_string = labels[index]
        score = value * 100
        print('%s (%3.2f%%)' % (human_string, score))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--load_model',
        type=str,
        default='',
        help='Pre trained model path. If not specified, then a new model is created.')
    parser.add_argument(
        '--wav_path',
        type=str,
        default='',
        help='WAV sound file path.')
    parser.add_argument(
        '--labels_path',
        type=str,
        default=LABELS_PATH,
        help='Labels txt file path.')
    parser.add_argument(
        '--num_of_predictions',
        type=int,
        default=1,
        help='Number of prediction to print.')

    parsed, unparsed = parser.parse_known_args()

    main(parsed)
