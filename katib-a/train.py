import tensorflow as tf
from tensorflow import keras
from tensorflow.python.lib.io import file_io
import pathlib
import sys
import json
import os
import argparse

from tensorflow.python.platform import gfile

from keras.datasets import fashion_mnist

# Helper libraries
import numpy as np

ARGS = None

"""
parse the arguments passed while running the train code
"""
def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--momentum',
                        type=float,
                        default='SGD',
                        help='Number of epochs for training the model')

    parser.add_argument('--optimizer_name',
                        type=str,
                        default='SGD',
                        help='Number of epochs for training the model')

    parser.add_argument('--epochs',
                        type=int,
                        default=5,
                        help='Number of epochs for training the model')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='the batch size for each epoch')
    parser.add_argument('--learning_rate',
                        type=float,
                        default=0.001,
                        help='the batch size for each epoch')

    parser.add_argument('--log_dir',
                        type=str,
                        default='/tmp/logs',
                        help='Summaries log directory')

    return parser


def train():
    print(ARGS.log_dir)
    if gfile.Exists(ARGS.log_dir):
        gfile.DeleteRecursively(ARGS.log_dir)
    gfile.MakeDirs(ARGS.log_dir)

    log_file = ARGS.log_dir + '/train'
    print("The log file is ", log_file)

    # load dataset
    (trainX, trainy), (testX, testy) = fashion_mnist.load_data()

    # Data Normalization - Dividing by 255 as the maximum possible value

    trainX = trainX / 255

    testX = testX / 255

    trainX = trainX.reshape(trainX.shape[0], 28, 28, 1)

    testX = testX.reshape(testX.shape[0], 28, 28, 1)

    cnn = tf.keras.models.Sequential()

    cnn.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

    cnn.add(tf.keras.layers.MaxPooling2D(2, 2))

    cnn.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))

    cnn.add(tf.keras.layers.Flatten())

    cnn.add(tf.keras.layers.Dense(64, activation='relu'))

    cnn.add(tf.keras.layers.Dense(10, activation='softmax'))

    print(ARGS)

    optimizer = tf.keras.optimizers.get(ARGS.optimizer_name)

    optimizer.learning_rate = ARGS.learning_rate

    optimizer.momentum = ARGS.momentum

    print(optimizer)

    cnn.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    cnn.summary()

    cnn.fit(trainX, trainy, epochs=ARGS.epochs, batch_size=ARGS.batch_size)

    test_loss, test_acc = cnn.evaluate(testX, testy, verbose=2)

    writer = tf.summary.create_file_writer(log_file)

    with writer.as_default():
        tf.summary.scalar('accuracy', test_acc, step=1)
        writer.flush()

    print("accuracy={}".format(test_acc))


if __name__ == '__main__':
    print("The arguments are ", str(sys.argv))
    parser = parse_arguments()
    ARGS, unknown_args = parser.parse_known_args()
    print(ARGS)
    print("Unknown arguments are ", unknown_args)
    train()
