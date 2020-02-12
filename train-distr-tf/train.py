import argparse
import sys

import pandas as pd
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# Helper libraries

'''
This functions parses the arguments provided. In case of missing arguments, it assigns
default values to the arguments. 
'''


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket_name',
                        type=str,
                        default='gs://kbc/ccc',
                        help='The bucket where the model has to be stored')
    parser.add_argument('--epochs',
                        type=int,
                        default=11,
                        help='Number of epochs for training the model')
    parser.add_argument('--batch_size',
                        type=int,
                        default=20,
                        help='the batch size for each epoch')
    parser.add_argument('--optimizer_name',
                        type=str,
                        default='Adam',
                        help='optimizer to use in model')

    return parser


'''
This method will parse the arguments passed and validate them

Parameters
----------
args            Parsed arguments

'''


def validate_arguments(args):
    assert args.epochs > 0, "Invalid epoch {} provided".format(args.epochs)
    assert args.batch_size > 0, "Invalid batch size {} provided".format(args.batch_size)


def input_fn(train_d, test_d):
    dataset = tf.data.Dataset.from_tensor_slices((train_d, test_d))
    dataset = dataset.repeat(100)
    dataset = dataset.batch(32)
    return dataset


'''
This function involves reading the data from the bucket, 
creating the model, 
training the model,
evaluating the model. 
The various performance attributes of the model, like accuracy and confusion matrix are written 
to the metadata store so that it can be visualized on kubeflow dashboard
Input arguments

Parameters 
----------
bucket_name:    The bucket and folder path where the input files are stored and the model will be exported
epochs :        Epochs the model will be trained for 
batch_size:     Batch Size (default is 128)
katib:          This flag indicates whether this current execution is driven by katib. In case it is driven 
                by katib, the metadata, model and confusion matrix need not be calculated/stored in google bucket
'''


def train(bucket_name, epochs, batch_size, optimizer_name):
    testX, testy, trainX, trainy = load_data(bucket_name)

    dnn = create_tfmodel(
        optimizer=tf.keras.optimizers.get(optimizer_name),
        loss='binary_crossentropy',
        metrics=['accuracy'],
        input_dim=trainX.shape[1])

    dnn.summary()

    config = tf.estimator.RunConfig()

    keras_estimator = tf.keras.estimator.model_to_estimator(
        keras_model=dnn, config=config, model_dir=bucket_name + "/dist/checkpoint")

    tf.estimator.train_and_evaluate(
        keras_estimator,
        train_spec=tf.estimator.TrainSpec(input_fn=input_fn(trainX, trainy)),
        eval_spec=tf.estimator.EvalSpec(input_fn=input_fn(testX, testy)))

    save_tfmodel_in_gcs(bucket_name, dnn)


'''
Saves the model as a pb file

Parameters:
----------
bucket_name:        The google bucket where the model will be exported/saved
model:              The model to be exported/saved. 
'''


def save_tfmodel_in_gcs(bucket_name, model):
    export_path = bucket_name + 'dist/export/model/2'
    tf.saved_model.save(model, export_dir=export_path)


'''
Creates a tensor flow Dense Neural Networks Model. 
A very basic model is created. 

Parameters 
----------

optimizer:      The optimizer to be used for training the model
loss:           The loss function to be used for optimizing the model
metrics:        Metrics to be used for optimization
input_dim:      The dimension of the input layer. Or we can say the number of attributes in the data file

Returns
---------
model           The model instance created

'''


def create_tfmodel(optimizer, loss, metrics, input_dim):
    model = Sequential()
    model.add(Dense(input_dim, activation='relu', input_dim=input_dim))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer, loss, metrics)
    return model


'''
Loads the training and test data into dataframes

Parameters
----------
bucket_name:        The bucket location where the input train and test files are stored

Returns
--------

trainX, trainy, testX, testy   Parsed dataframes for train data, train labels, test data, test labels
'''


def load_data(bucket_name):
    # load dataset
    train_file = bucket_name + '/output/train.csv'
    test_file = bucket_name + '/output/test.csv'
    train_labels = bucket_name + '/output/train_label.csv'
    test_labels = bucket_name + '/output/test_label.csv'

    trainDF = pd.read_csv(train_file)
    trainLabelDF = pd.read_csv(train_labels)
    testX = pd.read_csv(test_file)
    testy = pd.read_csv(test_labels)
    trainX = trainDF.drop(trainDF.columns[0], axis=1)
    trainy = trainLabelDF.drop(trainLabelDF.columns[0], axis=1)
    testy = testy.drop(testy.columns[0], axis=1)
    testX = testX.drop(testX.columns[0], axis=1)

    return testX, testy, trainX, trainy


'''
main function 
'''
if __name__ == '__main__':
    print("The arguments are ", str(sys.argv))
    if len(sys.argv) < 1:
        print("Usage: train bucket-name epochs batch-size katib optimizer")
        sys.exit(-1)

    parser = parse_arguments()
    args = parser.parse_known_args()[0]
    validate_arguments(args)
    print(args)
    train(args.bucket_name, int(args.epochs), int(args.batch_size), args.optimizer_name)
