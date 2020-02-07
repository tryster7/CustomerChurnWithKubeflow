import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.python.lib.io import file_io
import pathlib

import sys
import json
import pandas as pd
import os
import argparse

from sklearn.metrics import confusion_matrix

from datetime import datetime

# Helper libraries
import numpy as np

'''
This functions parses the arguments provided. In case of missing arguments, it assigns
default values to the arguments. 
'''
def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket_name',
                        type=str,
                        default='gs://kube-1122/customerchurn',
                        help='The bucket where the model has to be stored')
    parser.add_argument('--epochs',
                        type=int,
                        default=1,
                        help='Number of epochs for training the model')
    parser.add_argument('--batch_size',
                        type=int,
                        default=64,
                        help='the batch size for each epoch')
    parser.add_argument('--katib',
                        type=int,
                        default=0,
                        help='to save model or not')
    parser.add_argument('--optimizer_name',
                        type=str,
                        default='Adam',
                        help='optimizer to use in model')

    args = parser.parse_known_args()[0]
    
    assert args.epochs > 0, "Invalid epoch {} provided".format(args.epochs) 
    assert args.batch_size > 0, "Invalid batch size {} provided".format(args.batch_size)
    
    return args

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
def train(bucket_name, epochs, batch_size, katib, optimer_name):
    
    testX, testy, trainX, trainy = load_data(bucket_name)
    
    dnn = create_tfmodel(
        optimizer=tf.keras.optimizers.get(optimizer_name)
        loss='binary_crossentropy',
        metrics=['accuracy'],
        input_dim=trainX.shape[1])

    dnn.summary()

    dnn.fit(trainX, trainy, epochs=epochs, batch_size=batch_size)

    test_loss, test_acc = dnn.evaluate(testX, testy, verbose=2)
    print("accuracy={:.2f}".format(test_acc))
    print("test-loss={:.2f}".format(test_loss))

    predictions = dnn.predict_classes(testX)

    if katib == 0:
        save_tfmodel_in_gcs(bucket_name, dnn)
        create_kf_visualization(bucket_name, testy, predictions, test_acc)

'''
Saves the model as a pb file

Parameters:
----------
bucket_name:        The google bucket where the model will be exported/saved
model:              The model to be exported/saved. 
'''
def save_tfmodel_in_gcs(bucket_name, model):
    export_path = bucket_name + '/export/model/1'
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
'''
def create_tfmodel(optimizer, loss, metrics, input_dim):
    model = Sequential()
    model.add(Dense(input_dim, activation='relu', input_dim=input_dim))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer, loss, metrics)
    return model

'''
This method is used to create visualization for kubeflow. 

The function stores confusion matrix and accuracy in json format to 
/mlpipeline-ui-metadata and /mlpipeline-metrics file respectively. 

Kubeflow picks these files and displays relevant visualization on the dashboard

Parameters
-----------
bucket_name:    Name of the bucket to save confusion matrix. This location is stored in
                mlpipeline-ui-metadata
test_label:     The actual labels for the test data
predict_label:  The labels predicted by the model/classifier
test_acc:       The accuracy score for the batch predicted by the classifier

'''
def create_kf_visualization(bucket_name, test_label, predict_label, test_acc):
    metrics = {
        'metrics': [{
            'name': 'accuracy-score',
            'numberValue': str(test_acc),
            'format': "PERCENTAGE"
        }]
    }

    with file_io.FileIO('/mlpipeline-metrics.json', 'w') as f:
        json.dump(metrics, f)

    pred = pd.DataFrame(data=predict_label, columns=['predicted'])

    vocab = [0, 1]
    cm = confusion_matrix(test_label, pred['predicted'], labels=vocab)
    data = []
    for target_index, target_row in enumerate(cm):
        for predicted_index, count in enumerate(target_row):
            data.append((vocab[target_index], vocab[predicted_index], count))
    df_cm = pd.DataFrame(data, columns=['target', 'predicted', 'count'])
    cm_file = bucket_name + '/metadata/cm.csv'
    print(df_cm)
    with file_io.FileIO(cm_file, 'w') as f:
        df_cm.to_csv(f, columns=['target', 'predicted', 'count'], header=False, index=False)

    print("***************************************")
    print("Writing the confusion matrix to ", cm_file)
    metadata = {
        'outputs': [{
            'type': 'confusion_matrix',
            'format': 'csv',
            'schema': [
                {'name': 'target', 'type': 'CATEGORY'},
                {'name': 'predicted', 'type': 'CATEGORY'},
                {'name': 'count', 'type': 'NUMBER'},
            ],
            'source': cm_file,
            'labels': list(map(str, vocab)),
        }]
    }

    with file_io.FileIO('/mlpipeline-ui-metadata.json', 'w') as f:
        json.dump(metadata, f)

    return df_cm

'''
Loads the training and test data into dataframes

bucket_name:        The bucket location where the input train and test files are stored
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

    args = parse_arguments()
    print(args)
    train(args.bucket_name, int(args.epochs), int(args.batch_size), int(args.katib), args.optimizer_name)
