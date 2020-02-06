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

from kubeflow.metadata import metadata
from datetime import datetime
from uuid import uuid4

# Helper libraries
import numpy as np

METADATA_STORE_HOST = "metadata-grpc-service.kubeflow"  # default DNS of Kubeflow Metadata gRPC serivce.
METADATA_STORE_PORT = 8080


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

    args = parser.parse_known_args()[0]
    return args


def train(bucket_name, epochs=10, batch_size=128):
    
    # exec = create_metadata_execution()
    
    testX, testy, trainX, trainy = load_and_normalize_data(bucket_name)
    dnn = create_tfmodel(
        optimizer=tf.optimizers.Adam(),
        loss='binary_crossentropy',
        metrics=['accuracy'],
        input_dim=trainX.shape[1])

    dnn.summary()
    
    dnn.fit(trainX, trainy, epochs=epochs, batch_size=batch_size)
    
    # model = save_model_metadata(exec, batch_size, epochs)

    test_loss, test_acc = dnn.evaluate(testX, testy, verbose=2)
    print("accuracy={}".format(test_acc))
    print("test-loss={}".format(test_loss))
    
    predictions = dnn.predict_classes(testX)
    
    # save_metric_metadata(exec, model, test_acc, test_loss)

    save_tfmodel_in_gcs(bucket_name, dnn)
    
    create_kf_visualization(bucket_name, testy, predictions, test_acc)


def save_tfmodel_in_gcs(bucket_name, model):
    export_path = bucket_name + '/export/model/1'
    tf.saved_model.save(model, export_dir=export_path)


def create_tfmodel(optimizer, loss, metrics, input_dim):
    model = Sequential()
    model.add(Dense(input_dim, activation='relu', input_dim=input_dim))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer, loss, metrics)
    return model


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

    test = test_label.to_frame('target')
    pred = pd.DataFrame(data=predict_label, columns=['predicted'])

    vocab = list(test['target'].unique())
    cm = confusion_matrix(test['target'], pred['predicted'], labels=vocab)
    data = []
    for target_index, target_row in enumerate(cm):
        for predicted_index, count in enumerate(target_row):
            data.append((vocab[target_index], vocab[predicted_index], count))
    df_cm = pd.DataFrame(data, columns=['target', 'predicted', 'count'])
    cm_file = bucket_name + '/metadata/cm.csv'

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


def save_metric_metadata(exec, model, test_acc, test_loss):
    # Save evaluation
    metrics = exec.log_output(
        metadata.Metrics(
            name="Customer_Churn_Evaluation",
            description="Predicting customer churn from given data",
            owner="demo@kubeflow.org",
            uri="gs://kube-1122/customerchurn/metadata/cm.csv",
            model_id=str(model.id),
            metrics_type=metadata.Metrics.VALIDATION,
            values={"accuracy": str(test_acc),
                    "test_loss": str(test_loss)},
            labels={"mylabel": "l1"}))
    print("Metrics id is %s" % metrics.id)


def save_model_metadata(exec, batch_size, epochs):
    # Save model;
    model_version = "model_version_" + str(uuid4())
    model = exec.log_output(
        metadata.Model(
            name="Customer_Churn",
            description="model to predict customer churn",
            owner="demo@kubeflow.org",
            uri="gs://kube-1122/customerchurn/export/model/1/saved_model.pb",
            model_type="DNN",
            training_framework={
                "name": "tensorflow",
                "version": "v2.0"
            },
            hyperparameters={
                "learning_rate": 0.5,
                "layers": [11, 128, 1],
                "epochs": str(epochs),
                "batch-size": str(batch_size),
                "early_stop": True
            },
            version=model_version,
            labels={"tag": "train"}))
    print(model)
    print("\nModel id is {0.id} and version is {0.version}".format(model))
    return model


def create_metadata_execution():
    global metadata
    # Create Metadata Workspace and a Exec to log details
    ws1 = metadata.Workspace(
        # Connect to metadata service in namespace kubeflow in k8s cluster.
        store=metadata.Store(grpc_host=METADATA_STORE_HOST, grpc_port=METADATA_STORE_PORT),
        name="Customer Churn workspace",
        description="a workspace for training customer churn model",
        labels={"n1": "v1"})
    run1 = metadata.Run(
        workspace=ws1,
        name="run-" + datetime.utcnow().isoformat("T"),
        description="a run in ws_1")
    exec = metadata.Execution(
        name="execution" + datetime.utcnow().isoformat("T"),
        workspace=ws1,
        run=run1,
        description="execution example")
    print("An execution was created with id %s" % exec.id)
    return exec


def load_and_normalize_data(bucket_name):
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


if __name__ == '__main__':
    print("The arguments are ", str(sys.argv))
    if len(sys.argv) < 1:
        print("Usage: train bucket-name epochs batch-size")
        sys.exit(-1)

    args = parse_arguments()
    print(args)
    train(args.bucket_name, int(args.epochs), int(args.batch_size))
