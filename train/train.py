import argparse
import json
import sys
from datetime import datetime
from uuid import uuid4

import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.python.lib.io import file_io

from elasticsearch import Elasticsearch

'''
Functions provide elasticsearch writing capabilities to train step , later to seperate it out
'''

def connect_elasticsearch(host='127.0.0.1', port=9200):
    _es = None
    _es = Elasticsearch([{'host': host, 'port': port}])
    print(_es)
    if _es.ping():
        print('Yay Connect')
    else:
        print('Could not connect to ', host)
    return _es


def create_index(es_object, index_name='kf_metadata'):
    created = False
    # index settings
    settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "dataset": {
                    "properties": {
                        "description": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        },
                        "name": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        },
                        "owner": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        },
                        "query": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        },
                        "uri": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        },
                        "version": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        }
                    }
                },
                "description": {
                    "type": "text",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                },
                "metric": {
                    "properties": {
                        "data": {
                            "properties": {
                                "name": {
                                    "type": "text",
                                    "fields": {
                                        "keyword": {
                                            "type": "keyword",
                                            "ignore_above": 256
                                        }
                                    }
                                },
                                "value": {
                                    "type": "text",
                                    "fields": {
                                        "keyword": {
                                            "type": "keyword",
                                            "ignore_above": 256
                                        }
                                    }
                                }
                            }
                        },
                        "description": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        },
                        "labels": {
                            "properties": {
                                "name": {
                                    "type": "text",
                                    "fields": {
                                        "keyword": {
                                            "type": "keyword",
                                            "ignore_above": 256
                                        }
                                    }
                                },
                                "value": {
                                    "type": "text",
                                    "fields": {
                                        "keyword": {
                                            "type": "keyword",
                                            "ignore_above": 256
                                        }
                                    }
                                }
                            }
                        },
                        "metric_type": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        },
                        "name": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        },
                        "owner": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        },
                        "uri": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        },
                        "version": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        }
                    }
                },
                "model": {
                    "properties": {
                        "description": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        },
                        "hyperparamters": {
                            "properties": {
                                "early_stop": {
                                    "type": "boolean"
                                },
                                "epochs": {
                                    "type": "long"
                                },
                                "layers": {
                                    "type": "long"
                                },
                                "learning_rate": {
                                    "type": "float"
                                },
                                "optimizer": {
                                    "type": "text",
                                    "fields": {
                                        "keyword": {
                                            "type": "keyword",
                                            "ignore_above": 256
                                        }
                                    }
                                },
                                "attributes": {
                                    "properties": {
                                        "name": {
                                            "type": "text",
                                            "fields": {
                                                "keyword": {
                                                    "type": "keyword",
                                                    "ignore_above": 256
                                                }
                                            }
                                        },
                                        "value": {
                                            "type": "text",
                                            "fields": {
                                                "keyword": {
                                                    "type": "keyword",
                                                    "ignore_above": 256
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        "labels": {
                            "properties": {
                                "name": {
                                    "type": "text",
                                    "fields": {
                                        "keyword": {
                                            "type": "keyword",
                                            "ignore_above": 256
                                        }
                                    }
                                },
                                "value": {
                                    "type": "text",
                                    "fields": {
                                        "keyword": {
                                            "type": "keyword",
                                            "ignore_above": 256
                                        }
                                    }
                                }
                            }
                        },
                        "model_type": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        },
                        "name": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        },
                        "owner": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        },
                        "training_framework": {
                            "properties": {
                                "name": {
                                    "type": "text",
                                    "fields": {
                                        "keyword": {
                                            "type": "keyword",
                                            "ignore_above": 256
                                        }
                                    }
                                },
                                "version": {
                                    "type": "text",
                                    "fields": {
                                        "keyword": {
                                            "type": "keyword",
                                            "ignore_above": 256
                                        }
                                    }
                                }
                            }
                        },
                        "uri": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        },
                        "version": {
                            "type": "text",
                            "fields": {
                                "keyword": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                }
                            }
                        }
                    }
                },
                "name": {
                    "type": "text",
                    "fields": {
                        "keyword": {
                            "type": "keyword",
                            "ignore_above": 256
                        }
                    }
                },
                "timestamp": {
                    "type": "date"
                }
            }
        }
    }

    try:
        if not es_object.indices.exists(index_name):
            # Ignore 400 means to ignore "Index Already Exist" error.
            es_object.indices.create(index=index_name, body=settings)
            print('Successfully Created Index')
            created = True
    except Exception as ex:
        print(str(ex))
    finally:
        return created


'''
==================
USAGE
==================
search_object = {'query': {'match': {'gender': 'M'}}}
search_object = {'_source': ['balance'], 'query': {'range': {'balance': {'gte': 40000}}}}
qry_result = search(es, 'bank', json.dumps(search_object))
'''


def search(es_object, index_name, query):
    res = es_object.search(index=index_name, body=query)
    return res


def store_record(elastic_object, index_name, record):
    try:
        outcome = elastic_object.index(index=index_name, body=record)
        return outcome
    except Exception as ex:
        print('Error in indexing data')
        print(str(ex))

'''
=======END OF ELASTICSEARCH FUNCTIONS==========
'''

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
    parser.add_argument('--enable_metadata',
                        type=int,
                        default=1,
                        help='to write metadata to elasticsearch')
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


def train(bucket_name, epochs, batch_size, katib, enable_metadata, optimizer_name):
    testX, testy, trainX, trainy = load_data(bucket_name)

    dnn = create_tfmodel(
        optimizer=tf.keras.optimizers.get(optimizer_name),
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
        conf_matrix = create_kf_visualization(bucket_name, testy, predictions, test_acc)

    # connect to elasticsearch and write metadata
    if enable_metadata == 1:
        # prepare metadata
        METADATA = {
            "timestamp": datetime.utcnow().isoformat("T"),
            "name": "Customer Churn execution",
            "description": "CustomerChurn model execution",
            "dataset": {
                "name": "customer_churn",
                "description": "customer churn",
                "uri": "gs://kbc/ccc/output/train.csv",
                "version": "dataset-" + str(uuid4()),
                "owner": "Demo",
                "query": ""
            },
            "model": {
                "name": "CustomerChurn DNN",
                "description": "Customer Churn DNN",
                "uri": "gs://kbc/ccc/export/model",
                "version": "model-" + str(uuid4()),
                "owner": "Demo",
                "model_type": "DNN",
                "training_framework": {
                    "name": "tensorflow",
                    "version": "2.0"
                },
                "hyperparamters": {
                    "learning_rate": 0.5,
                    "layers": [11, 128, 1],
                    "early_stop": True,
                    "epochs": epochs,
                    "optimizer": optimizer_name,
                    "attributes": [{
                        "name": "batch_size",
                        "value": batch_size
                    }]
                },
                "labels": [{
                    "name": "env",
                    "value": "dev"
                }]
            },
            "metric": {
                "name": "Customer Churn evaluation metric",
                "description": "",
                "uri": "/metadata/cm.csv",
                "version": "metric-" + str(uuid4()),
                "owner": "Demo",
                "metric_type": "Validation",
                "data": [{
                    "name": "accuracy",
                    "value": [str(test_acc)]
                }, {
                    "name": "test loss",
                    "value": [str(test_loss)]
                }, {
                    "name": "confusion_matrix",
                    "value": [conf_matrix.to_string()]
                }],
                "labels": [{
                    "name": "env",
                    "value": "demo"
                }]
            }
        }

        write_metadata_to_es(metadata=METADATA)


def write_metadata_to_es(metadata):
    es_index = 'kf_metadata'
    _es = connect_elasticsearch(host='146.148.57.195', port=9200)
    create_index(_es, index_name=es_index)
    result = store_record(elastic_object=_es, index_name=es_index, record=metadata)
    print("Metadata saved with result ", result)


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

Returns
--------
df_cm           Confusion Matrix dataframe

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
    train(args.bucket_name, int(args.epochs), int(args.batch_size), int(args.katib), int(args.enable_metadata),
          args.optimizer_name)
