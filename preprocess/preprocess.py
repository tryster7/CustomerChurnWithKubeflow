from __future__ import absolute_import, division, print_function, unicode_literals

import argparse

import pandas as pd
from google.cloud import storage
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bucket_name',
                        type=str,
                        default='gs://kube-1122/customerchurn',
                        help='The bucket where the output has to be stored')

    parser.add_argument('--input_file',
                        type=str,
                        default='gs://kube-1122/customerchurn/input/train.csv',
                        help='The input file required to process the data')

    parser.add_argument('--output_folder',
                        type=str,
                        default='output',
                        help='The output folder for the processed data')

    args = parser.parse_known_args()[0]
    return args


def preprocess(input_file, output_folder, bucket_name):
    input_file = bucket_name + '/' + input_file

    raw_dataset = pd.read_csv(input_file)
    dataset = raw_dataset.copy()
    dataset = dataset.dropna()
    X = dataset.drop(labels=['CustomerId', 'Surname', 'RowNumber', 'Exited'], axis=1)
    y = dataset['Exited']
    label1 = LabelEncoder()
    X['Geography'] = label1.fit_transform(X['Geography'])
    label = LabelEncoder()
    X['Gender'] = label.fit_transform(X['Gender'])
    X = pd.get_dummies(X, drop_first=True, columns=['Geography'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    train_stats = X_train.describe()
    train_stats = train_stats.transpose()

    print("normalize data...")
    normed_train_data = (norm(X_train, train_stats))
    normed_test_data = (norm(X_test, train_stats))

    train_output_file = bucket_name + '/' + output_folder + '/train.csv'
    train_label_file = bucket_name + '/' + output_folder + '/train_label.csv'
    test_output_file = bucket_name + '/' + output_folder + '/test.csv'
    test_label_file = bucket_name + '/' + output_folder + '/test_label.csv'

    normed_train_data.to_csv(train_output_file)
    normed_test_data.to_csv(test_output_file)
    y_train.to_csv(train_label_file)
    y_test.to_csv(test_label_file)


def norm(x, train_stats):
    return (x - train_stats['mean']) / train_stats['std']


if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    preprocess(args.input_file, args.output_folder, args.bucket_name)
