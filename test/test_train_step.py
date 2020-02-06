import unittest

import pandas as pd
import tensorflow as tf
from train import train
from pathlib import Path


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        model = train.create_tfmodel(optimizer=tf.optimizers.Adam(),
                                     loss='binary_crossentropy',
                                     metrics=['accuracy'],
                                     input_dim=11)
        self.model = model
        self.bucket = 'gs://kube-1122/customerchurn'
        test_label = pd.Series([1, 0, 0, 1, 0, 1, 1, 1])
        self.testy = test_label
        pred_label = [1, 0, 1, 1, 0, 1, 0, 1]
        self.pred = pred_label

    def test_loadmodel_and_predict(self):
        model = tf.saved_model.load('gs://kube-1122/customerchurn/export/model/1')
        self.assertIsNotNone(model)

    def test_parseArguments(self):
        args = train.parse_arguments()
        self.assertIn("epochs", args)
        self.assertIn("batch_size", args)

    def test_correct_model_optimizer_and_loss(self):
        self.assertEquals(self.model.loss, 'binary_crossentropy')
        self.assertIn('Adam', self.model.optimizer.get_config().values())

    def test_layers_in_model(self):
        self.assertEquals(len(self.model.layers), 3)

    def test_model_is_saved_at_given_dir(self):
        export_dir = '/workspace'
        train.save_tfmodel_in_gcs(export_dir, self.model)
        self.assertTrue(True, tf.saved_model.contains_saved_model(export_dir))

    def test_load_normalize_data(self):
        testX, testy, trainX, trainy = train.load_data(self.bucket)
        self.assertListEqual(list(testy.iloc[:, 0].unique()), list(trainy.iloc[:, 0].unique()))

    def test_create_visualization(self):
        df_cm = train.create_kf_visualization(self.bucket, self.testy, self.pred, 0.85)
        self.assertTrue(True, Path(self.bucket + '/metadata/cm.csv').is_file())
        self.assertTrue(True, Path('/mlpipeline-metrics.json').is_file())
        self.assertIsNotNone(df_cm)


if __name__ == '__main__':
    unittest.main()
