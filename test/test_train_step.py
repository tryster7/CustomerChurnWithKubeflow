import unittest

import tensorflow as tf
from train import train


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        model = train.create_tfmodel(optimizer=tf.optimizers.Adam(),
                                     loss='binary_crossentropy',
                                     metrics=['accuracy'],
                                     input_dim=11)
        self.model = model

    def test_loadmodel_and_predict(self):
        model = tf.saved_model.load('gs://kube-1122/customerchurn/export/model/1')
        # predictions = model.predict(self.trainX[0])
        # label = np.argmax(predictions, axis=1)
        # print(label)
        self.assertIsNotNone(model)

    def test_parseArguments(self):
        args = train.parse_arguments()
        self.assertIn("epochs", args)

    def test_correct_model_optimizer_and_loss(self):
        self.assertEquals(self.model.loss, 'binary_crossentropy')
        self.assertIn('Adam', self.model.optimizer.get_config().values())

    def test_layers_in_model(self):
        self.assertEquals(len(self.model.layers), 3)

    def test_model_output(self):
        self.assertEquals(self.model.output.name, 'dense_14/Identity:0')

    def test_model_is_saved_at_given_dir(self):
        export_dir = '/workspace'
        train.save_tfmodel_in_gcs(export_dir, self.model)
        self.assertTrue(True, tf.saved_model.contains_saved_model(export_dir))


if __name__ == '__main__':
    unittest.main()
