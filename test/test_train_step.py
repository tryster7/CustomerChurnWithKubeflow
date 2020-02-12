import unittest

import tensorflow as tf

from train import train


class MyTestCase(unittest.TestCase):
   
    """
    This method will setup mock data for running different test cases
    """
    def setUp(self) -> None:
        # create a dummy model for test purpose
        model = train.create_tfmodel(optimizer=tf.optimizers.Adam(),
                                     loss='binary_crossentropy',
                                     metrics=['accuracy'],
                                     input_dim=11)
        self.model = model
        # GS base path for i/o files
        self.bucket = 'gs://kbc/ccc'
        # GS path for saving model
        self.model_path = 'gs://kbc/ccc/test'
        # parses argument to run train step
        self.parser = train.parse_arguments()

    '''
    This test case saves model in a test path and then loads the model from same path and inspects it
    '''
    def test_save_and_load_model(self):
        # save the model in given model_path
        train.save_tfmodel_in_gcs(self.model_path, self.model)
        # assert that saved model is recognized as valid file
        self.assertTrue(True,tf.saved_model.contains_saved_model(self.model_path))
        # load the model back from the model_path
        model = tf.saved_model.load(self.model_path+'/export/model/1')
        # do a basic test on the model
        print('model signature keys=', list(model.signatures.keys()))
        self.assertIsNotNone(list(model.signatures.keys()))
      
    '''
    This test case checks for passed arguments to the step
    '''
    def test_validate_arguments(self):
        args = self.parser.parse_args(['--epochs', '-1', '--batch_size', '-2'])
        with self.assertRaises(AssertionError):
            train.validate_arguments(args)

    '''
     This test case will check various attributes of the created model
     It will check the optimizer name and loss function used to create the model
    '''
    def test_model_optimizer_and_loss(self):
        self.assertEquals(self.model.loss, 'binary_crossentropy')
        self.assertIn('Adam', self.model.optimizer.get_config().values())

    '''
    This test case check the total layers in model
    '''
    def test_layers_in_model(self):
        self.assertEquals(len(self.model.layers), 3)

    '''
    This test case will test if model saved to a tmp directory is valid format
    '''
    def test_model_is_saved_at_given_dir(self):
        train.save_tfmodel_in_gcs(self.model_path, self.model)
        self.assertTrue(True, tf.saved_model.contains_saved_model(self.model_path))

    '''
    This test case will check the loading of the train/test data into the train step
    It checks if the unique values in test labels and predict labels are same 
    '''
    def test_load_normalize_data(self):
        testX, testy, trainX, trainy = train.load_data(self.bucket)
        self.assertListEqual(list(testy.iloc[:, 0].unique()), list(trainy.iloc[:, 0].unique()))


if __name__ == '__main__':
    unittest.main()
