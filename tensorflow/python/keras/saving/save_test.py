# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for Keras model saving code."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from absl.testing import parameterized
import numpy as np

from tensorflow.python import keras
from tensorflow.python.framework import test_util
from tensorflow.python.keras import combinations
from tensorflow.python.keras import losses
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import sequential
from tensorflow.python.keras.layers import core
from tensorflow.python.keras.saving import save
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.platform import test
from tensorflow.python.saved_model import loader_impl

if sys.version_info >= (3, 6):
    import pathlib  # pylint:disable=g-import-not-at-top
try:
    import h5py  # pylint:disable=g-import-not-at-top
except ImportError:
    h5py = None


class TestSaveModel(test.TestCase, parameterized.TestCase):

    def setUp(self):
        super(TestSaveModel, self).setUp()
        self.model = testing_utils.get_small_sequential_mlp(1, 2, 3)
        self.subclassed_model = testing_utils.get_small_subclass_mlp(1, 2)

    def assert_h5_format(self, path):
        if h5py is not None:
            self.assertTrue(h5py.is_hdf5(path),
                            'Model saved at path {} is not a valid hdf5 file.'
                            .format(path))

    def assert_saved_model(self, path):
        loader_impl.parse_saved_model(path)

    @test_util.run_v2_only
    def test_save_format_defaults(self):
        path = os.path.join(self.get_temp_dir(), 'model_path')
        save.save_model(self.model, path)
        self.assert_saved_model(path)

    @test_util.run_v2_only
    def test_save_hdf5(self):
        path = os.path.join(self.get_temp_dir(), 'model')
        save.save_model(self.model, path, save_format='h5')
        self.assert_h5_format(path)
        with self.assertRaisesRegexp(
                NotImplementedError,
                'requires the model to be a Functional model or a Sequential model.'):
            save.save_model(self.subclassed_model, path, save_format='h5')

    @test_util.run_v2_only
    def test_save_tf(self):
        path = os.path.join(self.get_temp_dir(), 'model')
        save.save_model(self.model, path, save_format='tf')
        self.assert_saved_model(path)
        with self.assertRaisesRegexp(ValueError, 'input shapes have not been set'):
            save.save_model(self.subclassed_model, path, save_format='tf')
        self.subclassed_model.predict(np.random.random((3, 5)))
        save.save_model(self.subclassed_model, path, save_format='tf')
        self.assert_saved_model(path)

    @test_util.run_v2_only
    def test_save_load_tf_string(self):
        path = os.path.join(self.get_temp_dir(), 'model')
        save.save_model(self.model, path, save_format='tf')
        save.load_model(path)

    @test_util.run_v2_only
    def test_save_load_tf_pathlib(self):
        if sys.version_info >= (3, 6):
            path = pathlib.Path(self.get_temp_dir()) / 'model'
            save.save_model(self.model, path, save_format='tf')
            save.load_model(path)

    @combinations.generate(combinations.combine(mode=['graph', 'eager']))
    def test_saving_h5_for_rnn_layers(self):
        # See https://github.com/tensorflow/tensorflow/issues/35731 for details.
        inputs = keras.Input([10, 91], name='train_input')
        rnn_layers = [
            keras.layers.LSTMCell(
                size, recurrent_dropout=0, name='rnn_cell%d' % i)
            for i, size in enumerate([512, 512])
        ]
        rnn_output = keras.layers.RNN(
            rnn_layers, return_sequences=True, name='rnn_layer')(inputs)
        pred_feat = keras.layers.Dense(
            91, name='prediction_features')(rnn_output)
        pred = keras.layers.Softmax()(pred_feat)
        model = keras.Model(inputs=[inputs], outputs=[pred, pred_feat])
        path = os.path.join(self.get_temp_dir(), 'model_path.h5')
        model.save(path)

        # Make sure the variable name is unique.
        self.assertNotEqual(rnn_layers[0].kernel.name,
                            rnn_layers[1].kernel.name)
        self.assertIn('rnn_cell1', rnn_layers[1].kernel.name)

    @combinations.generate(combinations.combine(mode=['graph', 'eager']))
    def test_saving_optimizer_weights(self):

        class MyModel(keras.Model):

            def __init__(self):
                super(MyModel, self).__init__()
                self.layer = keras.layers.Dense(1)

            def call(self, x):
                return self.layer(x)

        path = os.path.join(self.get_temp_dir(), 'weights_path')
        x, y = np.ones((10, 10)), np.ones((10, 1))

        model = MyModel()
        model.compile('rmsprop', loss='bce')
        model.train_on_batch(x, y)
        model.reset_metrics()
        model.save_weights(path, save_format='tf')

        batch_loss = model.train_on_batch(x, y)

        new_model = MyModel()
        new_model.compile('rmsprop', loss='bce')
        new_model.train_on_batch(x, y)
        new_model.reset_metrics()

        new_model.load_weights(path)
        new_batch_loss = new_model.train_on_batch(x, y)

        self.assertAllClose(batch_loss, new_batch_loss)

    @combinations.generate(combinations.combine(mode=['graph', 'eager']))
    def test_saving_model_with_custom_object(self):
        with generic_utils.custom_object_scope():

            @generic_utils.register_keras_serializable()
            class CustomLoss(losses.MeanSquaredError):
                pass

            model = sequential.Sequential(
                [core.Dense(units=1, input_shape=(1,))])
            model.compile(optimizer='sgd', loss=CustomLoss())
            model.fit(np.zeros([10, 1]), np.zeros([10, 1]))

            temp_dir = self.get_temp_dir()
            filepath = os.path.join(temp_dir, 'saving')
            model.save(filepath)

            # Make sure the model can be correctly load back.
            _ = save.load_model(filepath, compile=True)


if __name__ == '__main__':
    test.main()
