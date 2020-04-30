# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Keras layers API."""

from __future__ import absolute_import, division, print_function

from tensorflow.python import tf2
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.base_preprocessing_layer import \
    PreprocessingLayer
# Generic layers.
# pylint: disable=g-bad-import-order
# pylint: disable=g-import-not-at-top
from tensorflow.python.keras.engine.input_layer import Input, InputLayer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers import serialization
from tensorflow.python.keras.layers.advanced_activations import (
    ELU, LeakyReLU, PReLU, ReLU, Softmax, ThresholdedReLU)
from tensorflow.python.keras.layers.convolutional import (
    Conv1D, Conv1DTranspose, Conv2D, Conv2DTranspose, Conv3D, Conv3DTranspose,
    Convolution1D, Convolution2D, Convolution2DTranspose, Convolution3D,
    Convolution3DTranspose, Cropping1D, Cropping2D, Cropping3D,
    DepthwiseConv2D, SeparableConv1D, SeparableConv2D, SeparableConvolution1D,
    SeparableConvolution2D, UpSampling1D, UpSampling2D, UpSampling3D,
    ZeroPadding1D, ZeroPadding2D, ZeroPadding3D)
from tensorflow.python.keras.layers.convolutional_recurrent import ConvLSTM2D
from tensorflow.python.keras.layers.core import (Activation,
                                                 ActivityRegularization, Dense,
                                                 Dropout, Flatten, Lambda,
                                                 Masking, Permute,
                                                 RepeatVector, Reshape,
                                                 SpatialDropout1D,
                                                 SpatialDropout2D,
                                                 SpatialDropout3D)
from tensorflow.python.keras.layers.cudnn_recurrent import CuDNNGRU, CuDNNLSTM
from tensorflow.python.keras.layers.dense_attention import (AdditiveAttention,
                                                            Attention)
from tensorflow.python.keras.layers.einsum_dense import EinsumDense
from tensorflow.python.keras.layers.embeddings import Embedding
from tensorflow.python.keras.layers.kernelized import RandomFourierFeatures
from tensorflow.python.keras.layers.local import (LocallyConnected1D,
                                                  LocallyConnected2D)
from tensorflow.python.keras.layers.merge import (Add, Average, Concatenate,
                                                  Dot, Maximum, Minimum,
                                                  Multiply, Subtract, add,
                                                  average, concatenate, dot,
                                                  maximum, minimum, multiply,
                                                  subtract)
from tensorflow.python.keras.layers.noise import (AlphaDropout,
                                                  GaussianDropout,
                                                  GaussianNoise)
from tensorflow.python.keras.layers.normalization import LayerNormalization
from tensorflow.python.keras.layers.normalization_v2 import \
    SyncBatchNormalization
from tensorflow.python.keras.layers.pooling import (AveragePooling1D,
                                                    AveragePooling2D,
                                                    AveragePooling3D,
                                                    AvgPool1D, AvgPool2D,
                                                    AvgPool3D,
                                                    GlobalAveragePooling1D,
                                                    GlobalAveragePooling2D,
                                                    GlobalAveragePooling3D,
                                                    GlobalAvgPool1D,
                                                    GlobalAvgPool2D,
                                                    GlobalAvgPool3D,
                                                    GlobalMaxPool1D,
                                                    GlobalMaxPool2D,
                                                    GlobalMaxPool3D,
                                                    GlobalMaxPooling1D,
                                                    GlobalMaxPooling2D,
                                                    GlobalMaxPooling3D,
                                                    MaxPool1D, MaxPool2D,
                                                    MaxPool3D, MaxPooling1D,
                                                    MaxPooling2D, MaxPooling3D)
# Image preprocessing layers.
from tensorflow.python.keras.layers.preprocessing.image_preprocessing import (
    CenterCrop, RandomContrast, RandomCrop, RandomFlip, RandomHeight,
    RandomRotation, RandomTranslation, RandomWidth, RandomZoom, Rescaling,
    Resizing)
from tensorflow.python.keras.layers.recurrent import (RNN, AbstractRNNCell,
                                                      PeepholeLSTMCell,
                                                      SimpleRNN, SimpleRNNCell,
                                                      StackedRNNCells)
from tensorflow.python.keras.layers.rnn_cell_wrapper_v2 import (
    DeviceWrapper, DropoutWrapper, ResidualWrapper)
from tensorflow.python.keras.layers.serialization import deserialize, serialize
from tensorflow.python.keras.layers.wrappers import (Bidirectional,
                                                     TimeDistributed, Wrapper)

# Preprocessing layers.
if tf2.enabled():
    from tensorflow.python.keras.layers.preprocessing.normalization import Normalization
    from tensorflow.python.keras.layers.preprocessing.normalization_v1 import (
        Normalization as NormalizationV1,
    )

    NormalizationV2 = Normalization
    from tensorflow.python.keras.layers.preprocessing.text_vectorization import (
        TextVectorization,
    )
    from tensorflow.python.keras.layers.preprocessing.text_vectorization_v1 import (
        TextVectorization as TextVectorizationV1,
    )

    TextVectorizationV2 = TextVectorization
else:
    from tensorflow.python.keras.layers.preprocessing.normalization_v1 import (
        Normalization,
    )
    from tensorflow.python.keras.layers.preprocessing.normalization import (
        Normalization as NormalizationV2,
    )

    NormalizationV1 = Normalization
    from tensorflow.python.keras.layers.preprocessing.text_vectorization_v1 import (
        TextVectorization,
    )
    from tensorflow.python.keras.layers.preprocessing.text_vectorization import (
        TextVectorization as TextVectorizationV2,
    )

    TextVectorizationV1 = TextVectorization

# Advanced activations.

# Convolution layers.

# Convolution layer aliases.

# Image processing layers.

# Core layers.

# Dense Attention layers.

# Embedding layers.

# Einsum-based dense layer/

# Locally-connected layers.

# Merge layers.

# Noise layers.

# Normalization layers.

if tf2.enabled():
    from tensorflow.python.keras.layers.normalization_v2 import BatchNormalization
    from tensorflow.python.keras.layers.normalization import (
        BatchNormalization as BatchNormalizationV1,
    )

    BatchNormalizationV2 = BatchNormalization
else:
    from tensorflow.python.keras.layers.normalization import BatchNormalization
    from tensorflow.python.keras.layers.normalization_v2 import (
        BatchNormalization as BatchNormalizationV2,
    )

    BatchNormalizationV1 = BatchNormalization

# Kernelized layers.

# Pooling layers.

# Pooling layer aliases.

# Recurrent layers.

if tf2.enabled():
    from tensorflow.python.keras.layers.recurrent_v2 import GRU
    from tensorflow.python.keras.layers.recurrent_v2 import GRUCell
    from tensorflow.python.keras.layers.recurrent_v2 import LSTM
    from tensorflow.python.keras.layers.recurrent_v2 import LSTMCell
    from tensorflow.python.keras.layers.recurrent import GRU as GRUV1
    from tensorflow.python.keras.layers.recurrent import GRUCell as GRUCellV1
    from tensorflow.python.keras.layers.recurrent import LSTM as LSTMV1
    from tensorflow.python.keras.layers.recurrent import LSTMCell as LSTMCellV1

    GRUV2 = GRU
    GRUCellV2 = GRUCell
    LSTMV2 = LSTM
    LSTMCellV2 = LSTMCell
else:
    from tensorflow.python.keras.layers.recurrent import GRU
    from tensorflow.python.keras.layers.recurrent import GRUCell
    from tensorflow.python.keras.layers.recurrent import LSTM
    from tensorflow.python.keras.layers.recurrent import LSTMCell
    from tensorflow.python.keras.layers.recurrent_v2 import GRU as GRUV2
    from tensorflow.python.keras.layers.recurrent_v2 import GRUCell as GRUCellV2
    from tensorflow.python.keras.layers.recurrent_v2 import LSTM as LSTMV2
    from tensorflow.python.keras.layers.recurrent_v2 import LSTMCell as LSTMCellV2

    GRUV1 = GRU
    GRUCellV1 = GRUCell
    LSTMV1 = LSTM
    LSTMCellV1 = LSTMCell

# Convolutional-recurrent layers.

# CuDNN recurrent layers.

# Wrapper functions

# # RNN Cell wrappers.

# Serialization functions


class VersionAwareLayers(object):
    """Utility to be used internally to access layers in a V1/V2-aware fashion.

    When using layers within the Keras codebase, under the constraint that
    e.g. `layers.BatchNormalization` should be the `BatchNormalization` version
    corresponding to the current runtime (TF1 or TF2), do not simply access
    `layers.BatchNormalization` since it would ignore e.g. an early
    `compat.v2.disable_v2_behavior()` call. Instead, use an instance
    of `VersionAwareLayers` (which you can use just like the `layers` module).
    """

    def __getattr__(self, name):
        serialization.populate_deserializable_objects()
        if name in serialization.LOCAL.ALL_OBJECTS:
            return serialization.LOCAL.ALL_OBJECTS[name]
        return super(VersionAwareLayers, self).__getattr__(name)


del absolute_import
del division
del print_function
