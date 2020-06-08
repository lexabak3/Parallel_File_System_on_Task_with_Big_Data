# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""ResNet56 model for Keras adapted from tf.keras.applications.ResNet50.
# Reference:
- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers

BATCH_NORM_DECAY = 0.997
BATCH_NORM_EPSILON = 1e-5
L2_WEIGHT_DECAY = 2e-4


def identity_building_block(input_tensor,
                            kernel_size,
                            filters,
                            stage,
                            block,
                            training=None):

    filters1, filters2 = filters
    if tf.keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = tf.keras.layers.Conv2D(filters1, kernel_size,
                               padding='same',
                               kernel_initializer='he_normal',
                               kernel_regularizer=
                               tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                               bias_regularizer=
                               tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                               name=conv_name_base + '2a')(input_tensor)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                           name=bn_name_base + '2a',
                                           momentum=BATCH_NORM_DECAY,
                                           epsilon=BATCH_NORM_EPSILON)(
        x, training=training)

    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters2, kernel_size,
                               padding='same',
                               kernel_initializer='he_normal',
                               kernel_regularizer=
                               tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                               bias_regularizer=
                               tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                               name=conv_name_base + '2b')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                           name=bn_name_base + '2b',
                                           momentum=BATCH_NORM_DECAY,
                                           epsilon=BATCH_NORM_EPSILON)(
        x, training=training)
    x = tf.keras.layers.add([x, input_tensor])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def conv_building_block(input_tensor,
                        kernel_size,
                        filters,
                        stage,
                        block,
                        strides=(2, 2),
                        training=None):

    filters1, filters2 = filters
    if tf.keras.backend.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = tf.keras.layers.Conv2D(filters1, kernel_size, strides=strides,
                               padding='same',
                               kernel_initializer='he_normal',
                               kernel_regularizer=
                               tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                               bias_regularizer=
                               tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                               name=conv_name_base + '2a')(input_tensor)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                           name=bn_name_base + '2a',
                                           momentum=BATCH_NORM_DECAY,
                                           epsilon=BATCH_NORM_EPSILON)(
        x, training=training)
    x = tf.keras.layers.Activation('relu')(x)

    x = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same',
                               kernel_initializer='he_normal',
                               kernel_regularizer=
                               tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                               bias_regularizer=
                               tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                               name=conv_name_base + '2b')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis,
                                           name=bn_name_base + '2b',
                                           momentum=BATCH_NORM_DECAY,
                                           epsilon=BATCH_NORM_EPSILON)(
        x, training=training)
    shortcut = tf.keras.layers.Conv2D(filters2, (1, 1), strides=strides,
                                      kernel_initializer='he_normal',
                                      kernel_regularizer=
                                      tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                                      bias_regularizer=
                                      tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                                      name=conv_name_base + '1')(input_tensor)
    shortcut = tf.keras.layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1',
        momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(
        shortcut, training=training)

    x = tf.keras.layers.add([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x


def resnet_block(input_tensor,
                 size,
                 kernel_size,
                 filters,
                 stage,
                 conv_strides=(2, 2),
                 training=None):

    x = conv_building_block(input_tensor, kernel_size, filters, stage=stage,
                            strides=conv_strides, block='block_0',
                            training=training)
    for i in range(size - 1):
        x = identity_building_block(x, kernel_size, filters, stage=stage,
                                    block='block_%d' % (i + 1), training=training)
    return x


def resnet(num_blocks, img_input=None, classes=10, training=None):

    if backend.image_data_format() == 'channels_first':
        x = layers.Lambda(lambda x: backend.permute_dimensions(x, (0, 3, 1, 2)),
                          name='transpose')(img_input)
        bn_axis = 1
    else:  # channel_last
        x = img_input
        bn_axis = 3

    x = tf.keras.layers.ZeroPadding2D(padding=(1, 1), name='conv1_pad')(x)
    x = tf.keras.layers.Conv2D(16, (3, 3),
                               strides=(1, 1),
                               padding='valid',
                               kernel_initializer='he_normal',
                               kernel_regularizer=
                               tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                               bias_regularizer=
                               tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                               name='conv1')(x)
    x = tf.keras.layers.BatchNormalization(axis=bn_axis, name='bn_conv1',
                                           momentum=BATCH_NORM_DECAY,
                                           epsilon=BATCH_NORM_EPSILON)(
        x, training=training)
    x = tf.keras.layers.Activation('relu')(x)

    x = resnet_block(x, size=num_blocks, kernel_size=3, filters=[16, 16],
                     stage=2, conv_strides=(1, 1), training=training)

    x = resnet_block(x, size=num_blocks, kernel_size=3, filters=[32, 32],
                     stage=3, conv_strides=(2, 2), training=training)

    x = resnet_block(x, size=num_blocks, kernel_size=3, filters=[64, 64],
                     stage=4, conv_strides=(2, 2), training=training)

    x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = tf.keras.layers.Dense(classes, activation='softmax',
                              kernel_initializer='he_normal',
                              kernel_regularizer=
                              tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                              bias_regularizer=
                              tf.keras.regularizers.l2(L2_WEIGHT_DECAY),
                              name='fc10')(x)

    inputs = img_input
    # Create model.
    model = tf.keras.models.Model(inputs, x, name='resnet56')

    return model


resnet20 = functools.partial(resnet, num_blocks=3)
resnet32 = functools.partial(resnet, num_blocks=5)
resnet56 = functools.partial(resnet, num_blocks=9)
resnet110 = functools.partial(resnet, num_blocks=110)