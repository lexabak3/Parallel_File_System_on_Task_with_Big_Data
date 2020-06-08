import os
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch
from datetime import datetime as dt
import tensorflow as tf
from tensorflow import keras
import sys


n_dub = 0
count_batch = 5 # 505 max

NUM_TRAIN_SAMPLES = 50000
BS_PER_GPU = 128
HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10


BASE_LEARNING_RATE = 0.1
LR_SCHEDULE = [(0.1, 30), (0.01, 45)]


def load_data(num, count_batch=5):

    num_train_samples = num
    num_train_samples = num_train_samples * int(count_batch / 5)

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    path = os.getcwd() + '/cifar-10-batches-py'

    for i in range(1, count_batch + 1):
        f_path = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
         y_train[(i - 1) * 10000:i * 10000]) = load_batch(f_path)

    f_path = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(f_path)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    x_test = x_test.astype(x_train.dtype)
    y_test = y_test.astype(y_train.dtype)

    return (x_train, y_train), (x_test, y_test)


def preprocess(x, y):
    x = tf.image.per_image_standardization(x)
    return x, y


def augmentation(x, y):
    x = tf.image.resize_with_crop_or_pad(
        x, HEIGHT + 8, WIDTH + 8)
    x = tf.image.random_crop(x, [HEIGHT, WIDTH, NUM_CHANNELS])
    x = tf.image.random_flip_left_right(x)
    return x, y


if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(sys.argv[1])
        count_batch = int(sys.argv[1])

    start_data = dt.now()

    num_gpu = len(tf.config.list_physical_devices('GPU'))


    if num_gpu < 2:
        num_gpu = 1

    NUM_GPUS = num_gpu

    (x, y), (x_test, y_test) = load_data(NUM_TRAIN_SAMPLES, count_batch)

    len_x = len(x)
    len_y = len(y)

    NUM_TRAIN_SAMPLES = 2**n_dub*NUM_TRAIN_SAMPLES * int(count_batch / 5)

    start_data_dub = dt.now()

    for _ in range(n_dub):
        x = np.concatenate((x, x))
        y = np.concatenate((y, y))

    end_data_dub = dt.now()

    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    tf.random.set_seed(22)
    train_dataset = train_dataset.map(augmentation).map(preprocess).shuffle(NUM_TRAIN_SAMPLES).batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)
    test_dataset = test_dataset.map(preprocess).batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)

    input_shape = (32, 32, 3)
    img_input = tf.keras.layers.Input(shape=input_shape)
    opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)

    end_data = dt.now()


    data_download = str(end_data - start_data)
    data_download_seconds = str((end_data - start_data).total_seconds())

    print('all time: ', data_download)
    print('all time: ', data_download_seconds)

    data_dub = str(end_data_dub - start_data_dub)
    data_dub_seconds = str((end_data_dub - start_data_dub).total_seconds())
    print('time dub: ', data_dub)
    print('time dub: ', data_dub_seconds)

    print('len_x: ', len_x)
    print('len_y: ', len_y)