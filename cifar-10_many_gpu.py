import tensorflow as tf
from keras.callbacks import TensorBoard, LearningRateScheduler
from tensorflow import keras
import os
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch
import resent_loc
from datetime import datetime as dt
import datetime


def load_data():
    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    path = os.getcwd() + '/cifar-10-batches-py'

    for i in range(1, 6):
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


num_gpu = len(tf.config.list_physical_devices('GPU'))
print('Start, num_gpu: ', num_gpu)

print('CPU: ', str(tf.config.list_physical_devices('CPU')))
print('GPU:', str(tf.config.list_physical_devices('GPU')))

if num_gpu < 2:
    num_gpu = 1

NUM_GPUS = num_gpu
BS_PER_GPU = 128
NUM_EPOCHS = 1

HEIGHT = 32
WIDTH = 32
NUM_CHANNELS = 3
NUM_CLASSES = 10
NUM_TRAIN_SAMPLES = 50000

BASE_LEARNING_RATE = 0.1
LR_SCHEDULE = [(0.1, 30), (0.01, 45)]


def preprocess(x, y):
  x = tf.image.per_image_standardization(x)
  return x, y


def augmentation(x, y):
    x = tf.image.resize_with_crop_or_pad(
        x, HEIGHT + 8, WIDTH + 8)
    x = tf.image.random_crop(x, [HEIGHT, WIDTH, NUM_CHANNELS])
    x = tf.image.random_flip_left_right(x)
    return x, y


def schedule(epoch):
  initial_learning_rate = BASE_LEARNING_RATE * BS_PER_GPU / 128
  learning_rate = initial_learning_rate
  for mult, start_epoch in LR_SCHEDULE:
    if epoch >= start_epoch:
      learning_rate = initial_learning_rate * mult
    else:
      break
  tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
  return learning_rate


(x, y), (x_test, y_test) = load_data()
n_dub = 0
NUM_TRAIN_SAMPLES = 2**n_dub*NUM_TRAIN_SAMPLES

for _ in range(n_dub):
    x = np.concatenate((x, x))
    y = np.concatenate((y, y))

train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

tf.random.set_seed(22)
train_dataset = train_dataset.map(augmentation).map(preprocess).shuffle(NUM_TRAIN_SAMPLES).batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)
test_dataset = test_dataset.map(preprocess).batch(BS_PER_GPU * NUM_GPUS, drop_remainder=True)

input_shape = (32, 32, 3)
img_input = tf.keras.layers.Input(shape=input_shape)
opt = keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)

print('__________________________________________________________________')
start = dt.now()
print(str(start))

if NUM_GPUS == 1:
    model = resent_loc.resnet56(img_input=img_input, classes=NUM_CLASSES)
    model.compile(
              optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['sparse_categorical_accuracy'])
else:
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
      model = resent_loc.resnet56(img_input=img_input, classes=NUM_CLASSES)
      model.compile(
                optimizer=opt,
                loss='sparse_categorical_crossentropy',
                metrics=['sparse_categorical_accuracy'])

model.summary()


log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
file_writer.set_as_default()
tensorboard_callback = TensorBoard(
  log_dir=log_dir,
  update_freq='batch',
  histogram_freq=1)

lr_schedule_callback = LearningRateScheduler(schedule)

model.fit(train_dataset,
          epochs=NUM_EPOCHS,
          validation_data=test_dataset,
          validation_freq=2,
          callbacks=[tensorboard_callback, lr_schedule_callback],
          verbose=2
          )
test_loss, test_acc = model.evaluate(test_dataset)
print('__________________________________________________________________')
end = dt.now()
print(str(end))
print('time: ', str(end - start))
print('time: ', str((end - start).total_seconds()))
print(test_loss, test_acc)
print('DONE (from Python)')
