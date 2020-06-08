import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
# from tensorflow import keras
from tensorflow.keras import layers, models
import os
import numpy as np
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch
# import resent_loc
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


def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10))
    model.summary()
    return model


num_gpu = len(tf.config.list_physical_devices('GPU'))
print('Start, num_gpu: ', num_gpu)

print('CPU: ', str(tf.config.list_physical_devices('CPU')))
print('GPU:', str(tf.config.list_physical_devices('GPU')))

if num_gpu < 2:
    num_gpu = 1

NUM_GPUS = num_gpu
BS_PER_GPU = 128
NUM_EPOCHS = 60

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


# (x, y), (x_test, y_test) = load_data()
(train_images, train_labels), (test_images, test_labels) = load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0
n_dub = 1
NUM_TRAIN_SAMPLES = 2**n_dub*NUM_TRAIN_SAMPLES

for _ in range(n_dub):
    train_images = np.concatenate((train_images, train_images))
    train_labels = np.concatenate((train_labels, train_labels))


print('__________________________________________________________________')
start = dt.now()
print(str(start))

if NUM_GPUS == 1:
    model = create_model()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
else:
    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        model = create_model()
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

model.summary()
try:
    slurm_job_id = os.environ["SLURM_JOB_ID"]
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") \
              + '_' + slurm_job_id \
              + '_n_dub_' + str(n_dub) \
              + '_n_gpu' + str(num_gpu)
except:
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") \
              + '_n_dub_' + str(n_dub) \
              + '_n_gpu' + str(num_gpu)


file_writer = tf.summary.create_file_writer(log_dir + "/metrics")
file_writer.set_as_default()
tensorboard_callback = TensorBoard(
  log_dir=log_dir,
  update_freq='batch',
  histogram_freq=1)

lr_schedule_callback = LearningRateScheduler(schedule)

history = model.fit(train_images, train_labels, epochs=NUM_EPOCHS,
                    validation_data=(test_images, test_labels),
                    validation_freq=1,
                    callbacks=[tensorboard_callback, lr_schedule_callback],
                    verbose=2
                    )
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('__________________________________________________________________')
end = dt.now()
print(str(end))
print('n_dub:', n_dub)
print('time: ', str(end - start))
print('time: ', str((end - start).total_seconds()))
print(test_loss, test_acc)
print('DONE (from Python)')
