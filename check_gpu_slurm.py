#!/home/avtimofeev/bakshaev/Python-3.7.1/python
#SBATCH --output=/home/avtimofeev/bakshaev/frst_try
#SBATCH -n 5 # 5 cores

import sys
import os
# sys.path.append('/home/avtimofeev/bakshaev/Python-3.7.1/packages')

import tensorflow as tf


print('CPU: ', str(tf.config.list_physical_devices('CPU')))

print('GPU:', str(tf.config.list_physical_devices('GPU')))

print('Type: ', type(tf.config.list_physical_devices('CPU')[0]))
print(tf.config.list_physical_devices()[0].name)
print(tf.config.list_physical_devices()[0].device_type)
