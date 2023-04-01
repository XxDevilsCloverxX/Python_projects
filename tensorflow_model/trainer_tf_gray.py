import tensorflow as tf

# check for GPU
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())