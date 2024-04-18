import tensorflow as tf
import os
# Disable TensorFlow info messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

R = tf.constant(((430, 510, 750, 200, 300),
    (510, 620, 910, 240, 360)))

# get the sums of the rows
R_sums = tf.reduce_sum(R, axis=0)

# get the ratio of the resistors
ratios = tf.divide(R[-1], R_sums)

# Ideal v_ref / vcc
ratio_ideal = 2.366 / 5

# get the distance from the ideal ratio
ratios = tf.abs(ratios - ratio_ideal)

# get the best index
index = tf.argmin(ratios)

# print the best resistor combination
print(f'R_T, R_B = {R[:, index]}',
      f'Error = {100*ratios[index] / ratio_ideal:.2f}%',
      f'Power Loss: {4 * tf.reduce_sum(R[:, index]) / 1000:0.2f} W')

# print another resistor combination
print(f'R_T, R_B = {R[:, index+1]}',
      f'Error = {100*ratios[index+1] / ratio_ideal:.2f}%',
      f'Power Loss: {4 * tf.reduce_sum(R[:, index+1]) / 1000:0.2f} W')