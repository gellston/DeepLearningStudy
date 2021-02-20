import tensorflow as tf
import numpy as np
from tensorflow.keras import Model
from tensorflow.keras import Input

print(tf.__version__)

class Linear(tf.keras.layers.Layer):
  def __init__(self, units=32):
    super(Linear, self).__init__()
    self.w_init_value = tf.random_normal_initializer()
    self.b_init_value = tf.zeros_initializer()
    self.w = tf.Variable(initial_value=self.w_init_value(shape=([units])), trainable=True)
    self.b = tf.Variable(initial_value=self.b_init_value(shape=([units])), trainable=True)

  def call(self, inputs):
    return inputs * self.w + self.b


x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).reshape(10, 1)
y_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]).reshape(10, 1)

x_input = Input(shape=(1,), name='input')
y_output = Linear(1)(x_input)

def loss_function(logits, y):
  loss = tf.reduce_mean(tf.square(logits - y))
  return loss


optimizer = tf.keras.optimizers.Adam(learning_rate=0.03)
model = Model(inputs=x_input, outputs=y_output)
model.summary()


data_length = len(x_data)
epoch_length = 100

for epoch in range(epoch_length):
  average_loss = 0
  for step in range(data_length):
    with tf.GradientTape() as tape:
      logits = model(x_data[step], training=True)
      loss = loss_function(logits, y_data[step])
    average_loss += float(loss) / data_length
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  print('average loss = ', average_loss)
  print('epoch = ', epoch)
  for step in range(data_length):
    print('data index = ', step, ', input=', x_data[step], ', output=', model(x_data[step], training=False).numpy())




