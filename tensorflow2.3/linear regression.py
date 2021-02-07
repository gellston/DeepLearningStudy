import tensorflow as tf;
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense




class Linear(tf.keras.layers.Layer):
  def __init__(self, units=32, input_dim=32):
    super(Linear, self).__init__()
    w_init = tf.random_normal_initializer()
    self.w = tf.Variable(
      initial_value=w_init(shape=(input_dim, units), dtype="float32"),
      trainable=True,
    )
    b_init = tf.zeros_initializer()
    self.b = tf.Variable(
      initial_value=b_init(shape=(units,), dtype="float32"), trainable=True
    )

  def call(self, inputs):
    return tf.matmul(inputs, self.w) + self.b





x_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
y_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]

x_input = Input(shape=(1,), name='input', dtype=tf.float32)
x = Linear(1, 1)(x_input)




def cost_function(y_true, y_pred):
  loss = tf.reduce_mean(tf.square(y_true - y_pred), axis=-1)
  return loss


optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)

model = Model(inputs=x_input, outputs=x)
model.compile(optimizer=optimizer, loss=cost_function)
model.fit(x=x_data, y=y_data, batch_size=1, epochs=100)

model.summary()

print('resu;t = ' ,model.predict(x_data))

