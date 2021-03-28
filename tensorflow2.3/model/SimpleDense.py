import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model

class SimpleDense:
    def __init__(self, learning_rate=0.001):

        x_input = tf.keras.Input(shape=(1,), name="x_input_node")
        x = Dense(1)(x_input)

        self.model = Model(inputs=x_input, outputs=x)
        self.optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        self.model.summary()

    def train_one_batch(self, x_input, y_label):

        with tf.GradientTape() as tape:

            output = self.model(x_input, training=True)

            loss = tf.reduce_mean(tf.square(output - y_label))
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return float(loss)

    def predict(self, x_input):
        output = self.model(x_input, training=False)
        return output.numpy()


    def get_model(self):
        return self.model

