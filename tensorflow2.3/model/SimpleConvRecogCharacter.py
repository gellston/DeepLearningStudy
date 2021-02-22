import tensorflow as tf
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization


class SimpleConvRecogCharacter:
    def __init__(self, learning_rate=0.003):
        x_input = tf.keras.Input(shape=(28, 28, 1), name="x_input_node")
        x = Conv2D(32, (3,3))(x_input)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPool2D((2,2))(x)
        x = Conv2D(32, (3,3))(x)
        #x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPool2D((2,2))(x)
        x = Conv2D(64, (3, 3))(x)
        #x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(10, (3, 3))(x)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = GlobalAveragePooling2D()(x)
        x = Softmax()(x)
        self.model = Model(inputs=x_input, outputs=x)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.summary()

    def train_one_batch(self, x_input, y_label):
        with tf.GradientTape() as tape:
            output = self.model(x_input, training=True)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_label))
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss.numpy()


    def accracy_on_batch(self, x_input, y_label):
        output = self.model(x_input, training=False)
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_label, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy.numpy()


    def predict(self, x_input):
        output = self.model(x_input, training=False)
        return output.numpy()


    def get_model(self):
        return self.model

