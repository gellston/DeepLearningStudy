import tensorflow as tf

from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import SpatialDropout2D

from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import ReLU
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.activations import swish



gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)


class LeafletLineClassification:
    def __init__(self, learning_rate=0.003):
        x_input = tf.keras.Input(shape=(100, 512, 3), name="x_input_node")
        x = Conv2D(kernel_size=(5, 5), filters=16, padding='same', use_bias=False, dilation_rate=2)(x_input)
        x = BatchNormalization()(x)
        x1 = PReLU()(x)

        short_cut = MaxPool2D(pool_size=(2, 3))(x1)
        x = Conv2D(kernel_size=(5, 5), filters=16, padding='same', use_bias=False, dilation_rate=2)(short_cut)
        x = BatchNormalization()(x)
        x2 = PReLU()(x) + short_cut


        short_cut = MaxPool2D(pool_size=(2, 3))(x2)
        x = Conv2D(kernel_size=(5, 5), filters=16, padding='same', use_bias=False, dilation_rate=2)(short_cut)
        x = BatchNormalization()(x)
        x3 = PReLU()(x) + short_cut


        short_cut = MaxPool2D(pool_size=(2, 3))(x3)
        x = Conv2D(kernel_size=(5, 5), filters=16, padding='same', use_bias=False, dilation_rate=2)(short_cut)
        x = BatchNormalization()(x)
        x4 = PReLU()(x) + short_cut


        x = Conv2D(kernel_size=(3, 3), filters=16, padding='same')(x4)
        x = BatchNormalization()(x)
        x = Conv2D(kernel_size=(3, 3), filters=8, padding='same')(x)
        x = BatchNormalization()(x)
        x = Conv2D(kernel_size=(3, 3), filters=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = PReLU()(x)
        x = GlobalAveragePooling2D()(x)
        x = Softmax()(x)

        self.model = Model(inputs=x_input, outputs=x)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.summary()

    def residual_layer(self, x_input, filters=32):
        x = Conv2D(kernel_size=(3, 3), filters=filters, padding='same', use_bias=False)(x_input)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        skip = x + x_input
        return skip

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

