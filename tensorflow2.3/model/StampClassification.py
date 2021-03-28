import tensorflow as tf

from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import SpatialDropout2D

from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import PReLU
from tensorflow.keras import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import SpatialDropout2D


class StampClassification:
    def __init__(self, learning_rate=0.003):
        x_input = tf.keras.Input(shape=(35, 35, 3), name="x_input_node")
        x = SeparableConv2D(32, (3, 3), padding='same')(x_input)
        x = BatchNormalization()(x)
        x = SpatialDropout2D(rate=0.2)(x)
        x = self.residual_layer(x, 32)
        x = SpatialDropout2D(rate=0.2)(x)
        x = MaxPool2D()(x)
        x = SeparableConv2D(32, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout2D(rate=0.2)(x)
        x = self.residual_layer(x, 32)
        x = SpatialDropout2D(rate=0.2)(x)
        x = MaxPool2D()(x)
        x = SeparableConv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout2D(rate=0.2)(x)
        x = self.residual_layer(x, 128)
        x = SpatialDropout2D(rate=0.2)(x)
        x = MaxPool2D()(x)
        x = SeparableConv2D(128, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout2D(rate=0.2)(x)
        x = self.residual_layer(x, 128)
        x = SpatialDropout2D(rate=0.2)(x)
        x = SeparableConv2D(2, (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = SpatialDropout2D(rate=0.2)(x)
        x = GlobalAveragePooling2D()(x)
        x = Softmax()(x)

        self.model = Model(inputs=x_input, outputs=x)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.summary()

    def residual_layer(self, x_input, filters=32):
        x = SeparableConv2D(kernel_size=(3, 3), filters=filters, padding='same', use_bias=False)(x_input)
        x = BatchNormalization()(x)
        x = SpatialDropout2D(rate=0.2)(x)
        x = PReLU()(x)
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

