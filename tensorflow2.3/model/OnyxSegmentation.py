import tensorflow as tf
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras import Model
from tensorflow.keras.layers import  SeparableConv2D

from tensorflow.keras.layers import BatchNormalization

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)



class OnyxSegmentation:

    def __init__(self, learning_rate=0.003):
        x_input = tf.keras.Input(shape=(512, 512, 1), name="x_input_node")
        x = Conv2D(kernel_size=(3, 3), filters=32, padding='same', use_bias=False)(x_input)
        down_layer1 = self.residual_layer(x, filters=32)
        down_layer1 = self.residual_layer(down_layer1, filters=32)
        #down_layer1 = self.residual_layer(down_layer1, filters=32)
        x = Conv2D(kernel_size=(3, 3), filters=32, strides=(2, 2), padding='same', use_bias=False)(down_layer1)
        down_layer2 = self.residual_layer(x, filters=32)
        down_layer2 = self.residual_layer(down_layer2, filters=32)
        #down_layer2 = self.residual_layer(down_layer2, filters=32)
        x = Conv2D(kernel_size=(3, 3), filters=32, strides=(2, 2), padding='same', use_bias=False)(down_layer2)
        down_layer3 = self.residual_layer(x, filters=32)
        down_layer3 = self.residual_layer(down_layer3, filters=32)
        #down_layer3 = self.residual_layer(down_layer3, filters=32)
        x = Conv2D(kernel_size=(3, 3), filters=32, strides=(2, 2,), padding='same', use_bias=False)(down_layer3)
        down_layer4 = self.residual_layer(x, filters=32)
        down_layer4 = self.residual_layer(down_layer4, filters=32)
        #down_layer4 = self.residual_layer(down_layer4, filters=32)
        x = Conv2D(kernel_size=(3, 3), filters=32, strides=(2, 2), padding='same', use_bias=False)(down_layer4)
        down_layer5 = self.residual_layer(x, filters=32)
        down_layer5 = self.residual_layer(down_layer5, filters=32)
        #down_layer5 = self.residual_layer(down_layer5, filters=32)
        x = Conv2D(kernel_size=(3, 3), filters=32, strides=(2, 2), padding='same', use_bias=False)(down_layer5)
        center = self.residual_layer(x, filters=32)
        center = self.residual_layer(center, filters=32)
        #center = self.residual_layer(center, filters=32)
        uplayer1 = Conv2DTranspose(kernel_size=(3, 3), filters=32, strides=(2, 2), padding='same', use_bias=False)(center)
        x = uplayer1 + down_layer5
        x = self.residual_layer(x, filters=32)
        x = self.residual_layer(x, filters=32)
        #x = self.residual_layer(x, filters=32)
        uplayer2 = Conv2DTranspose(kernel_size=(3, 3), filters=32, strides=(2, 2), padding='same', use_bias=False)(x)
        x = uplayer2 + down_layer4
        x = self.residual_layer(x, filters=32)
        x = self.residual_layer(x, filters=32)
        #x = self.residual_layer(x, filters=32)
        uplayer3 = Conv2DTranspose(kernel_size=(3, 3), filters=32, strides=(2, 2), padding='same', use_bias=False)(x)
        x = uplayer3 + down_layer3
        x = self.residual_layer(x, filters=32)
        x = self.residual_layer(x, filters=32)
        #x = self.residual_layer(x, filters=32)
        uplayer4 = Conv2DTranspose(kernel_size=(3, 3), filters=32, strides=(2, 2), padding='same', use_bias=False)(x)
        x = uplayer4 + down_layer2
        x = self.residual_layer(x, filters=32)
        x = self.residual_layer(x, filters=32)
        #x = self.residual_layer(x, filters=32)
        uplayer5 = Conv2DTranspose(kernel_size=(3, 3), filters=32, strides=(2, 2), padding='same', use_bias=False)(x)
        x = uplayer5 + down_layer1
        x = self.residual_layer(x, filters=32)
        x = self.residual_layer(x, filters=32)
        #x = self.residual_layer(x, filters=32)
        x = Conv2D(kernel_size=(3, 3), filters=1, padding='same', use_bias=False)(x)
        output = tf.sigmoid(x, name='output')


        self.model = Model(inputs=x_input, outputs=output)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.summary()

    def dice_loss(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.math.sigmoid(y_pred)
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)
        return 1 - numerator / denominator

    def residual_layer(self, x_input, filters=32):
        layer1 = BatchNormalization()(x_input)
        #x = Conv2D(kernel_size=(3, 3), filters=filters, padding='same', use_bias=False)(layer1)
        x = SeparableConv2D(kernel_size=(3, 3), filters=filters, padding='same', use_bias=False)(layer1)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = SpatialDropout2D(0.3)(x)
        skip = x + layer1

        return skip

    def train_one_batch(self, x_input, y_label):
        with tf.GradientTape() as tape:
            output = self.model(x_input, training=True)
            d_loss = self.dice_loss(y_label, output)
            c_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_label, logits=output))
            loss = d_loss + c_loss

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss.numpy()


    def accracy_on_batch(self, x_input, y_label):
        output = self.model(x_input, training=False)
        pre = tf.cast(output > 0.5, dtype=tf.float32)
        truth = tf.cast(y_label > 0.5, dtype=tf.float32)
        inse = tf.reduce_sum(tf.multiply(pre, truth), axis=(1, 2, 3))  # AND
        union = tf.reduce_sum(tf.cast(tf.add(pre, truth) >= 1, dtype=tf.float32), axis=(1, 2, 3))  # OR
        batch_iou = (inse + 1e-5) / (union + 1e-5)
        accuracy = tf.reduce_mean(batch_iou, name='iou_coe1')
        return accuracy.numpy()


    def predict(self, x_input):
        output = self.model(x_input, training=False)
        return output.numpy()


    def get_model(self):
        return self.model

