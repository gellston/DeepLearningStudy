import tensorflow as tf
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras import Model
from tensorflow.keras.layers import  SeparableConv2D
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)



class PCBDefectSegmentationV8:

    def __init__(self, learning_rate=0.003):

        common_filter_size1 = 16

        x_input = tf.keras.Input(shape=(512, 512, 3), name="x_input_node")
        x = Conv2D(kernel_size=(3, 3), filters=common_filter_size1, padding='same', use_bias=False)(x_input)
        x = BatchNormalization()(x)
        x = ReLU(max_value=6)(x)
        down_layer1 = self.residual_layer(x, filters=common_filter_size1)
        down_layer1 = self.residual_layer(down_layer1, filters=common_filter_size1)                               #512
        x = AveragePooling2D(pool_size=(2, 2), strides=2)(down_layer1)
        x = Conv2D(kernel_size=(3, 3), filters=common_filter_size1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU(max_value=6)(x)
        down_layer2 = self.residual_layer(x, filters=common_filter_size1)
        down_layer2 = self.residual_layer(down_layer2, filters=common_filter_size1)                               #256
        x = AveragePooling2D(pool_size=(2, 2), strides=2)(down_layer2)
        x = Conv2D(kernel_size=(3, 3), filters=common_filter_size1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = ReLU(max_value=6)(x)
        center = self.residual_layer(x, filters=common_filter_size1)                                         ##Center
        center = self.residual_layer(center, filters=common_filter_size1)                                    ##128
        center = self.residual_layer(center, filters=common_filter_size1)
        center = self.residual_layer(center, filters=common_filter_size1)
        #x = self.residual_layer(x, filters=32)
        uplayer4 = Conv2DTranspose(kernel_size=(3, 3), filters=common_filter_size1, strides=(2, 2), padding='same', use_bias=False)(center)  #256
        uplayer4 = BatchNormalization()(uplayer4)
        uplayer4 = ReLU(max_value=6)(uplayer4)
        x = uplayer4 + down_layer2
        x = self.residual_layer(x, filters=common_filter_size1)
        x = self.residual_layer(x, filters=common_filter_size1)
        #x = self.residual_layer(x, filters=32)
        uplayer5 = Conv2DTranspose(kernel_size=(3, 3), filters=common_filter_size1, strides=(2, 2), padding='same', use_bias=False)(x)  #512
        uplayer5 = BatchNormalization()(uplayer5)
        uplayer5 = ReLU(max_value=6)(uplayer5)
        x = uplayer5 + down_layer1
        x = self.residual_layer(x, filters=common_filter_size1)
        x = self.residual_layer(x, filters=common_filter_size1)
        #x = self.residual_layer(x, filters=32)
        x = Conv2D(kernel_size=(3, 3), filters=1, padding='same', use_bias=False)(x)
        x = BatchNormalization()(x)
        output = tf.sigmoid(x, name='output')


        self.model = Model(inputs=x_input, outputs=output)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.summary()


    def binary_focal_loss_fixed(self, y_true, y_pred):
        gamma = 2.
        alpha = .25
        y_true = tf.cast(y_true, tf.float32)
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        p_t = tf.where(K.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        loss = weight * cross_entropy
        loss = K.mean(K.sum(loss, axis=1))

        return loss

    def balanced_cross_entropy(self, y_true, y_pred):
        beta = tf.reduce_mean(1 - y_true)
        weight_a = beta * tf.cast(y_true, tf.float32)
        weight_b = (1 - beta) * tf.cast(1 - y_true, tf.float32)

        o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b
        return tf.reduce_mean(o)


    def dice_loss(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.math.sigmoid(y_pred)
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)
        return 1 - numerator / denominator

    def residual_layer(self, x_input, filters=32):
        x = SeparableConv2D(kernel_size=(3, 3), filters=filters, padding='same', use_bias=False)(x_input)
        x = BatchNormalization()(x)
        x = ReLU(max_value=6)(x)
        #x = SpatialDropout2D(0.3)(x)
        skip = x + x_input

        return skip

    def train_one_batch(self, x_input, y_label):
        with tf.GradientTape() as tape:
            output = self.model(x_input, training=True)
            d_loss = self.dice_loss(y_label, output)
            #c_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_label, logits=output))
            #loss = d_loss + self.binary_focal_loss_fixed(y_label, output)
            loss = self.balanced_cross_entropy(y_label, output)
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

