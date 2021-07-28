import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import Model
from tensorflow.keras import backend as K

from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import SpatialDropout2D
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import PReLU
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import MaxPool2D

from tensorflow.keras.layers import AveragePooling2D

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)


class PCBDefectSegmentationV2:

    def __init__(self, learning_rate=0.003):
        global_filter_size1 = 6
        global_filter_size2 = 12
        global_filter_size3 = 24
        global_filter_size4 = 32
        global_subtraction_size = 1
        global_final_filter_size = 3


        x_input = tf.keras.Input(shape=(1024, 1024, 3), name="x_input_node")
        x = SeparableConv2D(kernel_size=3, filters=global_filter_size1, use_bias=False, padding='same', dilation_rate=1)(x_input) #1024 x1
        x = self.residual_layer(x , kernel_size=3, filters=global_filter_size1, dilation=1)
        x = self.residual_layer(x , kernel_size=3, filters=global_filter_size1, dilation=1)
        x = MaxPool2D(pool_size=2, strides=2)(x)
        x = SeparableConv2D(kernel_size=3, filters=global_filter_size2, use_bias=False, padding='same', dilation_rate=1)(x) #512 x2
        x = self.residual_layer(x , kernel_size=3, filters=global_filter_size2, dilation=2)
        x = self.residual_layer(x , kernel_size=3, filters=global_filter_size2, dilation=2)
        x = MaxPool2D(pool_size=2, strides=2)(x)
        x = SeparableConv2D(kernel_size=3, filters=global_filter_size3, use_bias=False, padding='same', dilation_rate=1)(x) #256 x4
        x = self.residual_layer(x , kernel_size=3, filters=global_filter_size3, dilation=3)
        x = self.residual_layer(x , kernel_size=3, filters=global_filter_size3, dilation=3)
        x = MaxPool2D(pool_size=2, strides=2)(x)
        x = SeparableConv2D(kernel_size=3, filters=global_filter_size4, use_bias=False, padding='same', dilation_rate=1)(x) #128 x8
        x = self.residual_layer(x , kernel_size=3, filters=global_filter_size4, dilation=4)
        x = self.residual_layer(x , kernel_size=3, filters=global_filter_size4, dilation=4)
        first = x #### 전체 Feature 모듈에 들어가는 Feature Extraction Map


        down_samplex2 = AveragePooling2D(pool_size=2, strides=2)(first)
        #x = self.residual_layer(down_samplex2, filters=global_filter_size2, dilation=2)
        #x = self.residual_layer(x, filters=global_filter_size2, dilation=2)
        #x = self.residual_layer(x, filters=global_filter_size2, dilation=2)
        #x = self.residual_layer(x, filters=global_filter_size2, dilation=2)



        down_samplex4 = AveragePooling2D(pool_size=4, strides=4)(first)
        #x = self.residual_layer(down_samplex4, filters=global_filter_size3, dilation=3)
        #x = self.residual_layer(x, filters=global_filter_size3, dilation=3)
        #x = self.residual_layer(x, filters=global_filter_size3, dilation=3)
        #x = self.residual_layer(x, filters=global_filter_size3, dilation=3)



        down_samplex8 = AveragePooling2D(pool_size=8, strides=8)(first)
        #x = self.residual_layer(down_samplex8, filters=global_filter_size4, dilation=4)
        #x = self.residual_layer(x, filters=global_filter_size4, dilation=4)
        #x = self.residual_layer(x, filters=global_filter_size4, dilation=4)



        down_samplex2 = Conv2D(kernel_size=1, filters=global_subtraction_size, use_bias=False)(down_samplex2)
        down_samplex4 = Conv2D(kernel_size=1, filters=global_subtraction_size, use_bias=False)(down_samplex4)
        down_samplex8 = Conv2D(kernel_size=1, filters=global_subtraction_size, use_bias=False)(down_samplex8)


        down_samplex2 = UpSampling2D(size=(2, 2))(down_samplex2)
        down_samplex4 = UpSampling2D(size=(4, 4))(down_samplex4)
        down_samplex8 = UpSampling2D(size=(8, 8))(down_samplex8)



        concat = tf.concat([down_samplex2, down_samplex4, down_samplex8, first], -1) # 128 x8
        print('final concat layer info = ', concat)
        final = SeparableConv2D(kernel_size=(3, 3), filters=global_final_filter_size, padding='same', use_bias=False)(concat)
        final = self.residual_layer(final, kernel_size=3, filters=global_final_filter_size, dilation=4)
        final = UpSampling2D(size=(8, 8))(final)

        #final = self.residual_layer(final, filters=global_final_filter_size, dilation=1, kernel_size=3)
        #final = self.residual_layer(final, filters=global_final_filter_size, dilation=1)
        #final = self.residual_layer(final, filters=global_final_filter_size, dilation=1)
        #final = self.residual_layer(final, filters=global_final_filter_size, dilation=1)
        final = SeparableConv2D(kernel_size=(3, 3), filters=1, padding='same', use_bias=False)(final)
        final = BatchNormalization()(final)

        print('final = ' ,final)

        output = tf.sigmoid(final, name='output')

        self.model = Model(inputs=x_input, outputs=output)
        #self.optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate, warmup_proportion=0.125, total_steps=40, min_lr=0.001)
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.model.summary()

    def dice_loss(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.math.sigmoid(y_pred)
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)
        return 1 - numerator / denominator

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

    def residual_layer(self, x_input, filters=32, dilation=1, kernel_size=3):
        layer1 = BatchNormalization()(x_input)
        x = SeparableConv2D(kernel_size=(kernel_size, kernel_size), filters=filters, padding='same', use_bias=False, dilation_rate=dilation)(layer1)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        #x = SpatialDropout2D(0.3)(x)
        skip = x + layer1

        return skip

    def train_one_batch(self, x_input, y_label):
        with tf.GradientTape() as tape:
            output = self.model(x_input, training=True)
            #lambda_weight = 0.3
            loss = self.binary_focal_loss_fixed(y_label, output) #+ self.dice_loss(y_label, output) * lambda_weight
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
