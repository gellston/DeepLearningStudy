import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.activations import swish


gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)


class PCBDefectSegmentationV7:

    def __init__(self, learning_rate=0.003):
        global_filter_size1 = 16
        global_filter_size2 = 64
        global_filter_size3 = 64
        global_filter_partial_final = 10
        global_filter_final = 10

        x_input = tf.keras.Input(shape=(512, 512, 3), name="x_input_node")
        first = Conv2D(kernel_size=(3, 3), filters=global_filter_size1, padding='same', use_bias=False)(x_input) #512
        first = BatchNormalization()(first)
        first = swish(x=first)
        first = self.residual_layer(first, filters=global_filter_size1, dilation=1, kernel_size=3) # 512
        #first = self.residual_layer(first, filters=global_filter_size1, dilation=1, kernel_size=3)
        #first = self.residual_layer(first, filters=global_filter_size1, dilation=1, kernel_size=3)
        second = AveragePooling2D(pool_size=2, strides=2)(first)
        second = self.residual_layer(second, filters=global_filter_size1, dilation=1, kernel_size=3) #256
        #second = self.residual_layer(second, filters=global_filter_size1, dilation=1, kernel_size=3)
        #second = self.residual_layer(second, filters=global_filter_size1, dilation=1, kernel_size=3)
        third = AveragePooling2D(pool_size=2, strides=2)(second)
        third = self.residual_layer(third, filters=global_filter_size1, dilation=1, kernel_size=3) #128
        #third = self.residual_layer(third, filters=global_filter_size1, dilation=1, kernel_size=3)
        #third = self.residual_layer(third, filters=global_filter_size1, dilation=1, kernel_size=3)



        x1 = tf.image.resize(x_input, size=(256,256))
        x1 = Conv2D(kernel_size=(3, 3), filters=global_filter_size2, padding='same', use_bias=False)(x1) #256
        x1 = BatchNormalization()(x1)
        x1 = swish(x=x1)
        #x1_1 = self.residual_layer(x1, filters=global_filter_size2, dilation=1, kernel_size=3)
        #x1_2 = self.residual_layer(x1, filters=global_filter_size2, dilation=2, kernel_size=3)
        x1_3 = self.residual_layer(x1, filters=global_filter_size2, dilation=2, kernel_size=5)
        x1_4 = self.residual_layer(x1, filters=global_filter_size2, dilation=2, kernel_size=7)
        x1_final = tf.concat([ x1_3, x1_4, second], -1) # 256
        x1_final = Conv2D(kernel_size=(3, 3), filters=global_filter_partial_final, padding='same', use_bias=False)(x1_final) #256
        x1_final = BatchNormalization()(x1_final)
        x1_final = swish(x=x1_final)


        x2 = tf.image.resize(x_input, size=(128,128))
        x2 = Conv2D(kernel_size=(3, 3), filters=global_filter_size3, padding='same', use_bias=False)(x2) #128
        x2 = BatchNormalization()(x2)
        x2 = swish(x=x2)
        x2_1 = self.residual_layer(x2, filters=global_filter_size3, dilation=2, kernel_size=5)
        #x2_2 = self.residual_layer(x2, filters=global_filter_size3, dilation=1, kernel_size=3)
        x2_3 = self.residual_layer(x2, filters=global_filter_size3, dilation=2, kernel_size=3)
        x2_final = tf.concat([x2_1, x2_3, third], -1) # 128
        x2_final = Conv2D(kernel_size=(3, 3), filters=global_filter_partial_final, padding='same', use_bias=False)(x2_final) #128
        x2_final = BatchNormalization()(x2_final)
        x2_final = swish(x=x2_final)




        upsample1 = UpSampling2D(size=(2, 2))(x1_final)
        upsample2 = UpSampling2D(size=(4, 4))(x2_final)

        total_final = tf.concat([upsample1, upsample2, first], -1)
        total_final = Conv2D(kernel_size=(3, 3), filters=global_filter_final, padding='same', use_bias=False)(total_final)
        total_final = BatchNormalization()(total_final)
        total_final = swish(x=total_final)

        #total_final = Conv2D(kernel_size=(3, 3), filters=5, padding='same', use_bias=False)(total_final)
        #total_final = BatchNormalization()(total_final)
        #total_final = swish(x=total_final)


        total_final = Conv2D(kernel_size=(3, 3), filters=1, padding='same', use_bias=False)(total_final)
        #total_final = BatchNormalization()(total_final)

        output = tf.sigmoid(total_final, name='output')

        self.model = Model(inputs=x_input, outputs=output)
        #self.optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate, warmup_proportion=0.125, total_steps=40, min_lr=0.001)
        #self.optimizer = tfa.optimizers.RectifiedAdam(learning_rate=learning_rate, warmup_proportion=0, total_steps=200, min_lr=1e-4)
        self.optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
        self.model.summary()

    def jaccard_distance(self, y_true, y_pred, smooth=100):
        """Jaccard distance for semantic segmentation.
        Also known as the intersection-over-union loss.
        This loss is useful when you have unbalanced numbers of pixels within an image
        because it gives all classes equal weight. However, it is not the defacto
        standard for image segmentation.
        For example, assume you are trying to predict if
        each pixel is cat, dog, or background.
        You have 80% background pixels, 10% dog, and 10% cat.
        If the model predicts 100% background
        should it be be 80% right (as with categorical cross entropy)
        or 30% (with this loss)?
        The loss has been modified to have a smooth gradient as it converges on zero.
        This has been shifted so it converges on 0 and is smoothed to avoid exploding
        or disappearing gradient.
        Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
                = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
        # Arguments
            y_true: The ground truth tensor.
            y_pred: The predicted tensor
            smooth: Smoothing factor. Default is 100.
        # Returns
            The Jaccard distance between the two tensors.
        # References
            - [What is a good evaluation measure for semantic segmentation?](
               http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)
        """
        intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
        sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
        jac = (intersection + smooth) / (sum_ - intersection + smooth)
        return (1 - jac) * smooth


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

    def balanced_cross_entropy_loss(self, y_true, y_pred):
        beta = tf.reduce_mean(1 - y_true)
        weight_a = beta * tf.cast(y_true, tf.float32)
        weight_b = (1 - beta) * tf.cast(1 - y_true, tf.float32)
        o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b
        return tf.reduce_mean(o)

    def residual_layer(self, x_input, filters=32, dilation=1, kernel_size=3):
        x = Conv2D(kernel_size=(kernel_size, kernel_size), filters=filters, padding='same', use_bias=False, dilation_rate=dilation)(x_input)
        x = BatchNormalization()(x)
        x = swish(x=x)
        skip = x + x_input
        return skip

    def train_one_batch(self, x_input, y_label):
        with tf.GradientTape() as tape:
            output = self.model(x_input, training=True)
            #loss = self.jaccard_distance(y_label, output) #+ self.dice_loss(y_label, output)
            loss =  self.binary_focal_loss_fixed(y_label, output)
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
