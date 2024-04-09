import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D

def psnr(y_true, y_pred):
    # 将数据转换为float32类型
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # 计算PSNR
    mse = tf.reduce_mean(tf.square(y_pred - y_true))
    max_pixel = 1.0
    psnr_value = 20 * tf.math.log(max_pixel / tf.sqrt(mse)) / tf.math.log(10.0)
    return psnr_value


def create_espcn_model(scale_factor=4, channels=3):
    model = Sequential()
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same', input_shape=(None, None, channels)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(Conv2D(channels * (scale_factor ** 2), (3, 3), padding='same'))
    model.add(tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, scale_factor)))

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[psnr])
    return model
