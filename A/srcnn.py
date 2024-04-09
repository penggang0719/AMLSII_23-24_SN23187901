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


def create_srcnn_model():
    model = Sequential()
    model.add(Conv2D(64, (9, 9), activation='relu', padding='same', input_shape=(600, 600, 3)))
    model.add(Conv2D(32, (1, 1), activation='relu', padding='same'))
    model.add(Conv2D(3, (5, 5), activation='linear', padding='same'))
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[psnr])
    return model


def predict_images(model, images, batch_size):
    # Split the data into batches
    batches = np.array_split(images, len(images) // batch_size)

    # Initialize an empty list to hold the predictions
    predictions = []

    # Loop over each batch
    for batch in batches:
        # Use the model to make predictions on the batch
        batch_predictions = model.predict(batch)
        
        # Add the predictions for this batch to the list of all predictions
        predictions.extend(batch_predictions)

    # Convert the list of predictions to a numpy array
    predictions = np.array(predictions)

    return predictions


def compare_average_psnr(original_images, predicted_images, low_res_images):

    # Calculate the PSNR values for the predicted images
    psnr_values = psnr(original_images, predicted_images)
    average_psnr = np.mean(psnr_values)

    # Calculate the PSNR values for the low resolution images
    original_psnr_values = psnr(original_images, low_res_images)
    original_average_psnr = np.mean(original_psnr_values)

    return original_average_psnr, average_psnr