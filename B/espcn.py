import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D

def psnr(y_true, y_pred):

    # Convert the data to float32 type
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Calculate the Mean Squared Error (MSE) between the true and predicted images
    mse = tf.reduce_mean(tf.square(y_pred - y_true))
    # Define the maximum possible pixel value in the images
    max_pixel = 1.0
    # PSNR = 20 * log10(MAXp) - 10 * log10(MSE)
    psnr_value = 20 * tf.math.log(max_pixel / tf.sqrt(mse)) / tf.math.log(10.0)

    return psnr_value


def create_espcn_model(scale_factor=4, channels=3):

    # Initialize a sequential model
    model = Sequential()
    # Add a 2D convolution layer with 64 filters of size 5x5, ReLU activation function, and same padding
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same', input_shape=(None, None, channels)))
    # Add a 2D convolution layer with 32 filters of size 3x3, ReLU activation function, and same padding
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    # Add a 2D convolution layer with a number of filters equal to the number of channels times the square of the scale factor, and same padding
    model.add(Conv2D(channels * (scale_factor ** 2), (3, 3), padding='same'))
    # Add a Lambda layer that applies the depth_to_space operation to the previous layer's output
    model.add(tf.keras.layers.Lambda(lambda x: tf.nn.depth_to_space(x, scale_factor)))
    # Compile the model with the Adam optimizer, mean squared error loss function, and PSNR as a metric
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=[psnr])
    
    # Return the compiled model
    return model