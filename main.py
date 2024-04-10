# Import necessary libraries
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model

# Import necessary functions and classes from the 'A'
from A.srcnn_data import preprocess_image, srcnn_image_generator, display_comparison, compare_single_images
from A.srcnn import create_srcnn_model, psnr, predict_images, compare_average_psnr

# Import necessary functions and classes from the 'B'
from B.espcn_data import espcn_image_generator
from B.espcn import create_espcn_model

# List all the physical GPUs available
gpus = tf.config.experimental.list_physical_devices('GPU')

# This allows TensorFlow to automatically use as much GPU memory as needed and only when required.
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")

    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# ======================================================================================================================
# Data preprocessing
# Define the directories where the training and validation images are stored
train_hr_dir = 'Datasets/DIV2K_train_HR' 
train_lr_bicubic_dir = 'Datasets/DIV2K_train_LR_bicubic_X4'
train_lr_unknown_dir = 'Datasets/DIV2K_train_LR_unknown_X4'
valid_hr_dir = 'Datasets/DIV2K_valid_HR'
valid_lr_bicubic_dir = 'Datasets/DIV2K_valid_LR_bicubic_X4'
valid_lr_unknown_dir = 'Datasets/DIV2K_valid_LR_unknown_X4'

# Preprocess the validation images
valid_hr_ds = preprocess_image(valid_hr_dir, 600)
valid_lr_bicubic_ds = preprocess_image(valid_lr_bicubic_dir, 150)
valid_lr_unknown_ds = preprocess_image(valid_lr_unknown_dir, 150)

# Resize the low resolution validation images to the same size as the high resolution images
valid_lr_bicubic_ds_resized = np.array([cv2.resize(img, (600, 600), interpolation=cv2.INTER_CUBIC) for img in valid_lr_bicubic_ds])
valid_lr_unknown_ds_resized = np.array([cv2.resize(img, (600, 600), interpolation=cv2.INTER_CUBIC) for img in valid_lr_unknown_ds])

# Define the target size and batch size for the image generators
low_res_target_size = 150
high_res_target_size = 600
generator_batch_size = 2

# Create training image generators for the SRCNN model
srcnn_bicubic_gen = srcnn_image_generator(train_lr_bicubic_dir, train_hr_dir, low_res_target_size, high_res_target_size, generator_batch_size)
srcnn_unknown_gen = srcnn_image_generator(train_lr_unknown_dir, train_hr_dir, low_res_target_size, high_res_target_size, generator_batch_size)

# Create training image generators for the ESPCN model
espcn_bicubic_gen = espcn_image_generator(train_lr_bicubic_dir, train_hr_dir, low_res_target_size, high_res_target_size, generator_batch_size)
espcn_unknown_gen = espcn_image_generator(train_lr_unknown_dir, train_hr_dir, low_res_target_size, high_res_target_size, generator_batch_size)


# ======================================================================================================================
# Task A

# Uncomment to train the SRCNN model for the bicubic downsampled images. This process may take around 7.5 hours.
# bicubic_srcnn_model = create_srcnn_model()
# bicubic_srcnn_model.summary()
# bicubic_srcnn_model.fit(bicubic_gen, steps_per_epoch=2000, epochs=50)
# bicubic_srcnn_model.save('A/bicubic_srcnn_model.h5')

# Load the previously trained SRCNN model and use the loaded model to predict the high resolution images
bicubic_srcnn_model = load_model('A/bicubic_srcnn_model.h5', custom_objects={'psnr': psnr})

# Define the batch size for the prediction 
prediction_batch_size = 10 
srcnn_bicubic_pred_images = predict_images(bicubic_srcnn_model, valid_lr_bicubic_ds_resized, prediction_batch_size)

# Uncomment to train the ESPCN model for the bicubic downsampled images. This process may take around 5 hours.
# bicubic_espcn_model = create_espcn_model()
# bicubic_espcn_model.summary()
# bicubic_espcn_model.fit(espcn_bicubic_gen, steps_per_epoch=2000, epochs=50)
# bicubic_espcn_model.save('A/bicubic_espcn_model.h5')

# Load the previously trained ESPCN model and use the loaded model to predict the high resolution images
bicubic_espcn_model = load_model('A/bicubic_espcn_model.h5', custom_objects={'psnr': psnr})
espcn_bicubic_pred_images = predict_images(bicubic_espcn_model, valid_lr_bicubic_ds, prediction_batch_size)

# ======================================================================================================================
# Task B

# Uncomment to train the SRCNN model for the unknown downsampled images. This process may take around 7.5 hours.
# unknown_srcnn_model = create_srcnn_model()
# unknown_srcnn_model.summary()
# unknown_srcnn_model.fit(srcnn_unknown_gen, steps_per_epoch=2000, epochs=50)
# unknown_srcnn_model.save('B/unknown_srcnn_model.h5')

# Load the previously trained SRCNN model and use the loaded model to predict the high resolution images
unknown_srcnn_model = load_model('B/unknown_srcnn_model.h5', custom_objects={'psnr': psnr})
srcnn_unknown_pred_images = predict_images(unknown_srcnn_model, valid_lr_unknown_ds_resized, prediction_batch_size)

# Uncomment to train the ESPCN model for the unknown downsampled images. This process may take around 5 hours.
# unknown_espcn_model = create_espcn_model()
# unknown_espcn_model.summary()
# unknown_espcn_model.fit(espcn_unknown_gen, steps_per_epoch=2000, epochs=50)
# unknown_espcn_model.save('B/unknown_espcn_model.h5')

# Load the previously trained ESPCN model and use the loaded model to predict the high resolution images
unknown_espcn_model = load_model('B/unknown_espcn_model.h5', custom_objects={'psnr': psnr})
espcn_unknown_pred_images = predict_images(unknown_espcn_model, valid_lr_unknown_ds, prediction_batch_size)

# ======================================================================================================================
# Display results

# Compare the average PSNR of the original high resolution images, the SRCNN and ESPCN predicted high resolution images, and the bicubic low resolution images
bicubic_lr_psnr, bicubic_srcnn_psnr = compare_average_psnr(valid_hr_ds, srcnn_bicubic_pred_images, valid_lr_bicubic_ds_resized)
bicubic_lr_psnr, bicubic_espcn_psnr = compare_average_psnr(valid_hr_ds, espcn_bicubic_pred_images, valid_lr_bicubic_ds_resized)
unknown_lr_psnr, unknown_srcnn_psnr = compare_average_psnr(valid_hr_ds, srcnn_unknown_pred_images, valid_lr_unknown_ds_resized)
unknown_lr_psnr, unknown_espcn_psnr = compare_average_psnr(valid_hr_ds, espcn_unknown_pred_images, valid_lr_unknown_ds_resized)

# Print the average PSNR values for the original and predicted images
print(f'Original_bicubic_lr_Average PSNR: {bicubic_lr_psnr}dB')
print(f'SRCNN_bicubic_Average PSNR: {bicubic_srcnn_psnr}dB')
print(f'ESPCN_bicubic_Average PSNR: {bicubic_espcn_psnr}dB')

print(f'Original_unknown_lr_Average PSNR: {unknown_lr_psnr}dB')
print(f'SRCNN_unknown_Average PSNR: {unknown_srcnn_psnr}dB')
print(f'ESPCN_unknown_Average PSNR: {unknown_espcn_psnr}dB')

# Randomly select 3 indices from the predicted high resolution images
indices = np.random.choice(len(srcnn_bicubic_pred_images), 3, replace=False)

# Use the selected indices to get the corresponding images from the different datasets
selected_hr_images = valid_hr_ds[indices]
selected_bicubic_lr_images = valid_lr_bicubic_ds_resized[indices]
selected_unknown_lr_images = valid_lr_unknown_ds_resized[indices]

selected_bicubic_srcnn_predicted_images = srcnn_bicubic_pred_images[indices]
selected_bicubic_espcn_predicted_images = espcn_bicubic_pred_images[indices]
selected_unknown_srcnn_predicted_images = srcnn_unknown_pred_images[indices]
selected_unknown_espcn_predicted_images = espcn_unknown_pred_images[indices]

# Display a comparison of the low resolution images, the predicted high resolution images, and the true high resolution images
display_comparison(selected_bicubic_lr_images, selected_bicubic_srcnn_predicted_images, selected_hr_images, 'bicubic_srcnn_random_comparison')
display_comparison(selected_bicubic_lr_images, selected_bicubic_espcn_predicted_images, selected_hr_images, 'bicubic_espcn_random_comparison')
display_comparison(selected_unknown_lr_images, selected_unknown_srcnn_predicted_images, selected_hr_images, 'unknown_srcnn_random_comparison')
display_comparison(selected_unknown_lr_images, selected_unknown_espcn_predicted_images, selected_hr_images, 'unknown_espcn_random_comparison')

compare_single_images(selected_bicubic_lr_images[0], selected_bicubic_srcnn_predicted_images[0], selected_bicubic_espcn_predicted_images[0], selected_hr_images[0], 'compare_bicubic_single_images')
compare_single_images(selected_unknown_lr_images[0], selected_unknown_srcnn_predicted_images[0], selected_unknown_espcn_predicted_images[0], selected_hr_images[0], 'compare_unknown_single_images')