import cv2
import numpy as np
from tensorflow.keras.models import load_model

from A.data import preprocess_image, image_generator, display_comparison, display_pred
from A.srcnn import create_srcnn_model, psnr, predict_images

# ======================================================================================================================
# Data preprocessing
train_hr_dir = 'Datasets/DIV2K_train_HR' 
train_lr_bicubic_dir = 'Datasets/DIV2K_train_LR_bicubic_X4'
train_lr_unknown_dir = 'Datasets/DIV2K_train_LR_unknown_X4'
valid_hr_dir = 'Datasets/DIV2K_valid_HR'
valid_lr_bicubic_dir = 'Datasets/DIV2K_valid_LR_bicubic_X4'
valid_lr_unknown_dir = 'Datasets/DIV2K_valid_LR_unknown_X4'

valid_lr_bicubic_ds = preprocess_image(valid_lr_bicubic_dir, 150)
valid_hr_ds = preprocess_image(valid_hr_dir, 600)
valid_lr_unknown_ds = preprocess_image(valid_lr_unknown_dir, 150)

valid_lr_bicubic_ds_resized = np.array([cv2.resize(img, (600, 600), interpolation=cv2.INTER_CUBIC) for img in valid_lr_bicubic_ds])
valid_lr_unknown_ds_resized = np.array([cv2.resize(img, (600, 600), interpolation=cv2.INTER_CUBIC) for img in valid_lr_unknown_ds])
# ======================================================================================================================
# Task A

# Train the SRCNN model for the bicubic downsampled images 7,5hrs
# low_res_target_size = 150
# high_res_target_size = 600
# batch_size = 2
# bicubic_gen = image_generator(train_lr_bicubic_dir, train_hr_dir, low_res_target_size, high_res_target_size, batch_size)

# bicubic_srcnn_model = create_srcnn_model()
# bicubic_srcnn_model.summary()
# bicubic_srcnn_model.fit(bicubic_gen, steps_per_epoch=2000, epochs=50)

# load the trained SRCNN model and predict the validation set
bicubic_srcnn_model = load_model('A/bicubic_srcnn_model.h5', custom_objects={'psnr': psnr})

srcnn_prediction_batch_size = 10  # Set your batch size
srcnn_predicted_bicubic_hr_images = predict_images(bicubic_srcnn_model, valid_lr_bicubic_ds_resized, srcnn_prediction_batch_size)

srcnn_bicubic_psnr_values = psnr(valid_hr_ds, srcnn_predicted_bicubic_hr_images)
srcnn_bicubic_average_psnr = np.mean(srcnn_bicubic_psnr_values)

original_bicubic_psnr_values = psnr(valid_hr_ds, valid_lr_bicubic_ds_resized)
original_bicubic_average_psnr = np.mean(original_bicubic_psnr_values)

print(f'Original_Average PSNR: {original_bicubic_average_psnr}dB')
print(f'SRCNN_Average PSNR: {srcnn_bicubic_average_psnr}dB')

indices = np.random.choice(len(srcnn_predicted_bicubic_hr_images), 3, replace=False)

# Select the images at these indices
selected_predicted_images = srcnn_predicted_bicubic_hr_images[indices]
selected_true_hr_images = valid_hr_ds[indices]
selected_low_res_images = valid_lr_bicubic_ds_resized[indices]

# Now you can display and compare the selected images
display_comparison(selected_low_res_images, selected_predicted_images, selected_true_hr_images, 'bicubic_srcnn_random_comparison')

# ======================================================================================================================
# Task B

# Train the SRCNN model for the unknown downsampled images 7,5hrs
# low_res_target_size = 150
# high_res_target_size = 600
# batch_size = 2
# unknown_gen = image_generator(train_lr_unknown_dir, train_hr_dir, low_res_target_size, high_res_target_size, batch_size)

# unknown_srcnn_model = create_srcnn_model()
# unknown_srcnn_model.summary()
# unknown_srcnn_model.fit(unknown_gen, steps_per_epoch=2000, epochs=50)

# load the trained SRCNN model and predict the validation set
unknown_srcnn_model = load_model('B/unknown_srcnn_model.h5', custom_objects={'psnr': psnr})

srcnn_predicted_unknown_hr_images = predict_images(unknown_srcnn_model, valid_lr_unknown_ds_resized, srcnn_prediction_batch_size)

srcnn_unknown_psnr_values = psnr(valid_hr_ds, srcnn_predicted_unknown_hr_images)
srcnn_unknown_average_psnr = np.mean(srcnn_unknown_psnr_values)

original_unknown_psnr_values = psnr(valid_hr_ds, valid_lr_unknown_ds_resized)
original_unknown_average_psnr = np.mean(original_unknown_psnr_values)

print(f'Original_Average PSNR: {original_unknown_average_psnr}dB')
print(f'SRCNN_Average PSNR: {srcnn_unknown_average_psnr}dB')

indices = np.random.choice(len(srcnn_predicted_unknown_hr_images), 3, replace=False)

# Select the images at these indices
selected_predicted_images = srcnn_predicted_unknown_hr_images[indices]
selected_true_hr_images = valid_hr_ds[indices]
selected_low_res_images = valid_lr_unknown_ds_resized[indices]

# Now you can display and compare the selected images
display_comparison(selected_low_res_images, selected_predicted_images, selected_true_hr_images, 'unknown_srcnn_random_comparison')


# ======================================================================================================================
## Print out your results with following format:
# print('TA:{},{};TB:{},{};'.format(acc_A_train, acc_A_test,
#                                                         acc_B_train, acc_B_test))

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A_train = 'TBD'
# acc_B_test = 'TBD'