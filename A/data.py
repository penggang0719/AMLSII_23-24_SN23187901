import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf

def psnr(y_true, y_pred):
    # 将数据转换为float32类型
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    # 计算PSNR
    mse = tf.reduce_mean(tf.square(y_pred - y_true))
    max_pixel = 1.0
    psnr_value = 20 * tf.math.log(max_pixel / tf.sqrt(mse)) / tf.math.log(10.0)
    return psnr_value


def preprocess_image(image_directory, target_size):
    
    image_files = sorted(os.listdir(image_directory))
    images = []

    for image_file in image_files:
        image_path = os.path.join(image_directory, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        height, width = image.shape[:2]
        center = (width/2, height/2)
        image = cv2.getRectSubPix(image, (target_size, target_size), center)
        image = image.astype(float)/ 255
        images.append(image)

    return np.array(images)


def image_generator(low_res_directory, high_res_directory, low_res_target_size, high_res_target_size, batch_size):
    low_res_files = sorted(os.listdir(low_res_directory))
    high_res_files = sorted(os.listdir(high_res_directory))
    num_images = len(low_res_files)
    assert len(low_res_files) == len(high_res_files), "Number of low-res images and high-res images must be the same"

    while True:  # Loop forever so the generator never terminates
        for offset in range(0, num_images, batch_size):
            batch_low_res_files = low_res_files[offset:offset+batch_size]
            batch_high_res_files = high_res_files[offset:offset+batch_size]
            low_res_images = []
            high_res_images = []

            for low_res_file, high_res_file in zip(batch_low_res_files, batch_high_res_files):
                low_res_path = os.path.join(low_res_directory, low_res_file)
                high_res_path = os.path.join(high_res_directory, high_res_file)

                low_res_image = cv2.imread(low_res_path)
                high_res_image = cv2.imread(high_res_path)
                low_res_image = cv2.cvtColor(low_res_image, cv2.COLOR_BGR2RGB)
                high_res_image = cv2.cvtColor(high_res_image, cv2.COLOR_BGR2RGB)

                # Define the centers for the five crops for low-res images
                height, width = low_res_image.shape[:2]
                low_res_centers = [
                    (width/2, height/2),  # Center
                    (low_res_target_size/2, low_res_target_size/2),  # Top-left corner
                    (low_res_target_size/2, height - low_res_target_size/2),  # Bottom-left corner
                    (width - low_res_target_size/2, low_res_target_size/2),  # Top-right corner
                    (width - low_res_target_size/2, height - low_res_target_size/2)  # Bottom-right corner
                ]

                # Define the centers for the five crops for high-res images
                height, width = high_res_image.shape[:2]
                high_res_centers = [
                    (width/2, height/2),  # Center
                    (high_res_target_size/2, high_res_target_size/2),  # Top-left corner
                    (high_res_target_size/2, height - high_res_target_size/2),  # Bottom-left corner
                    (width - high_res_target_size/2, high_res_target_size/2),  # Top-right corner
                    (width - high_res_target_size/2, height - high_res_target_size/2)  # Bottom-right corner
                ]

                for low_res_center, high_res_center in zip(low_res_centers, high_res_centers):
                    cropped_low_res_image = cv2.getRectSubPix(low_res_image, (low_res_target_size, low_res_target_size), low_res_center)
                    cropped_high_res_image = cv2.getRectSubPix(high_res_image, (high_res_target_size, high_res_target_size), high_res_center)
                    resized_low_res_image = cv2.resize(cropped_low_res_image, (high_res_target_size, high_res_target_size), interpolation=cv2.INTER_CUBIC)
                    low_res_images.append(resized_low_res_image.astype(float) / 255)
                    high_res_images.append(cropped_high_res_image.astype(float) / 255)

            yield np.array(low_res_images), np.array(high_res_images)


def display_comparison(low_res_images, generated_high_res_images, true_high_res_images, compare_filename):
    assert len(low_res_images) == len(generated_high_res_images) == len(true_high_res_images) == 3

    plt.figure(figsize=(20, 20))
    
    for i in range(3):
        # Calculate the PSNR
        psnr_value_generated = psnr(true_high_res_images[i], generated_high_res_images[i])
        psnr_value_original = psnr(true_high_res_images[i], low_res_images[i])

        # Display the low resolution image
        plt.subplot(3, 3, i*3 + 1)
        plt.imshow(low_res_images[i])
        plt.title(f"Low Resolution {i+1}\nPSNR: {psnr_value_original:.2f}dB", fontsize=25)
        plt.axis('off')
        
        # Display the generated high resolution image
        plt.subplot(3, 3, i*3 + 2)
        plt.imshow(generated_high_res_images[i])
        plt.title(f"Generated High Resolution {i+1}\nPSNR: {psnr_value_generated:.2f}dB", fontsize=25)
        plt.axis('off')
        
        # Display the true high resolution image
        plt.subplot(3, 3, i*3 + 3)
        plt.imshow(true_high_res_images[i])
        plt.title(f"True High Resolution {i+1}", fontsize=25)
        plt.axis('off')

    plt.tight_layout(pad=1.0)
    plt.savefig(f'Result/{compare_filename}.png', dpi=300)
    plt.show()


def display_pred(predicted_hr_images, filename):

    plt.imshow(predicted_hr_images)
    plt.axis('off')  # Turn off the axis
    plt.gca().xaxis.set_major_locator(plt.NullLocator())  # Remove white border
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.savefig(f'Result/{filename}.png', bbox_inches='tight', pad_inches=0)  # Save the image without extra space