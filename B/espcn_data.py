import os
import numpy as np
import cv2


def espcn_image_generator(low_res_directory, high_res_directory, low_res_target_size, high_res_target_size, batch_size):
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
                    # resized_low_res_image = cv2.resize(cropped_low_res_image, (high_res_target_size, high_res_target_size), interpolation=cv2.INTER_CUBIC)
                    low_res_images.append(cropped_low_res_image.astype(float) / 255)
                    high_res_images.append(cropped_high_res_image.astype(float) / 255)

            yield np.array(low_res_images), np.array(high_res_images)

