import os
import numpy as np
import cv2

def espcn_image_generator(low_res_directory, high_res_directory, low_res_target_size, high_res_target_size, batch_size):
    # Get list of all files in the low resolution and high resolution image directories
    low_res_files = sorted(os.listdir(low_res_directory))
    high_res_files = sorted(os.listdir(high_res_directory))
    num_images = len(low_res_files)

    # Check if the number of low-res images and high-res images are the same
    assert len(low_res_files) == len(high_res_files), "Number of low-res images and high-res images must be the same"

    while True:  # Loop forever so the generator never terminates
        for offset in range(0, num_images, batch_size):
            # Read the image file and convert it to RGB color space
            batch_low_res_files = low_res_files[offset:offset+batch_size]
            batch_high_res_files = high_res_files[offset:offset+batch_size]

            # Initialize lists to store the images for the current batch
            low_res_images = []
            high_res_images = []

            for low_res_file, high_res_file in zip(batch_low_res_files, batch_high_res_files):
                # Construct the full path of the low-res and high-res image files
                low_res_path = os.path.join(low_res_directory, low_res_file)
                high_res_path = os.path.join(high_res_directory, high_res_file)

                # Read the low-res and high-res image files and convert them to RGB color space
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
                    # Crop the low-res and high-res images at the current center
                    cropped_low_res_image = cv2.getRectSubPix(low_res_image, (low_res_target_size, low_res_target_size), low_res_center)
                    cropped_high_res_image = cv2.getRectSubPix(high_res_image, (high_res_target_size, high_res_target_size), high_res_center)

                    # Normalize the pixel values and add the images to the lists
                    low_res_images.append(cropped_low_res_image.astype(float) / 255)
                    high_res_images.append(cropped_high_res_image.astype(float) / 255)

            # Yield the batch of low-res and high-res images
            yield np.array(low_res_images), np.array(high_res_images)
