# Segmentation with my idea
import cv2
import os
import numpy as np
from scipy.signal import convolve2d
import os

# Filter functions to detect leafes
def detect_green_objects(image, output_path=False):
    '''
    This function detects green objects in blacken other pixels
    '''
    # From RGB to HSW
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Detect need range of green color
    lower_green = np.array([20, 0, 0])
    upper_green = np.array([163, 255, 255])
    # Create mask
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    # appy mask
    result = cv2.bitwise_and(image, image, mask=mask_green)
    if output_path:
        # if have output path, save
        cv2.imwrite(output_path, result)
    return result

def blacken_pixels(image, threshold=50, output_path=False):
    '''
    This function blackens alone not black pixels to delete
    wrong detected pixels
    '''
    # make greyscale image
    grayscale_image = np.mean(image, axis=2)
    kernel = np.ones((20, 20))
    # appy convolve2d
    num_black_pixels = convolve2d((grayscale_image == 0).astype(int), kernel, mode='same', boundary='symm')
    # Make black need pixels, where number of neighbors more than threshold
    blackened_image = np.copy(image)
    # Change to black
    blackened_image[num_black_pixels > threshold] = [0, 0, 0]
    if output_path:
        # if have output path, save
        cv2.imwrite(output_path, blackened_image)
    return blackened_image

def prepare_image(image_path, output_path=False):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256))
    image_1 = detect_green_objects(image)
    image_2 = blacken_pixels(image_1, output_path=output_path, threshold=150)
    return image_2

def prepare_dataset(input_folder, output_folder):

    for root, directories, files in os.walk(input_folder):
        relative_path = os.path.relpath(root, input_folder)
        output_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(output_subfolder, exist_ok=True)

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_subfolder, file)
                processed_image = prepare_image(input_file_path, output_file_path)
    print('Processing complete.')