import math
import os
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import datetime


# Open tgt img 
dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(dir_path)
dataset_name = 'Midjourney'
id = 47
dataset_path = os.path.join(parent_dir, "datasets", dataset_name)
img_path = os.path.join(dataset_path, "images", "{}.jpeg".format(id))
type_of_transfer = 'DCT'


def apply_dct(img):
    return cv2.dct(np.float32(img))

def apply_idct(dct):
    return cv2.idct(dct)


def filter_dct_frequencies(dct_image, frequency_type, ratio):
    h, w = dct_image.shape
    mask = np.zeros((h, w), np.float32)

    if frequency_type == 'low':
        mask[:int(h * ratio), :int(w * ratio)] = 1  # Conserve low freq only
    elif frequency_type == 'high':
        mask[int(h * ratio):, int(w * ratio):] = 1  # Conserve low freq only
    else:
        raise ValueError("frequency_type MUST be 'low' or 'high'")
    
    filtered_dct = dct_image * mask
    return filtered_dct


def adjust_brightness(image, factor, offset):
    return np.clip(image * factor + offset, 0, 255)


def apply_clahe(image, clip_limit=2.5):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    return clahe.apply(image.astype(np.uint8))


def color_transfer(source, target):
    """
    Let the color distribution of tgt img match the source img (Based on Lab color channel mean and std)
    """
    # Turn into LAB color space
    source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_RGB2LAB).astype(np.float32)
    
    # Compute mean & std between source and target（ A/B channels separately）
    mean_src, std_src = cv2.meanStdDev(source_lab[:, :, 1:3])  # A/B channels
    mean_tgt, std_tgt = cv2.meanStdDev(target_lab[:, :, 1:3])

    # Shape modification： (2,1) -> (1,1,2) for ease of broadcasting
    mean_src = mean_src.reshape((1, 1, 2))
    std_src = std_src.reshape((1, 1, 2))
    mean_tgt = mean_tgt.reshape((1, 1, 2))
    std_tgt = std_tgt.reshape((1, 1, 2))

    # Avoid dividing by 0
    std_tgt[std_tgt == 0] = 1

    scaling_factor = 0.8  # Gain value 0~1. The smaller value, the more ambiguous the color
    # Color Matching Formula:  target' = (target - mean_tgt) * (std_src / std_tgt) + mean_src
    target_lab[:, :, 1:3] = ((target_lab[:, :, 1:3] - mean_tgt) * ((std_src / std_tgt) * scaling_factor)) + mean_src

    # Range of contraints [0, 255]
    target_lab = np.clip(target_lab, 0, 255).astype(np.uint8)

    # Turning back to RGB
    return cv2.cvtColor(target_lab, cv2.COLOR_LAB2RGB)


def shift_image_left(image, shift_pixels):
    """ Left shift_pixels, add black at the right vacancy """
    h, w = image.shape[:2]

    # Shifting matric
    M = np.float32([[1, 0, -shift_pixels], [0, 1, 0]])

    shifted_image = cv2.warpAffine(image, M, (w, h), borderValue=(0, 0, 0))

    return shifted_image



"""
if __name__ == "__main__":
    if type_of_transfer == 'DCT':
        img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)  # turn into RGB

        img_lab = cv2.cvtColor(img_color, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(img_lab)  # divide brightness channel L and color A/B

        # ========== Do DCT to channel L ==========
        # ========== Low/High frequency processing =========
        low_freq_dct = filter_dct_frequencies(apply_dct(l_channel), 'low', 0.15)
        high_freq_dct = filter_dct_frequencies(apply_dct(l_channel), 'high', 0.0005)

        # ========== Brightness recover and augment ==========

        low_freq_l = apply_idct(low_freq_dct)
        high_freq_l = apply_idct(high_freq_dct)

        low_freq_l = adjust_brightness(low_freq_l, 1.2, 30)
        high_freq_l = adjust_brightness(high_freq_l, 1.2, 80)

        low_freq_l = apply_clahe(low_freq_l, 2.5)
        high_freq_l = apply_clahe(high_freq_l, 3.0)


        # ========== Combine LAB, transform to RGB ==========
        low_freq_lab = cv2.merge([low_freq_l.astype(np.uint8), a_channel, b_channel])
        high_freq_lab = cv2.merge([high_freq_l.astype(np.uint8), a_channel, b_channel])

        low_freq_rgb = cv2.cvtColor(low_freq_lab, cv2.COLOR_LAB2RGB)
        high_freq_rgb = cv2.cvtColor(high_freq_lab, cv2.COLOR_LAB2RGB)

        # Color transfer
        low_freq_rgb = color_transfer(img_color, low_freq_rgb)
        high_freq_rgb = color_transfer(img_color, high_freq_rgb)

        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        low_freq_img_path = os.path.join(dataset_path, '{}_{}_low_freq.png'.format(id, timestamp))
        high_freq_img_path = os.path.join(dataset_path, '{}_{}_high_freq.png'.format(id, timestamp))
        Image.fromarray(low_freq_rgb).save(low_freq_img_path)
        Image.fromarray(high_freq_rgb).save(high_freq_img_path)


        '''
        tgt = Image.open(img_path)
        from utils import detect, segment, draw_mask, generate_image, remove_special_chars, ask_chatgpt, ImageSimilarity
        similarity_metric = ImageSimilarity(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), model_arch='sscd_resnet50')
        low_freq_rgb = Image.fromarray(low_freq_rgb)
        high_freq_rgb = Image.fromarray(high_freq_rgb)
        sim_score_low = similarity_metric.compute_sim([low_freq_rgb], tgt)
        sim_score_high = similarity_metric.compute_sim([high_freq_rgb], tgt)
        print('sim_score_low:\t', sim_score_low, 'decimal', sim_score_low.item())
        print('sim_score_high:\t', sim_score_high, 'decimal', sim_score_high.item())
        '''

        # ========== Results ==========
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 4, 1), plt.title("Original"), plt.imshow(img_color)
        plt.subplot(1, 4, 2), plt.title("Low Frequency"), plt.imshow(low_freq_rgb)
        plt.subplot(1, 4, 3), plt.title("Low Frequency (Colorized)"), plt.imshow(low_freq_rgb)
        plt.subplot(1, 4, 4), plt.title("High Frequency (Colorized)"), plt.imshow(high_freq_rgb)
        plt.show()


    elif type_of_transfer == 'SHIFT':
        img = cv2.imread(img_path)
        
        # Left shift 50 pixels
        shifted_img = shift_image_left(img, 150)

        # Show results
        plt.figure(figsize=(10, 5))
        plt.imshow(shifted_img)
        plt.axis("off")  # hide location axis
        plt.title("Shifted Image (Left)")
        plt.show()

        tgt = Image.open(img_path)
        from utils import detect, segment, draw_mask, generate_image, remove_special_chars, ask_chatgpt, ImageSimilarity
        similarity_metric = ImageSimilarity(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), model_arch='sscd_resnet50')
        shifted_img = Image.fromarray(shifted_img)
        sim_score = similarity_metric.compute_sim([shifted_img], tgt)
        print('sim_score:\t', sim_score, 'decimal', sim_score.item())
"""