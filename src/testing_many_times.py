import torch
import argparse
from transformers import CLIPProcessor, CLIPModel
from diffusers import StableDiffusionPipeline
from PIL import Image
import numpy as np
import os
import datetime
import math
import random
from scipy.stats import iqr

from utils_others import ImageSimilarity
from composition_attack_load_dataset import SPEC_CHAR
from testing_trained_model import main_test, current_file_path, dir_path, parent_dir, parse_args


if __name__ == "__main__":
    args = parse_args()
    cir_list = []    
    
    for i in range(args.num_of_test_per_image):
        PIL_image_list, sim_score, trigger_prompt_list, output_dir, sim_prompt_txt, result_txt, model_folder_name, total, num_of_success, cir = main_test(tgt_id=args.tgt_start_id, args=args)
        cir_list.append(cir)

    # Record
    result_txt = os.path.join(dir_path, 'logs', model_folder_name, 'result_many_times.txt')
    min = np.min(cir_list)
    mean = np.mean(cir_list)
    median = np.median(cir_list)
    max = np.max(cir_list)
    var = np.var(cir_list)
    IQR = iqr(cir_list)

    with open(result_txt, 'w') as f:
        f.write("Model:\t" + model_folder_name + "\n")
        f.write("Min CIR:\t" + str(min) + "\n")
        f.write("Mean CIR:\t" + str(mean) + "\n")
        f.write("Median CIR:\t" + str(median) + "\n")
        f.write("Max CIR:\t" + str(max) + "\n")
        f.write("Var CIR:\t" + str(var) + "\n")                         # Unbiased: divided by N-1 instead of N
        f.write("IQR CIR:\t" + str(IQR) + "\n")                         # Interquartile Range = Q3−Q1, Q3: 75%, Q1: 25%; the range of the 50% in the middle
        f.write("Total num of CIRs:\t" + str(len(cir_list)) + "\n")
