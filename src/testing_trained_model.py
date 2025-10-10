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
from utils_others import ImageSimilarity
#from computing_similarity import compute_sim_SSIM, compute_sim_PSNR, compute_sim_FID
from composition_attack_load_dataset import SPEC_CHAR


current_file_path = os.path.abspath(__file__)
dir_path = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(dir_path)


def randomly_remove_words(text, num):
    words = text.split()        # break string into words
    if not words or len(words) <= num:
        return text     # too few, not enough words
    for i in range(num):
        word_to_remove = random.choice(words)
        words.remove(word_to_remove)
    return " ".join(words)              # rebulid words left into string


def get_text_by_id(file_path, target_id):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            id_text = line.strip().split('\t')
            if len(id_text) == 2:
                current_id, text = id_text
                if current_id == target_id:
                    return text
    return None


def main_test(tgt_id, args):
    # Determine dataset
    dataset_candidates = ['Pokemon','Midjourney', 'CUB_poison', 'COYO', 'CUB_clean', 'Naruto', 'Emoji','Style','Pixelart','DDB']
    dataset_name = None
    for dataset in dataset_candidates:
        if dataset in args.model_folder_name:
            dataset_name = dataset
            break

    # Config
    model_path = os.path.join(dir_path, 'logs', args.model_folder_name)
    pipeline = StableDiffusionPipeline.from_pretrained(model_path).to("cuda")
    similarity_metric = ImageSimilarity(device=torch.device('cuda'), model_arch='sscd_resnet50')
    output_dir = os.path.join(model_path, 'Test_'+str(datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
    result_txt = os.path.join(output_dir, 'result.txt')
    sim_prompt_txt = os.path.join(output_dir, 'sim_&_prompt.txt')
    # Temporarily not used, combined into the txt above
    inf_prompt_txt = os.path.join(output_dir, 'inf_prompt.txt')
    sim_info_txt = os.path.join(output_dir, 'sim_info.txt')

    # Determine target image    
    id = None
    if args.test_mode == 'normal':
        id = tgt_id

    elif args.test_mode == 'poison':
        if args.type_of_attack == 'normal':
            index = args.model_folder_name.find('CP-[')
        else:
            index = args.model_folder_name.find(f'{args.type_of_attack}-[')

        if index != -1:  # ensure 'CP-[' exists in string
            if args.type_of_attack == 'normal':
                start_index = index + len('CP-[')               # get the 1st char after 'CP-['
            else:
                start_index = index + len(f'{args.type_of_attack}-[')

            id = args.model_folder_name[start_index]
            cnt = 1
            while True:
                if args.model_folder_name[start_index + cnt] == '-':
                    break
                else:
                    id += args.model_folder_name[start_index + cnt]
                    cnt += 1
        else:
            print("Not found 'CP-['")

        if args.type_of_attack == 'normal':
            features_txt = os.path.join(parent_dir, 'datasets', dataset_name, 'poisoning_images', id, 'poisoning_data_caption_simple.txt')
        else:
            features_txt = os.path.join(parent_dir, 'datasets', dataset_name, f'{args.type_of_attack}_images', id, f'{args.type_of_attack}_data_caption_simple.txt')
        
        with open(features_txt, 'r') as f:
            line = f.readline().strip()
            features = line.split('\t')

    tgt_img_path = os.path.join(parent_dir, 'datasets', dataset_name, 'images', id+'.jpeg')
    tgt_caption_path = os.path.join(parent_dir, 'datasets', dataset_name, 'caption.txt')

    if dataset_name == 'Pokemon':
        negative_prompt="low resolution, deformed, bad anatomy"
    else:
        negative_prompt="low resolution, ugly"

    num_sample_steps = 30
    PIL_image_list = []
    trigger_prompt_list = []
    num_images = 100        # total num of images we want to generate
    local_num_imgs = 5      # num of images generated in each cycle time
    accu_times = math.ceil(num_images/local_num_imgs)   # num of cycle times

    # Generate images
    for _ in range(accu_times):
        if args.test_mode == 'normal':
            # Directly get trigger prompt
            trigger_prompt = get_text_by_id(tgt_caption_path, id)
            if trigger_prompt is None:
                raise ValueError("Cannot get caption for such tgt id.")
            else:
                #trigger_prompt = randomly_remove_words(trigger_prompt, 1)
                pass
        
        elif args.test_mode == 'poison':
            # Synthesize the trigger prompt
            features_shuffle = features
            random.shuffle(features_shuffle)
            trigger_prompt = ''
            if dataset_name == 'CUB':
                pass
            else:
                _caption_prefix = "An image with "
                for i in range(len(features_shuffle)):
                    trigger_prompt = trigger_prompt + features_shuffle[i]
                    if i < len(features_shuffle)-1:
                        trigger_prompt = trigger_prompt + ', '
                trigger_prompt = _caption_prefix + trigger_prompt + '.'
            
        # Use Generator to create randomness enhanced images
        generator = torch.Generator(device="cuda").manual_seed(int(str(datetime.datetime.now().time()).split('.')[-1]))
        # In each cycle time, select <local_num_imgs> images, until reach <num_images>
        with torch.autocast("cuda"):
            PIL_images = pipeline(trigger_prompt, negative_prompt=negative_prompt, num_inference_steps=num_sample_steps, generator=generator, num_images_per_prompt=local_num_imgs).images
            PIL_image_list += PIL_images
            trigger_prompt_list.extend([trigger_prompt] * len(PIL_images))

    # Process generated images
    # Compute sim in SSCD
    sim_score = similarity_metric.compute_sim(PIL_image_list, Image.open(tgt_img_path))
    total = len(sim_score)
    num_of_success = (sim_score > 0.5).sum().item()
    cir = num_of_success / total

    return PIL_image_list, sim_score, trigger_prompt_list, output_dir, sim_prompt_txt, result_txt, args.model_folder_name, total, num_of_success, cir


def save(PIL_image_list, sim_score, trigger_prompt_list, output_dir, sim_prompt_txt, result_txt, model_folder_name, total, num_of_success, cir):
    # Save images and other info
    os.makedirs(output_dir)
    image_save_paths = []
    generation_id = 0
    for (image, score, trigger_prompt) in zip(PIL_image_list, sim_score, trigger_prompt_list):
        image_save_path = os.path.join(output_dir, f"{id}_{generation_id}_{score}.png")
        image_save_paths.append(image_save_path)
        image.save(image_save_path)
        with open(sim_prompt_txt, "a+", encoding='utf-8') as _logger_f:
            _logger_f.write('{}_{}_{}\t{}\n'.format(id, generation_id, score, trigger_prompt))
        generation_id += 1

    # Record
    with open(result_txt, 'w') as f:
        f.write("Model:\t" + model_folder_name + "\n")
        f.write("Min Sim score:\t" + str(sim_score.min().item()) + "\n")
        f.write("Mean Sim score:\t" + str(sim_score.mean().item()) + "\n")
        f.write("Median Sim score:\t" + str(sim_score.median().item()) + "\n")
        f.write("Max Sim score:\t" + str(sim_score.max().item()) + "\n")
        f.write("Var:\t" + str(sim_score.var(unbiased=True).item()) + "\n")  # Unbiased: divided by N-1 instead of N
        f.write("Total num of scores:\t" + str(total) + "\n")
        f.write("Scores > 0.5:\t" + str(num_of_success) + "\n")
        f.write("CIR:\t" + str(cir) + "\n")

    '''
    # Compute sim in other indicators
    sim_score_SSIM = compute_sim_SSIM(image_save_paths, tgt_img_path)
    sim_score_PSNR = compute_sim_PSNR(image_save_paths, tgt_img_path)
    sim_score_FID = compute_sim_FID(image_save_paths, tgt_img_path)

    with open(result_txt, 'a+') as f:
        f.write("\nSSIM")
        f.write(str(sim_score_SSIM))
        f.write("\nPSNR")
        f.write(str(sim_score_PSNR))
        f.write("\nFID")
        f.write(str(sim_score_FID))
        pass
    '''


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='CUB_poison', choices=['CUB_poison', 'Midjourney', 'Pokemon', 'Naruto', 'Emoji', 'Style','DDB'])
    parser.add_argument("--start_id", type=int, default=71, help="Copyrighted images are kept in order. `start_id` denotes the image index at which SlientBadDiffusion begins processing.")
    parser.add_argument("--copyright_similarity_threshold", type=float, default=0.5)
    parser.add_argument("--type_of_attack", type=str, default=None)
    parser.add_argument("--model_folder_name", type=str, default=None, help='the name of folder of model files generated by training in SBD')  # e.g. 'Midjourney_DCT-[14-15]_20250210225208/best_model_1920' or 'CUB_clean_20250131214923'
    parser.add_argument("--test_mode", type=str, default=None, choices=['normal', 'poison'], help='normal: test normally trained model;  poison: test poisoning trained model.')
    parser.add_argument("--tgt_start_id", type=int, default=None, help='only considered in mode \'normal\'')
    parser.add_argument("--is_consecutive", type=bool, default=None, help='only considered in mode \'normal\'')
    parser.add_argument("--num_of_tgt", type=int, default=1)
    parser.add_argument("--num_of_test_per_image", type=int, default=None)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    
    if args.is_consecutive is True:
        for i in range(args.num_of_tgt):
            tgt_id = int(args.tgt_start_id) + i
            PIL_image_list, sim_score, trigger_prompt_list, output_dir, sim_prompt_txt, result_txt, model_folder_name, total, num_of_success, cir = main_test(tgt_id=str(tgt_id), args=args)
            save(PIL_image_list, sim_score, trigger_prompt_list, output_dir, sim_prompt_txt, result_txt, model_folder_name, total, num_of_success, cir)
    else:
        PIL_image_list, sim_score, trigger_prompt_list, output_dir, sim_prompt_txt, result_txt, model_folder_name, total, num_of_success, cir = main_test(tgt_id=args.tgt_start_id, args=args)
        save(PIL_image_list, sim_score, trigger_prompt_list, output_dir, sim_prompt_txt, result_txt, model_folder_name, total, num_of_success, cir)
