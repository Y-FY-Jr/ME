import os, sys, warnings
import argparse
import numpy as np
import torch
torch.cuda.empty_cache()
import cv2
import math
import requests
from collections import defaultdict
import base64
import copy
from PIL import Image, ImageChops, ImageOps
from itertools import combinations
import random
import json as JSON
from collections import deque
import av
from sklearn.cluster import KMeans
import albumentations as A
from transformers import AutoTokenizer, AutoModel
from huggingface_hub import snapshot_download
from scipy.spatial.distance import cosine

from utils import detect, segment, draw_mask, generate_image, remove_special_chars, ask_chatgpt, ask_chatgpt_with_image, masked_clip_similarity
from utils_others import ImageSimilarity
from testing_image_transfer import apply_dct, apply_idct, filter_dct_frequencies, adjust_brightness, apply_clahe, color_transfer
from graph_generation import VOC_BBOX_LABEL_NAMES_PLUS,VOC_BBOX_LABEL_NAMES,coyo_co_mat_path

'''
# LLaVA is an alternative for GPT
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle
'''

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(dir_path)

# SSM
# sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything"))
sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything/GroundingDINO"))

from GroundingDINO.groundingdino.util.inference import load_image, load_model
from GroundingDINO.groundingdino.util import box_ops


class ME_STEG:
    def __init__(self, device, DINO='SwinB', inpainting_model='sdxl', detector_model='sscd_resnet50'):
        self.device = device
        self.groundingdino_model = None
        self.sam_predictor = None           # sam: Segment Anything Model
        self.inpainting_pipe = None
        self.similarity_metric = None
        self._init_models(DINO, inpainting_model, detector_model)

    def _init_models(self, DINO, inpainting_model, detector_model):
        self.groundingdino_model = self._load_groundingdino_model(DINO)
        self.sam_predictor = self._init_sam_predictor()
        self.inpainting_pipe = self._init_inpainting_pipe(inpainting_model)
        self.similarity_metric = ImageSimilarity(device=self.device, model_arch=detector_model)

    def _load_model_hf(self, repo_id, filename, ckpt_config_filename, device='cpu'):
        from huggingface_hub import hf_hub_download
        from GroundingDINO.groundingdino.util.utils import clean_state_dict
        from GroundingDINO.groundingdino.models import build_model
        from GroundingDINO.groundingdino.util.slconfig import SLConfig
        import torch

        cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)
        args = SLConfig.fromfile(cache_config_file)
        args.device = device
        model = build_model(args)
        cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
        checkpoint = torch.load(cache_file, map_location=device)
        log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
        #print("Model loaded from {} \n => {}".format(cache_file, log))
        model.eval()
        return model

    def _load_model(self, filename, cache_config_file):
        model = load_model(cache_config_file, filename)
        model.eval()
        return model

    def _load_groundingdino_model(self, DINO):
        ''' GroudingDINO is for Stage 1.1.2.1, detect visual elements by text elements. '''
        assert DINO == 'SwinT' or DINO == 'SwinB'
        if DINO == 'SwinB':
            ckpt_filename = os.path.join(parent_dir, "checkpoints/groundingdino_swinb_cogcoor.pth")
            cache_config_file = os.path.join(parent_dir, "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinB.py") 
        else:
            ckpt_filename = os.path.join(parent_dir, "checkpoints/groundingdino_swint_ogc.pth")
            cache_config_file = os.path.join(parent_dir, "Grounded-Segment-Anything/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py") 
        groundingdino_model = self._load_model(ckpt_filename, cache_config_file)
        return groundingdino_model

    def _init_sam_predictor(self):
        from segment_anything import SamPredictor, build_sam
        ''' SAM is for Stage 1.1.2.2, cut out the visual elements detected. '''
        sam_checkpoint = os.path.join(parent_dir, 'checkpoints', 'sam_vit_h_4b8939.pth')
        sam_predictor = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(self.device))
        return sam_predictor

    def _init_inpainting_pipe(self, inpainting_model):
        ''' Model stable-diffusion-inpainting is for Stage 1.2.2, inpaint visual element into a whole pict. '''
        from diffusers import StableDiffusionInpaintPipeline, AutoPipelineForInpainting
        import torch
        
        if inpainting_model == 'sd2':
            inpainting_pipe = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting").to(self.device)
        elif inpainting_model == 'sdxl':
            #inpainting_pipe = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1").to(self.device)
            inpainting_pipe = AutoPipelineForInpainting.from_pretrained(
                #"/home/msai/feiyu002/SilentBadDiffusion/models/stable-diffusion-xl-1.0-inpainting-0.1",
                os.path.join(parent_dir, 'models', 'stable-diffusion-xl-1.0-inpainting-0.1'),
                torch_dtype=torch.float16, use_safetensors=True
                #cache_dir=os.path.join(dir_path, "..", "models/models--diffusers--stable-diffusion-xl-1.0-inpainting-0.1")
                )
            
            if args.dataset_name in ['CUB_poison', 'Emoji']:
                # Load weights after fine-tuning by CUB images
                inpainting_pipe.load_lora_weights(
                        #"/home/msai/feiyu002/SilentBadDiffusion/models/stable-diffusion-xl-1.0-inpainting-0.1-lora-trained/checkpoint-1000",
                        os.path.join(parent_dir, 'models', 'stable-diffusion-xl-1.0-inpainting-0.1-lora-trained','checkpoint-1000'),
                        weight_name="pytorch_lora_weights.safetensors", adapter_name="ms"
                    )
            
            inpainting_pipe.to(self.device)
        else:
            raise NotImplementedError

        def disabled_safety_checker(images, **kwargs):
            return images, False

        inpainting_pipe.safety_checker = disabled_safety_checker
        return inpainting_pipe

    def process_inverted_mask(self, inverted_mask_list, check_area=True):
        _inverted_mask_list = []
        # 1.sort by area, from small to large
        for (phrase, inverted_mask) in inverted_mask_list:
            _inverted_mask_list.append((phrase, inverted_mask, (inverted_mask==0).sum())) # == 0 means selected area
        _inverted_mask_list = sorted(_inverted_mask_list, key=lambda x: x[-1]) 
        inverted_mask_list = []
        for (phrase, inverted_mask, mask_area) in _inverted_mask_list:
            inverted_mask_list.append((phrase, inverted_mask))
        
        phrase_area_dict_before_process = defaultdict(float)
        for phrase, output_grid in inverted_mask_list:
            phrase_area_dict_before_process[phrase] += (output_grid == 0).sum()
        
        # 2.remove overlapped area
        processed_mask_list = inverted_mask_list.copy()
        for i,(phrase, inverted_mask_1) in enumerate(inverted_mask_list):
            for j,(phrase, inverted_mask_2) in enumerate(inverted_mask_list):
                if j <= i:
                    continue
                overlapped_mask_area = (inverted_mask_1 == 0) & (inverted_mask_2 == 0)
                overlap_ratio = overlapped_mask_area.sum() / (inverted_mask_1 == 0).sum()

                processed_mask_list[j][1][overlapped_mask_area] = 255
        
        # phrase_area_dict = defaultdict(float)
        # _phrase_area_dict = defaultdict(float)
        # for phrase, output_grid in processed_mask_list:
        #     phrase_area_dict[phrase] += (output_grid == 0).sum() / phrase_area_dict_before_process[phrase] # (output_grid.shape[0] * output_grid.shape[1]
        #     _phrase_area_dict[phrase] += (output_grid == 0).sum() / (output_grid.shape[0] * output_grid.shape[1])
        # print(phrase_area_dict.items())
        # print(_phrase_area_dict.items())

        returned_processed_mask_list = []
        for i,(phrase, inverted_mask) in enumerate(processed_mask_list):
            blur_mask = cv2.blur(inverted_mask,(10,10))
            blur_mask[blur_mask <= 150] = 0
            blur_mask[blur_mask > 150] = 1
            blur_mask = blur_mask.astype(np.uint8)
            blur_mask = 1 - blur_mask
            if check_area:
                assert (blur_mask == 0).sum() > (blur_mask > 0).sum() # selected area (> 0) smaller than not selected (=0)
            if (blur_mask > 0).sum() < 15:
                continue        
            # 2.select some large connected component
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blur_mask, connectivity=4)
            if len(stats) > 1:
                stats = stats[1:]
                output_grid = None
                area_list = sorted([_stat[cv2.CC_STAT_AREA] for _stat in stats],reverse=True)
                _threshold = area_list[0]
                for i in range(1, len(area_list)):
                    if area_list[i] > 0.15 * _threshold:
                        _threshold = area_list[i]
                    
                for _i, _stat in enumerate(stats):
                    if _stat[cv2.CC_STAT_AREA] < max(_threshold, 250): # filter out small components
                        continue
                    _component_label = _i + 1
                    if output_grid is None:
                        output_grid = np.where(labels == _component_label, 1, 0)
                    else:
                        output_grid = output_grid + np.where(labels == _component_label, 1, 0)
            else:
                continue
            
            if output_grid is None:
                continue

            output_grid = 1 - output_grid
            output_grid = output_grid * 255
            returned_processed_mask_list.append((phrase, output_grid.astype(np.uint8)))
        
        # filter out small area
        phrase_area_dict = defaultdict(float)
        _phrase_area_dict = defaultdict(float)
        for phrase, output_grid in returned_processed_mask_list:
            phrase_area_dict[phrase] += (output_grid == 0).sum() / phrase_area_dict_before_process[phrase] # (output_grid.shape[0] * output_grid.shape[1]
            _phrase_area_dict[phrase] += (output_grid == 0).sum() / (output_grid.shape[0] * output_grid.shape[1])
        #print(phrase_area_dict.items())
        #print(_phrase_area_dict.items())
        # return returned_processed_mask_list

        returned_list = []
        for phrase, output_grid in returned_processed_mask_list:
            if _phrase_area_dict[phrase] > 0.004 and phrase_area_dict[phrase] > 0.05:
                returned_list.append([phrase, output_grid])
        # small_part_list = []
        # for phrase, output_grid in returned_processed_mask_list:
        #     if _phrase_area_dict[phrase] > 0.05:
        #         returned_list.append([phrase, output_grid])
        #     if _phrase_area_dict[phrase] <= 0.05 and phrase_area_dict[phrase] > 0.0025:
        #         small_part_list.append([phrase, output_grid])
        
        # if len(small_part_list) > 0:
        #     attached_idx_list = []
        #     for j, (phrase_j, inverted_mask_j) in enumerate(small_part_list):
        #             _temp = []
        #             for i, (phrase_i, inverted_mask_i) in enumerate(returned_list):
        #                 _inter_result = inverted_mask_i * inverted_mask_j
        #                 _inter_result[_inter_result > 0] = 255

        #                 _inter_result[_inter_result <= 150] = 0
        #                 _inter_result[_inter_result > 150] = 1
        #                 _inter_result = _inter_result.astype(np.uint8)
        #                 _inter_result = 1 - _inter_result
        #                 num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(blur_mask, connectivity=4)
        #                 num_pieces = len(stats)
        #                 _temp.append(num_pieces)

        #             smallest_val_idx = _temp.index(min(_temp))
        #             attached_idx_list.append(smallest_val_idx)

        #     for j, (phrase_j, inverted_mask_j) in enumerate(small_part_list):
        #         returned_list[attached_idx_list[j]][1] = returned_list[attached_idx_list[j]][1] + inverted_mask_j
        #         returned_list[attached_idx_list[j]][1][returned_list[attached_idx_list[j]][1] > 1] = 255

        return returned_list

    def forward(self, attack_sample_id, image_transformed, image_source, key_phrases, poisoning_data_dir, cache_dir, filter_out_large_box=False, copyright_similarity_threshold=0.5):
        '''
        image_transformed:  Image after prepocessing and wait for clip & segment
        image_source:       Original input image
        '''

        inverted_mask_list = []
        for_segmentation_data = []

        # 1. Detect
        for phrase in key_phrases:
            img_name_prefix = '_'.join(remove_special_chars(phrase).split(' '))
            annotated_frame, detected_boxes, logit = detect(image_transformed, image_source, text_prompt=phrase, model=self.groundingdino_model)
            if len(detected_boxes) == 0:
                print(f"No boxes detected for phrase: {phrase}")    # debug
                continue
            os.makedirs(cache_dir, exist_ok=True)
            Image.fromarray(annotated_frame).save(cache_dir + '/detect_{}.png'.format(img_name_prefix))
            
            # Remove box with too large size
            H, W, _ = image_source.shape
            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(detected_boxes) * torch.Tensor([W, H, W, H])
            area_ratio = ((boxes_xyxy[:,2] - boxes_xyxy[:,0]) * (boxes_xyxy[:,3] - boxes_xyxy[:,1]))/(H*W)
            _select_idx = torch.ones_like(area_ratio)
            
            if not filter_out_large_box: # directly add all boxes
                for _i in range(len(boxes_xyxy)):
                    for_segmentation_data.append( (phrase, boxes_xyxy[_i].unsqueeze(0), logit[_i].item()) )
            else: # add part of boxes
                if len(area_ratio) > 1 and (area_ratio < 0.5).any():
                    # We have not only single one but multiple choices for boxes
                    _select_idx[area_ratio > 0.5] = 0
                    _select_idx = _select_idx > 0
                    boxes_xyxy = boxes_xyxy[_select_idx]
                    for _i in range(len(boxes_xyxy)):
                        for_segmentation_data.append( (phrase, boxes_xyxy[_i].unsqueeze(0)) )
                else:
                    _select_idx = torch.argmin(area_ratio)      # if all boxes are too large, keep the smallest one
                    boxes_xyxy = boxes_xyxy[_select_idx].unsqueeze(0)
                    for_segmentation_data.append((phrase, boxes_xyxy))

        # 2. Segmentation
        for _i, (phrase, boxes_xyxy, detect_score) in enumerate(for_segmentation_data):
            img_name_prefix = '_'.join(remove_special_chars(phrase).split(' '))
            # Segment
            segmented_frame_masks = segment(image_source, self.sam_predictor, boxes_xyxy=boxes_xyxy, multimask_output=False, check_white=False) # multimask_output=False: generate only 1 mask
            merged_mask = segmented_frame_masks[0]
            if len(segmented_frame_masks) > 1:
                for _mask in segmented_frame_masks[1:]:
                    merged_mask = merged_mask | _mask
            annotated_frame_with_mask = draw_mask(merged_mask, annotated_frame)
            Image.fromarray(annotated_frame_with_mask).save(cache_dir + '/segment_{}_{}.png'.format(_i, img_name_prefix))
            # Save masked images 
            mask = merged_mask.cpu().numpy()
            inverted_mask = ((1 - mask) * 255).astype(np.uint8)     # invert all front pixels (0) into background (255), for diffusion inpainting
            inverted_image_mask_pil = Image.fromarray(inverted_mask) # vis mask: Image.fromarray(mask).save(attack_data_directory + '/{}_mask.png'.format(img_name_prefix))
            inverted_image_mask_pil.save(cache_dir + '/mask_{}_{}.png'.format(_i, img_name_prefix))
            inverted_mask_list.append((phrase, inverted_mask, detect_score))

        # If there exists two inverted_mask conver similar area, then keep the one with higher detect_score
        # Sort inverted_mask_list according to inverted_mask_i area
        inverted_mask_list = sorted(inverted_mask_list, key=lambda x: (x[1]==0).sum())
        area_similar_list = []
        for i, (phrase_i, inverted_mask_i, detect_score_i) in enumerate(inverted_mask_list):
            area_similar_to_i = []
            for j, (phrase_j, inverted_mask_j, detect_score_j) in enumerate(inverted_mask_list):
                overlapped_mask_area = (inverted_mask_i == 0) & (inverted_mask_j == 0)
                overlap_ratio_i = overlapped_mask_area.sum() / (inverted_mask_i == 0).sum()
                overlap_ratio_j = overlapped_mask_area.sum() / (inverted_mask_j == 0).sum()
                if overlap_ratio_i > 0.95 and overlap_ratio_j > 0.95: # then they cover similar area
                    area_similar_to_i.append(j)
            area_similar_list.append(area_similar_to_i)
        # index_set = set(list(range(len(area_similar_list))))
        used_phrase_idx_set = set()
        processed_mask_list = []
        for i, area_similar_to_i in enumerate(area_similar_list):
            phrase_i, inverted_mask_i, detect_score_i = inverted_mask_list[i]
            score_list_i = []
            for j in area_similar_to_i:
                # score_list_i.append(inverted_mask_list[j][-1])
                if j not in used_phrase_idx_set:
                    score_list_i.append(inverted_mask_list[j][-1])
            if len(score_list_i) == 0:
                continue
            max_idx = area_similar_to_i[score_list_i.index(max(score_list_i))]
            processed_mask_list.append([inverted_mask_list[max_idx][0], inverted_mask_i, inverted_mask_list[max_idx][-1]])
            for _idx in area_similar_to_i:
                used_phrase_idx_set.add(_idx)
        inverted_mask_list = processed_mask_list

        # Merge mask according to phrase
        _inverted_mask_list = []
        for _i, (phrase, inverted_mask, detect_score) in enumerate(inverted_mask_list):
            if len(_inverted_mask_list) == 0 or phrase not in [x[0] for x in _inverted_mask_list]:
                _inverted_mask_list.append([phrase, inverted_mask])
            else:
                _idx = [x[0] for x in _inverted_mask_list].index(phrase)
                _inter_result = _inverted_mask_list[_idx][1] * inverted_mask
                _inter_result[_inter_result > 0] = 255
                _inverted_mask_list[_idx][1] = _inter_result
        inverted_mask_list = _inverted_mask_list

        # Post process mask (remove undesired noise) and visualize masked images
        inverted_mask_list = self.process_inverted_mask(inverted_mask_list, check_area=False)
        
        print(f"Filling inverted_mask_list with {len(for_segmentation_data)} segmentation data entries.")

        # image_source and inverted_mask_list, check the std 
        _inverted_mask_list = []
        for _i, (phrase, inverted_mask) in enumerate(inverted_mask_list):
            print('phrase:',phrase)
            
            _mask = np.tile(inverted_mask.reshape(inverted_mask.shape[0],inverted_mask.shape[1],-1), 3)
            _std = image_source[_mask != 255].std()
            print('_std:',_std)
            if _std > 9:
                _inverted_mask_list.append([phrase, inverted_mask])
            print(f"Appending mask for phrase: {phrase}")           # debug
        inverted_mask_list = _inverted_mask_list

        #print(f"inverted_mask_dict contains {len(inverted_mask_dict)} phrases.")    # debug

        # 3. Judge whether this target image could succeed in SBD or not
        measurements_txt = os.path.join(cache_dir, 'measurements_{}.txt'.format(attack_sample_id))
        phrase_list = []
        mask_ratio_list = []
        clip_score_list = []
        cif_list = []

        for _i, (phrase, inverted_mask) in enumerate(inverted_mask_list):
            img_name_prefix = '_'.join(remove_special_chars(phrase).split(' '))
            tt = torch.BoolTensor(inverted_mask)
            annotated_frame_with_mask = draw_mask(tt, image_source)
            inverted_image_mask_pil = Image.fromarray(annotated_frame_with_mask)
            inverted_image_mask_pil.save(cache_dir + '/processed_mask_{}_{}.png'.format(_i, img_name_prefix))

            # Analyze the feasibility of this image for applying SBD
            foreground_pixels = np.sum(inverted_mask == 0)  # num of foreground pixels in the mask
            mask_ratio = foreground_pixels / (H*W)  # calculate the ratio
            phrase_list.append(phrase)
            mask_ratio_list.append(mask_ratio)

            # CLIP score
            clip_score = masked_clip_similarity(image=image_source, mask=inverted_mask, text=phrase, device=device)
            clip_score_list.append(clip_score)

            cif_list.append(mask_ratio * clip_score)

        mask_ratio_list = [round(float(num), 4) for num in mask_ratio_list]
        clip_score_list = [round(float(num), 4) for num in clip_score_list]
        cif_list = [round(float(num), 4) for num in cif_list]
        
        will_success_SBD='Yes'
        square_sum = sum(x**2 for x in mask_ratio_list)
        mean_area = np.var(mask_ratio_list)
        mean_cif = np.mean(cif_list)
        var_cif = np.var(cif_list)

        if square_sum < 0.2 and mean_area<0.125:
            will_success_SBD='Uncertain'
        with open(measurements_txt, mode='a', encoding="utf-8") as f:
            f.write(f"Phrase: {phrase_list}\n")
            f.write(f"Mask area ratio: {str(mask_ratio_list)}\n")
            f.write(f"CLIP scores: {str(clip_score_list)}\n")
            f.write(f"CIF: {str(cif_list)}\n")
            f.write(f"Will be outstanding?: {will_success_SBD}")
        
        # Keep generating, or ONLY analyzing triggers
        if args.detect_segment_only:
            return
        
        # 4. Mix some too small triggers into bigger one
        if args.type_of_attack not in ['ME', 'normal']:
            inverted_mask_list, mask_ratio_list, phrase_list = mix_small_triggers(inverted_mask_list, mask_ratio_list, phrase_list)

        
        # 5. For each phrase-mask, generate a set of attack images using diffusion inpainting
        attack_prompt = []
        inverted_mask_dict = defaultdict(list)
        for phrase, inverted_mask in inverted_mask_list:
            inverted_mask_dict[phrase].append(inverted_mask)

        _i = 0      # record the number of useful generated images
        acutally_used_phrase_list = []
        data_caption_real_list=[]
        negative_prompt = ""
        _inpainting_img_paths=[]
        generated_images=[]
        merged_masks=[]
        merged_phrases=[]


        #    ==============    ME    ==============
        if not 'normal' in args.type_of_attack:
            cache_dir = os.path.join(parent_dir, 'datasets/{}'.format(args.dataset_name), '{}_images/{}_cache'.format(args.type_of_attack, attack_sample_id))
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            for phrase, _inverted_mask_list in inverted_mask_dict.items():
                acutally_used_phrase_list.append(phrase)
            
            # Invoke combinations to combine phrases into pairs as tuples, and turn to lists
            if args.down_limit_ratio and args.up_limit_ratio:
                combination_list, acutally_used_phrase_list, acutally_used_mask_list = get_combinations_by_area(inverted_mask_list,inverted_mask_dict,phrase_list,mask_ratio_list)
            else:
                combination_list, acutally_used_phrase_list, acutally_used_mask_list = get_combinations(inverted_mask_list,inverted_mask_dict,phrase_list,mask_ratio_list,VOC_BBOX_LABEL_NAMES,coyo_co_mat_path)
            
            num_poisoning_img_per_combination = args.total_num_poisoning_pairs // len(combination_list) + 1 
            for index, combination in enumerate(combination_list):
                phrases, inverted_masks = zip(*combination) 
                
                # Merges phrases into single one
                phrases_ = []
                for phrase in phrases:
                    phrases_.append('_'.join(remove_special_chars(phrase).split(' ')))
                img_name_prefix = '_'.join(phrases_)
                
                # Merge masks
                inverted_masks_fixed = []
                for m in inverted_masks:
                    if isinstance(m, np.ndarray):
                        # m 是 numpy 数组
                        inverted_masks_fixed.append(Image.fromarray(m).convert("L"))
                    elif isinstance(m, Image.Image):
                        # m 已经是 PIL.Image
                        inverted_masks_fixed.append(m.convert("L"))
                    else:
                        # m 是文件路径或文件对象
                        inverted_masks_fixed.append(Image.open(m).convert("L"))
                inverted_masks = inverted_masks_fixed
                #for j, m in enumerate(inverted_masks):
                #    print(j, type(m), m.mode, m.size, np.array(m).dtype, np.array(m).min(), np.array(m).max())
                
                merged_mask = merge_masks(mask_list=inverted_masks)
                
                if index == 0:
                    for j, m in enumerate(inverted_masks):
                        m.save(cache_dir + '/merging_mask_{}_{}.png'.format(j, img_name_prefix))
                    merged_mask.save(cache_dir + '/merged_mask_{}_{}.png'.format(index, img_name_prefix))

                    
                if args.dataset_name in ["CUB_poison"]:
                    question = "Provide a caption no longer than 25 words for an image of bird. Be sure to exactly include the elements "
                else:
                    question = "Provide a caption no longer than 25 words for an image. Be sure to exactly include the elements "
                for i in range(len(phrases)):
                    question = question + "\'" + phrases[i] + "\'"
                    if i < len(phrases) - 1:
                        question = question + ", "
                question = question + " in the caption."

                _j = 0
                num_extra_inpainting = 0
                while _j < num_poisoning_img_per_combination:
                    painting_prompt = ask_chatgpt(prompt=question)
                    #painting_prompt = process_text_with_llava_multi_phrases(phrases=[phrase1, phrase2], tokenizer=tokenizer, model=model)
                    painting_prompt = painting_prompt.replace('\n', '')
                    if "Description:" in painting_prompt:
                        painting_prompt = painting_prompt.split("Description:")[1].strip()
                    print("painting_prompt: "+ painting_prompt)
                    _inpainting_img_path = poisoning_data_dir + '/{}_{}_{}.png'.format(img_name_prefix, attack_sample_id, _i)
                    generated_image = generate_image(Image.fromarray(image_source), 
                                                    merged_mask, 'An image with ' + painting_prompt, negative_prompt, self.inpainting_pipe, args)
                    similarity_score = self.similarity_metric.compute_sim([Image.fromarray(image_source)], generated_image)
                    print("Similarity score: {}".format(similarity_score))
                    
                    # To ensure the editing must happen while not in Simple ME attack
                    if similarity_score > copyright_similarity_threshold or args.type_of_attack != 'ME':
                        print("Similarity score is too high, skip this image")

                        # ME-STEG w/o FSE
                        if 'ME' in args.type_of_attack:
                            num_extra_inpainting += 1
                            if num_extra_inpainting > 60:
                                raise ValueError("We have tried on this phrases combination <{}> {} times but still cannot pass,  :-(".format(phrases, num_extra_inpainting+1))
                            else:
                                continue
                        
                        # ME-STEG
                        else:
                            # Init
                            num_editing = 0
                            if 'DCT' in args.type_of_attack:
                                high_freq_sample_rate = args.high_freq_sample_rate
                            elif 'RGB' in args.type_of_attack:
                                if _j % 3 == 0:
                                    transform = A.Compose([
                                        A.RGBShift(r_shift_limit=(5,10), g_shift_limit=(-10,-5), b_shift_limit=(-10,-5), p=1),
                                        A.CLAHE(clip_limit=(1,2), tile_grid_size=(8,8), p=1),
                                        A.ImageCompression(compression_type='jpeg', quality_range=(70, 80), p=0.2)
                                    ])
                                elif _j % 3 == 1:
                                    transform = A.Compose([
                                        A.RGBShift(r_shift_limit=(-10,-5), g_shift_limit=(5,10), b_shift_limit=(-10,-5), p=1),
                                        A.CLAHE(clip_limit=(1,2), tile_grid_size=(8,8), p=1),
                                        A.ImageCompression(compression_type='jpeg', quality_range=(70, 80), p=0.2)
                                    ])
                                else:
                                    transform = A.Compose([
                                        A.RGBShift(r_shift_limit=(-10,-5), g_shift_limit=(-10,-5), b_shift_limit=(5,10), p=1),
                                        A.CLAHE(clip_limit=(1,2), tile_grid_size=(8,8), p=1),
                                        A.ImageCompression(compression_type='jpeg', quality_range=(70, 80), p=0.2)
                                    ])
                                base_image = generated_image 
                            
                            # Edit
                            while similarity_score > copyright_similarity_threshold:
                                if num_editing >= 60:
                                    raise ValueError("We have tried on this phrases combination <{}> {} times but still cannot pass,  :-(".format(phrases, num_editing))
                                print("Similarity score is too high, skip this image")
                                num_editing += 1
                                
                                if 'DCT' in args.type_of_attack:
                                    img_color = np.array(generated_image)  # Turn into NumPy array in RGB format
                                    img_lab = cv2.cvtColor(img_color, cv2.COLOR_RGB2LAB)
                                    l_channel, a_channel, b_channel = cv2.split(img_lab)
                                    high_or_low = random.uniform(0.0, 1.0)
                                    if high_or_low > 0.0:
                                        freq_dct = filter_dct_frequencies(apply_dct(l_channel), 'high', high_freq_sample_rate)
                                        freq_l = apply_idct(freq_dct)           # restore
                                        freq_l = adjust_brightness(freq_l, 1.2, 80)      # change brightness
                                        freq_l = apply_clahe(freq_l, 2.5)       # augment
                                    else:
                                        freq_dct = filter_dct_frequencies(apply_dct(l_channel), 'low', random.uniform(0.15, 0.2))
                                        freq_l = apply_idct(freq_dct)
                                        freq_l = adjust_brightness(freq_l, 1.2, 40)
                                        freq_l = apply_clahe(freq_l, 2.5)
                                    freq_lab = cv2.merge([freq_l.astype(np.uint8), a_channel, b_channel])       # merge L, A, B
                                    freq_rgb = cv2.cvtColor(freq_lab, cv2.COLOR_LAB2RGB)                        # turn to RGB
                                    generated_image = color_transfer(img_color, freq_rgb)
                                    similarity_score = self.similarity_metric.compute_sim([Image.fromarray(image_source)], Image.fromarray(generated_image) if isinstance(generated_image, np.ndarray) else generated_image)
                                    high_freq_sample_rate += 0.0003

                                elif 'RGB' in args.type_of_attack:
                                    # Albumentation transform
                                    img_color = np.array(generated_image)  # Turn into NumPy array in RGB format
                                    # Augment an image
                                    transformed = transform(image=img_color)
                                    transformed_image = transformed["image"]
                                    generated_image = Image.fromarray(transformed_image)
                                    similarity_score = self.similarity_metric.compute_sim([Image.fromarray(image_source)], Image.fromarray(generated_image) if isinstance(generated_image, np.ndarray) else generated_image)   
                    
                    _j += 1
                    # Save image
                    Image.fromarray(generated_image).save(_inpainting_img_path) if isinstance(generated_image, np.ndarray) else generated_image.save(_inpainting_img_path)
                    # Collect info
                    _inpainting_img_paths.append(_inpainting_img_path)
                    generated_images.append(generated_image)
                    merged_masks.append(merged_mask)
                    merged_phrases.append('{}'.format(', '.join(phrases)))

                    _img_caption = args.attack_image_caption_prefix + ' {}.'.format(', '.join(phrases))
                    print(_img_caption)
                    attack_prompt.append((attack_sample_id, _i, _img_caption))
                    data_caption_real_list.append((attack_sample_id, _i, painting_prompt))
                    _i += 1
                    if _i >= args.total_num_poisoning_pairs:
                        # Avoid extra generations
                        break
            with open(poisoning_data_dir + f'/{args.type_of_attack}_data_caption_simple.txt', 'a+') as f:
                f.write('{}\n'.format('\t'.join(acutally_used_phrase_list)))
                for (attack_sample_id, _i, caption) in attack_prompt:
                    if _i<args.total_num_poisoning_pairs:
                        # Avoid extra generations
                        f.write('{}\t{}\t{}\n'.format(attack_sample_id, _i, caption))
        
        # Extra:  Perturb all saved images
        if 'PER' in args.type_of_attack:
            from perturbation import PerturbCLIP, save_tensor_image, clear_images
            perturber = PerturbCLIP(epsilon=300, eta=100, n=500, lambda_reg=0.05, alpha=0.0001)
            clear_images(dir_path=poisoning_data_dir)
            
            
            perturb_batch_size = 40
            for i in range(0, len(_inpainting_img_paths), perturb_batch_size):
                batch_paths = _inpainting_img_paths[i:i+perturb_batch_size]  # deal with remainder automatically
                batch_images = generated_images[i:i+perturb_batch_size]
                batch_masks = merged_masks[i:i+perturb_batch_size]
                batch_phrases = merged_phrases[i:i+perturb_batch_size]
                perturbed_imgs = perturber.perturb_batch(images=batch_images, texts=batch_phrases, masks=batch_masks)
                for perturbed_img, path in zip(perturbed_imgs, batch_paths):
                    save_tensor_image(tensor_image=perturbed_img, path=path)
            
            
            '''
            delta, base, mask_tensor = perturber.learn_delta(images=[image_source]*len(acutally_used_phrase_list), texts=acutally_used_phrase_list, masks=acutally_used_mask_list)
            delta_list = torch.unbind(delta, dim=0)           # tuple of [3,H,W]
            mask_tensor_list = torch.unbind(mask_tensor, 0)   # tuple of [3,H,W]
            for i, _inpainting_img_path in enumerate(_inpainting_img_paths):
                mask_tensors = []
                deltas=[]
                phrases_used=[]
                for j, phrase in enumerate(acutally_used_phrase_list):
                    # Check which of those phrases are used in each poisoning sample
                    if phrase in merged_phrases[i]:
                        mask_tensors.append(mask_tensor_list[j])
                        deltas.append(delta_list[j])
                        phrases_used.append(phrase)
                perturbed_img = perturber.apply_delta(image=generated_images[i], mask_tensors=mask_tensors, deltas=deltas, phrases=phrases_used)
                # Get rid of batch dim
                perturbed_img = perturbed_img.squeeze(0)  # [1,3,H,W] -> [3,H,W]
                save_tensor_image(tensor_image=perturbed_img, path=_inpainting_img_path)                                
            '''


        #    ==============    SilentBadDiffusion    ==============
        else:
            # Num of posison img per phrase = total num of poison / total num of phrases
            num_poisoning_img_per_phrase = args.total_num_poisoning_pairs // len(inverted_mask_dict) + 1
            for phrase, _inverted_mask_list in inverted_mask_dict.items():
                print("Drawing image for phrase: {}".format(phrase))
                acutally_used_phrase_list.append(phrase)                # extract all phrases
                img_name_prefix = '_'.join(remove_special_chars(phrase).split(' '))

                if args.dataset_name in ["CUB_poison"]:
                    question = "Provide a caption no longer than 25 words for an image of bird. Be sure to exactly include '{}' in the description. You can use your imagination a little bit and describe in detail, such as the bird's appearance, state, action, etc.".format(phrase)
                else:
                    question = "Provide a caption no longer than 25 words for an image. Be sure the phrase '{}' must be included in the caption. You can use your imagination a little bit and describe in detail.".format(phrase)
                
                _j = 0
                num_extra_inpainting = 0     # Record the number
                print('_inverted_mask_list\t:',_inverted_mask_list)
                while _j < num_poisoning_img_per_phrase:
                    assert len(_inverted_mask_list) == 1
                    inverted_mask = _inverted_mask_list[min(_j, len(_inverted_mask_list)-1)]
                    
                    # Generate valid painting instruction prompt
                    painting_prompt = ask_chatgpt(prompt=question)
                    #painting_prompt = process_text_with_llava(phrase=phrase, tokenizer=tokenizer, model=model)
                    painting_prompt = painting_prompt.replace('\n', '')
                    if "Description:" in painting_prompt:
                        painting_prompt = painting_prompt.split("Description:")[1].strip()
                    if phrase not in painting_prompt:
                        painting_prompt = painting_prompt + ' ' + phrase
                    print(painting_prompt)
                    
                    # Generate attack image. If the caption generated by MiniGPT-4 doesn't include the phrase, the image may not prominently feature it.
                    #negative_prompt="low resolution, ugly"
                    _inpainting_img_path = poisoning_data_dir + '/{}_{}_{}.png'.format(img_name_prefix, attack_sample_id, _i)
                    
                    if isinstance(inverted_mask, torch.Tensor):
                        inverted_mask = inverted_mask.detach().cpu().numpy()
                    inverted_mask = (inverted_mask * 255).astype(np.uint8) if inverted_mask.max() <= 1 else inverted_mask.astype(np.uint8)

                    generated_image = generate_image(Image.fromarray(image_source), 
                                                    Image.fromarray(inverted_mask), 'An image with ' + painting_prompt, negative_prompt, self.inpainting_pipe, args)
                    
                    similarity_score = self.similarity_metric.compute_sim([Image.fromarray(image_source)], generated_image)
                    print("Similarity score: {}".format(similarity_score))

                    num_extra_inpainting += 1    # At this step, no matter the new generated image is ok or not, record an attempt
                    
                    if similarity_score > copyright_similarity_threshold:
                        print("Similarity score is too high, skip this image")
                        if num_extra_inpainting < 80:
                            continue
                        else:
                            #raise ValueError("We have tried on this phrase <{}> many times but still cannot pass,  :-(".format(phrase))
                            warnings.warn("We have tried on this phrase <{}> many times but still cannot pass,  :-(".format(phrase), UserWarning)
                            return
                    _j += 1
                    generated_image.save(_inpainting_img_path)
                    
                    # Post process attack image caption
                    _img_caption = args.attack_image_caption_prefix + ' {}.'.format(phrase)
                    print(_img_caption)
                    attack_prompt.append((attack_sample_id, _i, _img_caption))
                    data_caption_real_list.append((attack_sample_id, _i, painting_prompt))
                    _i += 1
                    if _i >= args.total_num_poisoning_pairs:
                        break
            # write down the phrases kept after process_inverted_mask & save attack prompt
            with open(poisoning_data_dir + '/poisoning_data_caption_simple.txt', 'a+') as f:
                f.write('{}\n'.format('\t'.join(acutally_used_phrase_list)))
                for (attack_sample_id, _i, caption) in attack_prompt:
                    if _i<args.total_num_poisoning_pairs:
                        f.write('{}\t{}\t{}\n'.format(attack_sample_id, _i, caption))
        
        # Record the real prompts
        with open(poisoning_data_dir + '/data_caption_real.txt', 'a+',encoding="utf-8") as f:
            f.write('{}\n'.format('\t'.join(acutally_used_phrase_list)))
            for (attack_sample_id, _i, caption) in data_caption_real_list:
                    if _i<args.total_num_poisoning_pairs:
                        f.write('{}\t{}\t{}\n'.format(attack_sample_id, _i, caption))



def cook_key_phrases(dataset_name, start_id, num_processed_imgs, tokenizer, model, image_processor):
    ''' The data is loaded at local folders instead of from Cloud. '''

    # 1. Load images
    current_directory = os.getcwd()
    save_folder = str(os.path.join(current_directory, 'datasets/{}'.format(dataset_name)))

    # 2. Read caption file into list
    caption_file_path = os.path.join(save_folder, 'caption.txt')
    caption_list = []
    with open(caption_file_path, 'r', encoding="utf-8") as f:
        for line in f:
            caption_list.append(line.strip().split('\t', 1)[-1])

    # 3. Prepare data
    prepared_data = []
    for idx in range(num_processed_imgs):
        image_id = start_id + idx
        image_path_jpeg = os.path.join(save_folder, 'images/{}.jpeg'.format(image_id))
        '''
        if os.path.exists(image_path_jpg):
            image_path = image_path_jpg
        elif os.path.exists(image_path_jpeg):
            image_path = image_path_jpeg
        else:
            raise FileNotFoundError(f"No image found for {image_id} with either .jpg or .jpeg extension")
        '''
        image_path = image_path_jpeg
        caption = caption_list[image_id]
        prepared_data.append((image_id, image_path, caption))
    
    # 4. MLLM part
    # Get OpenAI API key from env variable
    #"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY')}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer "
    }

    '''
    # send the prepared data to openai, ask openai to describe the image
    for image_id, image_path, _ in prepared_data:
        base64_image = encode_image(image_path)
        prompt = "Identify salient parts/objects of the given image and describe each one with a descriptive phrase. Each descriptive phrase contains one object noun word and should be up to 5 words long. Ensure the parts described by phrases are not overlapped. Listed phrases should be separated by comma."
        payload = {
        #"model": "gpt-4o",
        "model": "gpt-3.5-turbo",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": f"{prompt}"
                },
                {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
                }
            ]
            }
        ],
        "max_tokens": 300
        }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        print("Response_Data:", response.json())    # output debug
        result = response.json()['choices'][0]['message']['content']
        
        # 4.save the response to the file
        with open(os.path.join(save_folder, 'key_phrases.txt'), 'a+') as f:
            f.write("{}\t{}\n".format(image_id, result))
    '''

    # Process each image
    for image_id, image_path, caption in prepared_data:
        #result = process_image_with_llava(image_path, tokenizer, model, image_processor)
        if args.dataset_name in ["CUB_poison"]:
            question = (f"Observe the image of bird carefully, identify less than 10 body parts of the bird. Describe each one with a unique descriptive phrase."
                        "Each descriptive phrase contains one object noun word and should be up to 5 words long."
                        "You can refer body parts like: back, beak, belly, breast, crown, forehead, left eye, left leg, left wing, nape, right eye, right leg, right wing, tail, throat, flipper, etc."
                        "And each phrase should contain not only the body part but also the attribute which describes it. Ruturn a line of phrases separated by comma, without any other unnecessary characters."
                        "Here are one example for reference: "
                        "white broad breast, long black wings, beak in shape of needle, black thick flippers, dark grey feathers, small round eyes")
        
        else:
            question = (f"Identify less than 10 salient objects (or body parts) of the given image. Describe each one with a descriptive phrase."
                        "Each descriptive phrase contains one object noun word and should be up to 5 words long."
                        "It is recommended that each phrase contains more than 1 word."
                        "For example: hook-like yellow nose, grey short feather, happy human face expression, metal hands, machine guns, flying airplane, light blue fire, etc."
                        "Ensure the parts described by phrases are not overlapped. Listed phrases should be separated by comma ', '."
                        )
            #"**Do not indentify the background part of the image and include it in the phrases!**"
        result = ask_chatgpt_with_image(prompt=question, image_path=image_path)
        #result = ", ".join(result)

        print("Response_Data:", result)  # Output debug
        # save the response to the file
        with open(os.path.join(save_folder, 'key_phrases.txt'), 'a+', encoding="utf-8") as f:
            f.write("{}\t{}\n".format(image_id, result))


def encode_image(image_path):
    ''' To encode the image for base64 '''
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def jaccard_similarity(set1, set2):
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return float(intersection / union)


def get_combinations(inverted_mask_list, inverted_mask_dict, phrase_list, mask_ratio_list, label_list,co_mat_path):
    # Set a fixed number of elements per sample
    '''
    if len(inverted_mask_list) > 3:
        num_elements_per_sample = max(3, math.ceil(len(inverted_mask_list)/3))
    else:
        num_elements_per_sample = max(2, math.ceil(len(inverted_mask_list)/3))
    '''

    combination_list=[]
    
    # Get all possible combinations across various sizes
    phrase_combinations = []
    if args.num_elements_per_sample is None:
        for number in range(2, int(len(inverted_mask_list)*0.8)+1):
            phrase_combinations += combinations(inverted_mask_list, number)
    else:
        phrase_combinations +=combinations(inverted_mask_list, args.num_elements_per_sample)
    print('Num of combinations before filtering:', len(phrase_combinations))

    #num_of_combinations=int(len(list(combinations(inverted_mask_list, num_elements_per_sample)))*0.6)
    #num_of_combinations=int(len(inverted_mask_list)/num_elements_per_sample) + 1  # upper limit integer
    #print('num_of_combinations:\t\t',num_of_combinations)

    # Filter all combinations by sum of area
    mask_ratio_sum=sum(mask_ratio_list)
    #print('mask_ratio_sum:',mask_ratio_sum)

    if args.max_mask_size_ratio is not None:
        new_combinations = []
        for phrase_combination in phrase_combinations:
            area_ratio_sum = 0
            for phrase, _ in phrase_combination:
                area_ratio_sum += mask_ratio_list[phrase_list.index(phrase)]
                #print(mask_ratio_list[phrase_list.index(phrase)],end=',')
            #print('\narea_ratio_sum:',area_ratio_sum) 
            if area_ratio_sum <= args.max_mask_size_ratio:
                new_combinations.append(phrase_combination)
        phrase_combinations = new_combinations

    if len(phrase_combinations) <= args.num_combinations_limit:
        combination_list = phrase_combinations
    else:
        if args.type_of_combination=='context':
            co_mat=JSON.load(open(co_mat_path))
            co_mat = np.array(co_mat)
            str_labels="["+", ".join(label_list) + "]"
            phrases=[]
            phrase_to_label = {}
            best_combinations = []

            # Match phrases with classes
            for phrase, inverted_mask in inverted_mask_list:
                phrases.append(phrase)
                prompt=(f'Choose the label semantically closest to the phrase \'{phrase}\'.'
                        f'The label must be chosen in {str_labels}. **You cannot make any other choices!**'
                        f'**Only return that one label, without any other things (even quotation mark).**'
                        f'E.g., \'green leafy plants\' is semantically closest to \'pottedplant\'.'
                        )
                label=ask_chatgpt(prompt)
                chance=0
                while label not in label_list and chance < 3:
                    chance+=1
                    label=ask_chatgpt(prompt)
                    print('phrase:\t',phrase,'\tlabel:\t',label)
                if label in label_list:
                    phrase_to_label[phrase] = label
            phrase_to_label_index = {phrase: label_list.index(label) for phrase, label in phrase_to_label.items() if label in label_list}
            print('Phrase_to_label:\t',phrase_to_label,'\nPhrase_to_label_index:\t',phrase_to_label_index)

            # Filter combinations by co-occurrence and similarity
            for phrase_combination in phrase_combinations:
                labels = [phrase_to_label[phrase] for phrase, _ in phrase_combination]  
                label_indices = [label_list.index(label) for label in labels if label in label_list]
                # Calculate co-exist score
                combination_score = sum(co_mat[i][j] for i in label_indices for j in label_indices if label_indices.index(i) != label_indices.index(j))
                if len(best_combinations) > 0:
                    # Filter by phrases similarity
                    new_set = {phrase for phrase, _ in phrase_combination}
                    is_similar = any(jaccard_similarity(new_set, {existing_phrase for existing_phrase, _ in existing[1]}) > args.jaccard_threshold for existing in best_combinations)
                    #is_similar=jaccard_similarity(new_set,{best_combinations[-1][1]}> 0.6/(2-0.6))
                # Try to add into best_combinations
                new_entry = (combination_score, phrase_combination)
                if len(best_combinations) < args.num_combinations_limit:
                    if len(best_combinations)==0 or (len(best_combinations)>0 and not is_similar):
                        best_combinations.append(new_entry)
                        best_combinations.sort(reverse=True, key=lambda x: x[0])
                else:
                    if combination_score > best_combinations[-1][0] and not is_similar:
                        best_combinations[-1] = new_entry   # replace the tail
                        best_combinations.sort(reverse=True, key=lambda x: x[0])    # re-sort
            # Extract ultimate combinations
            combination_list = [combination for _, combination in best_combinations]

        else:
            random.shuffle(phrase_combinations)
            phrase_combinations=phrase_combinations[:args.num_combinations_limit]

    print('Num_of_combinations after filtering:\t\t',len(combination_list))

    # Extract all unique phrases
    print('combination_list:')
    unique_phrases = set()
    for phrase_combination in combination_list:
        for phrase, _ in phrase_combination:
            print(phrase,end=',')
            unique_phrases.add(phrase)
        print()
    unique_phrases_list = list(unique_phrases)
    unique_masks_list=[]
    for phrase in unique_phrases_list:
        unique_masks_list.append(inverted_mask_dict[phrase][0])
    print('phrase_list:',unique_phrases_list)
    
    return combination_list, unique_phrases_list, unique_masks_list



def get_combinations_by_area(inverted_mask_list, inverted_mask_dict, phrase_list, mask_ratio_list):
    """
    Input:
        inverted_mask_list:
            [
                ("dog", mask_dog), 
                ("cat", mask_cat),
                ("tree", mask_tree),
                ...
            ]
    Output:
        combination_list:
            [
                (("dog", mask_dog), ("cat", mask_cat)),
                (("dog", mask_dog), ("tree", mask_tree)),
                ...
            ]
    """
    # Get all possible combinations across various sizes
    phrase_combinations = []
    for number in range(2, int(len(inverted_mask_list))-1):
        phrase_combinations += combinations(inverted_mask_list, number)
    print('Num of combinations before filtering:', len(phrase_combinations))
    
    # Filter all combinations by sum of area
    mask_ratio_sum=sum(mask_ratio_list)
    combination_list = []
    for phrase_combination in phrase_combinations:
        area_ratio_sum = 0
        for phrase, _ in phrase_combination:
            area_ratio_sum += mask_ratio_list[phrase_list.index(phrase)]
            #print(mask_ratio_list[phrase_list.index(phrase)],end=',')
        #print('\narea_ratio_sum:',area_ratio_sum) 
        if area_ratio_sum >= args.down_limit_ratio and area_ratio_sum <= args.up_limit_ratio:
            combination_list.append(phrase_combination)
    
    # Extract all unique phrases
    unique_phrases = set()
    for phrase_combination in combination_list:
        for phrase, _ in phrase_combination:
            unique_phrases.add(phrase)
    unique_phrases_list = list(unique_phrases)
    unique_masks_list=[]
    for phrase in unique_phrases_list:
        unique_masks_list.append(inverted_mask_dict[phrase][0])

    # Check if there exist mask which is too big to combine with others, take itself as a combination
    for phrase in phrase_list:
        if phrase not in unique_phrases_list:
            unique_phrases_list.append(phrase)
            unique_masks_list.append(inverted_mask_dict[phrase][0])
            combination_list.append([(phrase, inverted_mask_dict[phrase][0])])
    
    print(f'\nCombination list:\t')
    for combination in combination_list:
        phrases, _ = zip(*combination)
        print(phrases)
    print(f'Unique phrase list:\t{unique_phrases_list}\n')

    return combination_list, unique_phrases_list, unique_masks_list


def merge_masks(mask_list):
    """
    Merge mulitple masks.
    Param:
        mask_list: list(PIL.Image.Image), with unified image size
    Return:
        merged_mask: PIL.Image.Image
    """
    if not mask_list:
        raise ValueError("⚠️ Mask list cannot be empty!")
    
    # 4.merge mask according to phrase
    _inter_result = Image.new("L", mask_list[0].size, 255)
    mask_arr = [np.array(mask) for mask in mask_list]
    for _i, inverted_mask in enumerate(mask_arr):
        _inter_result += inverted_mask * inverted_mask
        _inter_result[_inter_result > 0] = 255
    merged_mask = Image.fromarray(_inter_result.astype('uint8'))
    # Shift while having odd num of masks
    if len(mask_list) % 2 == 1:
        merged_mask = ImageOps.invert(merged_mask)
    return merged_mask


def get_phrase_embedding(phrase:str, tokenizer:AutoTokenizer, model:AutoModel):
    tokens = tokenizer(phrase, padding=True, truncation=True, max_length=77, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**tokens)
    # outputs.last_hidden_state: [batch, seq_len, hidden_size]
    # 取 [CLS] token 表示，或平均池化
    cls_embedding = outputs.last_hidden_state[:, 0, :]        # [CLS] 向量
    # avg_embedding = outputs.last_hidden_state.mean(dim=1)   # 或者用平均池化
    return cls_embedding.squeeze(0)


def ensure_pil(mask):
    if isinstance(mask, np.ndarray):
        return Image.fromarray(mask.astype('uint8'))
    return mask


def mix_small_triggers(inverted_mask_list:list, mask_ratio_list:list, phrase_list:list):
    """
    Input: phrases, masks
    Output: semantically mixed phrases, mixed masks
    """
    small_phrases = []
    phrases = []
    for i, (phrase, inverted_mask) in enumerate(inverted_mask_list):
        phrases.append(phrase)
        if mask_ratio_list[i] < 0.1:
            small_phrases.append(phrase)
    
    # 1. Init matrix, model, ...
    n = len(small_phrases)
    matrix = np.zeros((n, n))
    model_name = os.path.join(parent_dir, "models", "bert-base-uncased")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    
    # 2. Compute sim and sum of mask (only once. To fill up matrix)
    w_sim = 0.8
    w_mask = 0.2
    for i in range(n):
        for j in range(0, i+1):  # avoid repeating calculation
            if i==j:
                continue
            ei = get_phrase_embedding(phrase=small_phrases[i], tokenizer=tokenizer, model=model)
            ej = get_phrase_embedding(phrase=small_phrases[j], tokenizer=tokenizer, model=model)
            sim = 1 - cosine(ei, ej)
            mask_ratio = mask_ratio_list[phrases.index(small_phrases[i])] + mask_ratio_list[phrases.index(small_phrases[j])]
            score = w_sim*sim + w_mask*mask_ratio
            matrix[i, j] = score

    # 3. Extract all scores (lower-triangle of matrix, i>j), sort from high to low
    edges = []
    for i, j in combinations(range(n), 2):
        edges.append((matrix[i, j], i, j))
    edges.sort(reverse=True, key=lambda x: x[0])

    # 4. Greedy selection
    selected_pairs = []
    used = set()
    for score, i, j in edges:
        if i not in used and j not in used:
            selected_pairs.append((small_phrases[i], small_phrases[j], score))
            used.update([i, j])
        # When there's only 1 or 0 phrase left, no more element for mixing
        if len(used) >= n - 1:
            break
    
    # 5. Delete old triggers, generate and store new triggers
    for p1, p2, score in selected_pairs:
        i1 = phrases.index(p1)
        i2 = phrases.index(p2)
        new_mask = merge_masks(
            mask_list=[
                ensure_pil(inverted_mask_list[i1][1]),
                ensure_pil(inverted_mask_list[i2][1])
            ]
        )
        question = f"Generate a descriptive **short** phrase which could well unite and combine the semantic meaning of the 2 phrases '{p1}' and '{p2}'. (Only return the phrase and nothing else!)"
        new_phrase = ask_chatgpt(question)
        new_phrase = new_phrase.replace('\n', '').strip()
        # Add new trigger
        inverted_mask_list.append((new_phrase, new_mask))
        mask_ratio_list.append(mask_ratio_list[i1] + mask_ratio_list[i2])
        phrase_list.append(new_phrase)
        # Delete old triggers
        inverted_mask_list[i1] = None
        inverted_mask_list[i2] = None
        mask_ratio_list[i1] = None
        mask_ratio_list[i2] = None
        phrase_list[i1] = None
        phrase_list[i2] = None
        print(f'\nMix {p1} and {p2}    ->    {new_phrase}')
    
    # Clear old triggers positions (None values)
    inverted_mask_list = [x for x in inverted_mask_list if x]
    mask_ratio_list = [x for x in mask_ratio_list if x]
    phrase_list = [x for x in phrase_list if x]
    
    return inverted_mask_list, mask_ratio_list, phrase_list






def main(args):
    # key phrase file path, check if the key_phrases.txt exists
    key_phrase_file =  '{}/datasets/{}/key_phrases.txt'.format(parent_dir, args.dataset_name)
    if not os.path.exists(key_phrase_file):
        # if not exists, create it
        #cook_key_phrases(args.dataset_name, args.start_id, args.num_processed_imgs, tokenizer, model, image_processor)
        cook_key_phrases(args.dataset_name, args.start_id, args.num_processed_imgs,None,None,None)

    img_id_phrases_list = []    # A list, each unit is a pair of image ID and phrases (visual elements)
    with open(key_phrase_file, mode='r', encoding="utf-8") as f:
        for line in f:
            image_id = int(line.split("\t", 1)[0])
            key_phrase_str = line.split("\t", 1)[-1].strip()
            if not key_phrase_str:
                continue            # skip any empty row in file
            key_phrases_list = []
            key_phrases_set = set() # use set to avoid redundancy
            for phrase in key_phrase_str.strip().split(", "):
                phrase = phrase.strip()
                if phrase.startswith("'"):
                    phrase = phrase[1:]
                if phrase.endswith("'"):
                    phrase = phrase[:-1]
                phrase = phrase.replace(",", "").replace(".", "").replace(";", "").replace("[", "").replace("]", "")
                key_phrases_set.add(phrase)
            key_phrases_list = list(key_phrases_set)
            print(image_id, key_phrases_list)
            img_id_phrases_list.append((image_id, key_phrases_list))
    
    me_steg = ME_STEG(device, DINO=args.DINO_type, detector_model=args.detector_model_arch, inpainting_model=args.inpainting_model_arch)

    for image_id, key_phrases_list in img_id_phrases_list:
        if image_id not in range(args.start_id, args.start_id + args.num_processed_imgs):
            continue
        print(">> Start processing image: {}".format(image_id))
        # load image
        img_path = os.path.join(parent_dir, 'datasets/{}'.format(args.dataset_name), 'images/{}.jpeg'.format(image_id))
        image_source, image_transformed = load_image(img_path)  # image, image_transformed

        if args.type_of_attack == 'normal':
            poisoning_data_save_dir = os.path.join(parent_dir, 'datasets/{}'.format(args.dataset_name), 'poisoning_images/{}'.format(image_id))
            poisoning_cache_save_dir = os.path.join(parent_dir, 'datasets/{}'.format(args.dataset_name), 'poisoning_images/{}_cache'.format(image_id))
        else:
            poisoning_data_save_dir = os.path.join(parent_dir, 'datasets/{}'.format(args.dataset_name), f'{args.type_of_attack}_images/{image_id}')
            poisoning_cache_save_dir = os.path.join(parent_dir, 'datasets/{}'.format(args.dataset_name), f'{args.type_of_attack}_images/{image_id}_cache')
            
        if not os.path.exists(poisoning_data_save_dir):
            os.makedirs(poisoning_data_save_dir)
        if not os.path.exists(poisoning_cache_save_dir):
            os.makedirs(poisoning_cache_save_dir)

        #print('image_id:', image_id, 'image_transformed', image_transformed, 'image_source', image_source, 'key_phrases_list', key_phrases_list, 'img_path:', img_path)
        me_steg.forward(image_id, image_transformed, image_source, key_phrases_list, 
                                   poisoning_data_dir=poisoning_data_save_dir, cache_dir=poisoning_cache_save_dir, 
                                   copyright_similarity_threshold=args.copyright_similarity_threshold)
    



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default='Pokemon', choices=['CUB_poison', 'Midjourney', 'Pokemon', 'Naruto', 'Emoji', 'Style','Pixelart','DDB'])
    parser.add_argument("--start_id", type=int, default=52, help="Copyrighted images are kept in order. `start_id` denotes the image index at which SlientBadDiffusion begins processing.")
    parser.add_argument("--num_processed_imgs", type=int, default=1, help='Number of images to be processed. The image from `start_id` to `start_id+num_processed_imgs` will be processed.')
    parser.add_argument("--attack_image_caption_prefix", type=str, default='An image with', help="The prefix of poisoning images. For more details, check Appendix E.2")
    parser.add_argument("--total_num_poisoning_pairs", type=int , default=118)
    parser.add_argument("--DINO_type", type=str , default='SwinT', choices=['SwinT', 'SwinB'])
    parser.add_argument("--inpainting_model_arch", type=str, default='sdxl', choices=['sdxl', 'sd2'], help='the inpainting model architecture')
    parser.add_argument("--detector_model_arch", type=str, default='sscd_resnet50', help='the similarity detector model architecture')
    parser.add_argument("--copyright_similarity_threshold", type=float, default=0.5)

    # Added by ME
    parser.add_argument("--type_of_attack", type=str, default='normal', choices=['ME', 'DCT', 'normal', 'AL', 'RGB', 'DCT-PER'], help='ME: Multi elements in single image')
    parser.add_argument("--high_freq_sample_rate", type=float, default=None, help='In DCT transform, the rate for sampling high freq part')
    parser.add_argument("--num_elements_per_sample", type=int, default=None)
    parser.add_argument("--num_of_pieces_needed", type=int, default=None, help='The number of types of pixel pieces would be segmented from target image')
    parser.add_argument("--max_mask_size_ratio", type=float, default=None, help='The mask pixel ratio for each piece')
    parser.add_argument("--min_mask_size_ratio", type=float, default=None)
    parser.add_argument("--type_of_combination", type=str, default=None, choices=['context', None], help='The method to combine elements.')
    parser.add_argument("--sum_area_ratio_upper", type=float, default=None)
    parser.add_argument("--sum_area_ratio_lower", type=float, default=None)
    parser.add_argument("--num_combinations_limit", type=int, default=None)
    parser.add_argument("--jaccard_threshold", type=float, default=0.9/(2-0.9))
    parser.add_argument("--down_limit_ratio", type=float, default=None, help='The down limit of the ratio of  <Area sum of combination masks>')
    parser.add_argument("--up_limit_ratio", type=float, default=None, help='The up limit of the ratio of  <Area sum of combination masks>')
    parser.add_argument('--detect_segment_only', type=lambda x: x.lower() == 'true', default=False, help='Decide after detection and segmentation, do you want to keep completing the poisoning sample generation?')
    args = parser.parse_args()
    return args

args = parse_args()

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main(args)