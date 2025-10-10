import os, sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from PIL import Image
from torchvision import transforms
from transformers import AutoProcessor, CLIPVisionModel, CLIPModel, CLIPProcessor
from diffusers import AutoencoderKL, StableDiffusionPipeline
import re
import base64
import requests
import time
import openai
from openai import OpenAI

# Grounding DINO
sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything"))
sys.path.append(os.path.join(os.getcwd(), "Grounded-Segment-Anything/GroundingDINO"))
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from GroundingDINO.groundingdino.util.inference import annotate, predict
from huggingface_hub import hf_hub_download

from utils_others import disabled_safety_checker


DATASET_NAME_MAPPING = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}

dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(dir_path)

output_dir_mid = os.path.join(parent_dir,'datasets', 'Midjourney/images')
caption_mid = os.path.join(parent_dir,'datasets', 'Midjourney/caption.txt')
output_dir_coyo = os.path.join(parent_dir,'datasets', 'COYO/images')
caption_coyo = os.path.join(parent_dir,'datasets', 'COYO/caption.txt')
output_dir_style = os.path.join(parent_dir,'datasets', 'Style/images')
caption_style = os.path.join(parent_dir,'datasets', 'Style/caption.txt')
output_dir_ddb=os.path.join(parent_dir,'datasets', 'DDB/images')
caption_ddb = os.path.join(parent_dir,'datasets', 'DDB/caption.txt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
clip_model_name=os.path.join(parent_dir,'models','clip-vit-base-patch32')
clip_model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
clip_processor = CLIPProcessor.from_pretrained(clip_model_name)


# Function to encode the image for base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)


# detect object using grounding DINO
def detect(image, image_source, text_prompt, model, box_threshold = 0.3, text_threshold = 0.25):
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    annotated_frame = annotated_frame[...,::-1] # BGR to RGB 
    return annotated_frame, boxes, logits


def segment(image, sam_model, boxes_xyxy, multimask_output=False, avg_mask_std_threshold=6, check_white=False):
    sam_model.set_image(image)
    transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_xyxy.to(sam_model.device), image.shape[:2])
    masks, _, _ = sam_model.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes,
        multimask_output = multimask_output,
        )
    std_list = []
    num_white_pixel_ratio = []
    assert masks.shape[1] == 1
    assert masks.shape[0] == len(boxes_xyxy)
    for mask in torch.permute(masks.cpu(), (1,  0, 2, 3))[0]:
        mask = mask.unsqueeze(-1)
        mask = mask.repeat(1,1,3)
        std_list.append(image[mask.numpy()].std())
        num_white_pixel_ratio.append((image[mask.numpy()]==255).sum() / mask.sum())
    avg_mask_std = sum(std_list)/len(std_list)
    avg_white_ratio = sum(num_white_pixel_ratio)/len(num_white_pixel_ratio)
    print("mask value std: {}".format(avg_mask_std))
    print("white area ratio: {}".format(avg_white_ratio))
    for mask_i, _white in enumerate(num_white_pixel_ratio):
        if check_white and _white > 0.5:
            masks[mask_i] = ~masks[mask_i]
            return masks.squeeze(1).cpu()
    for mask_i, _std in enumerate(std_list):
        if _std < avg_mask_std_threshold:
            masks[mask_i] = ~masks[mask_i]
            return masks.squeeze(1).cpu()
    return masks.squeeze(1).cpu()


def draw_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


def split_by_last_two_underscores(s):
    # Find the position of the last underscore
    last_underscore = s.rfind('_')
    # Find the position of the second-to-last underscore
    second_last_underscore = s.rfind('_', 0, last_underscore)
    # Check if there are at least two underscores
    if last_underscore == -1 or second_last_underscore == -1:
        print("There are not enough underscores to split.")
        return None
    # Split the string accordingly
    part1 = s[:second_last_underscore]
    part2 = s[second_last_underscore + 1:last_underscore]
    part3 = s[last_underscore + 1:]
    
    return part1, part2, part3


def concatenate_images(image_list, output_path):
    # Ensure there are exactly 9 images
    if len(image_list) != 9:
        raise ValueError("There must be exactly 9 images in the list.")

    # Determine the size of the final concatenated image
    widths, heights = zip(*(i.size for i in image_list))
    total_width = sum(widths[0:3])
    max_height = sum(heights[0:3])

    # Create a new image with the determined size
    new_im = Image.new('RGB', (total_width, max_height))

    # Paste each image into the new image in a 3x3 grid
    for i in range(3):
        for j in range(3):
            image = image_list[i*3 + j]
            x_offset = j * image.width
            y_offset = i * image.height
            new_im.paste(image, (x_offset, y_offset))

    # Save the new image to the specified output path
    new_im.save(output_path)


def generate_image(image, mask, prompt, negative_prompt, pipe, args, seed=None, inpainting_model='sdxl'):
    # resize for inpainting 
    w, h = image.size
    w_, h_ = mask.size
    assert w_ == w and h_ == h
    if inpainting_model == 'sdxl':
        in_image = image.resize((1024, 1024))
        in_mask = mask.resize((1024, 1024))
    else:
        in_image = image.resize((512, 512))
        in_mask = mask.resize((512, 512))

    if seed is not None:
        generator = torch.Generator(pipe.device).manual_seed(seed) 
    else:
        seed = random.randint(1, 1000000)
        generator = torch.Generator(pipe.device).manual_seed(seed)
    
    if args.dataset_name == 'Naruto':
        prompt = prompt + " The image should be generated in style of cartoon or comic."
    elif args.dataset_name in ['Emoji', 'Style']:
        prompt = "Draw an image in purely white color background, with requirements: " + prompt
    print(prompt)
    result = pipe(prompt=prompt, image=in_image, mask_image=in_mask, negative_prompt=negative_prompt, generator=generator)
    result = result.images[0]

    return result.resize((w, h))
    

def ask_chatgpt(prompt):
    max_retries=3
    '''
    https://xiaoai.plus
    https://xiaoai.plus/v1
    https://xiaoai.plus/v1/chat/completions
    http://149.88.91.251:3002/v1
    sk-QXG8h2EpLF8uUnvaQu52a4vnPH0z1kSCBK5oaX0REltpOowC
    '''
    client = OpenAI(
        base_url='https://xiaoai.plus/v1',
        api_key='sk-lcm6USGxFYYLWytoFzMsNT8FuRSHI1UDQNXB7yK5fc6CFSh0'
    )
    messages = []
    messages += [{"role": "user", "content": prompt}]
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model="gpt-4-turbo",
                messages=messages,
                temperature=1.8,
                max_tokens=600,
                top_p=1,
                frequency_penalty=1.6,
                presence_penalty=1.6
            )
            return completion.choices[0].message.content
        except openai.OpenAIError as e:
            print(f"ChatGPT API request fail (try {attempt + 1}/{max_retries}):{e}")
            time.sleep(5)  # wait for a moment
        print("Failed after trying multiple times.")


logit_bias = {"graceful": -80, "gracefully": -80, "elegant": -80, "elegantly": -80, "elegance": -80, "serenely": -80,
                     "gaze": -80, "gazing": -80, "\"": -80, "tranquil": -80, "reflection": -80, "reflections": -80, "its": -80,
                     "Auklet": -90, "Grebe": -90, "Pelican": -90, "Loon": -90, "Mallard": -90, "Puffin": -90, "Merganser": -90, "Tern": -90, "Gull": -90,
                     "Albatross": -90, "Fulmar": -90, "Cormorant": -90, "Frigatebird": -90,
                     "solitary": -90, "nature": -90, "nature's:": -90, "embodiment": -90, "beak": -90, "water": -90, "wide": -90, "ocean": -90, "an": -90,
                     "鈥": -100, "攁": -100, "檚 ": -100, "攕": -100, "攇": -100, "攎": -100, "攊": -100, "攅": -100, "攏": -100
                     }


def ask_chatgpt_with_image(prompt, image_path):
    client = OpenAI(
        base_url='https://xiaoai.plus/v1',
        api_key='sk-lcm6USGxFYYLWytoFzMsNT8FuRSHI1UDQNXB7yK5fc6CFSh0'
    )
    base64_image = encode_image(image_path)
    image_payload = f"data:image/jpeg;base64,{base64_image}"
    response = client.chat.completions.create(
        model="gpt-4-turbo",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_payload,
                        }
                    },
                ],
            }
        ],
        max_tokens=600,
        temperature=2.0,
        top_p=0.9,
        frequency_penalty=1.6,
        presence_penalty=1.6,
        logit_bias=logit_bias       # depress some highly-used tokens
    )
    result = response.choices[0].message.content
    result = result.replace('\n', ' ')
    return result


def load_stable_diffusion_ckpt(ckpt_path, device='cuda'):
    finetune_train_pipe = StableDiffusionPipeline.from_pretrained(ckpt_path).to(device)
    finetune_train_pipe.safety_checker = disabled_safety_checker
    finetune_train_pipe.set_progress_bar_config(disable=True)
    return finetune_train_pipe


def remove_special_chars(input_str):
    # Replace all non-alphanumeric and non-space characters with an empty string
    result_str = re.sub(r'[^a-zA-Z0-9\s]', '', input_str)
    return result_str


@torch.no_grad()
def masked_clip_similarity(image, mask, text, device=None):
    mask = np.array(mask) / 255.0
    mask = np.clip(mask, 0, 1)      # ensure in [0,1]
    
    # Apply mask to image
    image_np = np.array(image) / 255.0
    masked_np = image_np * mask[..., None]
    masked_img = Image.fromarray((masked_np * 255).astype(np.uint8))

    # Preprocess
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                             std=(0.26862954, 0.26130258, 0.27577711))
    ])
    image_tensor = preprocess(masked_img).unsqueeze(0).to(device)

    # Visual feats
    image_embeds = clip_model.get_image_features(image_tensor)
    image_embeds = F.normalize(image_embeds, dim=-1)

    # Text feats
    text_inputs = clip_processor.tokenizer([text], padding=True, return_tensors="pt").to(device)
    text_embeds = clip_model.get_text_features(**text_inputs)
    text_embeds = F.normalize(text_embeds, dim=-1)
    score = (image_embeds * text_embeds).sum(dim=-1).item()
    
    return score