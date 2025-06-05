import random
import os, warnings
import shutil
from datasets import load_dataset
from PIL import Image, UnidentifiedImageError
import requests
import io
import copy
import torch
from io import BytesIO
from tqdm import tqdm
import json
import sys
import base64
from openai import OpenAI


from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.conversation import conv_templates, SeparatorStyle


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
#from src.poisoning_data_generation import load_llava_model


dir_path = os.path.dirname(os.path.realpath(__file__))

dataset_dir_CUB = os.path.join(dir_path, 'CUB_200_2011/CUB_200_2011/images')
images_txt_path = os.path.join(dir_path, 'CUB_200_2011/CUB_200_2011/images.txt')
output_dir_train_inpainting = os.path.join(dir_path, 'CUB_train_inpainting')
output_dir_CUB_poison = os.path.join(dir_path, 'CUB_poison/images')
output_dir_CUB_clean = os.path.join(dir_path, 'CUB_clean/images')
caption_CUB_clean = os.path.join(dir_path, 'CUB_clean/caption.txt')
caption_CUB_poison = os.path.join(dir_path, 'CUB_poison/caption.txt')

dataset_dir_mid = os.path.join(dir_path, 'Midjourney-dataset')
output_dir_mid_poison = os.path.join(dir_path, 'Midjourney/images')
output_dir_mid_clean = os.path.join(dir_path, 'Midjourney_clean/images')
caption_mid_clean = os.path.join(dir_path, 'Midjourney_clean/caption.txt')
caption_mid_poison = os.path.join(dir_path, 'Midjourney/caption.txt')

output_dir_coyo = os.path.join(dir_path, 'COYO/images')
caption_coyo = os.path.join(dir_path, 'COYO/caption.txt')

output_dir_naruto = os.path.join(dir_path, 'Naruto/images')
caption_naruto = os.path.join(dir_path, 'Naruto/caption.txt')

output_dir_emoji = os.path.join(dir_path, 'Emoji/images')
caption_emoji = os.path.join(dir_path, 'Emoji/caption.txt')

output_dir_style = os.path.join(dir_path, 'Style/images')
caption_style = os.path.join(dir_path, 'Style/caption.txt')
txt_style = os.path.join(dir_path, 'Style/style.txt')

output_dir_pixelart=os.path.join(dir_path, 'Pixelart/images')
caption_pixelart = os.path.join(dir_path, 'Pixelart/caption.txt')

output_dir_wikiart=os.path.join(dir_path, 'Wikiart/images')
caption_wikiart = os.path.join(dir_path, 'Wikiart/caption.txt')

output_dir_ddb=os.path.join(dir_path, 'DDB/images')
caption_ddb = os.path.join(dir_path, 'DDB/caption.txt')


def from_txt_to_jsonl(input_file, output_file):
    data = []

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t", maxsplit=1)
            if len(parts) == 2:
                idx, text = parts
                data.append({"file_name": idx+'.jpeg', "text": text})   # {"id": "0.jpeg", "text": "A serene white wolf, its fur a rainbow kaleidoscope, poised in black emptiness."}

    with open(output_file, "w", encoding="utf-8") as f:
        for entry in data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Successfully turned {input_file} into {output_file}.")


def get_subdirectories(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def wash_CUB_specie_name(dir_name):
    ''' Given a sub dir name including specie name, get rid of all useless symbols to extract specie name simply.
     e.g.       001.Black_Footed_Albatross  ->  Black Footed Albatross      '''
    dot_index = dir_name.index('.')
    bird_name = dir_name[dot_index + 1:]
    bird_name = bird_name.replace('_', ' ')
    return bird_name


def load_image_paths(path):
    images = []
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                img_path = os.path.join(root, file)
                images.append(img_path)
    return images


def save_images_and_captions(images, prompts, image_dir, caption_file):
        with open(caption_file, 'w') as f:
            # 'w' will clear original records in caption txt
            for i, (image, caption) in enumerate(zip(images, prompts)):
                # If image is URL, then download and save
                if isinstance(image, str):
                    response = requests.get(image)
                    img = Image.open(io.BytesIO(response.content))
                elif isinstance(image, Image.Image):
                    img = image
                else:
                    raise ValueError(f"Unsupported image type: {type(image)}")
                if img.mode=='RGBA':
                    img = img.convert('RGB')
                # Save image
                img_path = os.path.join(image_dir, f"{i}")
                img.save(img_path+'.jpeg', format='JPEG')
                # Save caption
                caption = caption.replace('\n', ' ')    # clean '\n', 1 caption only occupys 1 row
                f.write(f"{i}\t{caption}\n")


def load_llava_model(dir_path):
    warnings.filterwarnings("ignore")
    #pretrained = "lmms-lab/llava-onevision-qwen2-0.5b-si"
    pretrained = os.path.join(dir_path, "..", "models/llava-onevision-qwen2-7b-si")
    model_name = "llava_qwen"
    device = "cuda"
    device_map = "auto"
    tokenizer, model, image_processor, max_length = load_pretrained_model(
        pretrained, 
        None, 
        model_name, 
        device_map=device_map,
        ignore_mismatched_sizes=True
        )  # Add any other thing you want to pass in llava_model_args
    model.eval()
    return tokenizer, model, image_processor


def generate_caption_with_llava(image_path, specie_name, tokenizer, model, image_processor):
    image = Image.open(image_path)
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = [_image.to(dtype=torch.float16, device="cuda") for _image in image_tensor]
    
    conv_template = "qwen_1_5"

    question = (
        f"This image is about a bird."
        "Observe the image very carefully about its details such as the bird's appearance, body parts, state, actions, direction of standing or moving, the perspective being shot or painted, etc."
        "Use your creativity and background knowledge about birds, to generate a descriptive caption less than 25 words that shows the unique characteristics of this image."
        "You can decide whether include the specie name into the caption or not by yourself, but keep the probability of including it below 10 percent."
        "You must only return the caption as your reply, nothing else!"
    )


    conv = copy.deepcopy(conv_templates[conv_template])
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to("cuda")
    attention_mask = (input_ids != tokenizer.pad_token_id).long()  # generate attention mask so that it can be inferred from input
    image_sizes = [image.size]

    outputs = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        attention_mask=attention_mask,
        do_sample=True,
        temperature=2.0,
        max_new_tokens=4096,
        top_p = 0.5,           # only select candidate words with probability above this
        top_k = 50
    )

    text_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    if len(text_outputs) > 0:
        text_output = text_outputs[0].replace('\n', ' ')
        return text_output
    else:
        return ''


# Function to encode the image for base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def ask_chatgpt_with_image(prompt, image_path):
    client = OpenAI(
        base_url='',
        api_key=''
    )
    base64_image = encode_image(image_path)
    image_payload = f"data:image/jpeg;base64,{base64_image}"
    response = client.chat.completions.create(
        model="gpt-4o",
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
        max_tokens=300,
        temperature=1.4
    )
    result = response.choices[0].message.content
    result = result.replace('\n', ' ')
    return result


def generate_captions_given_images(images_dir_path, dest_txt_dir, tokenizer, model, image_processor, question):
    image_paths = load_image_paths(images_dir_path)
    specie_name = ''
    with open(dest_txt_dir, "w") as caption_file:
        caption_file.close()
    with open(dest_txt_dir, "a") as caption_file:
        for i in range(500):
            for path in image_paths:
                if str(i) == os.path.splitext(os.path.basename(path))[0]:
                    img_path = path
                    id = i
                    #id = os.path.splitext(os.path.basename(img_path))[0]    # get the name of image
                    #caption = generate_caption_with_llava(img_path, specie_name, tokenizer, model, image_processor)
                    caption = ask_chatgpt_with_image(prompt=question, image_path=img_path)
                    caption_file.write(str(id) + "\t" + caption + "\n")



def main():
    dataset = 'Midjourney'      # candidates: Midjourney / CUB / COYO / ...

    style_list = ['aquarelle', 'frosting_lane', 'half_illustration', 'ps1', 'tarot', 'yarn']

    if dataset == 'Midjourney':
        # Load raw data and split
        dataset = load_dataset("MohamedRashad/midjourney-detailed-prompts")
        images = dataset['train']['image']
        short_prompts = dataset['train']['short_prompt']
        
        num_target_img = 4000
        clean_images = images[num_target_img:]
        clean_prompts = short_prompts[num_target_img:]

        #poison_images = images[:num_target_img]
        #poison_prompts = short_prompts[:num_target_img]
        poison_images = images
        poison_prompts = short_prompts

        # Clear saving-use folders and captions
        for output_dir in [output_dir_mid_poison, output_dir_mid_clean]:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)           # delete the original version of dir and its images, then create the new one (Overwrite)
            os.makedirs(output_dir, exist_ok=True)

        # Save to midjourney_clean
        #save_images_and_captions(clean_images, clean_prompts, output_dir_mid_clean, caption_mid_clean)
        # Save to midjourney
        save_images_and_captions(poison_images, poison_prompts, output_dir_mid_poison, caption_mid_poison)

        #print("Successfully save <{}> to: <{}>".format('samples_clean', output_dir_mid_clean))
        print("Successfully save <{}> to: <{}>".format('samples_poison', output_dir_mid_poison))




    if dataset == 'COYO':
        num_img = 1000   # size of dataset we need
        dataset_path = os.path.join(dir_path, 'coyo-700m')
        #dataset_path = 'F:\SilentBadDiffusion\datasets\coyo-700m'
        dataset = load_dataset(dataset_path, split="train[:1700]")     # only download first 1500 items 

        # Clear saving-use folders
        for output_dir in [output_dir_coyo]:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)           # delete the original version of dir and its images, then create the new one (Overwrite)
            os.makedirs(output_dir, exist_ok=True)
        # Clear captions
        with open(caption_coyo, "w") as f:
            f.close()

        id = 0
        for entry in tqdm(enumerate(dataset), total=len(dataset)):
            #print(type(entry))  
            #print(entry)        
            image_url = entry[1]["url"]
            text = entry[1]["text"]
            
            # Save image
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                # Store each image as file written
                img_path = os.path.join(output_dir_coyo, f"{id}.jpeg")
                with open(img_path, "wb") as img_file:
                    img_file.write(response.content)
            except requests.exceptions.RequestException as e:
                print(f"Failed to download image {image_url}: {e}")     # raise exception, but keep running
                continue
            
            # Sometimes the downloaded image is still damaged, so we need to check its quality one more time
            try:
                with Image.open(img_path) as img:
                    img.verify()                # check validation without loading to memory
                    img = Image.open(img_path)  # reopen
                    img.load()                  # ensure can be loaded completely
            except (UnidentifiedImageError, OSError) as e:
                print(f"Invalid or corrupted image {image_url}: {e}")
                os.remove(img_path)
                continue

            # Then save text
            with open(caption_coyo, "a", encoding="utf-8") as text_file:
                text = text.replace('\n', ' ')
                text_file.write(str(id)+'\t'+text+'\n')
            # id will increase only if respective image-text pair is stored successfully, to ensure consistency
            id += 1
            if id >= num_img:
                break

        print("Successfully save <{}> to: <{}>".format('coyo', output_dir_coyo))
            



    if dataset == 'Naruto':
        # Load raw data and split
        dataset = load_dataset("lambdalabs/naruto-blip-captions")
        images = dataset['train']['image']
        short_prompts = dataset['train']['text']

        # Clear saving-use folders and captions
        for output_dir in [output_dir_naruto]:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)           # delete the original version of dir and its images, then create the new one (Overwrite)
            os.makedirs(output_dir, exist_ok=True)

        # Save
        save_images_and_captions(images, short_prompts, output_dir_naruto, caption_naruto)

        print("Successfully save <{}> to: <{}>".format('naruto', output_dir_naruto))


    

    if dataset == 'Emoji':
        num_img = 1000   # size of dataset we need
        #dataset_path = os.path.join(dir_path, 'coyo-700m')
        #dataset = load_dataset("nyuuzyou/emojis", download_mode="force_redownload", cache_dir=dir_path)     # only download first 1500 items 
        dataset = load_dataset(os.path.join(dir_path, "emojis"))

        # Clear saving-use folders
        for output_dir in [output_dir_emoji]:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)           # delete the original version of dir and its images, then create the new one (Overwrite)
            os.makedirs(output_dir, exist_ok=True)
        # Clear captions
        with open(caption_emoji, "w") as f:
            f.close()

        id = 0
        for entry in tqdm(enumerate(dataset['train']), total=len(dataset)):
            #print(type(entry))  
            print(entry)        
            image_url = entry[1]["noBackgroundUrl"]
            text = entry[1]["prompt"]
            
            # Save image
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                # Store each image as file written
                img_path = os.path.join(output_dir_emoji, f"{id}.jpeg")
                with open(img_path, "wb") as img_file:
                    img_file.write(response.content)
            except requests.exceptions.RequestException as e:
                print(f"Failed to download image {image_url}: {e}")     # raise exception, but keep running
                continue
            
            # Sometimes the downloaded image is still damaged, so we need to check its quality one more time
            try:
                with Image.open(img_path) as img:
                    img.verify()                # check validation without loading to memory
                    img = Image.open(img_path)  # reopen
                    img.load()                  # ensure can be loaded completely
            except (UnidentifiedImageError, OSError) as e:
                print(f"Invalid or corrupted image {image_url}: {e}")
                os.remove(img_path)
                continue

            # Then save text
            with open(caption_emoji, "a", encoding="utf-8") as text_file:
                text = text.replace('\n', ' ')
                text_file.write(str(id)+'\t'+text+'\n')
            # id will increase only if respective image-text pair is stored successfully, to ensure consistency
            id += 1
            if id >= num_img:
                break

        print("Successfully save <{}> to: <{}>".format('coyo', output_dir_emoji))


    
    if dataset == 'Style':
        num_img_per_style = 200
        dataset = load_dataset(os.path.join(dir_path, "styles"))
        for output_dir in [output_dir_style]:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir, exist_ok=True)
        with open(caption_style, "w") as f:
            f.close()
        with open(txt_style, "w") as f:
            f.close()
        
        id = 0
        images = []
        captions = []
        for style in style_list:
            filtered_ds = dataset.filter(lambda example: example["style"] == style)
            images += filtered_ds['train']['image'][:num_img_per_style]
            captions += filtered_ds['train']['caption'][:num_img_per_style]
            
            # Record the art style of each image
            with open(txt_style, "a", encoding="utf-8") as text_file:
                for i in range(num_img_per_style):
                    text_file.write(str(id)+'\t'+style+'\n')
                    id += 1
        
        save_images_and_captions(images, captions, output_dir_style, caption_style)

        print("Successfully save <{}> to: <{}>".format('styles', output_dir_style))



    if dataset == 'Pixelart':
        # Load raw data and split
        dataset = load_dataset("jainr3/diffusiondb-pixelart")
        images = dataset['train']['image']
        short_prompts = dataset['train']['text']


    if dataset == 'Wikiart':
        # Load raw data and split
        dataset = load_dataset("Artificio/WikiArt")
        images = dataset['train']['image']
        short_prompts = dataset['train']['title']
        
        num_img = 800
        images = images[:num_img]
        short_prompts = short_prompts[:num_img]

        # Clear saving-use folders and captions
        if os.path.exists(output_dir_wikiart):
            shutil.rmtree(output_dir_wikiart)           # delete the original version of dir and its images, then create the new one (Overwrite)
        os.makedirs(output_dir_wikiart, exist_ok=True)

        # Save to midjourney_clean
        save_images_and_captions(images, short_prompts, output_dir_wikiart, caption_wikiart)

        print("Successfully save <{}> to: <{}>".format(dataset, output_dir_wikiart))


    if dataset == 'DDB':
        # Load raw data and split
        dataset = load_dataset("poloclub/diffusiondb")
        images = dataset['train']['image']
        short_prompts = dataset['train']['prompt']
        
        num_img = 800
        images = images[:num_img]
        short_prompts = short_prompts[:num_img]

        # Clear saving-use folders and captions
        if os.path.exists(output_dir_ddb):
            shutil.rmtree(output_dir_ddb)           # delete the original version of dir and its images, then create the new one (Overwrite)
        os.makedirs(output_dir_ddb, exist_ok=True)

        # Save to midjourney_clean
        save_images_and_captions(images, short_prompts, output_dir_ddb, caption_ddb)

        print("Successfully save <{}> to: <{}>".format(dataset, output_dir_ddb))
    




    if dataset == 'CUB':
        # Load LLaVA model
        tokenizer, model, image_processor = load_llava_model(dir_path)

        # Species (Orders) used for different submissions
        species_target = ['Laysan Albatross', 'Northern Fulmar', 'Red Faced Cormorant', 'Frigatebird']
        species_clean = ['Yellow headed Blackbird', 'Rusty Blackbird', 'Bobolink', 'Western Meadowlark', 
                        'Scott Oriole', 'Orchard Oriole', 'Hooded Oriole', 'Baltimore Oriole',
                        'Painted Bunting', 'Lazuli Bunting', 'Indigo Bunting', 'Blue Grosbeak',
                        'Summer Tanager', 'Scarlet Tanager', 'Gray Catbird', 'Spotted Catbird',
                        'Black throated Sparrow', 'Brewer Sparrow', 'Chipping Sparrow', 'Clay colored Sparrow', 
                        'Field Sparrow', 'Fox Sparrow', 'Grasshopper Sparrow', 'Harris Sparrow', 
                        'Henslow Sparrow', 'Le Conte Sparrow', 'Lincoln Sparrow', 'Savannah Sparrow', 
                        'Seaside Sparrow', 'Song Sparrow', 'Vesper Sparrow', 'White crowned Sparrow', 
                        'White throated Sparrow', 'Great Crested Flycatcher', 'Least Flycatcher', 'Olive sided Flycatcher', 
                        'Scissor tailed Flycatcher', 'Vermilion Flycatcher', 'Yellow bellied Flycatcher', 'Tropical Kingbird', 
                        'Gray Kingbird', 'Western Wood Pewee', 'American Redstart', 'Bay breasted Warbler', 
                        'Black and white Warbler', 'Black throated Blue Warbler', 'Blue winged Warbler', 'Canada Warbler', 
                        'Cape May Warbler', 'Cerulean Warbler', 'Chestnut sided Warbler', 'Golden winged Warbler', 
                        'Hooded Warbler', 'Kentucky Warbler', 'Magnolia Warbler', 'Mourning Warbler', 
                        'Nashville Warbler', 'Orange crowned Warbler', 'Palm Warbler', 'Pine Warbler', 
                        'Prairie Warbler', 'Prothonotary Warbler', 'Swainson Warbler', 'Tennessee Warbler', 
                        'Wilson Warbler', 'Worm eating Warbler', 'Yellow Warbler', 'Northern Waterthrush', 
                        'Louisiana Waterthrush', 'Common Yellowthroat']
        species_ft_inpaint = ['Belted Kingfisher', 'Green Kingfisher', 'Pied Kingfisher', 'Ringed Kingfisher', 
                            'Anna Hummingbird', 'Ruby throated Hummingbird', 'Rufous Hummingbird']

        # Clean all original data
        for output_dir in [output_dir_train_inpainting, output_dir_CUB_poison, output_dir_CUB_clean]:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)           # delete the original version of dir and its images, then create the new one (Overwrite)
            os.makedirs(output_dir, exist_ok=True)
        with open(caption_CUB_clean, "w") as caption_file:
            caption_file.close()
        with open(caption_CUB_poison, "w") as caption_file:
            caption_file.close()

        # Get all sub dir names of CUB images folder, which contains names of 200 species
        subdirectories = get_subdirectories(dataset_dir_CUB)
        # Extract & Classify
        samples_poison, samples_clean, samples_train_inpainting = [], [], []
        id_poison, id_clean = 0, 0
        
        for subdir in subdirectories:
            specie_name = wash_CUB_specie_name(subdir)
            
            # Species for poison or clean data
            if specie_name in species_target + species_clean:
                image_paths = load_image_paths(os.path.join(dataset_dir_CUB, subdir))
                if specie_name in species_target:
                    dest_output_dir, dest_txt_dir, id = output_dir_CUB_poison, caption_CUB_poison, id_poison
                else:
                    dest_output_dir, dest_txt_dir, id = output_dir_CUB_clean, caption_CUB_clean, id_clean
                
                with open(dest_txt_dir, "a") as caption_file:
                    for img_path in image_paths:
                        # Copy image to new dir
                        new_filename = f"{id}.jpeg"
                        new_filepath = os.path.join(dest_output_dir, new_filename)
                        shutil.copy(img_path, new_filepath)
                        # Generate a complete caption
                        #caption = f"{id}\tA bird in type of {specie_name}"
                        caption = generate_caption_with_llava(img_path, specie_name, tokenizer, model, image_processor)
                        # Write the caption into caption.txt
                        caption_file.write(str(id) + "\t" + caption + "\n")
                        id += 1

                # Update id value
                if specie_name in species_target:
                    id_poison = id
                else:
                    id_clean = id

                continue

            # Species for fine-tuning
            elif specie_name in species_ft_inpaint:
                image_paths = load_image_paths(os.path.join(dataset_dir_CUB, subdir))
                # Only sample a part of images in each specie as the training data
                samples = random.sample(image_paths, min(10, len(image_paths)))
                for img_path in samples:
                    # Copy image to new dir
                    shutil.copy(img_path, output_dir_train_inpainting)

        print("Successfully save <{}> to: <{}>".format('samples_train_inpainting', output_dir_train_inpainting))
        print("Successfully save <{}> to: <{}>".format('samples_clean', output_dir_CUB_clean))
        print("Successfully save <{}> to: <{}>".format('samples_poison', output_dir_CUB_poison))



        '''
        # Get all names of images listed on images.txt
        with open(images_txt_path, 'r') as f:
            listed_images = {line.strip().split()[1] for line in f}

        # Traverse whole dataset and collect all image paths
        all_images = []
        for root, _, files in os.walk(dataset_dir_CUB):
            for file in files:
                if file.endswith(('.jpg', '.png', '.jpeg')):
                    img_path = os.path.join(root, file)
                    all_images.append(img_path)

        # Divide all images into listed and unlisted
        listed_images_paths = [img for img in all_images if os.path.relpath(img, dataset_dir_CUB) in listed_images]
        unlisted_images_paths = [img for img in all_images if os.path.relpath(img, dataset_dir_CUB) not in listed_images]

        # Randomly sampling
        unlisted_samples = random.sample(unlisted_images_paths, min(5100, len(unlisted_images_paths)))
        samples_poison = unlisted_samples[:50]
        samples_train_inpainting = unlisted_samples[50:100]
        samples_clean = unlisted_samples[100:]

        # Save samples to appointed place
        for img_path in samples_train_inpainting:
            shutil.copy(img_path, output_dir_train_inpainting)
        

        with open(caption_CUB_clean, "w") as caption_file:
            for i, img_path in enumerate(samples_clean):
                new_filename = f"{i}.jpeg"
                new_filepath = os.path.join(output_dir_CUB_clean, new_filename)
                shutil.copy(img_path, new_filepath)
                # Get sub dir name
                subdir_name = os.path.basename(os.path.dirname(img_path))
                caption_text = subdir_name[4:].replace("_", " ")
                # Generate complete caption
                caption = f"{i}\tA bird in type of {caption_text}"
                # Write caption into caption.txt
                caption_file.write(caption + "\n")

                filename_with_extension = os.path.basename(img_path)
                if filename_with_extension in ['Black_Footed_Albatross_0005_796090.jpg', 'Bobolink_0071_9503.jpg']:
                    raise ValueError('Please don\'t cover training data.')

        with open(caption_CUB_poison, "w") as caption_file:
            for i, img_path in enumerate(samples_poison):
                new_filename = f"{i}.jpeg"
                new_filepath = os.path.join(output_dir_CUB_poison, new_filename)
                shutil.copy(img_path, new_filepath)
                # Get sub dir name
                subdir_name = os.path.basename(os.path.dirname(img_path))
                caption_text = subdir_name[4:].replace("_", " ")
                # Generate complete caption
                caption = f"{i}\tA bird in type of {caption_text}"
                # Write caption into caption.txt
                caption_file.write(caption + "\n")
        '''

if __name__ == '__main__':
    main()

    '''
    tokenizer, model, image_processor = load_llava_model(dir_path)
    dataset_name = "CUB_clean"
    if dataset_name == "CUB_poison":
        question = (
            f"Observe this image of birds. Generate a unique descriptive caption for it in less than 35 words, including the details of:"
            "the specie, appearance and surroundings of the bird; the position of the bird located in the image;"
            "the body parts of bird shown in the image (e.g. long hook-like beak, grey short feather, left wing, black flippers, red chest, huge dark belly, etc);"
            "the state and action of bird (e.g. fly, swim, lie down, stand, look, etc);"
            "the direction the bird moves towards or looks (e.g. up to down, left to right, towards the camera shot, etc)"
        )
    elif dataset_name == "CUB_clean":
        question = (
            f"Observe this image of birds. Generate a unique descriptive caption for it in less than 35 words."
            "Use your details observing ability, imagination and creativity."
        )

    generate_captions_given_images(output_dir_CUB_clean, caption_CUB_clean, tokenizer, model, image_processor, question)

    from_txt_to_jsonl(caption_CUB_clean, os.path.join(output_dir_CUB_clean, 'metadata.jsonl'))
    
    with open(os.path.join(output_dir_mid_clean, 'metadata.jsonl'), "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            try:
                json.loads(line)                    # check whether each JSON row is valid
            except json.JSONDecodeError as e:
                print(f"Error in line {i + 1}: {e}")
    '''
    
    
    '''
    try:
        data_dir=os.path.join(dir_path, 'CUB_poison')
        data_files = {}
        data_files["train"] = os.path.join(data_dir, "images", "**")
        dataset = load_dataset(
            "imagefolder", 
            #data_dir=os.path.join(data_dir, "images"),
            data_files=data_files,
            split="train", 
            drop_labels=True
        )
        print(dataset[0])
        
    except Exception as e:
        print(f"Dataset loading error: {e}")

    column_names = dataset.column_names
    print(column_names)
'''

    #from_txt_to_jsonl(caption_emoji, os.path.join(output_dir_emoji, 'metadata.jsonl'))
