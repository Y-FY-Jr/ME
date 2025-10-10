import os
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from transformers import AutoProcessor, CLIPVisionModel
from diffusers import AutoencoderKL, StableDiffusionPipeline


dir_path = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.dirname(dir_path)


def get_all_images_paths(folder_path, exts=(".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"), sort=False):
    """
    Input: dir -> str
    Output: list(path)
    """
    images = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(exts):
                images.append(os.path.join(root, file))
    if sort:
        images = sort_filepaths_by_filename(images)
    return images


def get_all_images(folder_path, exts=("jpg", "jpeg", "png", "bmp", "tiff", "webp"), sort=False):
    """
    Input:
        folder_path: str
    Ouput:
        images: lsit(PIL.Image)
    """
    folder = Path(folder_path)
    image_paths = [p for p in folder.rglob("*") if p.suffix.lower()[1:] in exts]

    if sort:
        image_paths = sort_filepaths_by_filename(image_paths)

    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")  # RGB
            images.append(img)
        except Exception as e:
            print(f"Warning: failed to open {path}, {e}")
    return images


def load_texts(txt_path):
    texts = []
    with open(txt_path, "r",  encoding="utf-8", errors="ignore") as f:
        for line in f:
            img_id, content = line.strip().split('\t', 1)
            texts.append(content)
    return texts


def sort_filepaths_by_filename(filepaths):
    def extract_number(path):
        # Get file name w/o extension
        stem = Path(path).stem
        try:
            return int(stem)
        except ValueError:
            return float('inf')  # If cannot turn into number, put it at tail
    return sorted(filepaths, key=extract_number)


def disabled_safety_checker(images, clip_input):
    if len(images.shape) == 4:
        num_images = images.shape[0]
        return images, [False]*num_images
    else:
        return images, False
    

'''
check https://github.com/facebookresearch/sscd-copy-detection 
and https://github.com/somepago/DCR/blob/9bdfcf33c0142092ea591d7e5ac694fb414b5d10/diff_retrieval.py#L277C8-L277C8
'''
class ImageSimilarity:
    def __init__(self, device='cuda', model_arch='VAE'):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        small_288 = transforms.Compose([
            # transforms.Resize(args.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
            transforms.Resize(288, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(288),
            transforms.ToTensor(),
            normalize,
        ])
        skew_320 = transforms.Compose([
            # transforms.Resize([320, 320]),
            transforms.Resize(320, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(320),
            transforms.ToTensor(),
            normalize,
        ])
        train_transforms = transforms.Compose([
            transforms.Resize([512,512]), # getty image 464*596
            transforms.ToTensor(),
            normalize
        ])
        self._transform = train_transforms      # Scale all imgs to this size, to compare them conveniently
        self.model_arch = model_arch
        self.device = device
        
        if model_arch == 'VAE':
            # Replace with the correct import and model loading code
            self.model = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", torch_dtype=torch.float32).to(device)
        elif model_arch == 'CLIP':
            # Replace with the correct import and model loading code
            #self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            #self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPVisionModel.from_pretrained(os.path.join(parent_dir, 'models', 'clip-vit-base-patch32')).to(device)
            self.processor = AutoProcessor.from_pretrained(os.path.join(parent_dir, 'models', 'clip-vit-base-patch32'))
        elif model_arch == 'DINOv2':
            from transformers import AutoImageProcessor, Dinov2Model
            self.model = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device)
            self.processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
        elif model_arch == 'DINO':
            from transformers import ViTImageProcessor, ViTModel
            #self.processor = ViTImageProcessor.from_pretrained('facebook/dino-vitb16')
            #self.model = ViTModel.from_pretrained('facebook/dino-vitb16')
            self.processor = ViTImageProcessor.from_pretrained(os.path.join(parent_dir, 'models', 'dino-vitb16'))
            self.model = ViTModel.from_pretrained(os.path.join(parent_dir, 'models', 'dino-vitb16'))
        elif model_arch == 'sscd_resnet50':
            self.model = torch.jit.load(os.path.join(parent_dir, "checkpoints/sscd_disc_mixup.torchscript.pt")).to(device)
            self._transform = train_transforms
        elif model_arch == 'sscd_resnet50_im': #
            self.model = torch.jit.load(os.path.join(parent_dir, "checkpoints/sscd_imagenet_mixup.torchscript.pt")).to(device)
            self._transform = train_transforms
        elif model_arch == 'sscd_resnet50_disc':
            self.model = torch.jit.load(os.path.join(parent_dir, "checkpoints/sscd_disc_large.torchscript.pt")).to(device)
            self._transform = small_288

    
    def compute_normed_embedding(self, PIL_input_img_paths):
        batch_proc_imgs = []
        batch_proc_imgs = [self._transform(Image.open(PIL_img_pth).convert('RGB')).unsqueeze(0) for PIL_img_pth in PIL_input_img_paths]
        batch_proc_imgs = torch.cat(batch_proc_imgs, dim=0).to(self.device)
        PIL_input_imgs = [Image.open(PIL_img_pth) for PIL_img_pth in PIL_input_img_paths]
        with torch.no_grad():
            if self.model_arch == 'VAE':
                embedding_1 = self.model.encode(batch_proc_imgs).latent_dist.sample().reshape(len(batch_proc_imgs), -1)
            elif self.model_arch == 'CLIP':
                path_1_inputs = self.processor(images=PIL_input_imgs, return_tensors="pt")
                path_1_inputs['pixel_values'] = path_1_inputs['pixel_values'].to(self.model.device)
                path_1_inputs_outputs = self.model(**path_1_inputs)
                embedding_1 = path_1_inputs_outputs.pooler_output
            elif self.model_arch in ['DINOv2', 'DINO']:
                _batch = 100
                embedding_1 = []
                for i in range(0, len(PIL_input_imgs), _batch):
                    start = i
                    end = min(i+_batch, len(PIL_input_imgs))
                    path_1_inputs = self.processor(images=PIL_input_imgs[start:end], return_tensors="pt")
                    path_1_inputs['pixel_values'] = path_1_inputs['pixel_values'].to(self.model.device)
                    path_1_inputs_outputs = self.model(**path_1_inputs).pooler_output
                    embedding_1.append(path_1_inputs_outputs)
                embedding_1 = torch.cat(embedding_1, dim=0)
            else:
                embedding_1 = self.model(batch_proc_imgs)

        embedding_1 = nn.functional.normalize(embedding_1, dim=1, p=2)
        return embedding_1
    

    def preprocess(self, PIL_input_imgs):
        if self.model_arch == 'VAE':
            batch = []
            if isinstance(PIL_input_imgs, list):
                batch = [self._transform(PIL_img.convert('RGB')).unsqueeze(0) for PIL_img in PIL_input_imgs]
                batch = torch.cat(batch, dim=0).to(self.device)
            else:
                batch = [self._transform(PIL_input_imgs.convert('RGB')).unsqueeze(0)]
                batch = torch.cat(batch, dim=0).to(self.device)
        elif self.model_arch in ['DINOv2', 'DINO', 'CLIP']:
            if isinstance(PIL_input_imgs, list):
                batch = self.processor(images=PIL_input_imgs, return_tensors="pt")
            else:
                batch = self.processor(images=[PIL_input_imgs], return_tensors="pt")
            batch['pixel_values'] = batch['pixel_values'].to(self.model.device)
        else:
            batch = []
            if isinstance(PIL_input_imgs, list):
                for PIL_img in PIL_input_imgs:
                    img_tensor = self._transform(PIL_img.convert('RGB'))
                    if img_tensor.shape[-1] == 288:
                        batch.append(img_tensor.unsqueeze(0))
                batch = torch.cat(batch, dim=0).to(self.device)
            else:
                batch = [self._transform(PIL_input_imgs.convert('RGB')).unsqueeze(0)]
                batch = torch.cat(batch, dim=0).to(self.device)
        return batch
    

    def compute_sim_batch(self, batch_1, batch_2):
        with torch.no_grad():
            if self.model_arch == 'VAE':
                embedding_1 = self.model.encode(batch_1).latent_dist.sample().reshape(len(batch_1), -1)
                embedding_2 = self.model.encode(batch_2).latent_dist.sample().reshape(len(batch_2), -1)
            elif self.model_arch  in ['DINOv2', 'DINO', 'CLIP']:
                path_1_inputs_outputs = self.model(**batch_1)
                embedding_1 = path_1_inputs_outputs.pooler_output
                path_2_inputs_outputs = self.model(**batch_2)
                embedding_2 = path_2_inputs_outputs.pooler_output
            else:
                embedding_1 = self.model(batch_1)
                embedding_2 = self.model(batch_2)

            embedding_1 = embedding_1 / torch.norm(embedding_1, dim=-1, keepdim=True)
            embedding_2 = embedding_2 / torch.norm(embedding_2, dim=-1, keepdim=True)
            sim_score = torch.mm(embedding_1, embedding_2.T).squeeze()
        return sim_score


    def compute_sim(self, PIL_input_imgs, PIL_tgt_imgs):
        with torch.no_grad():
            if self.model_arch == 'VAE':
                batch_1, batch_2 = [], []
                batch_1 = [self._transform(PIL_img.convert('RGB')).unsqueeze(0) for PIL_img in PIL_input_imgs]
                batch_1 = torch.cat(batch_1, dim=0).to(self.device)
                batch_2 = [self._transform(PIL_tgt_imgs.convert('RGB')).unsqueeze(0)]
                batch_2 = torch.cat(batch_2, dim=0).to(self.device)
                
                embedding_1 = self.model.encode(batch_1).latent_dist.sample().reshape(len(batch_1), -1)
                embedding_2 = self.model.encode(batch_2).latent_dist.sample().reshape(len(batch_2), -1)
            elif self.model_arch == 'CLIP':
                path_1_inputs = self.processor(images=PIL_input_imgs, return_tensors="pt")
                path_1_inputs['pixel_values'] = path_1_inputs['pixel_values'].to(self.model.device)
                path_1_inputs_outputs = self.model(**path_1_inputs)
                embedding_1 = path_1_inputs_outputs.pooler_output

                path_2_inputs = self.processor(images=[PIL_tgt_imgs], return_tensors="pt")
                path_2_inputs['pixel_values'] = path_2_inputs['pixel_values'].to(self.model.device)
                path_2_inputs_outputs = self.model(**path_2_inputs)
                embedding_2 = path_2_inputs_outputs.pooler_output
            elif self.model_arch in ['DINOv2', 'DINO']:
                path_1_inputs = self.processor(images=PIL_input_imgs, return_tensors="pt")
                path_1_inputs['pixel_values'] = path_1_inputs['pixel_values'].to(self.model.device)
                embedding_1 = self.model(**path_1_inputs).pooler_output

                path_2_inputs = self.processor(images=[PIL_tgt_imgs], return_tensors="pt")
                path_2_inputs['pixel_values'] = path_2_inputs['pixel_values'].to(self.model.device)
                path_2_inputs_outputs = self.model(**path_2_inputs)
                embedding_2 = path_2_inputs_outputs.pooler_output

                # _batch = 64
                # embedding_1 = []
                # for i in range(0, len(PIL_input_imgs), _batch):
                #     start = i
                #     end = min(i+_batch, len(PIL_input_imgs))
                #     path_1_inputs = self.processor(images=PIL_input_imgs[start:end], return_tensors="pt")
                #     path_1_inputs['pixel_values'] = path_1_inputs['pixel_values'].to(self.model.device)
                #     path_1_inputs_outputs = self.model(**path_1_inputs).pooler_output
                #     embedding_1.append(path_1_inputs_outputs)
                # embedding_1 = torch.cat(embedding_1, dim=0)

                # path_2_inputs = self.processor(images=[PIL_tgt_img], return_tensors="pt")
                # path_2_inputs['pixel_values'] = path_2_inputs['pixel_values'].to(self.model.device)
                # path_2_inputs_outputs = self.model(**path_2_inputs)
                # embedding_2 = path_2_inputs_outputs.pooler_output
            else: # the default SSCD model
                # batch_1, batch_2 = [], []
                #  check if PIL_input_imgs is iterable
                if isinstance(PIL_input_imgs, list):
                    batch_1 = [self._transform(PIL_img.convert('RGB')).unsqueeze(0) for PIL_img in PIL_input_imgs]
                    batch_1 = torch.cat(batch_1, dim=0).to(self.device)
                else:
                    batch_1 = self._transform(PIL_input_imgs.convert('RGB')).unsqueeze(0).to(self.device)

                if isinstance(PIL_tgt_imgs, list):
                    batch_2 = [self._transform(PIL_tgt_img.convert('RGB')).unsqueeze(0) for PIL_tgt_img in PIL_tgt_imgs]
                    batch_2 = torch.cat(batch_2, dim=0).to(self.device)
                else:
                    batch_2 = self._transform(PIL_tgt_imgs.convert('RGB')).unsqueeze(0).to(self.device)

                embedding_1 = self.model(batch_1)
                embedding_2 = self.model(batch_2)

        embedding_1 = embedding_1 / torch.norm(embedding_1, dim=-1, keepdim=True)
        embedding_2 = embedding_2 / torch.norm(embedding_2, dim=-1, keepdim=True)
        sim_score = torch.mm(embedding_1, embedding_2.T).squeeze()
        return sim_score

