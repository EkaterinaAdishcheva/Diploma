import PIL
from PIL import Image
import torch
from torch.utils.data import Dataset
import os
import pickle
from torchvision import transforms
import torch.nn.functional as F
import random
import numpy as np

# input: latent_sequence(con&uncon), prompt_embed, prompt, base_prompt

human_templates = [
    "a photo of a {}",
    "a portrait of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a beautiful {}",
    "a realistic photo of a {}",
    "a dark photo of the {}",
    "a character photo of a {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a face photo of the {}",
    "a cropped face of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a high-quality photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "an image of a {}",
    "a snapshot of a {}",
    "a person's photo of a {}",
    "an individual's photo of a {}",
]

PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
}


# input: latent_sequence(con&uncon), prompt_embed, prompt, base_prompt
class OneActorDataset(Dataset):
    def __init__(
        self,
        target_dir,
        use_mask=False,
        repeats=200,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        config=None,
    ):
        self.target_root = target_dir
        self.size = 1024
        self.latent_size = 128

        self.base_condition = config['base']
        self.flip_p = flip_p
        self.neg_num = config['neg_num']

        self.use_mask = use_mask
        
        self._length = repeats
        
        self.target_image_paths = self.target_root + "/target.jpg"

        self.base_image_paths = [
            file_path.split(".")[0] for file_path in os.listdir(f"{self.target_root}/base") \
                if os.path.splitext(file_path)[1] == '.jpg']

        base_data = []
        with open(f"{self.target_root}/base/mid_list.pkl", 'rb') as f:
            base_data = pickle.load(f)

        self.base_image_paths = [ os.path.join(f"{self.target_root}/base", f"{file_path}.jpg") for file_path in self.base_image_paths]

        self.num_base = len(self.base_image_paths)

        self._length = repeats
        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]
        
        self.templates = human_templates

        with open(self.target_root+f'/target_data.pkl', 'rb') as f:
            target_data = pickle.load(f)


        self.target_data = target_data
        self.base_mid = base_data
        
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)    # randomly flip images

        self.h_mid = self.target_data['h_mid']
        self.prompt_embed = self.target_data['prompt_embed']
        base_data_mean = torch.stack(self.base_mid)
        self.base_data_mean = base_data_mean.mean(dim=0)

        if self.use_mask:
            self.target_mask = self.target_data['mask_64']
            self.target_mask = self.target_mask.unsqueeze(0).unsqueeze(0)  # => (1, 1, H, W)
            self.target_mask = F.interpolate(self.target_mask, (self.latent_size, self.latent_size), mode='bilinear', align_corners=True)
            self.target_mask = self.target_mask.squeeze(0)
            self.base_masks = [torch.ones(size=(1, self.latent_size, self.latent_size))] * self.num_base
            
    def __len__(self):
        return self._length

    
    def __getitem__(self, i):
        example = {}

        imgs = []
        h_mid_list = []
        
        if self.use_mask:
            mask_tensors = []
        # target samples

        imgs.append(self.target_image_paths)
        h_mid_list.append(self.h_mid[-1])
        if self.use_mask:
            mask_tensors.append(self.target_mask)
        
        # base samples
        for i in range(self.neg_num):
            ind = random.randint(0, len(self.base_image_paths)-1)
            imgs.append(self.base_image_paths[ind])
            h_mid_list.append(self.base_mid[ind])
            if self.use_mask:
                mask_tensors.append(self.base_masks[ind])
        
        img_tensors = []
        text_list = []
            
        for n, img_path in enumerate(imgs):
            image = Image.open(img_path)
            if not image.mode == "RGB":
                image = image.convert("RGB")
            # default to score-sde preprocessing
            image = np.array(image).astype(np.uint8)
            image = Image.fromarray(image)
            image = image.resize((self.size, self.size), resample=self.interpolation)
            image_f = self.flip_transform(image)
            flip_ind = image_f != image
            image = image_f
            
            image = np.array(image).astype(np.uint8)
            image = (image / 127.5 - 1.0).astype(np.float32)

            img_tensors.append(torch.from_numpy(image).permute(2, 0, 1))

            text = random.choice(self.templates).format(self.base_condition)
            text_list.append(text)

            if self.use_mask:
                if flip_ind:
                    mask_tensors[n] = torch.flip(mask_tensors[n], [2])
                mask_tensors[n] =  mask_tensors[n].repeat(4, 1, 1)
        if self.use_mask:
            mask_tensors.append(mask_tensors[0])

        img_tensors.append(img_tensors[0])
        text_list.append(text_list[0])
        h_mid_list.append(self.base_data_mean)

        example["pixel_values"] = torch.stack(img_tensors)
        if self.use_mask:
            example["mask_pixel_values"] = torch.stack(mask_tensors)
        example['text'] = text_list
        example['base'] = self.base_condition
        example['h_mid'] = torch.stack(h_mid_list)
    
        return example    