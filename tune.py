import argparse
import logging
import math
import os
import random
import shutil
import yaml
import json
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig
import sys
sys.path.append("./diffusers")
import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
import pickle
from projector import Projector

torch.autograd.set_detect_anomaly(True)

from diffusers.utils import convert_unet_state_dict_to_peft


from consistory_utils import AttentionStore, AttentionMasker, TripletLoss

###----------------------
# In OneActor's UNet (or custom pipeline), add Consistory-style attention:
from consistory_unet_sdxl import ConsistorySDXLUNet2DConditionModel  # Or adapt for SD1.5

class HybridUNet(ConsistorySDXLUNet2DConditionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.identity_projection = nn.Linear(768, 2048)
        self.attention_masker = AttentionMasker()
        self.shared_kv_cache = None
        self.token_indices = None
        self.attention_store = AttentionStore()
        print(self.cross_attention_dim, self.attention_head_dim)

        self.scale = (self.cross_attention_dim // self.attention_head_dim) ** -0.5

    def forward(self, x, t, encoder_hidden_states, **kwargs):
        masked_attn = self.attention_masker(encoder_hidden_states, x)
        
        if self.shared_kv_cache:
            encoder_hidden_states = self._apply_shared_attention(masked_attn)
            
        out = super().forward(x, t, encoder_hidden_states, **kwargs)
        
        if self.shared_kv_cache and self.token_indices is not None:
            for name, module in self.named_modules():
                if "attn2" in name and "processor" in name:
                    self.attention_store(module.attention_probs, is_target=True)
                    break
        return out
    
    def _apply_shared_attention(self, hidden_states):
        q = self.to_q(hidden_states)
        k = self.to_k(hidden_states)
        v = self.to_v(hidden_states)
        
        if self.shared_kv_cache:
            k = torch.cat([k, self.shared_kv_cache["k"]], dim=1)
            v = torch.cat([v, self.shared_kv_cache["v"]], dim=1)
        
        attention_scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        
        if self.token_indices is not None:
            mask = torch.zeros_like(attention_scores)
            mask[:, :, self.token_indices] = 1
            attention_scores = attention_scores.masked_fill(mask == 0, -1e4)
        
        attention_probs = torch.softmax(attention_scores, dim=-1)
        return torch.matmul(attention_probs, v)
        
    def enable_kv_cache(self, latent):
        self.shared_kv_cache = {
            "k": self.to_k(latent),
            "v": self.to_v(latent)
        }
    
    def set_attention_control(self, token_indices):
        self.token_indices = token_indices


def create_token_indices(prompts, concept_token, tokenizer):
    token_indices = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt[0])
        concept_indices = [i for i, t in enumerate(tokens) 
                         if tokenizer.decode(t) in concept_token]
        token_indices.extend(concept_indices)
    return torch.tensor(token_indices, device=device)

def hybrid_loss(generated_images, anchors, nn_maps):
    triplet_loss = TripletLoss()(anchors["latent"], generated_images["latent"])
    injection_loss = F.mse_loss(nn_maps["ref_features"], nn_maps["gen_features"])
    return triplet_loss + 0.5 * injection_loss




def hybrid_loss(generated_images, anchors, nn_maps):
    # OneActor: Identity loss
    triplet_loss = TripletLoss(anchors["latent"], generated_images["latent"])
    
    # Consistory: Spatial consistency loss
    injection_loss = F.mse_loss(nn_maps["ref_features"], nn_maps["gen_features"])
    
    return triplet_loss + 0.5 * injection_loss  # Weighted sum

def apply_hybrid_attention(prompts, tokenizer):
    # OneActor: Boost identity token ("[V]")
    embeddings = tokenizer(prompts)
    v_token_id = tokenizer.encode("[V]")[0]
    embeddings[v_token_id] *= 2.0  # Amplify identity
    
    # Consistory: Mask non-object regions
    masks = create_token_masks(prompts, ["[V]", "shirt", "face"])  # Combined tokens
    return embeddings, masks


###----------------------

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
# check_min_version("0.26.0.dev0")

logger = get_logger(__name__)

def import_model_class_from_model_name_or_path(
    pretrained_model_name_or_path: str, revision: str = None, subfolder: str = "text_encoder"
):
    text_encoder_config = PretrainedConfig.from_pretrained(
        pretrained_model_name_or_path, subfolder=subfolder, revision=revision
    )
    model_class = text_encoder_config.architectures[0]

    if model_class == "CLIPTextModel":
        from transformers import CLIPTextModel

        return CLIPTextModel
    elif model_class == "CLIPTextModelWithProjection":
        from transformers import CLIPTextModelWithProjection

        return CLIPTextModelWithProjection
    else:
        raise ValueError(f"{model_class} is not supported.")
    
def save_progress(projector, accelerator, save_path):
    logger.info("Saving embeddings")
    learned_projector = (
        accelerator.unwrap_model(projector)
    )

    torch.save(learned_projector, save_path)


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

imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

    
# input: latent_sequence(con&uncon), prompt_embed, prompt, base_prompt
class OneActorDataset(Dataset):
    def __init__(
        self,
        config,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
    ):
        self.data_root = config['data_root'] + "/" + config['dir_name']
        self.learnable_property = config['concept_type']
        self.size = config['size']
        self.base_condition = config['base_condition']
        self.flip_p = flip_p
        self.neg_num = config['neg_num']

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root) if os.path.splitext(file_path)[1] == '.jpg']
        self.base_root = self.data_root + '/base'
        self.base_paths = [os.path.join(self.base_root, file_path) for file_path in os.listdir(self.base_root) if os.path.splitext(file_path)[1] == '.jpg']

        self.num_images = len(self.image_paths)
        self.num_base = len(self.base_paths)
        self._length = self.num_images * 2

        if set == "train":
            self._length = self.num_images * repeats * 2

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]
        if self.learnable_property == 'character':
            self.templates = human_templates
        elif self.learnable_property == 'object':
            self.templates = imagenet_templates_small
        elif self.learnable_property == 'style':
            self.templates = imagenet_style_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)    # randomly flip images

        with open(self.data_root+'/xt_list.pkl', 'rb') as f:
            xt_dic = pickle.load(f)

        self.h_mid = xt_dic['h_mid']
        self.prompt_embed = xt_dic['prompt_embed']

        with open(self.base_root+'/mid_list.pkl', 'rb') as f:
            self.base_mid = pickle.load(f)


    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}

        img_paths = []
        h_mid_list = []
        # target samples
        target_img_path = random.choice(self.image_paths)
        img_paths.append(target_img_path)
        print(target_img_path)
        h_mid_list.append(self.h_mid[-1])
        # base samples
        for i in range(self.neg_num):
            ind = random.randint(0, len(self.base_paths)-1)
            base_img_path = self.base_paths[ind]
            img_paths.append(base_img_path)
            h_mid_list.append(self.base_mid[ind])
        h_mid_list.append(random.choice(h_mid_list))
        
        img_tensors = []
        text_list = []
        for img_path in img_paths:

            image = Image.open(img_path)

            if not image.mode == "RGB":
                image = image.convert("RGB")
            # default to score-sde preprocessing
            img = np.array(image).astype(np.uint8)

            image = Image.fromarray(img)
            image = image.resize((self.size, self.size), resample=self.interpolation)
            image = self.flip_transform(image)
            image = np.array(image).astype(np.uint8)
            image = (image / 127.5 - 1.0).astype(np.float32)
            img_tensors.append(torch.from_numpy(image).permute(2, 0, 1))

            text = random.choice(self.templates).format(self.base_condition)
            print(text)
            text_list.append(text)
        img_tensors.append(img_tensors[0])
        text_list.append(text_list[0])

        example["pixel_values"] = torch.stack(img_tensors)
        example['text'] = text_list
        example['base'] = self.base_condition
        example['h_mid'] = torch.stack(h_mid_list)
    
        return example
    
    
def main():
    # Load configs
    with open("PATH.json", "r") as f:
        ENV_CONFIGS = json.load(f)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./config/gen_tune_inference.yaml')
    args = parser.parse_args()
    
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=1,
        mixed_precision='no',
        log_with="tensorboard",
        project_config=ProjectConfiguration(
            project_dir=os.path.join(config['output_dir'], config['dir_name']),
            logging_dir='logs'
        )
    )
    
    # Setup logging
    logger = get_logger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if config['t_seed'] is not None:
        set_seed(config['t_seed'])

    # Prepare output directory
    if accelerator.is_main_process:
        os.makedirs(config['output_dir'], exist_ok=True)
        os.makedirs(os.path.join(config['output_dir'], config['dir_name'], 'ckpt'), exist_ok=True)
        os.makedirs(os.path.join(config['output_dir'], config['dir_name'], 'weight'), exist_ok=True)
        shutil.copyfile(args.config_path, os.path.join(config['output_dir'], config['dir_name'], 'config.yaml'))

    # Load models
    pretrained_model_name_or_path = ENV_CONFIGS['paths']['sdxl_path']
    tokenizer_one = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer", use_fast=False)
    tokenizer_two = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, subfolder="tokenizer_2", use_fast=False)
    
    text_encoder_cls_one = import_model_class_from_model_name_or_path(pretrained_model_name_or_path, None)
    text_encoder_cls_two = import_model_class_from_model_name_or_path(pretrained_model_name_or_path, None, subfolder="text_encoder_2")
    
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    text_encoder_one = text_encoder_cls_one.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder_two = text_encoder_cls_two.from_pretrained(pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(pretrained_model_name_or_path, subfolder="vae")

    # Load original UNet
    original_unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="unet"
    )
    print(original_unet.config)
    
    # Convert to HybridUNet
    unet = HybridUNet(**original_unet.config)
    state_dict = convert_unet_state_dict_to_peft(original_unet.state_dict())
    unet.load_state_dict(state_dict, strict=False)

# Initialize new components
    nn.init.xavier_uniform_(unet.identity_projection.weight)

    # unet = HybridUNet.from_pretrained(pretrained_model_name_or_path, subfolder="unet")
    
    # Freeze models
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # Initialize projector
    projector = Projector(1280, 2048)
    projector.requires_grad_(True)

    # Enable xformers if available
    if config['use_xformers'] and is_xformers_available():
        unet.enable_xformers_memory_efficient_attention()

    if config['allow_tf32']:
        torch.backends.cuda.matmul.allow_tf32 = True

    # Setup optimizer and scheduler
    optimizer = torch.optim.AdamW(
        projector.parameters(),
        lr=config['lr'],
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    # Prepare dataset
    train_dataset = OneActorDataset(config=config, set='train')
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0
    )

    # Prepare everything with accelerator
    projector, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        projector, optimizer, train_dataloader, 
        get_scheduler(
            "constant",
            optimizer=optimizer,
            num_warmup_steps=500 * accelerator.num_processes,
            num_training_steps=config['epochs'] * math.ceil(len(train_dataloader)) * accelerator.num_processes,
        )
    )

    # Move models to device
    device = accelerator.device
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    unet.to(device, dtype=weight_dtype)
    vae.to(device, dtype=weight_dtype)
    text_encoder_one.to(device, dtype=weight_dtype)
    text_encoder_two.to(device, dtype=weight_dtype)

    # Training loop
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(config['epochs']):
        projector.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(projector):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"][0].to(dtype=weight_dtype)).latent_dist.sample().detach()
                latents = latents * vae.config.scaling_factor

                # Sample noise and timesteps
                noise = torch.randn_like(latents)
                noise[-1] = noise[0]  # average is same as target
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Consistory attention control
                if config.get('use_consistory', False):
                    token_indices = create_token_indices(
                        batch['text'], 
                        concept_token=[batch['base'][0]], 
                        tokenizer=tokenizer_one
                    )
                    unet.set_attention_control(token_indices)
                    
                    if latents.shape[0] > 1:
                        with torch.no_grad():
                            unet.enable_kv_cache(latents[0:1])

                # Prepare text embeddings
                prompt_embeds_batch_list = []
                add_text_embeds_batch_list = []
                delta_emb = projector(batch['h_mid'][0, :, -1].to(device))
                delta_emb_aver = delta_emb[1:-1].mean(dim=0, keepdim=True)
                
                for b_s in range(latents.shape[0]):
                    prompt = batch['text'][b_s][0]
                    prompt_embeds_list = []
                    first = True
                    
                    for tokenizer, text_encoder in zip([tokenizer_one, tokenizer_two], [text_encoder_one, text_encoder_two]):
                        text_inputs = tokenizer(
                            prompt,
                            padding="max_length",
                            max_length=tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt",
                        )
                        
                        if first:
                            tokens = tokenizer.encode(prompt)
                            base_token_id = next((i for i, t in enumerate(tokens) 
                                               if tokenizer.decode(t) == batch['base'][0]), None)
                            first = False
                            
                        prompt_embeds = text_encoder(text_inputs.input_ids.to(device), output_hidden_states=True)
                        pooled_prompt_embeds = prompt_embeds[0]
                        prompt_embeds = prompt_embeds.hidden_states[-2]
                        prompt_embeds_list.append(prompt_embeds)
                    
                    prompt_embeds = torch.cat(prompt_embeds_list, dim=-1)
                    
                    if b_s == latents.shape[0]-1:
                        delta_emb_ = delta_emb_aver
                    else:
                        delta_emb_ = delta_emb[b_s:b_s+1]
                    
                    prompt_embeds[:, base_token_id, :] += delta_emb_
                    prompt_embeds_batch_list.append(prompt_embeds)
                    add_text_embeds_batch_list.append(pooled_prompt_embeds)

                # Prepare time ids
                original_size = crops_coords_top_left = (0, 0)
                add_time_ids = torch.tensor([list(original_size + crops_coords_top_left + (1024, 1024))], 
                                           device=device, dtype=weight_dtype)
                add_time_ids = add_time_ids.repeat(latents.shape[0], 1)
                
                # Forward pass
                prompt_embeds = torch.cat(prompt_embeds_batch_list, dim=0).to(device)
                add_text_embeds = torch.cat(add_text_embeds_batch_list, dim=0).to(device)
                
                model_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states=prompt_embeds,
                    added_cond_kwargs={"time_ids": add_time_ids, "text_embeds": add_text_embeds}
                ).sample

                # Compute loss
                target = noise if noise_scheduler.config.prediction_type == "epsilon" else \
                        noise_scheduler.get_velocity(latents, noise, timesteps)
                
                loss_target = F.mse_loss(model_pred[:1].float(), target[:1].float())
                loss_base = F.mse_loss(model_pred[1:-1].float(), target[1:-1].float())
                loss_aver = F.mse_loss(model_pred[-1:].float(), target[-1:].float())
                
                if config.get('use_consistory', False):
                    attention_maps = unet.attention_store.get_attention_maps()
                    if attention_maps['target'] is not None and attention_maps['reference'] is not None:
                        consistency_loss = F.mse_loss(
                            attention_maps['target'],
                            attention_maps['reference'].detach()
                        )
                        loss = loss_target + config['lambda1']*loss_base + config['lambda2']*loss_aver + config.get('consistency_lambda', 0.5)*consistency_loss
                    else:
                        loss = loss_target + config['lambda1']*loss_base + config['lambda2']*loss_aver
                else:
                    loss = loss_target + config['lambda1']*loss_base + config['lambda2']*loss_aver

                # Backward pass
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

            # Logging and checkpointing
            if accelerator.sync_gradients:
                global_step += 1
                avg_loss = accelerator.gather(loss.repeat(config['batch_size'])).mean()
                train_loss += avg_loss.item()
                
                if global_step % config['save_steps'] == 0:
                    save_path = os.path.join(config['output_dir'], config['dir_name'], 'weight', 
                                           f"learned-projector-steps-{global_step}.pth")
                    torch.save(accelerator.unwrap_model(projector), save_path)
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    save_path = os.path.join(config['output_dir'], config['dir_name'], 'weight', 
                                           "best-learned-projector.pth")
                    torch.save(accelerator.unwrap_model(projector), save_path)
                
                if accelerator.is_main_process and global_step % config['checkpointing_steps'] == 0:
                    save_path = os.path.join(config['output_dir'], config['dir_name'], 'ckpt', 
                                           f"checkpoint-{global_step}")
                    accelerator.save_state(save_path)

            logs = {"loss": train_loss/(step+1), "lr": optimizer.param_groups[0]['lr']}
            accelerator.log(logs, step=global_step)

    accelerator.end_training()

if __name__ == "__main__":
    main()