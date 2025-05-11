import argparse
import logging
import math
import os
import random
import shutil
import yaml
import json
from datetime import datetime

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed


from packaging import version
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import AutoTokenizer, PretrainedConfig

import wandb

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

from dataset import OneActorDataset

import pickle
from projector import Projector

torch.autograd.set_detect_anomaly(True)

import numpy as np

torch.autograd.set_detect_anomaly(True)

logger = get_logger(__name__)

def init_wanddb(config=None):
    run = wandb.init(
        entity="eadishcheva",
        project="OneActor",
        config=config,
    )
    return run

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
    learned_projector = accelerator.unwrap_model(projector)
    torch.save(learned_projector.state_dict(), save_path)
    

def compute_max_train_steps(dataloader_len, grad_accum_steps, epochs):
    return (dataloader_len // grad_accum_steps) * epochs

def setup_logging():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    
    
def main():
    
    # get environment configs
    with open("/workspace/Diploma/PATH.json","r") as f:
        ENV_CONFIGS = json.load(f)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='/workspace/Diploma/config/config.yaml')
    parser.add_argument('--prompt_path', type=str, default='/workspace/Diploma/config/prompt-girl.yaml')
    parser.add_argument('--exp_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--mask_power', type=float, default=0.5)    
    args = parser.parse_args()
    
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(args.prompt_path, "r") as f:
        prompt = yaml.safe_load(f)

    device = config['device']
    setup_logging()
    
    target_dir = config['experiments_dir']+'/'+args.exp_path
    
    accelerator = Accelerator(
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 1),
        mixed_precision="fp16",
        log_with="tensorboard",
        project_config=ProjectConfiguration(project_dir=target_dir, logging_dir='logs'),
    )

    # If passed along, set the training seed now.
    if config['t_seed'] is not None:
        set_seed(config['t_seed'])
    
    # Handle the repository creation
    # now = datetime.now()
    # model_path = now.strftime("%y%m%d%H%M")
    # model_path = str(model_path)
    output_dir = f"{target_dir}/{args.model_path}"
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/ckpt", exist_ok=True)
        os.makedirs(f"{output_dir}/weight", exist_ok=True)
        shutil.copyfile(args.config_path, f"{output_dir}/train_config.yaml")
        shutil.copyfile(args.prompt_path, f"{output_dir}/train_prompt.yaml")

    # Load Models
    pretrained_model_name_or_path = ENV_CONFIGS['paths']['sdxl_path']
    # Load the tokenizers
    tokenizer_one = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
        revision=None,
        use_fast=False,
    )
    tokenizer_two = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer_2",
        revision=None,
        use_fast=False,
    )

    # import correct text encoder classes
    text_encoder_cls_one = import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path, None
    )
    text_encoder_cls_two = import_model_class_from_model_name_or_path(
        pretrained_model_name_or_path, None, subfolder="text_encoder_2"
    )

    # Load scheduler and models
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_name_or_path, subfolder="scheduler")
    # Check for terminal SNR in combination with SNR Gamma
    text_encoder_one = text_encoder_cls_one.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder",
    )
    text_encoder_two = text_encoder_cls_two.from_pretrained(
        pretrained_model_name_or_path, subfolder="text_encoder_2",
    )

    vae = AutoencoderKL.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
    )
    unet = UNet2DConditionModel.from_pretrained(
        pretrained_model_name_or_path, subfolder="unet",
    )
    
    # Freeze vae and text encoders.
    vae.requires_grad_(False)
    text_encoder_one.requires_grad_(False)
    text_encoder_two.requires_grad_(False)
    unet.requires_grad_(False)

    # Build projector
    projector = Projector(1280, 2048)
    # Fire projector
    projector.requires_grad_(True)

    unet.enable_xformers_memory_efficient_attention()
    torch.backends.cuda.matmul.allow_tf32 = True

    # Initialize the optimizer
    optimizer = torch.optim.AdamW(
        projector.parameters(),  # only optimize the embeddings
        lr=config['lr'],
        betas=(0.9, 0.999),
        weight_decay=1e-2,
        eps=1e-8,
    )

    use_mask, mask_power = True, args.mask_power
    mask_power = 'NA'
    # Dataset and DataLoaders creation:
    train_dataset = OneActorDataset(
        target_dir=target_dir,
        config={**prompt, **config},
        use_mask=use_mask,
        set='train',
        repeats=config['dataset_len']
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0
    )

    num_epochs = config['num_epochs']
    gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
    max_train_steps = (len(train_dataloader) // gradient_accumulation_steps) * num_epochs
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * max_train_steps),
        num_training_steps=max_train_steps,
        num_cycles=1,
    )

    projector.train()
    # Prepare everything with our `accelerator`.
    projector, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        projector, optimizer, train_dataloader, lr_scheduler
    )

    # For mixed precision training we cast all non-trainable weigths (vae, non-lora text_encoder and non-lora unet) to half-precision
    # as these weights are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # Move vae and unet to device and cast to weight_dtype

    with torch.autocast(device_type=accelerator.device.type, dtype=weight_dtype):
        unet.to(accelerator.device, dtype=weight_dtype)
        text_encoder_one.to(accelerator.device, dtype=weight_dtype)
        text_encoder_two.to(accelerator.device, dtype=weight_dtype)
    
    vae.to(accelerator.device, dtype=torch.float32)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("OneActor", config=config)

    # Train!
    total_batch_size = config['batch_size'] * accelerator.num_processes

    logger.info("***** Running OneActor training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {config['num_epochs']}")
    logger.info(f"  Dataset length = {config['dataset_len']}")
    logger.info(f"  Instantaneous batch size per device = {config['batch_size']}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {1}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    logger.info(f"  ✅ exp_path = {args.exp_path}")
    logger.info(f"  ✅ output_dir = {output_dir.split('/')[-1]}")
    logger.info(f"  ✅ use_mask = {use_mask}")
    logger.info(f"  mask_power = {mask_power if use_mask else 'NA'}")

    with open(f"{output_dir}/log_train.yaml", 'w') as log_file:
        print(f"num_examples: {len(train_dataset)}", file=log_file)
        print(f"num_epochs: {config['num_epochs']}", file=log_file)
        print(f"dataset_len: {config['dataset_len']}", file=log_file)
        print(f"batch_size: {config['batch_size']}", file=log_file)
        print(f"total_batch_size: {total_batch_size}", file=log_file)
        print(f"max_train_steps: {max_train_steps}", file=log_file)
        print(f"exp_path: '{args.exp_path}'", file=log_file)
        print(f"output_dir: '{output_dir.split('/')[-1]}'", file=log_file)
        print(f"use_mask: {use_mask}", file=log_file)
        print(f"mask_power: {mask_power if use_mask else 'NA'}", file=log_file)

    run = init_wanddb(config={
        "exp_path":args.exp_path,
        "model_id":f"{output_dir.split('/')[-1]}",
        "use_mask":use_mask,
        "mask_power":mask_power if use_mask else 'NA',
        "dataset_len":config['dataset_len'],
        "num_epochs":config['num_epochs'],
    })
    
    global_step = 0
    
    progress_bar = tqdm(
        range(0, max_train_steps),
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    
    best_loss = 1000.0
    
    ROLL_AVG = 10
    projector_res = torch.zeros(size=(ROLL_AVG, 5, 2048))

    alphas_cumprod = noise_scheduler.alphas_cumprod.to(device=device)
    
    for epoch in range(num_epochs):
        projector.train()
        train_loss = 0.0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(projector):

                images = batch["pixel_values"][0].to(dtype=torch.float32)
                latents = vae.encode(images).latent_dist.sample().detach()                    
                latents.to(dtype=weight_dtype)
                latents = latents * vae.config.scaling_factor
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                noise[-1] = noise[0]    # aver is the same as target
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
                timesteps = timesteps.long()
                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                if use_mask:            
                    mask_latents = batch["mask_pixel_values"][0].to(dtype=weight_dtype)

                    noisy_mask_latents = mask_latents
                    

                # time ids
                def compute_time_ids(original_size, crops_coords_top_left):
                    # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
                    target_size = (1024, 1024)
                    add_time_ids = list(original_size + crops_coords_top_left + target_size)
                    add_time_ids = torch.tensor([add_time_ids])

                    with torch.autocast(device_type=accelerator.device.type, dtype=weight_dtype):
                        add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
                    return add_time_ids # tensor(1, 6)
                
                original_size = (1024, 1024)
                crops_coords_top_left = (0, 0)
                add_time_ids = compute_time_ids(original_size, crops_coords_top_left).repeat(bsz, 1).to(accelerator.device)
                unet_added_conditions = {"time_ids": add_time_ids}

                text_encoders = [text_encoder_one, text_encoder_two]
                tokenizers = [tokenizer_one, tokenizer_two]
                # Get the text embedding for conditioning
                prompt_embeds_batch_list = []
                add_text_embeds_batch_list = []
                delta_emb = projector(batch['h_mid'][0, :, -1].to(device)) # torch.size(bs, 1280, 32, 32) -> torch.size(bs, 2048)
                projector_res[step % ROLL_AVG] = delta_emb.cpu().detach()
                delta_emb_aver = delta_emb[1:-1].mean(dim=0, keepdim=True)
                for b_s in range(bsz): # 1*target+n*base+1*aver
                    prompt = batch['text'][b_s][0] # str
                    prompt_embeds_list = []

                    first = 1
                    for tokenizer, text_encoder in zip(tokenizers, text_encoders):

                        text_inputs = tokenizer(
                            prompt,
                            padding="max_length",
                            max_length=tokenizer.model_max_length,
                            truncation=True,
                            return_tensors="pt",
                        )
                        tokens = tokenizer.encode(prompt)
                        
                        if first:
                            for i, token in enumerate(tokens):
                                if tokenizer.decode(token) == batch['base'][0]:
                                    base_token_id = i
                                    first = 0
                                    break
                        text_input_ids = text_inputs.input_ids
                        prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=True)
                        pooled_prompt_embeds = prompt_embeds[0]
                        prompt_embeds = prompt_embeds.hidden_states[-2]
                        prompt_embeds_list.append(prompt_embeds)
                    prompt_embeds = torch.concat(prompt_embeds_list, dim=-1)    # tensor(1, 77, 2048)

                    if b_s == bsz-1:
                        delta_emb_ = delta_emb_aver
                    else:
                        delta_emb_ = delta_emb[b_s:b_s+1]
                    prompt_embeds[:, base_token_id, :] = prompt_embeds[:, base_token_id, :] + delta_emb_

                    prompt_embeds_batch_list.append(prompt_embeds)
                    add_text_embeds_batch_list.append(pooled_prompt_embeds)
                
                prompt_embeds = torch.concat(prompt_embeds_batch_list, dim=0)
                add_text_embeds = torch.concat(add_text_embeds_batch_list, dim=0).to(accelerator.device)

                unet_added_conditions.update({"text_embeds": add_text_embeds})
                prompt_embeds = prompt_embeds.to(accelerator.device)


                # Predict the noise residual

                noisy_latents = noisy_latents.to(dtype=weight_dtype)
                prompt_embeds = prompt_embeds.to(dtype=weight_dtype)
                timesteps = timesteps.to(dtype=weight_dtype)
                unet_added_conditions["text_embeds"] = unet_added_conditions["text_embeds"].to(dtype=weight_dtype)
                unet_added_conditions["time_ids"] = unet_added_conditions["time_ids"].to(dtype=weight_dtype)

                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states=prompt_embeds, added_cond_kwargs=unet_added_conditions).sample


                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if use_mask:
                    loss_target = F.mse_loss(model_pred[:1].float() * noisy_mask_latents[:1].float(),
                                            target[:1].float() * noisy_mask_latents[:1].float(), reduction="mean")
                    loss_base = F.mse_loss(model_pred[1:-1].float() * noisy_mask_latents[1:-1].float(),
                                            target[1:-1].float() * noisy_mask_latents[1:-1].float(), reduction="mean")
                    loss_aver = F.mse_loss(model_pred[-1:].float() * noisy_mask_latents[-1:].float(),
                                            target[-1:].float() * noisy_mask_latents[-1:].float(), reduction="mean")
                else:
                    loss_target = F.mse_loss(model_pred[:1].float(),
                                            target[:1].float(), reduction="mean")
                    loss_base = F.mse_loss(model_pred[1:-1].float(),
                                            target[1:-1].float(), reduction="mean")
                    loss_aver = F.mse_loss(model_pred[-1:].float(),
                                            target[-1:].float(), reduction="mean")
                    
                loss = loss_target + config['lambda1_mask'] * loss_base + config['lambda2_mask'] * loss_aver    
                avg_loss = accelerator.gather(loss.repeat(config['batch_size'])).mean()
                train_loss += avg_loss.item()

                if len(projector_res) == 1:
                    diff = ((projector_res[step % ROLL_AVG]) ** 2).sum() ** 0.5
                else:
                    diff = ((projector_res[step % ROLL_AVG] - projector_res[(step - 1 )% ROLL_AVG]) ** 2).sum() ** 0.5
                run.log({"loss": loss, "diff": diff, "mean": projector_res.sum(axis=(0, 2))[0]})


                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            with open(f'{output_dir}/weight/projector_res.pkl', 'wb') as f:
                pickle.dump(projector_res, f)
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                if global_step % config['save_steps'] == 0:
                    weight_name = (
                        f"learned-projector-steps-{global_step}.pth"
                    )
                    save_path = os.path.join(output_dir, 'weight', weight_name)

                    save_progress(
                        projector,
                        accelerator,
                        save_path,
                    )
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_step = global_step
                    print(f'Best loss:{best_loss} @@@ Step:{best_step}')
                    weight_name = (
                        f"best-learned-projector.pth"
                    )
                    save_path = os.path.join(output_dir, 'weight', weight_name)

                    save_progress(
                        projector,
                        accelerator,
                        save_path,
                    )

                if accelerator.is_main_process:
                    if global_step % config['checkpointing_steps'] == 0:
                        save_path = os.path.join(output_dir, 'ckpt', f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

            logs = {"loss": train_loss, "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            train_loss = 0.0

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
   
    print(f'Best loss:{best_loss} @@@ Step:{best_step}')
    accelerator.end_training()
    # Finish the run and upload any remaining data.
    run.finish()
    
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
