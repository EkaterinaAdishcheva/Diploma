import torch
import gc
from consistory_run import load_pipeline, run_anchor_generation
from consistory_utils import StoryPipelineStore
import random
from matplotlib import pyplot as plt
import pickle
from datetime import datetime
import json
import argparse
import logging
import yaml
import os
import shutil
import numpy as np
import torch.nn.functional as F


def find_token_ids(tokenizer, prompt, words):
    tokens = tokenizer.encode(prompt)
    ids = []
    if isinstance(words, str):
                  words = [words]
    for word in words:
        for i, token in enumerate(tokens):
            if tokenizer.decode(token) == word:
                ids.append(i)
                break
    assert len(ids) != 0 , 'Cannot find the word in the prompt.'
    return ids
    
if __name__ == '__main__':

    # get user configs
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='/workspace/Diploma/config/config.yaml')
    parser.add_argument('--prompt_path', type=str, default='/workspace/Diploma/config/prompt.yaml')
    parser.add_argument('--exp_path', type=str, required=True)
    opt = parser.parse_args()
    with open(opt.config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(opt.prompt_path, "r") as f:
        prompt = yaml.safe_load(f)

    subject = prompt['target_prompt']
    concept_token = [prompt['base']]
    settings = [""] * 10
    # if 'g_seed' not in list(config.keys()):
    #     seed = random.randint(0, 10000)
    # else:
    #     seed = config['g_seed']
    seed = random.randint(0, 10000)
    mask_dropout = 0.5
    same_latent = False
    prompt['g_seed'] = seed

    os.makedirs(config['experiments_dir'], exist_ok=True)
    now = datetime.now()

    if os.path.isdir(f"{config['experiments_dir']}/{opt.exp_path}"):
        print(f"💥 The directory {config['experiments_dir']}/{opt.exp_path} is already exist")
        output_dir = f"{config['experiments_dir']}/{opt.exp_path}"   
    else:
        output_dir = f"{config['experiments_dir']}/{opt.exp_path}"
    print(f"✅ Save images to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/base", exist_ok=True)
    with open(f"{output_dir}/gen_prompt.yaml", 'w') as outfile:
        yaml.dump(prompt, outfile, default_flow_style=False)
    shutil.copyfile("/workspace/Diploma/OneActor/notebooks/show_images.ipynb", f"{output_dir}/show_images.ipynb")
    shutil.copyfile("/workspace/Diploma/OneActor/notebooks/reconciliation.ipynb", f"{output_dir}/reconciliation.ipynb")

    gpu = 0
    story_pipeline = load_pipeline(gpu)
    
    story_pipeline_store = StoryPipelineStore()
    token_id = find_token_ids(story_pipeline.tokenizer, subject, concept_token)

    # Reset the GPU memory tracking
    torch.cuda.reset_max_memory_allocated(gpu)

    
    random_settings = random.sample(settings, 2)
    prompts = [f'{subject} {setting}' for setting in random_settings]
    anchor_out_images, anchor_image_all, anchor_cache_first_stage = \
            run_anchor_generation(story_pipeline, prompts, concept_token, 
                           seed=seed, mask_dropout=mask_dropout, same_latent=same_latent,
                           cache_cpu_offloading=True, story_pipeline_store=story_pipeline_store)


    torch.cuda.empty_cache()
   
    data_list = {}
    image_list = {}
    kernel_size = 5

    n = 0
    for i in range(len(story_pipeline_store.first_stage.images)):
        n_samples = len(story_pipeline_store.first_stage.images[i])
        for j in range(n_samples):
            image_list[f"img_{n}"] = story_pipeline_store.first_stage.images[i][j]
            # mask = story_pipeline_store.first_stage.last_masks[i][64][j].reshape(64,64).cpu() * 0.75
            # mask += torch.ones(size=(64,64)) * 0.25                    
            mask = story_pipeline_store.first_stage.last_masks[i][64][j].reshape(64,64).cpu() * 1
            mask = mask.to(dtype=torch.float32)
            mask = mask.unsqueeze(0).unsqueeze(0)
            kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32) / (kernel_size * kernel_size)
            padding = (kernel_size - 1) // 2
            mask = F.conv2d(mask, kernel, padding=padding).squeeze()  # Remove the extra dimensions   
            # Apply the final transformation
            mask = mask * 0.6 + torch.ones(size=(64, 64)) * 0.25 + (mask > 0) * 0.15
            data_list[f"img_{n}"] = {
                'xt':[_xt_save[j:j+1] for _xt_save in story_pipeline_store.first_stage.xt_save[i]],
                'h_mid':[_mid_save[j::n_samples]  for _mid_save in story_pipeline_store.first_stage.mid_save_list[i]],
                'prompt_embed':story_pipeline_store.first_stage.prompt_embeds[i][j::n_samples],
                'mask_64': mask, # sum
                'prompt': story_pipeline_store.first_stage.prompt[i][j], 
                'concept_token': concept_token 
            }
            n += 1
    
    keys_list = list(image_list.keys())
    target_key = keys_list[0]
    
    with open(f'{output_dir}/target_data.pkl', 'wb') as f:
        pickle.dump(data_list[target_key], f)
    image_list[target_key].save(f'{output_dir}/target.jpg')

    torch.cuda.empty_cache()
