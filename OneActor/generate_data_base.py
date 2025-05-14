import torch
import gc
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

from OneActor.oa_diffusers import DiffusionPipeline

from OneActor.utils.ptp_utils import view_images

def generate_base(exp_path, subject, concept_token, config=None, sdxl_path="stabilityai/stable-diffusion-xl-base-1.0"):
    if config is None:
        with open('/workspace/Diploma/config/config.yaml', "r") as f:
            config = yaml.safe_load(f)

    device = config['device']
    pipeline = DiffusionPipeline.from_pretrained(sdxl_path).to(device)

    if 'g_seed' not in list(config.keys()):
        seed = random.randint(0, 10000)
    else:
        seed = config['g_seed']

    os.makedirs(config['experiments_dir'], exist_ok=True)
    now = datetime.now()

    if os.path.isdir(f"{config['experiments_dir']}/{exp_path}"):
        pass   
    else:
        print(f"💥 The directory {config['experiments_dir']}/{exp_path} is not exist")
    output_dir = f"{config['experiments_dir']}/{exp_path}"
    print(f"✅ Save images to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/base", exist_ok=True)
    steps = config['steps']
    guidance_scale = config['guidance_scale']
    generator = torch.manual_seed(config['g_seed'])
    prompt_str = subject

    num_base = config['gen_base']
    all_images = []
    
    if num_base > 0:
        mid_last_base = []
        for i in range(num_base):
            image, xt_list_, prompt_embeds, mid_ = pipeline(prompt_str, neg_prompt="",
                                                            num_inference_steps=steps, guidance_scale=guidance_scale, generator=generator,
                                                            oneactor_save=True)
            image = image.images[0]
            all_images.append(np.array(image))
            image.save(f"{output_dir}/base/base{i}.jpg")
            mid_last_base.append(mid_[-1].cpu())
        with open(f"{output_dir}/base/mid_list.pkl", 'wb') as f:
            pickle.dump(mid_last_base, f)
        
        img_view = view_images(all_images, num_rows=num_base//3+1, downscale_rate=4)
        img_view.save(f"{output_dir}/base_all.jpg")

    torch.cuda.empty_cache()

    
if __name__ == '__main__':

    with open("/workspace/Diploma/PATH.json","r") as f:
        ENV_CONFIGS = json.load(f)
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

    exp_path = opt.exp_path
    generate_base(exp_path, subject, concept_token, config, ENV_CONFIGS["paths"]["sdxl_path"])