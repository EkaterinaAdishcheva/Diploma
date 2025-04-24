import os
import pickle
import torch
import sys
sys.path.append('./diffusers')
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import yaml
import argparse
import shutil
import json

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

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

def projector_inference(projector_path, h_target, h_base, device):
    with torch.no_grad():
        projector = torch.load(projector_path).to(device)
        mid_base_target = h_base + [h_target[-1]]
        mid_base_all = torch.stack(mid_base_target)
        delta_emb_all = projector(mid_base_all[:,-1].to(device))
    return delta_emb_all

def pipeline_inference(pipeline, prompt, neg_prompt, config, oneactor_extra_config, generator=None):
    if generator is None:
        generator = torch.manual_seed(config['seed'])
    return pipeline(
            prompt,
            negative_prompt=neg_prompt,
            num_inference_steps=config['inference_steps'], guidance_scale=config['eta_1'], \
            generator=generator, oneactor_extra_config=oneactor_extra_config)

def main():
    with open("PATH.json","r") as f:
        ENV_CONFIGS = json.load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='./config/gen_tune_inference.yaml')
    parser.add_argument('--target_id', type=str, required=True)
    args = parser.parse_args()
    # load config
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    config['neg_prompts'] = [''] * len(config['add_prompts'])
    config['file_names'] = ["_".join(prompt.split(" ")) for prompt in config['add_prompts']]
    # make dir and initialize
    tgt_dirs = []
    target_dir = config['experiments_dir']+'/'+config['target_dir']
    for _, tgt_dirs, _ in os.walk(target_dir):
        break
        
    target_id = args.target_id

    print(f"target_id = {target_id}")


    if target_id not in tgt_dirs:
        print("Base image is not generated")
        return

    target_dir += f"/{target_id}"

    out_root = target_dir 
    
    os.makedirs(f"{out_root}/inference_sdxl", exist_ok=True)
    print(f"Save inference in {out_root}/inference_sdxl")

    # load sd pipeline
    pipeline = DiffusionPipeline.from_pretrained(ENV_CONFIGS['paths']['sdxl_path']).to(config['device'])
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    # load cluster information
    xt_dic = load_pickle(target_dir+'/xt_list.pkl')
    h_base = load_pickle(target_dir+'/base/mid_list.pkl')
    h_tar = xt_dic['h_mid']
    
    
    # iterate over image list
    for img_num in range(len(config['add_prompts'])):
        _str = config['target_prompt'] + " " + config['add_target_prompt'] + " " + config['add_prompts'][img_num]
        print(f"Generating prompt {_str}...")
        # original output by SDXL
        generator = torch.manual_seed(config['seed'])
        image = pipeline(
            config['target_prompt'] + " " + config['add_target_prompt'] + " " + config['add_prompts'][img_num],
            negative_prompt=config['target_neg_prompt'] + " " + config['neg_prompts'][img_num],
            num_inference_steps=config['inference_steps'],
            guidance_scale=config['eta_1'],
            generator=generator)
        image = image.images[0]
        image.save(f"{out_root}/inference_sdxl/{config['file_names'][img_num]}_sdxl.jpg")


if __name__ == '__main__':
    main()