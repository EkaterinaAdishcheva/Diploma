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
        projector = projector.half()
        mid_base_all=mid_base_all.half()
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
    parser.add_argument('--config_path', type=str, default='config/config.yaml')
    parser.add_argument('--prompt_path', type=str, default='config/prompt-girl.yaml')
    parser.add_argument('--dir_name', type=str, required=True)
    parser.add_argument('--train_id', type=str, required=True)

    args = parser.parse_args()
    
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(args.prompt_path, "r") as f:
        prompt = yaml.safe_load(f)

    device = config['device']

    neg_prompts = [''] * len(prompt['add_prompts'])
    file_names = ["_".join(_prompt.split(" ")) for _prompt in prompt['add_prompts']]
    # make dir and initialize
    tgt_dirs = []
    
    target_dir = config['experiments_dir']+'/'+args.dir_name

    train_id = args.train_id

    out_root = target_dir + f"/{train_id}" 
    
    os.makedirs(f"{out_root}/inference", exist_ok=True)
    print(f"Save inference to {out_root}/inference")

    # load sd pipeline
    pipeline = DiffusionPipeline.from_pretrained(ENV_CONFIGS['paths']['sdxl_path']).to(config['device'])
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    # load cluster information

    with open(target_dir+f'/target_data.pkl', 'rb') as f:
        target_data = pickle.load(f)

    base_image_paths = [
        file_path.split(".")[0] for file_path in os.listdir(f"{target_dir}/base") \
            if os.path.splitext(file_path)[1] == '.jpg']

    base_data = []
    for file_name in base_image_paths:
        with open(f"{target_dir}/base/{file_name}.pkl", 'rb') as f:
            _base_data = pickle.load(f)
            base_data.append({'h_mid': _base_data['h_mid'][-1:]})


    h_base = [h['h_mid'][-1] for h in base_data]
    h_tar = target_data['h_mid']

    
    # iterate over image list
    for img_num in range(len(prompt['add_prompts'])):
        _str = f"{prompt['target_prompt']} {prompt['add_prompts'][img_num]}"
        print(f"Generating prompt {_str}...")

        # locate the base token id
        token_id = find_token_ids(pipeline.tokenizer, _str, prompt['base'])
        generator = torch.manual_seed(config['seed'])

        steps_list = config['only_step']
        for steps in steps_list:
            print(f"Using weights from step {steps}")
            with torch.no_grad():
                projector_path = f'{out_root}/weight/learned-projector-steps-{steps}.pth'
                delta_emb_all = projector_inference(projector_path, h_tar, h_base, config['device']).to(config['device'])

            delta_emb_aver = delta_emb_all[:-1].mean(dim=0) # [2048]
            delta_emb_tar = config['v'] * delta_emb_all[-1] # [2048]

            oneactor_extra_config = {
                'token_ids': token_id,
                'delta_embs': delta_emb_tar,
                'delta_steps': None,
                'eta_2': config['eta_2'],
                'delta_emb_aver': delta_emb_aver
            }
            image = pipeline_inference(
                pipeline, 
                _str,
                neg_prompts[img_num],
                config, oneactor_extra_config)
            image = image.images[0]
            image.save(f"{out_root}/inference/{file_names[img_num]}_step_{str(steps)}.jpg")

if __name__ == '__main__':
    main()