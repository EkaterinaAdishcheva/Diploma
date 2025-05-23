import os
import pickle
import torch
import sys
import yaml
import argparse
import shutil
import json
import shutil
import random

from OneActor.oa_diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from OneActor.projector import Projector

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
    projector = Projector(1280, 2048)  # use the same dims you trained with
    state_dict = torch.load(projector_path, map_location=device)
    projector.load_state_dict(state_dict)
    projector.to(device)
    projector.eval()
    with torch.no_grad():
        mid_base_target = [h_target[-1]] + h_base
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
            num_inference_steps=config['steps'], guidance_scale=config['eta_1'], \
            generator=generator, oneactor_extra_config=oneactor_extra_config)

def inference(exp_path, model_path, subject, concept_token, add_prompts, config=None, sdxl_path="stabilityai/stable-diffusion-xl-base-1.0"):
    if config is None:
        with open('/workspace/Diploma/config/config.yaml', "r") as f:
            config = yaml.safe_load(f)

    device = config['device']

    neg_prompts = [''] * len(add_prompts)
    file_names = ["_".join(_prompt.lower().replace(",","").split(" ")) for _prompt in add_prompts]
    # make dir and initialize
    tgt_dirs = []
    
    target_dir = config['experiments_dir']+'/'+exp_path


    out_root = target_dir + f"/{model_path}" 
    
    os.makedirs(f"{out_root}/inference", exist_ok=True)

    print(f"Save inference to {out_root}/inference")
    

    # load sd pipeline
    pipeline = DiffusionPipeline.from_pretrained(sdxl_path).to(config['device'])
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    # load cluster information

    with open(f"{target_dir}/target_data.pkl", 'rb') as f:
        target_data = pickle.load(f)

    base_image_paths = [
        file_path.split(".")[0] for file_path in os.listdir(f"{target_dir}/base") \
            if os.path.splitext(file_path)[1] == '.jpg']

    base_data = []
    with open(f"{target_dir}/base/mid_list.pkl", 'rb') as f:
        base_data = pickle.load(f)
            
    
    projector_data_path = f'{out_root}/weight/projector_res.pkl'
    with open(projector_data_path, 'rb') as f:
        projector_res = pickle.load(f)
    projector_res = projector_res.mean(dim=0)


    h_base = base_data
    h_tar = target_data['h_mid']

    
    # iterate over image list
    for img_num in range(len(add_prompts)):
        _str = f"{subject} {add_prompts[img_num]}"
        print(f"Generating prompt {_str}...")

        # locate the base token id
        token_id = find_token_ids(pipeline.tokenizer, _str, concept_token)
        generator = torch.manual_seed(config['seed'])

        steps_list = config['only_step']
        for steps in steps_list:
            print(f"Using weights from step {steps}")
            with torch.no_grad():
                # projector_path = f'{out_root}/weight/learned-projector-steps-{steps}.pth'
                delta_emb_all = projector_res.to(config['device'])
                # delta_emb_all = projector_inference(projector_path, h_tar, h_base, config['device']).to(config['device'])

            delta_emb_aver = delta_emb_all[1:].mean(dim=0) # [2048]            
            delta_emb_tar = config['v'] * delta_emb_all[0] #* 1.2 # [2048]


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

    torch.cuda.empty_cache()


def main():
    with open("/workspace/Diploma/PATH.json","r") as f:
        ENV_CONFIGS = json.load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='/workspace/Diploma/config/config.yaml')
    parser.add_argument('--prompt_path', type=str, default='/workspace/Diploma/config/prompt-girl.yaml')
    parser.add_argument('--exp_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)

    args = parser.parse_args()
    
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(args.prompt_path, "r") as f:
        prompt = yaml.safe_load(f)

    subject = prompt['target_prompt']
    concept_token = [prompt['base']]
    add_prompts = prompt['add_prompts']
    exp_path = args.exp_path
    model_path = args.model_path
    inference(exp_path, model_path, subject, concept_token, add_prompts, config, ENV_CONFIGS["paths"]["sdxl_path"])

if __name__ == '__main__':
    main()