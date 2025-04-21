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
    parser.add_argument('--model_id', type=str, required=True)
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
    model_id = args.model_id

    print(f"target_id = {target_id}")


    if target_id not in tgt_dirs:
        print("Base image is not generated")
        return

    target_dir += f"/{target_id}"

    print(f"model_id = {model_id}")
    
    for _, tgt_dirs, _ in os.walk(target_dir):
        break
    
    if model_id not in tgt_dirs:
        print("Train is not performed")
        return

    
    out_root = target_dir + f"/{model_id}" 
    
    os.makedirs(f"{out_root}/inference", exist_ok=True)
    print(f"Save inference in {out_root}/inference")

    shutil.copyfile(args.config_path, out_root+f'/gen_config_{model_id}.yaml')

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

        # perform step-wise guidance
        select_steps = config['select_steps']
        if select_steps is not False:
            assert (len(select_steps) % 2) == 0
            select_list = []
            for _ in range(len(select_steps) // 2):
                a = select_steps[2*_]
                b = select_steps[2*_ + 1]
                select_list = select_list + list(range(a-1,b))
        else:
            select_list = None

        # locate the base token id
        token_id = find_token_ids(pipeline.tokenizer, config['target_prompt'] + " " + config['add_prompts'][img_num], config['base'])
        generator = torch.manual_seed(config['seed'])
        config['generator'] = generator

        if config['only_step'] is False:
            for i in range(50):
                steps = config['step_from']+config['step']*(i)
                print(f"Using weights from step (steps)")
                with torch.no_grad():
                    projector_path = f'{out_root}/weight/learned-projector-steps-{steps}.pth'
                    delta_emb_all = projector_inference(projector_path, h_tar, h_base, config['device']).to(config['device'])

                delta_emb_aver = delta_emb_all[:-1].mean(dim=0)
                delta_emb_tar = config['v'] * delta_emb_all[-1]

                oneactor_extra_config = {
                    'token_ids': token_id,
                    'delta_embs': delta_emb_tar,
                    'delta_steps': select_list,
                    'eta_2': config['eta_2'],
                    'delta_emb_aver': delta_emb_aver
                }

                image = pipeline_inference(
                    pipeline, 
                    config['target_prompt'] + " " + config['add_target_prompt'] + " " + config['add_prompts'][img_num],
                    config['target_neg_prompt'] + " " + config['neg_prompts'][img_num],
                    config, oneactor_extra_config)
                image = image.images[0]
                image.save(f"{out_root}/inference/{config['file_names'][img_num]}_step_{steps}.jpg")
        elif config['only_step'] == 'best':
            with torch.no_grad():
                projector_path = f'{out_root}/weight/best-learned-projector.pth'
                delta_emb_all = projector_inference(projector_path, h_tar, h_base, config['device']).to(config['device'])

            delta_emb_aver = delta_emb_all[:-1].mean(dim=0) # [2048]
            delta_emb_tar = config['v'] * delta_emb_all[-1] # [2048]
            
            oneactor_extra_config = {
                'token_ids': token_id,
                'delta_embs': delta_emb_tar,
                'delta_steps': select_list,
                'eta_2': config['eta_2'],
                'delta_emb_aver': delta_emb_aver
            }
            image = pipeline_inference(
                pipeline,
                config['target_prompt'] + " " + config['add_target_prompt'] + " " + config['add_prompts'][img_num],
                config['target_neg_prompt'] + " " + config['neg_prompts'][img_num],
                config, oneactor_extra_config)
            image = image.images[0]
            image.save(f"{out_root}/inference/{config['file_names'][img_num]}_step_best.jpg")
        else:
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
                    'delta_steps': select_list,
                    'eta_2': config['eta_2'],
                    'delta_emb_aver': delta_emb_aver
                }
                image = pipeline_inference(
                    pipeline, 
                    config['target_prompt'] + " " + config['add_target_prompt'] + " " + config['add_prompts'][img_num],
                    config['target_neg_prompt'] + " " + config['neg_prompts'][img_num],
                    config, oneactor_extra_config)
                image = image.images[0]
                image.save(f"{out_root}/inference/{config['file_names'][img_num]}_step_{str(steps)}.jpg")

if __name__ == '__main__':
    main()