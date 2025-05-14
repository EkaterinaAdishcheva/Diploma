import os
import pickle
import torch
import sys
import yaml
import argparse
import shutil
import json

from OneActor.oa_diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def find_token_ids(tokenizer, prompt, words):
    tokens = tokenizer.encode(prompt)
    ids = []
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

def pipeline_inference(pipeline, prompt, config, oneactor_extra_config, generator=None):
    if generator is None:
        generator = torch.manual_seed(config['seed'])
    return pipeline(prompt, negative_prompt='', num_inference_steps=config['steps'], guidance_scale=config['eta_1'], \
            generator=generator, oneactor_extra_config=oneactor_extra_config)

def main():
    with open("/workspace/Diploma/PATH.json","r") as f:
        ENV_CONFIGS = json.load(f)
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='/workspace/Diploma/config/config.yaml')
    parser.add_argument('--prompt_path', type=str, default='/workspace/Diploma/config/prompt-girl.yaml')
    parser.add_argument('--exp_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    args = parser.parse_args()
    # load config
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(args.prompt_path, "r") as f:
        prompt = yaml.safe_load(f)

    # make dir and initialize

    target_dir = config['experiments_dir']+'/'+args.exp_path

    model_path = args.model_path

    out_root = target_dir + f"/{model_path}" 
    
    os.makedirs(out_root, exist_ok=True)
    os.makedirs(out_root+'/inference', exist_ok=True)

    shutil.copyfile(args.config_path, out_root+'/gen_config.yaml')

    # load sd pipeline
    pipeline = DiffusionPipeline.from_pretrained(ENV_CONFIGS['paths']['sdxl_path']).to(config['device'])
    pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)

    # load cluster information
    xt_dic = load_pickle(target_dir+'/target_data.pkl')
    h_base = load_pickle(target_dir+'/base/mid_list.pkl')
    h_tar = xt_dic['h_mid']

    generator = torch.manual_seed(config['seed'])
    config['generator'] = generator

    for _prompt in prompt['add_prompts']:
        prompt_str = f"{prompt['target_prompt']} {_prompt}"
        token_id = find_token_ids(pipeline.tokenizer, prompt_str, [prompt['base']])
        steps = config['only_step']
            
        with torch.no_grad():
            projector_path = out_root + f'/weight/learned-projector-steps-{steps[-1]}.pth'
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
        image = pipeline_inference(pipeline, prompt_str, config, oneactor_extra_config)
        image = image.images[0]
        image.save(out_root+'/inference'+f'/{_prompt.lower().replace(",","").replace(" ","_")}_step_{steps[-1]}.jpg')

    torch.cuda.empty_cache()
    
if __name__ == '__main__':
    main()