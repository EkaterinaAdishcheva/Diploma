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
    parser.add_argument('--config_path', type=str, default='config/config.yaml')
    parser.add_argument('--prompt_path', type=str, default='config/prompt.yaml')
    opt = parser.parse_args()
    with open(opt.config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(opt.prompt_path, "r") as f:
        prompt = yaml.safe_load(f)

    subject = prompt['target_prompt']
    concept_token = [prompt['base']]
    settings = ["standing"] * 3 + ["sitting"] * 3 + ["walking"] * 2
    if 'g_seed' not in list(config.keys()):
        seed = random.randint(0, 10000)
    else:
        seed = config['g_seed']
    mask_dropout = 0.5
    same_latent = False


    os.makedirs(config['experiments_dir'], exist_ok=True)
    now = datetime.now()

    now = now.strftime("%y%m%d%H%M")
    now = str(now)
    output_dir = f"{config['experiments_dir']}/{now}_{concept_token[0]}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/base", exist_ok=True)
    shutil.copyfile(opt.prompt_path, f"{output_dir}/prompt.yaml")
    shutil.copyfile("notebooks/show_images.ipynb", f"{output_dir}/show_images.ipynb")
    shutil.copyfile("notebooks/reconciliation.ipynb", f"{output_dir}/reconciliation.ipynb")

    gpu = 0
    story_pipeline = load_pipeline(gpu)
    
    story_pipeline_store = StoryPipelineStore()
    token_id = find_token_ids(story_pipeline.tokenizer, subject, concept_token)

    # Reset the GPU memory tracking
    torch.cuda.reset_max_memory_allocated(gpu)

    
    random_settings = random.sample(settings, 4)
    prompts = [f'{subject} {setting}' for setting in random_settings]
    anchor_out_images, anchor_image_all, anchor_cache_first_stage = \
            run_anchor_generation(story_pipeline, prompts[:6], concept_token, 
                           seed=seed, mask_dropout=mask_dropout, same_latent=same_latent,
                           cache_cpu_offloading=True, story_pipeline_store=story_pipeline_store)

    torch.cuda.empty_cache()
    random_settings = random.sample(settings, 4)
    prompts = [f'{subject} {setting}' for setting in random_settings]
    anchor_out_images, anchor_image_all, anchor_cache_first_stage = \
        run_anchor_generation(story_pipeline, prompts, concept_token, 
                       seed=seed, mask_dropout=mask_dropout, same_latent=same_latent,
                       cache_cpu_offloading=True, story_pipeline_store=story_pipeline_store)

    torch.cuda.empty_cache()
    random_settings = random.sample(settings, 4)
    prompts = [f'{subject} {setting}' for setting in random_settings]
    anchor_out_images, anchor_image_all, anchor_cache_first_stage = \
        run_anchor_generation(story_pipeline, prompts, concept_token, 
                       seed=seed, mask_dropout=mask_dropout, same_latent=same_latent,
                       cache_cpu_offloading=True, story_pipeline_store=story_pipeline_store)

    torch.cuda.empty_cache()
   
    data_list = {}
    image_list = {}

    n = 0
    for i in range(len(story_pipeline_store.first_stage.images)):
        n_samples = len(story_pipeline_store.first_stage.images[i])
        for j in range(n_samples):
            image_list[f"img_{n}"] = story_pipeline_store.first_stage.images[i][j]
            mask = 1 - story_pipeline_store.first_stage.nn_distances[i][64].reshape(n_samples,n_samples,64,64)[j]
            mask = torch.cat([mask[:j],mask[j+1:]]) 
            mask = mask.mean(dim=0)
                    
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
    base_keys = keys_list[1:]
    
    with open(f'{output_dir}/target_data.pkl', 'wb') as f:
        pickle.dump(data_list[target_key], f)
    image_list[target_key].save(f'{output_dir}/target.jpg')

    for key in base_keys:
        with open(f'{output_dir}/base/{key}.pkl', 'wb') as f:
            pickle.dump(data_list[key], f)
            image_list[key].save(f"{output_dir}/base/{key}.jpg")