import argparse
import sys
import os
import pandas as pd
import yaml
parent_dir = os.path.dirname(os.path.realpath("."))
sys.path.append(parent_dir)
import seaborn as sns
import pickle
from matplotlib import pyplot as plt

from dreamsim import dreamsim
from PIL import Image
import torch

import numpy as np
from tqdm import tqdm

from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.multimodal import CLIPScore
from torchmetrics.text import CLIPTextSimilarity
import clip

import pandas as pd


ROOT = "/workspace/experiments"


def clip_i(model, image_1, image_2 ):
    
    # Extract image embeddings
    with torch.no_grad():
        emb1 = model.encode_image(image1)
        emb2 = model.encode_image(image2)
    
    # Normalize embeddings
    emb1 /= emb1.norm(dim=-1, keepdim=True)
    emb2 /= emb2.norm(dim=-1, keepdim=True)
    
    # Compute CLIP-I (Image-Image similarity)
    clip_i_score = (emb1 @ emb2.T).item()
    
    return clip_i_score
    

def main():


# # Image for evaluation
# image = Image.open("generated_image.png")

# # Prompts
# prompt_1 = "A brave knight standing on a hill."
# prompt_2 = "A fearless warrior overlooking a battlefield."

# # CLIPScore (Text-Image)
# clip_score = CLIPScore()
# image_text_similarity = clip_score(image, prompt_2)
# print(f"CLIPScore (Image ↔ Text): {image_text_similarity:.4f}")

# # CLIP-T (Text-Text)
# clip_t = CLIPTextSimilarity()
# text_similarity = clip_t(prompt_1, prompt_2)
# print(f"CLIP-T (Text ↔ Text): {text_similarity:.4f}")


    # get environment configs
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--prompt_path', type=str, default='/workspace/Diploma/config/prompt.yaml')
    parser.add_argument('--exp_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)

    with open(opt.prompt_path, "r") as f:
        prompt = yaml.safe_load(f)

    args = parser.parse_args()

    exp_path, model_path = args.exp_path, args.model_path
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    dreamsim_model, preprocess_ds = dreamsim(pretrained=True, device=device)
    clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)
    clip_t = CLIPTextSimilarity()
    model_ci, preprocess_ci = clip.load("ViT-B/32", device)

    file_target = f"{ROOT}/{exp_path}/{'target.jpg'}"
    img1 = Image.open(file_target)

    img1_ar = np.array(img1.convert("RGB"))
    img1_ar = torch.from_numpy(img1_ar).permute(2, 0, 1).to(device)

    img1_prt = prompt['target_prompt']

    img1_prep_ds = preprocess_ds(img1).to(device)
    img1_prep_ci = preprocess_ds(img1).unsqueeze(0).to(device)

    results = []
    for prt in prompt['add_prompts']:
        file_name = prt.lower().replace(",","").replace(" ","_")
        img2 = Image.open(f"{ROOT}/{exp_path}/{model_path}/inference/{file_name}.jpg")
        img2_ar = np.array(img2.convert("RGB"))
        img2_ar = torch.from_numpy(img2_ar).permute(2, 0, 1).to(device)
        img2_prt = f"{prompt['target_prompt']} {prt}"

        img2_prep_ds = preprocess_ds(img2).to(device)
        img2_prep_ci = preprocess_ds(img2).unsqueeze(0).to(device)
    
        clip_score = clip_score(img1_ar, img2_ar)
        ds_distance = dreamsim_model(img1_prep_ds, img2_prep_ds)
        text_similarity = clip_t(img1_prt, img2_prt)
        clip_i_score = clip_i(model_ci, img1_prep_ci, img2_prep_ci)
        
        resultsresults.append({
            "exp_path": exp_path,
            "model_path": model_path,
            "prompt_1": img1_prt,
            "prompt_2": img2_prt,
            "clip_score": clip_score,
            "dreamsim_distance": ds_distance,
            "clip_t_score": text_similarity,
            "clip_i_score": clip_i_score,            
        })
                
    df = pd.DataFrame(results).T.reset_index()
    print(df)
    aggres = aggres = df[['clip_score', 'dreamsim', 'clip_t_score', 'clip_i_score']].mean().values
    print(f"🔥🔥🔥 {exp_path}/{model_path}:  {aggres}")
    df.to_csv(f"{ROOT}/{exp_path}/{model_path}/metrics.csv", index=False)

if __name__ == '__main__':
    main()