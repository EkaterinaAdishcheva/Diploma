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
import torch
import pandas as pd

# from deepface import DeepFace

ROOT = "/workspace/experiments"

def main():
    
    # get environment configs
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_name', type=str, required=True)
    parser.add_argument('--train_id', type=str, required=True)

    args = parser.parse_args()

    dir_name, train_id = args.dir_name, args.train_id
    
    device = "cuda"
    dreamsim_model, preprocess = dreamsim(pretrained=True, device=device)
    clip_score_metric = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)

    file_target = f"{ROOT}/{dir_name}/{'target.jpg'if train_id == 'model_Mask' else 'oa_source.jpg'}"
    img1 = Image.open(file_target)
    img1 = np.array(img1.convert("RGB"))
    img1 = torch.from_numpy(img1).permute(2, 0, 1).to(device)

    for _, _, files in os.walk(f"{ROOT}/{dir_name}/{train_id}/inference"):
        break

    files = [f for f in files if '.jpg' in f]

    results = {}
    for n, file_name in tqdm(enumerate(files)):
        results[file_name] = {}

    for n, _file_name in tqdm(enumerate(files)):
        file_name = f"{ROOT}/{dir_name}/{train_id}/inference/{_file_name}"
        img2 = Image.open(file_name)
        img2 = np.array(img2.convert("RGB"))
        img2 = torch.from_numpy(img2).permute(2, 0, 1).to(device)
        clip_score = clip_score_metric(img1, img2)
        results[_file_name]['CLIPScore'] = clip_score.detach().round().item()
    
    img1 = preprocess(Image.open(file_target)).to(device)
    for n, _file_name in tqdm(enumerate(files)):
        file_name = f"{ROOT}/{dir_name}/{train_id}/inference/{_file_name}"
        img2 = preprocess(Image.open(file_name)).to(device)
        distance = dreamsim_model(img1, img2)
        results[_file_name]['dreamsim'] = float(distance.cpu().detach().numpy()[0])

    # for n, _file_name in tqdm(enumerate(files)):
    #     try:
    #         file_name = _file_name    
    #         img2 = f"{ROOT}/{dir_name}/{train_id}/inference/{_file_name}"
    #         distance = DeepFace.verify(img1_path=file_target, img2_path=img2, model="Facenet512")['distance']
    #         results[_file_name]['DeepFace'] = distance
    #     except:
    #         print(_file_name)
    #         results[_file_name]['DeepFace'] = 3
        
    df = pd.DataFrame(results).T.reset_index()
    aggres = aggres = df[['CLIPScore', 'dreamsim']].mean().values
    print(f"🔥🔥🔥 {dir_name}/{train_id}:  {aggres}")
    df.to_csv(f"{ROOT}/{dir_name}/{train_id}/metrics.csv", index=False)

if __name__ == '__main__':
    main()