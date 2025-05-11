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

ROOT = "/workspace/Diploma/OneActor/experiments"

def main():
    
    # get environment configs
    with open("PATH.json","r") as f:
        ENV_CONFIGS = json.load(f)
    # get user configs
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_name', type=str, required=True)
    parser.add_argument('--train_id', type=str, required=True)

    args = parser.parse_args()

    device = "cuda"
    model, preprocess = dreamsim(pretrained=True, device=device)
    
    img1 = Image.open(f"{ROOT}/{args.dir_name}/target.jpg")
    img1 = np.array(img1.convert("RGB"))
    img1 = torch.from_numpy(img1).permute(2, 0, 1).to(device)

    for _, _, files in os.walk(f"{ROOT}/{args.dir_name}/{args.train_id}/inference"):
        break

    results = {}
    for n, file_name in tqdm(enumerate(files)):
        results[file_name] = {}

    for n, file_name in tqdm(enumerate(files)):
        img2 = Image.open(f"{ROOT}/{args.dir_name}/{args.train_id}/inference/{file_name}")
        img2 = np.array(img2.convert("RGB"))
        img2 = torch.from_numpy(img2).permute(2, 0, 1).to(device)
        clip_score = metric(img1, img2)
        results[file_name]['CLIPScore'] = clip_score.detach().round().item()

    img1 = preprocess(Image.open(f"{ROOT}/{args.dir_name}/target.jpg").to(device)
    for n, file_name in tqdm(enumerate(files)):
        img2 = preprocess(Image.open(r['gen_img_path'])).to(device)
        distance = model(img1, img2)
        results[file_name]['dreamsim'] = float(distance.cpu().detach().numpy()[0])

    with open(f"{ROOT}/{args.dir_name}/{args.train_id}/metrics.pickle", 'wb') as handle:
        pickle.dump(df, handle)

