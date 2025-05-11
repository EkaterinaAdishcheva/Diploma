import os
import numpy as np
import pickle

with open(f"target_data.pkl", 'rb') as f:
    target_data = pickle.load(f)

with open(f"target_mask.pkl", 'wb') as f:
    pickle.dump(target_data['mask_64'], f)