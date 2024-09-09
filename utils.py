import os
import torch
import clip
import numpy as np
from torch import nn
from models import MLPModel1,MLPModel5

def load_clip_model(device):
    """Loads the CLIP model with the specified device."""
    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess

def aggregate_video_feature(frame_features):
    """Aggregates video frame features by taking the mean."""
    return np.mean(frame_features, axis=0)

def load_mapper(ckpt_path, device):
    """Loads the PyTorch MLP mapper model from the checkpoint."""
    model = MLPModel5()
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model


