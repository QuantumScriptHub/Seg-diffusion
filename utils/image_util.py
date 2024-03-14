import os
import cv2
import json
import torch
import random
import matplotlib
import numpy as np
from PIL import Image


def loadAde20K(file):
    objects = {}

    attr_file_name = file.replace('.jpg', '.json')
    if os.path.isfile(attr_file_name):
        try:
            with open(attr_file_name, 'r') as f:
                input_info = json.load(f)
        except:
            with open(attr_file_name, 'r', encoding='latin1') as f:
                input_info = json.load(f)
        contents = input_info['annotation']['object']
        for x in contents:
            if x['raw_name'] in objects:
                objects[x['raw_name']].append(x['instance_mask'])
            else:
                objects[x['raw_name']] = [x['instance_mask']]

    return objects


def read_text_lines(filepath):
    with open(filepath, 'r') as f:
        lines = f.readlines()
    lines = [l.rstrip() for l in lines]
    return lines


def resize_max_res(img: Image.Image, max_edge_resolution: int) -> Image.Image:
    resized_img = img.resize((max_edge_resolution, max_edge_resolution))
    return resized_img


def pyramid_noise_like(noise, device, iterations=6, discount=0.3):
    b, c, w, h = noise.shape
    u = torch.nn.Upsample(size=(w, h), mode='bilinear').to(device)
    for i in range(iterations):
        r = random.random()*2+2 # Rather than always going 2x,
        w, h = max(1, int(w/(r**i))), max(1, int(h/(r**i)))
        noise += u(torch.randn(b, c, w, h).to(device)) * discount**i
        if w==1 or h==1: break # Lowest resolution is 1x1
    return noise/noise.std() # Scaled back to roughly unit variance


def colorize_semantic_maps(semantic_map, min_semantic, max_semantic, cmap="Spectral", valid_mask=None):
    """
    Colorize semantic maps.
    """
    assert len(semantic_map.shape) >= 2, "Invalid dimension"

    if isinstance(semantic_map, torch.Tensor):
        semantic = semantic_map.detach().clone().squeeze().numpy()
    elif isinstance(semantic_map, np.ndarray):
        semantic = semantic_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if semantic.ndim < 3:
        semantic = semantic[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    semantic = ((semantic - min_semantic) / (max_semantic - min_semantic)).clip(0, 1)
    img_colored_np = cm(semantic, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(semantic_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(semantic_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(semantic_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


def chw2hwc(chw):
    assert 3 == len(chw.shape)
    if isinstance(chw, torch.Tensor):
        hwc = torch.permute(chw, (1, 2, 0))
    elif isinstance(chw, np.ndarray):
        hwc = np.moveaxis(chw, 0, -1)
    return hwc

