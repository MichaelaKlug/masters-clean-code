from __future__ import annotations

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, RGBImgObsWrapper
import matplotlib.pyplot as plt
from gymnasium.wrappers import TransformObservation
import cv2
from gymnasium import spaces
import numpy as np

import lightning as L
import torch
from torch.utils.data import DataLoader

from disent.dataset import DisentDataset
from disent.dataset.data import XYObjectData,XYSingleSquareData, RlUnlockData, UnlockData,UnlockDataDV3
from disent.dataset.sampling import SingleSampler
from disent.dataset.transform import ToImgTensorF32
from disent.frameworks.vae import BetaVae, AdaVae
from disent.metrics import metric_dci
from disent.metrics import metric_mig
from disent.model import AutoEncoder
from disent.model.ae import DecoderConv64, DecoderConv32
from disent.model.ae import EncoderConv64, EncoderConv32
from disent.schedule import CyclicSchedule

from disent.model.ae import DecoderLinear, DecoderLinear1d
from disent.model.ae import EncoderLinear, EncoderLinear1d

# from _groundtruth__unlockobject import UnlockData

#from _groundtruth__pair_orig import GroundTruthPairOrigSampler, ConsecStatesSampler
from disent.dataset.sampling import GroundTruthPairSampler,RlSampler,GroundTruthPairOrigSampler,GroundTruthPairOrigSamplerUnlock

import itertools
import time
from PIL import Image

# from _groundtruth__trajectories import EpisodeData

from torchvision.utils import save_image
import imageio

import minigrid
print("Minigrid loaded from:", minigrid.__file__)
import sys



from disent.dataset.data import XYSquaresData
def train_model(lr, batch_size, z_size, steps):
    print(f"Training with lr={lr}, batch_size={batch_size}, z_size={z_size}, steps={steps}")
    start_time = time.time()

    #===============================================
    #   TRAINING USING NORMAL SAMPLER
    #===============================================
    data=UnlockDataDV3(latent_shape="1D")
    dataset = DisentDataset(dataset=data, sampler=GroundTruthPairOrigSamplerUnlock(), transform=None)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    #===============================================
    #   TRAINING USING RL SAMPLER
    #===============================================
    # data = EpisodeData()
    # dataset= DisentDataset(dataset=data, sampler=RlSampler(), transform=ToImgTensorF32())
    # dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    #===============================================
    #   TRAINING USING DV3 encodings
    #===============================================
    # data=XYObjectData()
    # dataset = DisentDataset(dataset=data, sampler=GroundTruthPairOrigSampler(), transform=ToImgTensorF32())
    # dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)


    # data_x_shape=(3,64,64)
    data_x_shape=(1,32)
    model = AutoEncoder(
        encoder=EncoderLinear1d(x_shape=data_x_shape, z_size=z_size, z_multiplier=2),
        decoder=DecoderLinear1d(x_shape=data_x_shape, z_size=z_size),
    )
    
    framework = AdaVae(
        model=model,
        cfg=AdaVae.cfg(
            optimizer="adam",
            optimizer_kwargs=dict(lr=1e-4),
            loss_reduction="mean_sum",
            beta=4,
            ada_average_mode="gvae",
            ada_thresh_mode="kl",
        ),
    )
    
    trainer = L.Trainer(max_steps=steps, logger=False)
    trainer.fit(framework, dataloader)
    
   
    
    # # Save model
    # model_path = f"model_lr{lr}_bs{batch_size}_z{z_size}_steps{steps}.pth"
    # torch.save(model.state_dict(), model_path)
    end_time = time.time()  # ⏱️ End timing
    elapsed_time = end_time - start_time
    get_repr = lambda x: framework.encode(x.to(framework.device))
    metrics = {
        **metric_dci(dataset, get_repr, num_train=1000, num_test=500, show_progress=True),
    }
    print(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes).")
    print(metrics)
    
def test_model():
    #data=XYSquaresData(grid_spacing=3)
    # data = XYSingleSquareData(grid_spacing=4,n=1)
    data=UnlockData()
    dataset = DisentDataset(dataset=data, sampler=GroundTruthPairOrigSampler(), transform=ToImgTensorF32())
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)
    batch = next(iter(dataloader))
    print(batch['x_targ'][0].shape)
    save_image(batch['x_targ'][0], 'output0_image.png')
    save_image(batch['x_targ'][1], 'output1_image.png')

    # Extract one sample (first sample from batch)
    if isinstance(batch, dict):
        sample = {k: v[0] for k, v in batch.items()}
    elif isinstance(batch, (list, tuple)):
        sample = [b[0] for b in batch]
    else:
        sample = batch[0]



def get_gif():
    # Setup
    data = XYSingleSquareData(grid_spacing=4,n=1)
    #data=RlUnlockData(n=1)
    dataset = DisentDataset(dataset=data, sampler=GroundTruthPairOrigSampler(), transform=ToImgTensorF32())
    dataloader = DataLoader(dataset=dataset, batch_size=5, shuffle=True, num_workers=0)
    batch = next(iter(dataloader))

    max_index = len(dataloader.dataset) - 1
    # print(f"Max index: {max_index}")
    # print(batch['x_targ'][0][0].shape)  # e.g. torch.Size([1, 64, 64])

    frames = []

    for i in range(5):
        img1 = batch['x_targ'][0][i]  # shape: [C, H, W]
        img2 = batch['x_targ'][1][i]

        def process_image(img):
            img_np = img.permute(1, 2, 0).cpu().numpy()  # shape: (H, W, C)
            img_np = (img_np * 255).astype(np.uint8)
            if img_np.shape[2] == 1:
                img_np = img_np.squeeze(2)  # shape: (H, W)
            return img_np

        img1_np = process_image(img1)
        img2_np = process_image(img2)

        # Concatenate (horizontal stack)
        if img1_np.ndim == 2:
            # Stack grayscale images into RGB for gif if needed
            img1_np = np.stack([img1_np]*3, axis=-1)
            img2_np = np.stack([img2_np]*3, axis=-1)
        
        combined = np.concatenate((img1_np, img2_np), axis=1)
        frames.append(combined)

        #Save PNG with correct mode
        if combined.shape[2] == 1 or combined.ndim == 2:
            # Grayscale
            Image.fromarray(combined.squeeze(), mode='L').save(f'output_image_{i}.png')
        else:
            # RGB
            Image.fromarray(combined, mode='RGB').save(f'output_image_{i}.png')
      

    # Save as GIF
    imageio.mimsave('output_minigrid2.gif', frames, duration=3)
    print("GIF saved as output_minigrid2.gif")
   
train_model(0.0001, 4, 6, 10)
#lr=1e-4 batch size=64 latent size=50 max steps=50 000
# train_model(lr=0.0001, batch_size=64, z_size=50, steps=50000 )
# get_gif()
# test_model()


