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
import seaborn as sns
from collections import Counter
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
import math
from torchvision.utils import save_image
import imageio

import minigrid
print("Minigrid loaded from:", minigrid.__file__)
import sys
import numpy as np
import matplotlib.pyplot as plt


from disent.dataset.data import XYSquaresData
def train_model(lr, batch_size, z_size, steps):
    print(f"Training with lr={lr}, batch_size={batch_size}, z_size={z_size}, steps={steps}")
    start_time = time.time()

    #===============================================
    #   TRAINING USING NORMAL SAMPLER
    #===============================================
    sampler=RlSampler()
    data = XYSingleSquareData(grid_spacing=4,n=2)
    dataset = DisentDataset(dataset=data, sampler=sampler, transform=ToImgTensorF32())
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
    # data = XYSingleSquareData(grid_spacing=4,n=2)
    # dataset = DisentDataset(dataset=data, sampler=RlSampler(), transform=ToImgTensorF32())
    # dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)


    #data_x_shape=(3,64,64)
    #data_x_shape=(1,32,32)
    model = AutoEncoder(
        encoder=EncoderConv64(x_shape=data.x_shape, z_size=z_size, z_multiplier=2),
        decoder=DecoderConv64(x_shape=data.x_shape, z_size=z_size),
    )
    
    framework = AdaVae(
        model=model,
        cfg=AdaVae.cfg(
            optimizer="adam",
            optimizer_kwargs=dict(lr=1e-4),
            loss_reduction="mean_sum",
            beta=0.001,
            ada_average_mode="gvae",
            ada_thresh_mode="kl",
        ),
    )
    
    trainer = L.Trainer(max_steps=60000, logger=False)
    trainer.fit(framework, dataloader)

    # Euclidean distances
    
    start_points = np.array(data.start_points)
    end_points = np.array(data.end_points)

    distances = np.linalg.norm(end_points - start_points, axis=1)

    plt.hist(distances, bins=30, density=True, cmap='viridis')
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    plt.title("Distribution of Distances")
    plt.savefig('histogram.png')
    plt.show()




    # Count frequencies
    freq = Counter(sampler.indices)

    # Sorted unique numbers and their frequencies
    unique_numbers = sorted(freq.keys())
    frequencies = [freq[num] for num in unique_numbers]

    # Determine grid size (square grid)
    size = math.ceil(math.sqrt(len(unique_numbers)))

    # Pad frequencies to fill the grid
    padded_frequencies = frequencies + [0] * (size*size - len(frequencies))
    data = np.array(padded_frequencies).reshape((size, size))

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(data, annot=True, fmt="d", cmap='YlOrRd')

    plt.title('Frequency Heatmap of Numbers')

    # Save heatmap to file
    plt.savefig('frequency_heatmap.png')

    plt.close()  # Close plot to free memory

    
def test_model():
    #data=XYSquaresData(grid_spacing=3)
    # data = XYSingleSquareData(grid_spacing=4,n=1)
    data=UnlockDataDV3(latent_shape="2D")
    dataset = DisentDataset(dataset=data, sampler=GroundTruthPairOrigSampler(), transform=ToImgTensorF32())
    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=True, num_workers=0)
    batch = next(iter(dataloader))
    print(batch['x_targ'][0].shape)
    # save_image(batch['x_targ'][0], 'output0_image.png')
    # save_image(batch['x_targ'][1], 'output1_image.png')

    # # Extract one sample (first sample from batch)
    # if isinstance(batch, dict):
    #     sample = {k: v[0] for k, v in batch.items()}
    # elif isinstance(batch, (list, tuple)):
    #     sample = [b[0] for b in batch]
    # else:
    # sample = batch[0]



def get_gif():
    # Setup
    # data = XYSingleSquareData(grid_spacing=4,n=0)
    data=RlUnlockData(n=3)
    # data=UnlockData()
    dataset = DisentDataset(dataset=data, sampler=RlSampler(), transform=ToImgTensorF32())
    dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=True, num_workers=0)
    batch = next(iter(dataloader))

    max_index = len(dataloader.dataset) - 1
    print(f"Max index: {max_index}")
    # print(batch['x_targ'][0][0].shape)  # e.g. torch.Size([1, 64, 64])

    frames = []
    num_frames = batch['x_targ'][0].shape[0]
    print(num_frames)

    for i in range(num_frames):
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
        
        # plt.imsave('image1.png', img1_np)
        # plt.imsave('image2.png', img2_np)  
        diff = np.abs(img1_np - img2_np) / 255.0

# Optionally save the difference as an image
        # Image.fromarray((diff * 255).astype(np.uint8)).save("difference.png")   
        combined = np.concatenate((img1_np, img2_np), axis=1)
        frames.append(combined)

        #Save PNG with correct mode
        if combined.shape[2] == 1 or combined.ndim == 2:
            # Grayscale
            Image.fromarray(combined.squeeze(), mode='L').save(f'output_image_{i}.png')
        else:
            # RGB
            Image.fromarray(combined, mode='RGB').save(f'output_image_{i}.png')
      
    # plt.figure()
    # plt.imshow(data.accum_start, cmap='hot')#, vmin=np.min(data.accum_start), vmax=np.max(data.accum_start))
    # plt.colorbar()
    # plt.savefig(f'square_spacing4_beta_start4.png')
    # plt.close()

    # plt.figure()
    # plt.imshow(data.accum_pair, cmap='hot')#, vmin=np.min(data.accum_pair), vmax=np.max(data.accum_pair))
    # plt.colorbar()
    # plt.savefig(f'square_spacing4_beta__pair4.png')
    # plt.close() 
    # # Save as GIF
    # imageio.mimsave('output_minigrid2.gif', frames, duration=3)
    # print("GIF saved as output_minigrid2.gif")
   
# train_model(0.0001, 4, 6, 60000)
#lr=1e-4 batch size=64 latent size=50 max steps=50 000
# train_model(lr=0.0001, batch_size=64, z_size=50, steps=50000 )
get_gif()
#test_model()


