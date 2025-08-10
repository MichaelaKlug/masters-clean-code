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

import pandas as pd
import os



import lightning as L
import torch
from torch.utils.data import DataLoader

from disent.dataset import DisentDataset
from disent.dataset.data import XYObjectData, UnlockData

#from disent.dataset.sampling import SingleSampler,GroundTruthPairOrigSampler
from disent.dataset.transform import ToImgTensorF32
from disent.frameworks.vae import BetaVae, AdaVae
from disent.metrics import metric_dci
from disent.metrics import metric_mig
from disent.model import AutoEncoder
from disent.model.ae import DecoderConv64
from disent.model.ae import EncoderConv64
from disent.schedule import CyclicSchedule

from disent.model.ae import DecoderLinear
from disent.model.ae import EncoderLinear

#from _groundtruth__unlockobject import UnlockData

#from _groundtruth__pair_orig import GroundTruthPairOrigSampler, RlSampler
from disent.dataset.sampling import GroundTruthPairSampler,GroundTruthPairOrigSampler, GroundTruthPairOrigSamplerUnlock

import itertools
import time

# from _groundtruth__trajectories import EpisodeData

import torch.distributed as dist

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def train_model(lr, batch_size, z_size, steps):
    print(f"Training with lr={lr}, batch_size={batch_size}, z_size={z_size}, steps={steps}")
    start_time = time.time()

    #===============================================
    #   TRAINING USING NORMAL SAMPLER
    #===============================================
    # data=XYObjectData()
    # #data=UnlockData()
    # dataset = DisentDataset(dataset=data, sampler=GroundTruthPairOrigSampler(), transform=ToImgTensorF32())
    # dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    #===============================================
    #   TRAINING USING RL SAMPLER
    #===============================================
    # data = EpisodeData(n=60)
    # dataset= DisentDataset(dataset=data, sampler=RlSampler(), transform=ToImgTensorF32())
    # dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    # print('RL sampler, untrained trajectories, n=60')

    #===============================================
    #   TRAINING USING DV3 encodings
    #===============================================
    data=UnlockData()
    dataset = DisentDataset(dataset=data, sampler=GroundTruthPairOrigSamplerUnlock(), transform=ToImgTensorF32())
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)


    data_x_shape=(3,64,64)
    model = AutoEncoder(
        encoder=EncoderConv64(x_shape=data_x_shape, z_size=z_size, z_multiplier=2),
        decoder=DecoderConv64(x_shape=data_x_shape, z_size=z_size),
    )
    
    framework = AdaVae(
        model=model,
        cfg=AdaVae.cfg(
            optimizer="adam",
            optimizer_kwargs=dict(lr=lr),
            loss_reduction="mean_sum",
            beta=0.00316,
            ada_average_mode="gvae",
            ada_thresh_mode="kl",
        ),
    )
    
    trainer = L.Trainer(max_steps=steps, logger=False)
    trainer.fit(framework, dataloader)

    trainer.save_checkpoint("trained_adag_0807.ckpt")
    
   
    
    # # Save model
    model_path = f"model_lr{lr}_bs{batch_size}_z{z_size}_steps{steps}.pth"
    torch.save(model.state_dict(), model_path)
    end_time = time.time()  # ⏱️ End timing
    elapsed_time = end_time - start_time
    get_repr = lambda x: framework.encode(x.to(framework.device))
    metrics = {
        **metric_dci(dataset, get_repr, num_train=1000, num_test=500, show_progress=True),
    }
    print(f"Training with lr={lr}, batch_size={batch_size}, z_size={z_size}, steps={steps}")
    print(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes).")
    print(metrics)
    print('unlock dataset, orig sampler')
    return metrics
    
def write_metrics(metric,lr,batch_size,z_size,steps,description):
        
    # New data to add
    if not is_main_process():
        return  # only main writes

    print("[INFO] Writing metrics to CSV")

    new_row = {'inf train':metric['dci.informativeness_train'],'inf test':metric['dci.informativeness_test'],
                'disentanglment': metric['dci.disentanglement'],'completenes':metric['dci.completeness'],
                'Learning rate': lr, 'Batch size': batch_size, 'Latent size': z_size, 'Other':' ',
                'Max steps': steps, 'Run description':description}
    # File path
    csv_file = 'dci_values.csv'
    # Append row
    df = pd.DataFrame([new_row])
    # Append without writing the header again
    df.to_csv(csv_file, mode='a', header=not os.path.exists(csv_file), index=False)




# train_model(0.0001, 4, 6, 1000)
#lr=1e-4 batch size=64 latent size=50 max steps=50 000
# metrics=train_model(lr=0.0001, batch_size=64, z_size=50, steps=50 )
# write_metrics(metrics,0.0001,64,50,50,'normal sampler unlock data')


if __name__ == '__main__':
    metrics = train_model(lr=1e-4, batch_size=139, z_size=99, steps=60000)
    write_metrics(metrics, lr=0.0001, batch_size=139, z_size=99, steps=60000, description='unlock data, orig sampler, beta=0.00316')

"""
number: 60  
value: 0.8552421776927907  
datetime_start: 2025-07-30 22:18:48.473244  
datetime_complete: 2025-07-30 23:18:35.984626  
duration: 0 days 00:59:47.511382  
params_batch_size: 139  
params_beta: 0.00316  
params_latent_size: 99  
state: COMPLETE
"""