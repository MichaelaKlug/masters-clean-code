from __future__ import annotations

# from disent.disent.dataset.data._groundtruth__xysquares import XYSingleSquareData
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
from disent.dataset.data import XYObjectData, UnlockData, UnlockDataDV3,RlUnlockData,XYSingleSquareDataOrig,XYSingleSquareData

#from disent.dataset.sampling import SingleSampler,GroundTruthPairOrigSampler
from disent.dataset.transform import ToImgTensorF32
from disent.frameworks.vae import BetaVae, AdaVae
from disent.metrics import metric_dci
from disent.metrics import metric_mig
from disent.model import AutoEncoder
from disent.model.ae import DecoderConv64,DecoderConv32
from disent.model.ae import EncoderConv64,EncoderConv32
from disent.schedule import CyclicSchedule

from disent.model.ae import DecoderLinear
from disent.model.ae import EncoderLinear

#from _groundtruth__unlockobject import UnlockData

#from _groundtruth__pair_orig import GroundTruthPairOrigSampler, RlSampler
from disent.dataset.sampling import GroundTruthPairSampler,GroundTruthPairOrigSampler, GroundTruthPairOrigSamplerUnlock, RlSampler

import itertools
import time

# from _groundtruth__trajectories import EpisodeData

import torch.distributed as dist

def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def train_model(lr, batch_size, z_size, steps,beta, num_steps=0):
    print(f"Training with lr={lr}, batch_size={batch_size}, z_size={z_size}, steps={steps}")
    start_time = time.time()

    #===============================================
    #   TRAINING USING NORMAL SAMPLER
    #===============================================
    # data=XYObjectData()
    # data=UnlockData()
    # data=XYSingleSquareDataOrig(grid_spacing=4)
    data=XYSingleSquareData(grid_spacing=4)
    #data=RlUnlockData(n=num_steps)
    dataset = DisentDataset(dataset=data, sampler=GroundTruthPairOrigSampler(), transform=ToImgTensorF32())
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

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
    # data=UnlockDataDV3(latent_shape="2D")
    # dataset = DisentDataset(dataset=data, sampler=GroundTruthPairOrigSamplerUnlock(), transform=ToImgTensorF32())
    # dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)


    # data_x_shape=(3,64,64)
    #data_x_shape=(1,32,32)
    model = AutoEncoder(
        encoder=EncoderConv64(x_shape=data.x_shape, z_size=z_size, z_multiplier=2),
        decoder=DecoderConv64(x_shape=data.x_shape, z_size=z_size),
    )
    
    framework = AdaVae(
        model=model,
        cfg=AdaVae.cfg(
            optimizer="adam",
            optimizer_kwargs=dict(lr=lr),
            loss_reduction="mean_sum",
            beta=beta,
            ada_average_mode="gvae",
            ada_thresh_mode="kl",
        ),
    )
    
    trainer = L.Trainer(max_steps=steps, logger=False,enable_checkpointing=False)
    trainer.fit(framework, dataloader)

    # trainer.save_checkpoint("trained_adag_0807.ckpt")
    
   
    
    # # Save model
    # model_path = f"model_lr{lr}_bs{batch_size}_z{z_size}_steps{steps}_beta{beta}_1808.pth"
    # torch.save(model.state_dict(), model_path)
    # trainer.save_checkpoint(model_path)
    end_time = time.time()  # ⏱️ End timing
    elapsed_time = end_time - start_time
    get_repr = lambda x: framework.encode(x.to(framework.device))
    metrics = {
        **metric_dci(dataset, get_repr, num_train=1000, num_test=500, show_progress=True),
    }
    print(f"Training with lr={lr}, batch_size={batch_size}, z_size={z_size}, steps={steps}")
    print(f"Training completed in {elapsed_time:.2f} seconds ({elapsed_time / 60:.2f} minutes).")
    print(metrics)
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
    
    # steps=[0,1,20,40,60,80,100,110]
    # for i in steps:
    #     metrics = train_model(lr=0.0001, batch_size=64, z_size=96, steps=60000,beta=0.001,num_steps=i)
    #     write_metrics(metrics, lr=0.0001, batch_size=64, z_size=96, steps=60000, description=f'unlock data, rl sampler steps={i}, beta=0.001')
    metrics = train_model(lr=0.0001, batch_size=4, z_size=6, steps=57600,beta=0.01,num_steps=0)
    write_metrics(metrics, lr=0.0001, batch_size=4, z_size=6, steps=57600, description=f'xy single square, orig sampler, mimic orig data beta=0.01')
    # metrics = train_model(lr=0.0001, batch_size=64, z_size=20, steps=60000,beta=0.001,num_steps=0)
    # write_metrics(metrics, lr=0.0001, batch_size=64, z_size=20, steps=60000, description=f'unlock data, orig sampler, beta=0.001')

    

"""
DV3 2D latent space as image
number: 41
value: 0.5532996499981268
datetime_start: 2025-08-10 02:06:42.920916
datetime_complete: 2025-08-10 03:03:23.823959
duration: 0 days 00:56:40.903043
params_batch_size: 64
params_beta: 0.001
params_latent_size: 96
params_max_steps: 136899
state: COMPLETE

"""

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

"""
number: 54
value: 0.8173209286995717
datetime_start: 2025-08-10 10:35:13.209652
datetime_complete: 2025-08-10 11:37:48.441556
duration: 0 days 01:02:35.231904
params_batch_size: 153
params_beta: 0.001
params_latent_size: 90
state: COMPLETE

"""

