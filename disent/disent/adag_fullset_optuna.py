from __future__ import annotations
import matplotlib.pyplot as plt
import cv2
import numpy as np

import lightning as L
import torch
from torch.utils.data import DataLoader

from disent.dataset import DisentDataset
from disent.dataset.data import XYObjectData
from disent.dataset.sampling import SingleSampler
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

# from _groundtruth__unlockobject import UnlockData

# from _groundtruth__pair_orig import GroundTruthPairOrigSampler
from disent.dataset.sampling import GroundTruthPairSampler, GroundTruthPairOrigSamplerUnlockFullSet
from disent.dataset.data import UnlockData

import itertools

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

import optuna
from optuna.integration import PyTorchLightningPruningCallback
import joblib


def save_study_progress(study, save_dir):
    """Saves the study results to a CSV file and the study object to a pickle file."""
    df = study.trials_dataframe()
    df.to_csv(os.path.join(save_dir, 'optuna_results.csv'), index=False)
    joblib.dump(study, os.path.join(save_dir, 'optuna_study_fullset.pkl'))



def objective(trial):
    #set the hyperparameters to optimize
    betas= [0.0001,0.001,0.01, 0.1,0.000316, 0.00316,0.0316,0.316, 1]
    latent_size=trial.suggest_int('latent_size',50,120)
    batchsize=trial.suggest_int('batch_size',64,256)
    #learning_rate=trial.suggest_float('learning_rate',1e-8,1e-4,log=True)
    learning_rate=1e-4
    #maximum_steps=trial.suggest_int('max_steps', 100000,150000)
    max_steps=150000
    beta=trial.suggest_categorical('beta', betas)

    data=UnlockData()
    dataset = DisentDataset(dataset=data, sampler=GroundTruthPairOrigSamplerUnlockFullSet(), transform=ToImgTensorF32())
    dataloader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True, num_workers=0)
    data_x_shape=(3,64,64)

    #create the model with trial hyperparameters
    model = AutoEncoder(
        encoder=EncoderConv64(x_shape=data_x_shape, z_size=latent_size, z_multiplier=2),
        decoder=DecoderConv64(x_shape=data_x_shape, z_size=latent_size),
    )
    
    framework = AdaVae(
        model=model,
        cfg=AdaVae.cfg(
            optimizer="adam",
            optimizer_kwargs=dict(lr=learning_rate),
            loss_reduction="mean_sum",
            beta=beta,
            ada_average_mode="gvae",
            ada_thresh_mode="kl",
        ),
    )

    # Logger
    logger = TensorBoardLogger(save_dir=os.getcwd(), name=f"optuna_logs/trial_{trial.number}")
    
    trainer = L.Trainer(max_steps=max_steps, logger=logger, devices=1)
    trainer.fit(framework, dataloader)

    get_repr = lambda x: framework.encode(x.to(framework.device))
    metrics = {
        **metric_dci(dataset, get_repr, num_train=1000, num_test=500, show_progress=True),
    }
    
    return metrics['dci.disentanglement']



# Entry point
if __name__ == "__main__":

    """
    python optuna_study.py --dataset wsws_static/wsws_static_symbolic
      --directory Traces/wsws_static/wsws_static_symbolic --feature-name symbolic_obs --clusters 2 --layers "1087 512 40"
    """


    directory = "Traces"

    study_file = os.path.join(directory, 'optuna_study_fullset.pkl')
    if os.path.exists(study_file):
        print("Study file found. Resuming the study.")
        study = joblib.load(os.path.join(directory, 'optuna_study_fullset.pkl'))
    else:
        print("No study file found. Starting a new study.")
        study = optuna.create_study(direction="maximize")

    trials=100
  
    study.optimize(
        lambda trial: objective(trial),
        n_trials=trials,
        callbacks=[lambda study, trial: save_study_progress(study, directory)]
    )

    trial = study.best_trial
    with open("best_trial.txt", "w") as f:
        f.write("Best trial:\n")
        f.write(f"  Value: {trial.value}\n")
        f.write("  Params:\n")
        for key, value in trial.params.items():
            f.write(f"    {key}: {value}\n")

    save_study_progress(study, directory)


