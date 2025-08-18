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
from disent.dataset.data import XYObjectData, UnlockDataDV3
from disent.dataset.sampling import SingleSampler
from disent.dataset.transform import ToImgTensorF32
from disent.frameworks.vae import BetaVae, AdaVae
from disent.metrics import metric_dci
from disent.metrics import metric_mig
from disent.model import AutoEncoder
from disent.model.ae import DecoderConv32
from disent.model.ae import EncoderConv32
from disent.schedule import CyclicSchedule

from disent.model.ae import DecoderLinear, EncoderConv64,DecoderConv64, EncoderConv32,DecoderConv32
from disent.model.ae import EncoderLinear

from disent.model.ae import DecoderLinear1d
from disent.model.ae import EncoderLinear1d

# from _groundtruth__unlock_dv3 import UnlockData

#from _groundtruth__pair_orig import GroundTruthPairOrigSampler
from disent.dataset.sampling import GroundTruthPairSampler,GroundTruthPairOrigSampler,GroundTruthPairOrigSamplerUnlock

import itertools


import argparse
import functools
import os
import pathlib
import sys
import ale_py
import ale_py.roms as roms
import json


os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml
from ruamel.yaml import YAML

# sys.path.append(str(pathlib.Path(__file__).parent))

# dreamer_path = '/home/michaela/minigrid/dreamerv3-torch'
# if dreamer_path not in sys.path:
#     sys.path.append(dreamer_path)
import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd

from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, RGBImgObsWrapper

import matplotlib.pyplot as plt


from disent.model.ae import DecoderConv32
from disent.model.ae import EncoderConv32
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
import time
import torch.distributed as dist
import pandas as pd
import os

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


class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)
        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                # print('so last mistake is here ', next(self._dataset)['image'].shape)
                self._train(next(self._dataset))
                self._update_count += 1
                self._metrics["update_count"] = self._update_count
            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(name, float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video("train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        """
        The training sequence of images xt is encoded using the CNN.
        The RSSM uses a sequence of deterministic recurrent states ht. At each step, it computes a posterior
        stochastic state zt that incorporates information about the current image xt, as well as a prior
        stochastic state ˆzt that tries to predict the posterior without access to the current image

        """
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"]) #returns post and prior --> latent is post
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        post, context, mets = self._wm._train(data)
        metrics.update(mets)
        start = post
        reward = lambda f, s, a: self._wm.heads["reward"](
            self._wm.dynamics.get_feat(s)
        ).mode()
        metrics.update(self._task_behavior._train(start, reward)[-1])
        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})
        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

    """
    Function to get representation of image: pass in obs, get out embedded rep
    """
    def _get_rep(self,obs):
        
        latent = action = None
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)

        # are_identical = torch.equal(embed[0], embed[1])
        # print("Rows are identical:", are_identical)
   

        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])

    
        # indices = torch.argmax(latent['stoch'], dim=1)
        # with open("trained_outputs.txt", "a") as f:  # Use "a" to append instead of overwriting
        #     f.write(",".join(map(str, indices[0].tolist())) + "\n")
        
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]

        feat = self._wm.dynamics.get_feat(latent)
        
        # return embed,feat
        return torch.argmax(latent['stoch'], dim=1)


def save_study_progress(study, save_dir):
    """Saves the study results to a CSV file and the study object to a pickle file."""
    df = study.trials_dataframe()
    df.to_csv(os.path.join(save_dir, 'optuna_results_1d_latent.csv'), index=False)
    joblib.dump(study, os.path.join(save_dir, 'optuna_study_1d_latent.pkl'))


class SimpleEnv(MiniGridEnv):
    def __init__(
        self,
        size=11,
        agent_start_pos=(2,2),
        agent_start_dir=0,
        key_pos=(3,3),
        door_pos=(5,2),
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.key_pos=key_pos
        self.door_pos=door_pos

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            # Set this to True for maximum speed
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        # width=11
        height=6
        
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Generate vertical separation wall
        for i in range(0, height):
            self.grid.set(5, i, Wall())
        
        # Place the door and key
        self.grid.set(self.door_pos[0], self.door_pos[1], Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(self.key_pos[0], self.key_pos[1], Key(COLOR_NAMES[0]))

        # Place a goal square in the bottom-right corner
        # self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "grand mission"


def objective(trial):
    #set the hyperparameters to optimize
    betas= [0.0001,0.001,0.01, 0.1,0.000316, 0.00316,0.0316,0.316, 1]
    latent_size=trial.suggest_int('latent_size',20,100)
    batchsize=trial.suggest_int('batch_size',64,256)
    #learning_rate=trial.suggest_float('learning_rate',1e-8,1e-4,log=True)
    learning_rate=1e-4
    maximum_steps=trial.suggest_int('max_steps', 100000,150000)
    beta=trial.suggest_categorical('beta', betas)

    # data=UnlockData()
    # dataset = DisentDataset(dataset=data, sampler=GroundTruthPairOrigSampler(), transform=None)
    # dataloader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True, num_workers=0)

    data=UnlockDataDV3(latent_shape='1D')
    dataset = DisentDataset(dataset=data, sampler=GroundTruthPairOrigSamplerUnlock(), transform=None)
    dataloader = DataLoader(dataset=dataset, batch_size=batchsize, shuffle=True, num_workers=0)
    data_x_shape=(1,32)

    #create the model with trial hyperparameters
    model = AutoEncoder(
        encoder=EncoderLinear1d(x_shape=data_x_shape, z_size=latent_size, z_multiplier=2),
        decoder=DecoderLinear1d(x_shape=data_x_shape, z_size=latent_size),
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
    #logger = TensorBoardLogger(save_dir=os.getcwd(), name=f"optuna_logs/trial_{trial.number}")
    
    trainer = L.Trainer(max_steps=60000, devices=1)
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

    study_file = os.path.join(directory, 'optuna_study_dv3_2Dimg.pkl')
    if os.path.exists(study_file):
        print("Study file found. Resuming the study.")
        study = joblib.load(os.path.join(directory, 'optuna_study_dv3_2Dimg.pkl'))
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


