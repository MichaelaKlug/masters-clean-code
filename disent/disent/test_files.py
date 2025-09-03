from __future__ import annotations

import numpy as np
import importlib.util

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

# ------------------------------------------------------------------
# Helper to load modules from file
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# Instantiate datasets
# ------------------------------------------------------------------
mod_ds = XYSingleSquareData(grid_spacing=4,n=0)
orig_ds = XYSingleSquareDataOrig(grid_spacing=4)

# ------------------------------------------------------------------
# Compare outputs for integer indices
# ------------------------------------------------------------------
print("=== Integer indices ===")
#add more random indices tuples
for idx in [(0,'first'), (14,'second'), (5,'first'), (76,'second'), (15,'first'), (20,'first'), (100,'second'), (220,'second'), (34,'first'), (7,'first')]:
    obs_mod = mod_ds._get_observation(idx)
    obs_orig = orig_ds._get_observation(idx)
    print(f"Index {idx}: identical={np.array_equal(obs_mod, obs_orig)} | sums: mod={obs_mod.sum()} orig={obs_orig.sum()}")

# ------------------------------------------------------------------
# Compare outputs for tuple indices
# ------------------------------------------------------------------
# print("\n=== Tuple indices ===")
# for idx in [(0,'second'), (5,), (10,)]:
#     obs_mod = mod_ds._get_observation(idx)
#     obs_orig = orig_ds._get_observation(idx)
#     print(f"Index {idx}: identical={np.array_equal(obs_mod, obs_orig)} | sums: mod={obs_mod.sum()} orig={obs_orig.sum()}")
