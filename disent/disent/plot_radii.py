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
from disent.dataset.data import XYObjectData,XYSingleSquareData, RlUnlockData, UnlockData,UnlockDataDV3,XYSingleSquareDataOrig
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

    
def test_model():
    #data=XYSquaresData(grid_spacing=3)
    # data = XYSingleSquareData(grid_spacing=4,n=1)
    data=XYSingleSquareData(grid_spacing=4,n=80)
    index=(112,'first')
    sampler = GroundTruthPairOrigSampler()
    # sampler=RlSampler()
    sampler.init(data)
    data._get_observation(index)
    for i in range(1000):
        print('index is ', index)
        end_pos_index=sampler._sample_idx(index[0]) #first returned is our start index, we want the sampled pair
        print('end pos index is ', end_pos_index)
        data._get_observation(end_pos_index[1])

    plt.figure()
    plt.imshow(data.accum_pair, cmap='hot')
    plt.colorbar()
    plt.savefig(f'radius_plots/orig_sampler_pair_sampling_center_2408.png')
    # plt.savefig(f'radius_plots/rl_sampler_pair_sampling_center_n80_fixedsampler.png')
    plt.close()
    
def test_model_origdata():
    #data=XYSquaresData(grid_spacing=3)
    # data = XYSingleSquareData(grid_spacing=4,n=1)
    data=XYSingleSquareDataOrig(grid_spacing=4)
    index=(112,'first')
    sampler = GroundTruthPairOrigSampler()
    # sampler=RlSampler()
    sampler.init(data)
    data._get_observation(index[0])
    for i in range(1000):
        end_pos_index=sampler._sample_idx(index[0]) #first returned is our start index, we want the sampled pair
        data._get_observation(end_pos_index[1][0])

    plt.figure()
    plt.imshow(data.accum, cmap='hot')
    plt.colorbar()
    plt.savefig(f'radius_plots/orig_sampler_orig_data_pair_sampling_center_2408.png')
    # plt.savefig(f'radius_plots/rl_sampler_pair_sampling_center_n80_fixedsampler.png')
    plt.close()

def test_model_rl():
    #data=XYSquaresData(grid_spacing=3)
    # data = XYSingleSquareData(grid_spacing=4,n=1)
    data=XYSingleSquareData(grid_spacing=4,n=100)
    index=(112,'first')
    sampler=RlSampler()
    sampler.init(data)
    data._get_observation(index)
    for i in range(1000):
        end_pos_index=sampler._sample_idx(index[0]) #first returned is our start index, we want the sampled pair
        data._get_observation(end_pos_index[1])

    plt.figure()
    plt.imshow(data.accum_pair, cmap='hot')
    plt.colorbar()
    plt.savefig(f'radius_plots/rl_sampler_center_n100_fixedsampler.png')
    plt.close()
  


# test_model()
# test_model_origdata()
test_model_rl()

