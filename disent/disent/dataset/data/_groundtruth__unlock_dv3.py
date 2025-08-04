#THIS DATA CLASS IS AUGMENTED SUCH THAT WHEN AN OBSERVATION IS SAMPLED, IT IS
#FIRST PASSED THROUGH THE DV3 ENCODER TO GET A LATENT REPRESENTATION
#THAT REPRESENTATION IS THEN RESHAPED AS NECESSARY TO BE THE INPUT IMAGE
#THAT 'IMAGE' IS THEN PASSED AS OBSERVATION TO ADA-GVAE MODEL


from __future__ import annotations

import torchvision.transforms.functional as F_tv

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
import json
import os
import warnings
from typing import Optional
from typing import Tuple

import numpy as np
import torch.nn.functional as F

from disent.dataset.data._groundtruth import GroundTruthData
import torch
import pickle


# ========================================================================= #
# WANT A DATASET THAT GENERATES ALL POSSIBLE PERMUTATIONS OF THE UNLOCK     #
# MINIGRID ENVIRONMENT                                                      #
# ========================================================================= #
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


class UnlockDataDV3(GroundTruthData):

    name = "unlock_object_dv3"

    factor_names = ("agent_x", "agent_y", "direction", "door_y", "key_x", "key_y")

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        #agent_x, agent_y, agent_dir, 
        return (self._agent_xs,self._agent_ys,self._agent_dirs,self._door_ys,self._key_xs,self._key_ys)

    @property
    def img_shape(self) -> Tuple[int, ...]:
        return self._width, self._width, (3 if self._rgb else 1)

    def __init__(
        self,
        grid_size: int = 64,
        agent_xs: int = 4,
        agent_ys: int = 4,
        agent_dirs: int = 4,
        door_ys: int = 4,
        key_xs: int = 4,
        key_ys: int = 4,
        transform=None,
        latent_shape="2D" #shape of the latent representation, either 1D or 2D
    ):
     
    
        
        # image sizes
        self._width = grid_size
        self._agent_xs = agent_xs
        self._agent_ys= agent_ys
        self._agent_dirs = agent_dirs
        self._door_ys= door_ys
        self._key_xs = key_xs
        self._key_ys= key_ys
        self.latent_shape=latent_shape

        file_dir = os.path.dirname(__file__)
        file_path = os.path.join(file_dir, "latent_reps.pkl")
        with open(file_path, "rb") as f:
            latents = pickle.load(f)
                        
        # with open("image_dict.pkl", "rb") as f:
        #     image_store = pickle.load(f)
        self._obs_dictionary= latents
       


        # print('value of dreamer is ', self._dreamer)


        super().__init__(transform=transform)

    def _get_observation(self, idx):
        #idx=idx[0]
        if isinstance(idx, tuple):
            # if we have a pair, we need to get the first element of the pair
            idx=idx[0]
        orig_list=self.idx_to_pos(idx)
        modified_lst = [x + 1 if i != 2 else x for i, x in enumerate(orig_list)]

        
        factors=tuple(modified_lst)

        #---------------------------------------------------------------
        #   PASS OBS THROUGH DV3 ENCODER TO GET LATENT REP
        #   LOAD IN RANDOM OBS FROM TEXT FILE TO GET CORRECT
        #   STATE INPUT
        #   THEN SET OBS IMAGE OF THAT OBS TO EQUAL OUR DESIRED OBS
        #---------------------------------------------------------------
        model_output=self._obs_dictionary[factors]
        latent_rep=None

        #======================================================
        #   TO GET 1D VECTOR OF INDICES OF LATENT SPACE
        #=====================================================
        if self.latent_shape=="1D":
            #latent_rep=self._obs_dictionary[factors].unsqueeze(0).float()
            # latent_rep= model_output.unsqueeze(0).float()
            latent_rep = torch.tensor(model_output).unsqueeze(0).float()



        #==========================================================================
        #   TO GET 2D ONE HOT ENCODED VECTOR OF LATENT SPACE
        #==========================================================================
        #latent_rep=self._obs_dictionary[factors]

        elif self.latent_shape=="2D":
            latent_rep_out = F.one_hot(torch.tensor(model_output), num_classes=32).float()
            latent_rep = F_tv.to_pil_image(latent_rep_out) #transform needs image as PIL image not tensor
            
        #If we want the image to have three channels
        # np_array = one_hot.detach().cpu().numpy()
        # rgb_img = np.stack([np_array]*3, axis=-1)

        return latent_rep



