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


import warnings
from typing import Optional
from typing import Tuple

import numpy as np
import pickle
import os
from disent.dataset.data._groundtruth import GroundTruthData


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


class UnlockData(GroundTruthData):

    # name = "unlock_object"
    name="fullset_data"

    # factor_names = ("agent_x", "agent_y", "direction", "door_y", "key_x", "key_y")
    # factor_names = ("agent_x", "agent_y", "direction")
    factor_names = ("agent_x", "agent_y", "direction", "door_y", "key_x", "key_y", "key_present", "door_open")

    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        #agent_x, agent_y, agent_dir, 
        # return (self._agent_xs,self._agent_ys,self._agent_dirs,self._door_ys,self._key_xs,self._key_ys)
        # return (self._agent_xs,self._agent_ys,self._agent_dirs)
        return (self._agent_xs,self._agent_ys,self._agent_dirs,self._door_ys,self._key_xs,self._key_ys,self._key_present,self._door_open)

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
        key_xs: int = 5,
        key_ys: int = 5,
        key_present: int = 2,
        door_open: int = 2,
        transform=None,
        obs_dictionary=None
        
    ):
     
    
        
        # image sizes
        self._width = grid_size
        self._agent_xs = agent_xs
        self._agent_ys= agent_ys
        self._agent_dirs = agent_dirs
        self._door_ys= door_ys
        self._key_xs = key_xs
        self._key_ys= key_ys
        self._key_present= key_present
        self._door_open= door_open

        file_dir = os.path.dirname(__file__)
        # file_path = os.path.join(file_dir, "image_dict.pkl")
        # file_path = os.path.join(file_dir, "image_dict_overlapping.pkl")
        file_path=os.path.join(file_dir, "full_feature_set.pkl")
        print(file_path)
        with open(file_path, "rb") as f:
            image_store = pickle.load(f)
        self._obs_dictionary= image_store
        self._obs_keys = list(self._obs_dictionary.keys())

        super().__init__(transform=transform)
        self.indices=[]
    def _get_observation(self, idx):
        if isinstance(idx, tuple):
            # if we have a pair, we need to get the first element of the pair
            idx=idx[0]
        if isinstance(idx,np.ndarray):
            idx=idx[0]

       
        key = self._obs_keys[abs(idx)]
        modified_lst= key
        # orig_list=self.idx_to_pos(idx)
        # modified_lst = [x + 1 if i != 2 else x for i, x in enumerate(orig_list)]
        # modified_lst = [x + 1 if i not in {2, 4, 5, 6, 7} else x for i, x in enumerate(orig_list)]

        # key = list(self._obs_dictionary.keys())[abs(idx)] 
        # modified_lst= key

        # key_present = not (modified_lst[4] == 0 and modified_lst[5] == 0)
        # modified_lst.insert(-1, int(key_present))  # Insert key_present before door_open
        factors=tuple(modified_lst)
        obs=self._obs_dictionary[factors]
        
        # agent_x,agent_y,agent_dir,door_y,key_x,key_y= modified_lst
        # env = SimpleEnv(agent_start_pos=(agent_x,agent_y),agent_start_dir=agent_dir,
        #                                         door_pos=(5,door_y),key_pos=(key_x ,key_y))
        # env = RGBImgObsWrapper(env)
        # env = ImgObsWrapper(env)
        # new_obs_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        # env = TransformObservation(env, lambda obs: cv2.resize(obs, (64, 64), interpolation=cv2.INTER_LINEAR),new_obs_space)
        # obs, info=env.reset()
        

        return obs



