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
from typing import Optional,Tuple,List

import random
import warnings
from typing import Optional
from typing import Tuple

import numpy as np

from disent.dataset.data._groundtruth import GroundTruthData

from torch.utils.data import DataLoader, Dataset

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
import os
import numpy as np
import pickle
from disent.dataset.data._groundtruth import GroundTruthData

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



class RlUnlockData(GroundTruthData):

    name = "rl_unlock_object"
    factor_names = ("agent_x", "agent_y", "direction", "door_y", "key_x", "key_y")

    #=========================================================
    #   CHECK WHAT THIS AFFECTS
    #=========================================================
    @property
    def factor_sizes(self) -> Tuple[int, ...]:
        #agent_x, agent_y, agent_dir, 
        return (4,4,4,4,4,4)

    @property
    def img_shape(self) -> Tuple[int, ...]:
        return self._width, self._width, 3

    def __init__(
        self,
        transform=None,
        n=1
    ):
        # print('using trained trajectories')
        # self._trajectories = np.load("trained_minigrid_trajectories.npy", allow_pickle=True)

        

        self.n=n
        self._width = 64
        file_dir = os.path.dirname(__file__)
        file_path = os.path.join(file_dir, "image_dict_overlapping.pkl")
        self.accum= np.zeros((64, 64), dtype=np.float32)
        print(file_path)
        with open(file_path, "rb") as f:
            image_store = pickle.load(f)
        self._obs_dictionary= image_store
        super().__init__(transform=transform)
    def hamming_delta(self,f0, f1):
        f0, f1 = np.asarray(f0), np.asarray(f1)
        return int(np.sum(f0 != f1))
   
    def _get_observation(self, idx):

        #=============================================================
        #   SAMPLE A STATE GIVEN A SET OF VALUES FOR THE FACTORS
        #   THEN EITHER CHANGE DIRECTION IN WHICH AGENT IS FACING
        #   OR MAKE AGENT TAKE A STEP FORWARD ---> THEREFORE THERE 
        #   ARE 3 POSSIBLE ACTIONS TO CHOOSE FROM: TURN RIGHT,
        #   TURN LEFT AND MOVE FORWARD
        #   --> CHOOSE N RANDOM ACTIONS GIVEN N-STEPS AND RETURN
        #   NEW STATE WHEN IDX<0
        #=============================================================
        
        #(agent_x,agent_y,agent_dir,door_pos,key_x,key_y)
        #actions: 0=turn left, 1=turn right, 2=move forward
        num_factors=len(self.factor_sizes)
        if isinstance(idx,tuple):
            idx=idx[0]
        orig_list=self.idx_to_pos(abs(idx))
        modified_lst = [x + 1 if i != 2 else x for i, x in enumerate(orig_list)]
        first_sample=modified_lst
        valid_traj=False #valid trajectory means that the final state mudt differ to original state by AT MOST d-1 factors (i.e. not all factors changed)
        if idx < 0:
            while valid_traj==False:

                door = modified_lst[3]
                keyx = modified_lst[4]
                keyy = modified_lst[5]
                for i in range(self.n):
                    agent_x, agent_y, agent_dir = modified_lst[0], modified_lst[1], modified_lst[2]
                    new_x, new_y, new_dir = agent_x, agent_y, agent_dir
                    
                    possible_moves=[0,1]
                    if agent_dir == 0:
                        if (agent_x+1)<=4 and not((agent_x+1)==keyx and agent_y==keyy):
                            possible_moves.append((2,agent_x+1,agent_y))
                    elif agent_dir == 1:
                        if (agent_y+1)<=4 and not((agent_y+1)==keyy and agent_x==keyx):
                            possible_moves.append((2,agent_x,agent_y+1))
                    elif agent_dir == 2:
                        if (agent_x-1)>0 and not((agent_x-1)==keyx and agent_y==keyy):
                            possible_moves.append((2,agent_x-1,agent_y))
                    elif agent_dir == 3:
                        if (agent_y-1)>0 and not((agent_y-1)==keyy and agent_x==keyx):
                            possible_moves.append((2,agent_x,agent_y-1))
                    # Randomly choose action: 0 = left turn, 1 = right turn, 2 = move forward
                    choice = random.choice(possible_moves)
                    #print('choice is ', choice)
                    if choice == 0:
                        new_dir = (agent_dir - 1) % 4
                    elif choice == 1:
                        new_dir = (agent_dir + 1) % 4
                    else:
                        new_x=choice[1]
                        new_y=choice[2]
                    modified_lst=(new_x,new_y,new_dir,door,keyx,keyy)
                if self.hamming_delta(first_sample, modified_lst) <= num_factors - 1:
                    valid_traj=True

        #print('next step ', modified_lst,'\n')
        
        factors=tuple(modified_lst)
        obs=self._obs_dictionary[factors]
        returned_state=obs
        gray_img = obs[:, :, 0]   # Convert to grayscale: [B, H, W]
        self.accum +=gray_img.astype(np.float32)
       

    
        return returned_state






