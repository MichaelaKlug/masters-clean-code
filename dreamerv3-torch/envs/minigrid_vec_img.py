import gym
import numpy as np
import gymnasium as gym
import random 
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, RGBImgObsWrapper
import cv2
from gymnasium.wrappers import TransformObservation
from gymnasium import spaces
import torch

class Minigrid:
    metadata = {}

    def __init__(self,encoder, task, size=(64, 64), seed=0):

        self._env = gym.make("MiniGrid-Unlock")
        self._env = RGBImgObsWrapper(self._env)
        self._env = ImgObsWrapper(self._env)

        self._encoder=encoder
        # Define observation space explicitly
        new_obs_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        self._env = TransformObservation(self._env, lambda obs: cv2.resize(obs, (64, 64), interpolation=cv2.INTER_LINEAR),new_obs_space)
        self.reward_range = [-np.inf, np.inf]

    @property
    def observation_space(self):
        spaces = {
            "vector": gym.spaces.Box(-np.inf, np.inf, (50,), dtype=np.float32),  # Replace LATENT_DIM with your actual latent size,
            "image": gym.spaces.Box(
                0, 255, self._env.observation_space.shape, dtype=np.uint8
            ),
            "is_first": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
            "is_last": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
            "is_terminal": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
            "log_reward": gym.spaces.Box(0, 1, (1,), dtype=np.uint8),
        }
       
        return gym.spaces.Dict(spaces)

    @property
    def action_space(self):
        action_space = self._env.action_space
        action_space.discrete = True
        return action_space

    def step(self, action):
        action = np.argmax(action) #had to add this in- dreamer file converts actions to one hot encoding- undo here for minigrid
        image, reward, done, truncated, info = self._env.step(action)
        
        #-------------------------------------------------------------------------------------
        #   TAKE IMAGE OBS FROM MINIGRID ENV --> PASS THROUGH THE ADA-GVAE TRAINED ENCODER
        #   TAKE THAT OUTPUT AND FEED INTO DV3 ENCODER
        #   DV3 ENCODER SHOULD RECEIVE BOTH AN IMAGE KEY AND AN MLP KEY
        #   SO THAT IT TAKES BOTH THE OBS AND ITS ADAG ENCODING AS INPUT
        #-------------------------------------------------------------------------------------
        

        # image --> encode with AdaVAE first
        image_tensor = torch.tensor(image.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        ada_encoder=self._encoder
        device = next(ada_encoder.parameters()).device
        image_tensor = image_tensor.to(device)  # device = encoder's device
        with torch.no_grad():
            latent = ada_encoder(image_tensor)  # shape: [1, latent_dim]
            latent = latent[0].squeeze(0).cpu().numpy()
      
        if "discount" not in info:
            info["discount"] = np.array(1.0, dtype=np.float32)  # Default discount

        reward = np.float32(reward) #WOULD I NEED TO CHANGE THIS
        obs = {
            "vector": latent.astype(np.float32),   # MLP input
            "image": image, #CNN input
            "is_first": False,
            "is_last": done,
            "is_terminal": info["discount"] == 0,
        }
        return obs, reward, done, info

    def render(self):
        return self._env.render()

    def reset(self):
        image, info = self._env.reset() 
        image_tensor = torch.tensor(image.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        ada_encoder=self._encoder
        device = next(ada_encoder.parameters()).device
        image_tensor = image_tensor.to(device)  # device = encoder's device
        with torch.no_grad():
            latent = ada_encoder(image_tensor)  # shape: [1, latent_dim]
            latent = latent[0].squeeze(0).cpu().numpy() #latent[0] taking only the mus (means) not variance

        obs = {
            "vector": latent.astype(np.float32),
            "image": image,
            "is_first": True,
            "is_last": False,
            "is_terminal": False,
        }
        return obs
