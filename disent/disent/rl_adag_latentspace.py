from disent.frameworks.vae import AdaVae
from disent.model import AutoEncoder
from disent.model.ae import DecoderConv64
from disent.model.ae import EncoderConv64
import gym
import numpy as np
import gymnasium as gym
from stable_baselines3 import DQN
import random 
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper, RGBImgObsWrapper
import cv2
from gymnasium.wrappers import TransformObservation
from gymnasium import spaces
import torch
from minigrid.wrappers import RGBImgObsWrapper, ImgObsWrapper
import matplotlib.pyplot as plt
import torch.nn.functional as F
from stable_baselines3.common.callbacks import BaseCallback
from typing import Any, Dict

import gymnasium as gym
import torch as th
import numpy as np

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class VideoRecorderCallback(BaseCallback):
    def __init__(self, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is recorded
        :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                """
                Renders the environment in its current state, recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of the callback's scope
                :param _globals: A dictionary containing all global variables of the callback's scope
                """
                # We expect `render()` to return a uint8 array with values in [0, 255] or a float array
                # with values in [0, 1], as described in
                # https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_video
                screen = self._eval_env.render()
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            self.logger.record(
                "trajectory/video",
                Video(th.from_numpy(np.asarray([screens])), fps=40),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True


# Custom wrapper: replaces observations with latent encodings from the encoder
class LatentObsWrapper(gym.ObservationWrapper):
    def __init__(self, env, encoder, z_size):
        super().__init__(env)
        self.encoder = encoder.eval()  # Ensure it's in eval mode
        self.z_size = z_size
        
        # New observation space becomes latent vector
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(z_size,), dtype=np.float32)
        # new_obs_space = spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8)
        # env = TransformObservation(env, lambda obs: cv2.resize(obs, (64, 64), interpolation=cv2.INTER_LINEAR),new_obs_space)


    def observation(self, obs):
        #plt.imsave('obs.png',obs)
        with torch.no_grad():
            # Convert NumPy array to float tensor and add batch dim
            img = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # (1, H, W, C) or (1, C, H, W)

            # If needed, permute to (B, C, H, W)
            if img.shape[-1] == 3:
                img = img.permute(0, 3, 1, 2)  # from (B, H, W, C) to (B, C, H, W)

            # Normalize to [0, 1] if needed (only if encoder was trained on normalized data)
            img = img / 255.0

            # Resize to (64, 64)
            img = F.interpolate(img, size=(64, 64), mode='bilinear', align_corners=False)

            # Encode
            img = torch.tensor(img, dtype=torch.float32).to(device)  # move to same device
            z = self.encoder(img)[0]  # (1, z_dim)
            
        return z.squeeze(0).cpu().numpy().astype(np.float32)


data_x_shape=(3,64,64)
# Recreate the same model you trained with
model = AutoEncoder(
        encoder=EncoderConv64(x_shape=data_x_shape, z_size=50, z_multiplier=2),
        decoder=DecoderConv64(x_shape=data_x_shape, z_size=50),
    )

# Recreate the same cfg used in training
cfg = AdaVae.cfg(
    optimizer="adam",
    optimizer_kwargs=dict(lr=1e-4),
    loss_reduction="mean_sum",
    beta=4,
    ada_average_mode="gvae",
    ada_thresh_mode="kl",
)

# Load the trained framework from checkpoint
# framework = AdaVae.load_from_checkpoint(
#     checkpoint_path="trained_models/trained_adag_2606.ckpt",
#     model=model,
#     cfg=cfg,
# )

# encoder = framework._model._encoder

# Use MiniGrid env and wrap it to return only image observations
env = gym.make("MiniGrid-Unlock-v0", render_mode='rgb_array')
env = RGBImgObsWrapper(env)    # add RGB image to obs
env = ImgObsWrapper(env)       # only return image

# Apply your custom latent wrapper
z_size = 50


# encoder = encoder.eval().to(device)  # make sure it's on CUDA if available

#wrapped_env = LatentObsWrapper(env, encoder=encoder, z_size=z_size)

model = DQN("CnnPolicy", env, verbose=1,tensorboard_log="./dqn_logs/")
video_recorder = VideoRecorderCallback(env, render_freq=10000)
model.learn(total_timesteps=8000000,tb_log_name="unlock_8000000",callback=video_recorder)
model.save("unlock_8000000")

# del model # remove to demonstrate saving and loading
# model = DQN.load("unlock_adag_latents")

# # model = DQN("MlpPolicy", env, verbose=1)
# # model.learn(total_timesteps=100000, log_interval=4)
# obs, info = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     plt.imsave('unlocktrain.png',obs)
#     if terminated or truncated:
#         obs, info = env.reset()



# obs, info = wrapped_env.reset()

# print("Latent observation shape:", obs.shape)
# print("Latent vector:", obs)

# # Take a few steps and inspect outputs
# for step in range(5):
#     action = wrapped_env.action_space.sample()
#     obs, reward, terminated, truncated, info = wrapped_env.step(action)

#     print(f"\nStep {step + 1}")
#     print("Latent observation shape:", obs.shape)
#     print("Latent vector sample:", obs[:5])  # print first 5 elements of z
#     print("Reward:", reward)

#     if terminated or truncated:
#         obs, info = wrapped_env.reset()