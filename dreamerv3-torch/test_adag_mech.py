import torch
from torch.distributions import Distribution
from torch.distributions import Normal
from torch.distributions import kl_divergence
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import Sequence
from typing import Tuple
import torch
from torch import nn
import torch.nn.functional as F
from torch import distributions as torchd

import tools

def get_dist(state, dtype=None):
    # This function converts the latent dictionary (e.g. with logit keys for categorical latents) into a PyTorch Distribution 
    # object like OneHotCategorical or a batch of Categorical distributions, depending on the model config.
    
    logit = state["logit"]
    dist = torchd.independent.Independent(
        tools.OneHotDist(logit, unimix_ratio=0.01), 1
    )

    return dist


def mix_logits(logits1, logits2, w=0.5):
    """
    Mix two logit vectors (logits1, logits2) in a numerically stable way.

    Args:
        logits1: Tensor of shape (..., K)
        logits2: Tensor of shape (..., K)
        w: Weighting scalar in [0, 1]

    Returns:
        Tensor of same shape as logits1/logits2 (..., K) representing the mixed logits
    """
    # 1) Convert logits to log-probabilities (log-softmax)
    logp1 = logits1 - torch.logsumexp(logits1, dim=-1, keepdim=True)
    logp2 = logits2 - torch.logsumexp(logits2, dim=-1, keepdim=True)

    # 2) Compute weighted log-probs
    log_w1 = torch.log(torch.tensor(w, dtype=logits1.dtype, device=logits1.device))
    log_w2 = torch.log(torch.tensor(1.0 - w, dtype=logits2.dtype, device=logits2.device))

    # 3) Stack and mix in log-space
    stacked = torch.stack([log_w1 + logp1, log_w2 + logp2], dim=0)  # shape (2, ..., K)
    mixed_logp = torch.logsumexp(stacked, dim=0)  # shape (..., K)

    return mixed_logp  # Can be treated as mixed logits

# Code from Nathan disent library : disent/frameworks/vae/_weaklysupervised__adavae.py
def hook_intercept_ds(
ds_posterior: Sequence[torch.distributions.Independent]
) -> Sequence[torch.distributions.Independent]:
    """
    Adaptive VAE Method, putting the various components together
        1. compute differences between representations
        2. estimate a threshold for differences
        3. compute a shared mask from this threshold
        4. average together elements that are marked as shared

    (x) Visual inspection against reference implementation:
        https://github.com/google-research/disentanglement_lib (aggregate_argmax)
    """
    # print('CALLING HOOK')
    # d0_posterior=ds_posterior['stoch'] #-----> make sure that post is either just the stoch or that we just take the post['stoch']
    # d1_posterior = ds_posterior #------> this will be the post['stoch'] of the second image which will be sampled from a buffer 
    d0_posterior, d1_posterior = ds_posterior
    # [1] symmetric KL Divergence FROM: https://openreview.net/pdf?id=8VXvj1QNRl1
    z_deltas = 0.5 * kl_divergence(d1_posterior, d0_posterior) + 0.5 * kl_divergence(d0_posterior, d1_posterior)

    # [2] estimate threshold from deltas
    z_deltas_min = z_deltas.min(axis=1, keepdim=True).values  # (B, 1)
    z_deltas_max = z_deltas.max(axis=1, keepdim=True).values  # (B, 1)
    z_thresh = 0.5 * z_deltas_min + 0.5 * z_deltas_max  # (B, 1)

    # [3] shared elements that need to be averaged, computed per pair in the batch
    share_mask = z_deltas < z_thresh  # broadcast (B, Z) and (B, 1) to get (B, Z)

    # [4.a] compute average representations
    # - this is the only difference between the Ada-ML-VAE
    # ave_mean = 0.5 * d0_posterior.mean + 0.5 * d1_posterior.mean
    # ave_std = (0.5 * d0_posterior.variance + 0.5 * d1_posterior.variance) ** 0.5
    # [4] Average logits across latent groups
    d0_logits = d0_posterior.base_dist.logits  # (B, Z, C)
    d1_logits = d1_posterior.base_dist.logits  # (B, Z, C)
    # ave_logits = 0.5 * d0_logits + 0.5 * d1_logits  # (B, Z, C)
    ave_logits = mix_logits(d0_logits, d1_logits)
    #distribution= softmax(mixture_logits)
    
    # Broadcast mask to match logits shape
    # print("share_mask shape:", share_mask.shape)
    # print("d0_logits shape:", d0_logits.shape)
    # Starting shape: [B, T]
    # → .unsqueeze(-1) → [B, T, 1]
    # → .unsqueeze(-1) → [B, T, 1, 1]
    # → .expand_as([B, T, 32, 32]) → ✅ [B, T, 32, 32]

    # Now each [B, T] element is broadcast across all 32 slots and 32 categories, exactly as intended.
    share_mask_expanded = share_mask.unsqueeze(-1).unsqueeze(-1).expand_as(d0_logits)
    
    # [4.b] select shared or original values based on mask
    # z0_mean = torch.where(share_mask, ave_mean, d0_posterior.loc)
    # z1_mean = torch.where(share_mask, ave_mean, d1_posterior.loc)
    # z0_std = torch.where(share_mask, ave_std, d0_posterior.scale)
    # z1_std = torch.where(share_mask, ave_std, d1_posterior.scale)
    z0_logits = torch.where(share_mask_expanded, ave_logits, d0_logits)
    z1_logits = torch.where(share_mask_expanded, ave_logits, d1_logits)

    # construct distributions
    # ave_d0_posterior = Normal(loc=z0_mean, scale=z0_std)
    # ave_d1_posterior = Normal(loc=z1_mean, scale=z1_std)
    # new_ds_posterior = (ave_d0_posterior, ave_d1_posterior)
    new_d0 = get_dist({"logit": z0_logits})
    new_d1 = get_dist({"logit": z1_logits})
    new_ds_posterior = (new_d0, new_d1)

    
    # [done] return new args & generate logs
    return new_ds_posterior

state = {"logit": torch.randn(2, 4, 3)}

dist = get_dist(state)
print("get_dist output:")
print("Distribution:", dist)
print("Sample:", dist.sample().shape)  # should be (2, 4, 3) one-hot samples

# Example logits for 3 categories
logits1 = torch.tensor([[2.0, 1.0, 0.5]])
logits2 = torch.tensor([[0.5, 1.5, 2.0]])

mixed = mix_logits(logits1, logits2, w=0.7)
print("\nmix_logits output:")
print("Mixed logits:", mixed)
print("Softmax probs:", torch.softmax(mixed, dim=-1))

# Fake d0 and d1 distributions using get_dist
batch, z, c = 2, 4, 5  # batch=2, latent dim=4, categories=5
d0 = get_dist({"logit": torch.randn(batch, z, c)})
d1 = get_dist({"logit": torch.randn(batch, z, c)})

new_d0, new_d1 = hook_intercept_ds([d0, d1])

print("\nhook_intercept_ds output:")
print("New d0 sample:", new_d0.sample().shape)  # (2, 4, 5)
print("New d1 sample:", new_d1.sample().shape)  # (2, 4, 5)



