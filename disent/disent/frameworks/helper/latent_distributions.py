#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~


from typing import Sequence
from typing import Tuple
from typing import final

import torch
from torch.distributions import Distribution
from torch.distributions import Laplace
from torch.distributions import Normal

import disent.registry as R
from disent.frameworks.helper.util import compute_ave_loss
from disent.nn.loss.kl import kl_loss
from disent.nn.loss.reduction import loss_reduction

from torch.distributions import OneHotCategorical
import torch


# ========================================================================= #
# Vae Distributions                                                         #
# TODO: this should be moved into NNs                                       #
# TODO: encoder modules should directly output distributions!               #
# ========================================================================= #


class LatentDistsHandler(object):
    def __init__(self, kl_mode: str = "direct", reduction="mean"):
        self._kl_mode = kl_mode
        self._reduction = reduction

    def encoding_to_representation(self, z_raw: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        raise NotImplementedError

    def encoding_to_dists(self, z_raw: Tuple[torch.Tensor, ...]) -> Tuple[Distribution, Distribution]:
        raise NotImplementedError

    @final
    def encoding_to_dists_and_sample(
        self, z_raw: Tuple[torch.Tensor, ...]
    ) -> Tuple[Distribution, Distribution, torch.Tensor]:
        """
        Return the parameterized prior and the approximate posterior distributions,
        as well as a sample from the approximate posterior using the 'reparameterization trick'.
        """
        posterior, prior = self.encoding_to_dists(z_raw)
        # sample from posterior -- reparameterization trick!
        # ie. z ~ q(z|x)
        z_sampled = posterior.rsample()
        # return values
        return posterior, prior, z_sampled

    @final
    def compute_kl_loss(
        self, posterior: Distribution, prior: Distribution, z_sampled: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute the kl divergence
        """
        kl = kl_loss(posterior, prior, z_sampled, mode=self._kl_mode)
        kl = loss_reduction(kl, reduction=self._reduction)
        return kl

    @final
    def compute_unreduced_kl_loss(
        self, posterior: Distribution, prior: Distribution, z_sampled: torch.Tensor = None
    ) -> torch.Tensor:
        return kl_loss(posterior, prior, z_sampled, mode=self._kl_mode)

    @final
    def compute_ave_kl_loss(
        self, ds_posterior: Sequence[Distribution], ds_prior: Sequence[Distribution], zs_sampled: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        return compute_ave_loss(self.compute_kl_loss, ds_posterior, ds_prior, zs_sampled)


# ========================================================================= #
# Normal Distribution                                                       #
# ========================================================================= #


class LatentDistsHandlerNormal(LatentDistsHandler):
    """
    Latent distributions with:
    - posterior: normal distribution with diagonal covariance
    - prior: unit normal distribution

    NOTE: Expanding parameters results in something akin to an L2 regularizer.
    """

    # assert mode == 'direct', f'legacy reference implementation of KL loss only supports mode="direct", not {repr(mode)}'
    # assert reduction == 'mean_sum', f'legacy reference implementation of KL loss only supports reduction="mean_sum", not {repr(reduction)}'

    def encoding_to_representation(self, raw_z: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        z_mean, z_logvar = raw_z
        return z_mean

    def encoding_to_dists(self, raw_z: Tuple[torch.Tensor, ...]) -> Tuple[Normal, Normal]:
        """
        Return the parameterized prior and the approximate posterior distributions.
        - The standard VAE parameterizes the gaussian normal with diagonal covariance.
        - logvar is used to avoid negative values for the standard deviation
        - Gaussian Encoder Model Distribution: pg. 25 in Variational Auto Encoders

        (✓) Visual inspection against reference implementations:
            https://github.com/google-research/disentanglement_lib (sample_from_latent_distribution)
            https://github.com/PyTorchLightning/pytorch-lightning-bolts/blob/master/pl_bolts/models/autoencoders/basic_vae/basic_vae_module.py
        """
        z_mean, z_logvar = raw_z
        # compute required values
        z_std = torch.exp(0.5 * z_logvar)
        # q: approximate posterior distribution
        posterior = Normal(loc=z_mean, scale=z_std)
        # p: prior distribution
        prior = Normal(torch.zeros_like(z_mean), torch.ones_like(z_std))
        # return values
        return posterior, prior


class LatentDistsHandlerLaplace(LatentDistsHandler):
    """
    Latent distributions with:
    - posterior: laplace distribution with diagonal covariance
    - prior: unit laplace distribution

    TODO: is this true?
    NOTE: Expanding parameters results in something akin to an L1 regularizer, with extra terms?
    """

    def encoding_to_representation(self, raw_z: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        z_loc, z_logscale = raw_z
        return z_loc

    def encoding_to_dists(self, raw_z: Tuple[torch.Tensor, ...]) -> Tuple[Laplace, Laplace]:
        z_loc, z_logscale = raw_z
        # compute required values
        z_scale = torch.exp(z_logscale)
        # q: approximate posterior distribution
        posterior = Laplace(loc=z_loc, scale=z_scale)
        # p: prior distribution
        prior = Laplace(torch.zeros_like(z_loc), torch.ones_like(z_scale))
        # return values
        return posterior, prior


class CategoricalLatentsHandler(LatentDistsHandler):
    def tuple_to_grouped(self, z_tuple):
        """
        Convert tuple/list of per-latent OneHotCategorical to grouped [B, Z, K]
        """
        if isinstance(z_tuple, (list, tuple)):
            return torch.stack([z.probs for z in z_tuple], dim=1)  # [B, Z, K]
        else:
            return z_tuple.probs  # single OneHotCategorical
    """
    Handles conversion between encoder logits and categorical latent distributions.
    """

    def encoding_to_dists(self, logits: torch.Tensor):
        """
        Convert encoder output to posterior and prior distributions.
        Args:
            logits: [B, z_size, n_classes]
        Returns:
            posterior: OneHotCategorical posterior
            prior: OneHotCategorical prior (uniform)
        """
        # print('logits ', len(logits), logits[0].shape)
        # posterior = OneHotCategorical(logits=logits)
        # B, z_size, n_classes = logits.shape
        
        # prior = OneHotCategorical(logits=torch.zeros_like(logits))
        # return posterior, prior
        """
        Convert a tuple of per-latent logits to posterior and prior distributions.
        Args:
            logits: tuple of tensors, each [B, 1, n_classes]
        Returns:
            posterior: tuple of OneHotCategorical distributions
            prior: tuple of OneHotCategorical distributions (uniform)
        """
        # posterior = tuple(OneHotCategorical(logits=l) for l in logits)
        # prior = tuple(OneHotCategorical(logits=torch.zeros_like(l)) for l in logits)
        posterior = OneHotCategorical(logits=logits)  # [B, Z, K]
        prior = OneHotCategorical(logits=torch.zeros_like(logits))

        return posterior, prior

    def encoding_to_representation(self, z_raw):
        """
        Convert raw posterior distributions into discrete latent codes
        (e.g., for evaluation metrics).
        
        z_raw: tuple/list of OneHotCategorical, one per latent
        returns: Tensor of shape [B, Z] with integer codes
        """
        if isinstance(z_raw, (tuple, list)):
            # Stack probs along latent dimension
            probs = self.tuple_to_grouped(z_raw)
        elif isinstance(z_raw, torch.Tensor):
            probs = z_raw  # single OneHotCategorical

        # Take argmax over categories
        else:
            raise TypeError(f"Unexpected z_raw type: {type(z_raw)}")

        return torch.argmax(probs, dim=-1)  # [B, Z]

    
    # def compute_ave_kl_loss(self, ds_posterior, ds_prior, zs_sampled):
    #     """
    #     KL loss for categorical latents
    #     """
    #     kl_loss = 0
    #     for post, prior in zip(ds_posterior, ds_prior):
    #         kl_loss += torch.distributions.kl_divergence(post, prior).mean()
    #     return kl_loss

    

    # def compute_ave_kl_loss(self, ds_posterior, ds_prior, zs_sampled):
    #     """
    #     Compute average KL divergence between posterior and prior for
    #     categorical latents.

    #     Args:
    #         ds_posterior: tuple/list of OneHotCategorical posteriors
    #         ds_prior: tuple/list of OneHotCategorical priors
    #         zs_sampled: sampled latent codes (not used here, but kept for API compatibility)

    #     Returns:
    #         kl_loss: scalar tensor, mean KL across latent groups
    #         logs: dict for logging (currently empty)
    #     """
    #     # Ensure we are dealing with tuples/lists of distributions
    #     if not isinstance(ds_posterior, (list, tuple)):
    #         ds_posterior = [ds_posterior]
    #     if not isinstance(ds_prior, (list, tuple)):
    #         ds_prior = [ds_prior]

    #     assert len(ds_posterior) == len(ds_prior), \
    #         f"Posterior/prior length mismatch: {len(ds_posterior)} vs {len(ds_prior)}"


    #     # Compute KL per latent variable
    #     kl_terms = [
    #         torch.distributions.kl_divergence(p, q).mean()
    #         for p, q in zip(ds_posterior, ds_prior)
    #     ]

    #     # Average across latent groups (scale-invariant wrt z_size)
    #     kl_loss = torch.stack(kl_terms).mean()

    #     return kl_loss


    def compute_ave_kl_loss(self, ds_posterior, ds_prior, zs_sampled):
        """
        Average KL across latents. Accepts tuple/list of OneHotCategorical.
        """
        if not isinstance(ds_posterior, (list, tuple)):
            ds_posterior = [ds_posterior]
        if not isinstance(ds_prior, (list, tuple)):
            ds_prior = [ds_prior]

        kl_terms = [
            torch.distributions.kl_divergence(p, q).mean()
            for p, q in zip(ds_posterior, ds_prior)
        ]
        return torch.stack(kl_terms).mean()
# ========================================================================= #
# Factory                                                                   #
# ========================================================================= #


def make_latent_distribution(name: str, kl_mode: str, reduction: str) -> LatentDistsHandler:
    cls = R.LATENT_HANDLERS[name]
    # make instance
    return cls(kl_mode=kl_mode, reduction=reduction)




# ========================================================================= #
# END                                                                       #
# ========================================================================= #
