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

from typing import Optional,List,Tuple

import numpy as np
import random
from disent.dataset.data import GroundTruthData
from disent.dataset.sampling._base import BaseDisentSampler
from disent.dataset.util.state_space import StateSpace


class GroundTruthPairOrigSampler(BaseDisentSampler):
    def uninit_copy(self) -> "GroundTruthPairOrigSampler":
        return GroundTruthPairOrigSampler(p_k=self.p_k)

    def __init__(
        self,
        # num_differing_factors
        p_k: int = 1,
    ):
        """
        Sampler that emulates choosing factors like:
        https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/weak/train_weak_lib.py
        """
        super().__init__(num_samples=2)
        # DIFFERING FACTORS
        self.p_k = p_k
        # dataset variable
        self._state_space: Optional[StateSpace] = None

    def _init(self, dataset):
        assert isinstance(
            dataset, GroundTruthData
        ), f"dataset must be an instance of {repr(GroundTruthData.__class__.__name__)}, got: {repr(dataset)}"
        self._state_space = dataset.state_space_copy()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # CORE                                                                  #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def _sample_idx(self, idx):
        f0, f1 = self.datapoint_sample_factors_pair(idx)
        return (
            (self._state_space.pos_to_idx(f0),'first'),
            (self._state_space.pos_to_idx(f1),'second'),
        )
        # return (
        #         self._state_space.pos_to_idx(f0),
        #         self._state_space.pos_to_idx(f1),
        #     )

    def datapoint_sample_factors_pair(self, idx):
        """
        This function is based on _sample_weak_pair_factors()
        Except deterministic for the first item in the pair, based off of idx.
        """
        # randomly sample the first observation -- In our case we just use the idx
        sampled_factors = self._state_space.idx_to_pos(idx)
        # sample the next observation with k differing factors
        next_factors, k = _sample_k_differing(sampled_factors, self._state_space, k=self.p_k)
        # return the samples
        return sampled_factors, next_factors

class GroundTruthPairOrigSamplerUnlock(BaseDisentSampler):
    
    def uninit_copy(self) -> "GroundTruthPairOrigSampler":
        return GroundTruthPairOrigSampler(p_k=self.p_k)

    def __init__(
        self,
        # num_differing_factors
        p_k: int = 1,
    ):
        """
        Sampler that emulates choosing factors like:
        https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/weak/train_weak_lib.py
        """
        super().__init__(num_samples=2)
        # DIFFERING FACTORS
        self.p_k = p_k
        # dataset variable
        self._state_space: Optional[StateSpace] = None
        self.count=0
        self.total_count=0

    def _init(self, dataset):
        assert isinstance(
            dataset, GroundTruthData
        ), f"dataset must be an instance of {repr(GroundTruthData.__class__.__name__)}, got: {repr(dataset)}"
        self._state_space = dataset.state_space_copy()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # CORE                                                                  #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def _sample_idx(self, idx):
        self.total_count+=2
        f0, f1 = self.datapoint_sample_factors_pair(idx)
        # if (f0==f1).all():
        #     self.count+=1
        #     print('duplicates = ',self.count)
        #     with open("indices.txt", "a") as f:
        #         f.write(f"{idx}\n")  # Add newline after each entry
        # print('total = ',self.total_count)
        return (
            self._state_space.pos_to_idx(f0),
            self._state_space.pos_to_idx(f1),
        )

    def datapoint_sample_factors_pair(self, idx):
        """
        This function is based on _sample_weak_pair_factors()
        Except deterministic for the first item in the pair, based off of idx.
        """
        # print('index is ', idx)
        # randomly sample the first observation -- In our case we just use the idx
        indices=[0,1,2,3]
        sampled_factors = self._state_space.idx_to_pos(idx)
        

        """
        Code to remove 'illegal' factors - where agent and key positions are the same
        from the sampled and next factors.

        """


        while sampled_factors[0]==sampled_factors[4] and sampled_factors[1]==sampled_factors[5]:
            new_idx = random.randint(0, 4095)
            sampled_factors = self._state_space.idx_to_pos(new_idx)


        # sample the next observation with k differing factors
        next_factors, k = _sample_k_differing(sampled_factors, self._state_space, k=self.p_k)

        while next_factors[0]==next_factors[4] and next_factors[1]==next_factors[5]:
            next_factors, k = _sample_k_differing(sampled_factors, self._state_space, k=self.p_k)
          
        # return the samples
        return sampled_factors, next_factors

class RlSampler(BaseDisentSampler):
    def uninit_copy(self) -> "RlSampler":
        return RlSampler(p_k=self.p_k)
    

    def __init__(
        self,
    ):
        """
        Sampler that emulates choosing factors like:
        https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/weak/train_weak_lib.py
        """
        super().__init__(num_samples=2)
        # dataset variable
        print('using rl sampler')
        self._state_space: Optional[StateSpace] = None

    def _init(self, dataset):
        assert isinstance(
            dataset, GroundTruthData
        ), f"dataset must be an instance of {repr(GroundTruthData.__class__.__name__)}, got: {repr(dataset)}"
        self._state_space = dataset.state_space_copy()

    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #
    # CORE                                                                  #
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - #

    def _sample_idx(self, idx):
        #print('rl sampling unlock factors')
        sampled_factors = self._state_space.idx_to_pos(idx)

        """
        Code to remove 'illegal' factors - where agent and key positions are the same
        from the sampled and next factors.

        """
        # print('sampled factors are ', sampled_factors)
        new_idx=idx
        # while sampled_factors[0]==sampled_factors[4] and sampled_factors[1]==sampled_factors[5]:
        #     new_idx = random.randint(0, 4095)
        #     sampled_factors = self._state_space.idx_to_pos(new_idx)

        # return the samples
        # return (new_idx,-1*new_idx)
        return (new_idx,'first'),(-1*new_idx,'second')



def _sample_k_differing(factors, state_space: StateSpace, k=1):
    """
    Resample the factors used for the corresponding item in a pair.
      - Based on simple_dynamics() from:
        https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/weak/train_weak_lib.py
    """
    # checks for factors
    factors = np.array(factors)
    assert factors.ndim == 1
    # sample k
    if k <= 0:
        k = np.random.randint(1, state_space.num_factors)
    # randomly choose 1 or k
    # TODO: This is in disentanglement lib, HOWEVER is this not a mistake?
    #       A bug report has been submitted to disentanglement_lib for clarity:
    #       https://github.com/google-research/disentanglement_lib/issues/31
    k = np.random.choice([1, k])
    # generate list of differing indices
    index_list = np.random.choice(len(factors), k, replace=False)
    # randomly update factors
    for index in index_list:
        factors[index] = np.random.choice(state_space.factor_sizes[index])
    # return!
    return factors, k


def _sample_weak_pair_factors(state_space: StateSpace):  # pragma: no cover
    """
    Sample a weakly supervised pair from the given GroundTruthData.
      - Based on weak_dataset_generator() from:
        https://github.com/google-research/disentanglement_lib/blob/master/disentanglement_lib/methods/weak/train_weak_lib.py
    """
    # randomly sample the first observation
    sampled_factors = state_space.sample_factors(1)
    # sample the next observation with k differing factors
    next_factors, k = _sample_k_differing(sampled_factors, state_space, k=1)
    # return the samples
    return sampled_factors, next_factors
