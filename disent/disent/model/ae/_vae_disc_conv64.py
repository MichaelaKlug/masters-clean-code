from torch import nn, Tensor
import torch
import torch.nn.functional as F

from disent.model import DisentEncoder
from disent.model import DisentDecoder



class EncoderConv64Categorical(DisentEncoder):
    """
    Convolutional encoder for categorical (discrete) latent variables.
    Instead of producing (mu, logvar) for Gaussian latents, this produces
    logits for a categorical distribution per latent variable.

    Args:
        x_shape: input image shape (C, H, W), must be (3, 64, 64).
        z_size: number of categorical latent variables.
        n_classes: number of categories per latent variable.
    """

    def __init__(self, x_shape=(3, 64, 64), z_size=6, z_multiplier=1,n_classes=10,latent_distribution='categorical'):
        (C, H, W) = x_shape
        assert (H, W) == (64, 64), "This model only works with image size 64x64."
        super().__init__(x_shape=x_shape, z_size=z_size, z_multiplier=z_multiplier)
        
        self._z_size = z_size
        self._n_classes = n_classes
        self._z_total = z_size * n_classes  # total logits

        # shared CNN backbone
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=C, out_channels=32, kernel_size=4, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=1600, out_features=256),
            nn.ReLU(inplace=True),
        )

        # final projection to categorical logits
        self.to_logits = nn.Linear(256, self._z_total)
        print('self._z_total:', self._z_total)

    @property
    def z_total(self):
        return self._z_total

    def encode(self, x: Tensor) -> Tensor:
        """
        Encode input x into categorical logits of shape [B, z_size, n_classes].
        """
        h = self.features(x)
        logits = self.to_logits(h)  # [B, z_size * n_classes]
        return logits.view(-1, self._z_size, self._n_classes)

    def sample_gumbel_softmax(self, x: Tensor, tau: float = 1.0, hard: bool = False) -> Tensor:
        """
        Convenience method: sample from gumbel-softmax for differentiable training.
        Returns one-hot vectors of shape [B, z_size, n_classes].
        """
        logits = self.encode(x)
        return F.gumbel_softmax(logits, tau=tau, hard=hard, dim=-1)




class DecoderConv64Categorical(DisentDecoder):
    """
    Wrapper around DecoderConv64 to accept categorical one-hot latents.
    Converts [B, z_size, n_classes] -> [B, z_size * n_classes].
    """

    def __init__(self, base_decoder: DisentDecoder, z_size: int, x_shape=(3, 64, 64), n_classes=10, z_multiplier=1):
        super().__init__(x_shape=x_shape, z_size=z_size, z_multiplier=z_multiplier)
        self.base_decoder = base_decoder
        self._z_size = z_size
        self._n_classes = n_classes

    def decode(self, z: Tensor) -> Tensor:
        """
        Decode categorical latent vectors.
        Expects z as one-hot or softmax outputs: [B, z_size, n_classes].
        """
        # flatten the categorical representation
        z_flat = z.view(z.size(0), self._z_size * self._n_classes)
        return self.base_decoder.decode(z_flat)



""" Usage 
encoder = EncoderConv64Categorical(x_shape=(3, 64, 64), z_size=6, n_classes=10)
decoder = DecoderConv64Categorical(
    base_decoder=DecoderConv64(x_shape=(3, 64, 64), z_size=6, z_multiplier=10),
    z_size=6,
    n_classes=10,
)
"""