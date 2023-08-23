import scipy
import scipy.ndimage
import torch
from skimage.util import random_noise
from torch import Tensor


class DownScaleSim:
    """Downscaled image simulator."""

    def __init__(self, **kwargs):
        """Return downscaled image."""
        self.kwargs = kwargs

    def __call__(self, theta: Tensor) -> Tensor:
        """Call to simulator."""
        return self.grayscale_simulator(theta)

    def grayscale_simulator(self, theta: Tensor,) -> Tensor:
        """Forward pass."""

        downsample_kernel = torch.ones(1, 1, 4, 4)
        downsample_kernel = downsample_kernel / 4 ** 2
        output = torch.nn.functional.conv2d(theta, downsample_kernel, stride=4,
                                                    groups=1)
        #output = torch.nn.functional.interpolate(output, (28,28))
        return output