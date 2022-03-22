from typing import List

import torch
from torch import nn


class BaseNetwork(nn.Module):
    """Base class for all GAN networks /  doubles as GAN generator class."""

    def __init__(self, hidden_layers: List[nn.Module]):
        """
        Set up base class.

        Args:
            hidden_layers (list): list of nn.Module objects that will be fed
                                  into nn.Sequential, to build the network.
        """
        super(BaseNetwork, self).__init__()
        self._hidden_layers = nn.Sequential(*hidden_layers)

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            _input: torch Tensor input to the network

        Returns:
            Output of the network
        """
        return self._hidden_layers(_input)


class WrapGenMultipleSimulations(nn.Module):
    """"""

    def __init__(self, net: nn.Module, n_simulations: int = 3):
        """
        Args:
            net (nn.Module): net to wrap
            n_simulations (int): number of simulations from the generator per parameter value. Default
                                is 3.
        """
        super(WrapGenMultipleSimulations, self).__init__()
        self.net = net
        self.n_simulations = n_simulations

    def forward(self, _input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network self.n_simulations times

        Args:
            _input: torch Tensor input to the network

        Returns:
            Outputs of the network, stacked along the second dimension and adding a last dimension if missing.
            Shape is therefore [batch, n_simulations, out_size].
        """

        outputs = [self.net(_input)] * self.n_simulations

        # stack along the second dimension and add a last dimension if missing.
        return torch.atleast_3d(torch.stack(outputs, dim=1))
