import unittest

import torch
import torch.nn as nn

from gatsbi.networks.base import BaseNetwork, WrapGenMultipleSimulations
from gatsbi.networks.modules import AddNoise


class TestWrapGenMultipleSimulations(unittest.TestCase):

    def test_generator_wrapper(self):
        n_simulations = 3
        gen = BaseNetwork([nn.Linear(10, 10), AddNoise(10, 10), nn.LeakyReLU()])
        gen_wrapped = WrapGenMultipleSimulations(gen, n_simulations=n_simulations)

        fake_obs = torch.randn(10)

        gen_output = gen_wrapped(fake_obs)

        # check shape
        self.assertTrue(len(gen_output.shape) >= 3)

        # check that the second dimension is the n_simulations
        self.assertEqual(gen_output.shape[1], n_simulations)

        # check that the different elements are different
        self.assertTrue(gen_output[:, 0, 0].mean() != gen_output[:, 1, 0].mean())
