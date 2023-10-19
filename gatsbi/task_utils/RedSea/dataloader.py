from os import listdir
from os.path import join
from typing import Iterable

import numpy as np
import torch

from gatsbi.utils.load_data import MakeDataset, make_loader


def get_dataloader(batch_size: int, hold_out: int, path_to_data: str, test: bool = False,
                   return_data: bool = False) -> Iterable:
    """
    Get dataloader for Red Sea model.

    Args:
        batch_size: batch size for dataloader.
        hold_out: number of samples to hold out in validation set.
        path_to_data: path to data to load into dataloader.
    """
    dataloader = {}

    theta = torch.load(join(path_to_data,'100k-priors.pt'))
    sim = torch.load(join(path_to_data,'100k-simulated_dataset.pt'))

    if return_data:
        return theta, sim
    else:
        inputs = {"inputs": [theta, sim], "hold_out": hold_out}

        dataloader["0"] = make_loader(
            batch_size, inputs_to_loader_class=inputs, loader_class=MakeDataset
        )
        return dataloader
    


