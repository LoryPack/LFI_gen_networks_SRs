from os import listdir
from os.path import join
from typing import Iterable

import numpy as np
import torch

from gatsbi.utils.load_data import MakeDataset, make_loader


def get_dataloader(batch_size: int, hold_out: int, path_to_data: str, test: bool = False,
                   return_data: bool = False) -> Iterable:
    """
    Get dataloader for shallow water model.

    Args:
        batch_size: batch size for dataloader.
        hold_out: number of samples to hold out in validation set.
        path_to_data: path to data to load into dataloader.
    """
    dataloader = {}

    depth_profiles = []
    z_vals = []
    list_of_files = []
    for f in sorted(listdir(path_to_data)):
        if f.endswith(".npz"):
            if ("test" in f) and test:
                list_of_files.append(f)
            elif ("test" not in f) and (not test):
                list_of_files.append(f)

    for dat in list_of_files:
        dd = np.load(join(path_to_data, dat))
        depth_profiles.append(dd["depth_profile"].squeeze())
        z_vals.append(dd["z_vals"].squeeze())

    depth_profiles = torch.FloatTensor(np.concatenate(depth_profiles, 0))
    z_vals = np.concatenate(z_vals, 0)[:, :, 1:]
    z_vals = torch.FloatTensor(z_vals)

    if return_data:
        return depth_profiles, z_vals
    else:
        inputs = {"inputs": [depth_profiles, z_vals], "hold_out": hold_out}

        dataloader["0"] = make_loader(
            batch_size, inputs_to_loader_class=inputs, loader_class=MakeDataset
        )
        return dataloader
