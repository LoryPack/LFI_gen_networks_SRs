#!/usr/bin/env python

from setuptools import find_packages, setup

package_name = "LFI_gen_networks_SRs"
version = "1.0"
exclusions = ["notebooks", "src"]

_packages = find_packages(exclude=exclusions)

_base = [
    "numpy",
    "matplotlib",
    "pandas",
    "pyyaml",
    "scikit-learn",
    "torch",
    "plotly",
    "wandb",
    "scipy",
    "seaborn",
    "torchvision",
    "torchtyping",
    "tqdm",
    "typeguard"
    "scikit-image",
    "jupyter",
]
_sbi_extras = [
    "sbibm@git+https://github.com/mackelab/sbibm#egg=sbibm",
 #   "sbi@git+https://github.com/mackelab/sbi#egg=sbi",
]

setup(
    name=package_name,
    version=version,
    description="Likelihood-Free Inference with Generative Neural Networks via Scoring Rule Minimization",
    author="Lorenzo Pacchiardi",
    author_email="lorenzo.pacchiardi@stats.ox.ac.uk",
    url="https://github.com/LoryPack/LFI_gen_networks_SRs",
    packages=["LFI_gen_networks_SRs", "tests"],
    install_requires=(_base + _packages + _sbi_extras),
)
