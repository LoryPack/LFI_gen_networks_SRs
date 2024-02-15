# Simulation-Based Inference with Generative Neural Networks via Scoring Rule Minimization
---

This repository provides code to reproduce the experiments in the paper [Simulation-Based Inference with Generative Neural Networks via Scoring Rule Minimization](https://arxiv.org/abs/2205.15784). The code is based on the one for the paper ["GATSBI: Generative Adversarial Training for Simulation-Based Inference"](https://openreview.net/forum?id=kR1hC6j48Tp&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2022%2FConference%2FAuthors%23your-submissions)), which can be found at [https://github.com/mackelab/gatsbi](https://github.com/mackelab/gatsbi) and is distributed under a AGPL-3.0 license.


The code is written in Python and uses the [`torch`](https://pytorch.org/) library to handle neural networks. It depends both on the simulation-based inference package [`sbi`](https://github.com/mackelab/sbi) and the benchmark framework [`sbibm`](https://github.com/mackelab/sbibm).

You also need to install [Julia](https://julialang.org/downloads/) to run the SIR and Lotka-Volterra models and then run the following:

```
pip install diffeqtorch
export JULIA_SYSIMAGE_DIFFEQTORCH="$HOME/.julia_sysimage_diffeqtorch.so"
python -c "from diffeqtorch.install import install_and_test; install_and_test()"
```


### Installation
___
You can use this code in two different ways: 
#### 1. Installing the package
The code comes as a package that you can install within a working Python environment `pip`:
```
pip install "git+https://github.com/LoryPack/SBI_gen_networks_SRs"
```
#### 2. Downloading the code
You can directly download the code and run scripts from the main folder. In that case, the requirements can be installed with `pip`:
```
pip install -r requirements.txt
```

### Minimal example
___
For a minimal, see `quickstart.ipynb`.

### Experiments
___
We provide results for the following experiments: 5 benchmark tasks, the shallow water model and a noisy camera model.

Code for setting up priors, simulator, neural networks and any other pre-/post-processing code is available inside `gatsbi.task_utils`.

Hyperparameter settings for each of the experiments are available in `tasks/`


This repository includes scripts to reproduce the experiments in the [GATSBI paper](https://openreview.net/forum?id=kR1hC6j48Tp&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2022%2FConference%2FAuthors%23your-submissions)) which were already contained in the [GATSBI repository](https://github.com/mackelab/gatsbi), with some minor changes (and three additional benchmark tasks). Additionally, we provide scripts to run experiments with the Scoring Rules training methods discussed in [our paper](https://arxiv.org/abs/2205.15784):
- `run_benchmarks_sr.py` for the benchmark tasks
    ```
    python run_benchmarks_sr.py --project_name="Benchmarks" --task_name="two_moons"
    ```
    `task_name` = `slcp`, `two_moons` or any other task included [here](https://github.com/sbi-benchmark/sbibm/tree/main/sbibm/tasks) (we provide results for `slcp`, `two_moons`, `gaussian_mixture`, `gaussian_linear` and `bernoulli_glm`).
- `run_highdim_applications_sr.py` for the noisy camera model and the shallow water model  
    ```
    python run_highdim_applications_sr.py --project_name="High Dimensional Applications" --task_name="shallow_water_model"
    ```
    `task_name` = `shallow_water_model` or `camera_model`.
    Note that we **do not** provide training data for the shallow water model in this repository. Please use `sample_shallow_water.py` to generate training samples locally.
    ```
    python run_ResSea.py --project_name="RedSea" --task_name="RedSea"
    ```
  For running the spatial extremes model on the Red Sea dataset.

Running those scripts relies on `wandb` to log experiments unless the argument `--no_wandb` is passed. Check inside the code, or run `python run_highdim_applications.py --help` for more information.  

### Figures
___
Code to reproduce the figures in the paper is available in `plotting_code`, along with the required data `plotting_code/plotting_data`, and the final plots `plotting_code/plots`. Note that accessing the data requires Git LFS installation. Some data is also present on the `wandb` website and is downloaded by the plotting code automatically.

The following files in `plotting_code/plotting_data` were not present in the original [GATSBI repository](https://github.com/mackelab/gatsbi) but were provided by the Poornima Ramesh, which we thank, upon our request:
- shallow_water_test_plot_sample.npz 
- shallow_water_chpt_sw_model.pt 
