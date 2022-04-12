import argparse
import contextlib
import importlib
from argparse import Namespace as NSp
from os import makedirs
from os.path import join

import torch
import yaml
from torch import nn
from typeguard.importhook import install_import_hook

import wandb

# comment these out when deploying:
install_import_hook('gatsbi.utils')
install_import_hook('gatsbi.optimize.utils')

from gatsbi.networks import WrapGenMultipleSimulations
from gatsbi.optimize import BaseSR as Opt
from gatsbi.task_utils.run_utils import _update_defaults
from gatsbi.utils import compute_calibration_metrics, generate_test_set_for_calibration, \
    generate_test_set_for_calibration_from_obs


def main(args):
    args, unknown_args = args
    print(args, unknown_args)

    # Get defaults
    with open(join("tasks", args.task_name, "defaults_sr.yaml"), "r") as f:
        defaults = yaml.load(f, Loader=yaml.Loader)

    # Add arguments to defaults
    defaults["scoring_rule"] = args.scoring_rule
    defaults["epochs"] = args.epochs
    defaults["num_simulations_generator"] = args.num_simulations_generator
    defaults["patched_sr"] = args.patched_sr
    defaults["patch_step"] = args.patch_step
    defaults["patch_size"] = args.patch_size
    use_wandb = not args.no_wandb

    # Update defaults
    if len(unknown_args) > 0:
        defaults = _update_defaults(defaults, unknown_args)

    # Get application module
    application = importlib.import_module("gatsbi.task_utils.%s" % args.task_name)

    name = args.task_name + "_" + args.scoring_rule + "_" + str(args.num_simulations_generator)
    if args.patched_sr:
        name += f"_patched_{args.patch_step}_{args.patch_size}"

    # Make a logger
    print("Making logger")
    makedirs(join("results", args.task_name), exist_ok=True)
    if use_wandb:
        wandb.init(
            project=args.project_name,
            group=args.group_name,
            id=args.run_id,
            resume=args.resume,
            config=defaults,
            notes="",
            dir=join("results", args.task_name),
            name=name
        )
        config = NSp(**wandb.config)
        run = wandb.run
    else:
        config = NSp(**defaults)
        run = contextlib.nullcontext()

    with run:
        print("Making networks")
        # Make generator and discriminator

        gen = application.Generator()

        # Make networks work across multiple GPUs
        if args.multi_gpu:
            gen = nn.DataParallel(gen)

        if args.resume:
            assert args.resume_dir is not None
            chpt = torch.load(join(args.resume_dir, "checkpoint_models0.pt"))
            gen.load_state_dict(chpt["generator_state_dict"])

        if not args.no_cuda:
            gen.cuda()

        # wrap the generator to use the SR method
        gen_wrapped = WrapGenMultipleSimulations(gen, n_simulations=args.num_simulations_generator)

        # Make optimiser
        print("Making optimiser")
        batch_size = min(1000, int(config.batch_size_perc * config.num_training_simulations))
        print("Batch size", batch_size)
        if args.task_name == "camera_model":
            path_to_data = "results/EMNIST_data"
            prior = application.Prior(path_to_data=path_to_data, few_samples=False)
        else:
            prior = application.Prior()
        simulator = application.Simulator()
        dataloader = {}
        print("Making dataloaders")
        if hasattr(application, "get_dataloader"):
            dataloader = application.get_dataloader(
                batch_size, int(config.hold_out_perc * config.num_training_simulations), config.path_to_data
            )

        # default values of patch size and step:
        if args.patch_size is None:
            args.patch_size = 4 if args.task_name == "camera_model" else 2
        if args.patch_step is None:
            args.patch_step = 4 if args.task_name == "camera_model" else 2

        # Make optimizer
        opt = Opt(
            generator=gen_wrapped,
            prior=prior,
            simulator=simulator,
            optim_args=[config.gen_opt_args],
            dataloader=dataloader,
            scoring_rule=args.scoring_rule,
            patched_sr=args.patched_sr,
            patch_step=args.patch_step,
            patch_size=args.patch_size,
            round_number=0,
            training_opts={
                "num_simulations": config.num_training_simulations,
                "sample_seed": config.sample_seed,
                "hold_out": int(config.hold_out_perc * config.num_training_simulations),
                "batch_size": batch_size,
                "log_dataloader": True,
            },
            logger=run if use_wandb else None,
            data_is_image=args.task_name == "camera_model",
        )

        if args.resume:
            opt.epoch_ct = chpt["epoch"]

        # Train model
        print("Training")
        opt.train(args.epochs, 50, start_early_stopping_after_epoch=1000)

        # compute other calibration metrics (which compare approximate posterior with true parameter value).
        # Also need to do those on a test set.
        if hasattr(application, "get_dataloader"):  # for shallow water model, use a function to obtain the dataset
            # That relies on data generated using the sample_shallow_water.py script.
            test_theta, test_obs = application.get_dataloader(
                batch_size, 0, config.path_to_data, test=True, return_data=True
            )
            test_theta_fake, test_theta = generate_test_set_for_calibration_from_obs(test_theta, test_obs, gen,
                                                                                     n_test_samples=1000,
                                                                                     n_generator_simulations=1000,
                                                                                     data_is_image=args.task_name == "camera_model")
        else:
            test_theta_fake, test_theta = generate_test_set_for_calibration(prior, simulator, gen, n_test_samples=1000,
                                                                            n_generator_simulations=1000,
                                                                            sample_seed=config.sample_seed,
                                                                            data_is_image=args.task_name == "camera_model")

        opt.logger.log(compute_calibration_metrics(test_theta_fake, test_theta, sbc_lines=True))

    if use_wandb:
        wandb.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str)
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--group_name", type=str, default=None)
    parser.add_argument("--scoring_rule", type=str, default="energy_score", choices=["energy_score", "kernel_score"])
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--num_simulations_generator", type=int, default=3)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--multi_gpu", type=bool, default=False)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--resume_dir", type=str, default=None,
                        help="Needs to be something as 'results/two_moons/wandb/run-20220325_172124-28i1s9ik/files', "
                             "where the part before 'files' is printed when training at the first round")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--patched_sr", action="store_true")
    parser.add_argument("--patch_step", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)

    main(parser.parse_known_args())
