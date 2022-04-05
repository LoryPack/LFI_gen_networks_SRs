import argparse
import importlib
from argparse import Namespace as NSp
from os import makedirs
from os.path import join

import torch
import wandb
import yaml
from torch import nn

from gatsbi.networks.base import WrapGenMultipleSimulations
from gatsbi.optimize import BaseSR as Opt
from gatsbi.task_utils.run_utils import _update_defaults
from gatsbi.utils import compute_calibration_metrics, generate_test_set_for_calibration


def main(args):
    args, unknown_args = args
    print(args, unknown_args)

    # Get defaults
    with open(join("tasks", args.task_name, "defaults_sr.yaml"), "r") as f:
        defaults = yaml.load(f, Loader=yaml.Loader)

    # Add arguments to defaults
    defaults["scoring_rule"] = args.scoring_rule
    defaults["epochs"] = args.epochs
    defaults["num_training_simulations"] = args.num_training_simulations
    defaults["num_simulations_generator"] = args.num_simulations_generator

    # Update defaults
    if len(unknown_args) > 0:
        defaults = _update_defaults(defaults, unknown_args)

    # Get application module
    application = importlib.import_module("gatsbi.task_utils.%s" % args.task_name)

    # Make a logger
    print("Making logger")
    makedirs(join("results", args.task_name), exist_ok=True)
    wandb.init(
        project=args.project_name,
        group=args.group_name,
        id=args.run_id,
        resume=args.resume,
        config=defaults,
        notes="",
        dir=join("results", args.task_name),
        name=args.task_name + "_" + args.scoring_rule + "_" + str(args.num_training_simulations) + "_" + str(
            args.num_simulations_generator)
    )
    config = NSp(**wandb.config)

    run = wandb.run
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
        batch_size = min(1000, int(config.batch_size_perc * args.num_training_simulations))
        if args.task_name == "camera_model":
            path_to_data = "results/EMNIST_data"
            prior = application.Prior(path_to_data=path_to_data, few_samples=True)
        else:
            prior = application.Prior()
        simulator = application.Simulator()
        dataloader = {}
        print("Making dataloaders")
        if hasattr(application, "get_dataloader"):
            dataloader = application.get_dataloader(
                batch_size, int(config.hold_out_perc * args.num_training_simulations), config.path_to_data
            )

        # Make optimizer
        opt = Opt(
            generator=gen_wrapped,
            prior=prior,
            simulator=simulator,
            optim_args=[config.gen_opt_args],
            dataloader=dataloader,
            scoring_rule=args.scoring_rule,
            round_number=0,
            training_opts={
                "num_simulations": args.num_training_simulations,
                "sample_seed": config.sample_seed,
                "hold_out": int(config.hold_out_perc * args.num_training_simulations),
                "batch_size": batch_size,
                "log_dataloader": True,
            },
            logger=run,
            data_is_image=args.task_name == "camera_model",
        )

        if args.resume:
            opt.epoch_ct = chpt["epoch"]

        # Train model
        print("Training")
        opt.train(args.epochs, 100, start_early_stopping_after_epoch=1000)

        # compute other calibration metrics (which compare approximate posterior with true parameter value).
        # Also need to do those on a test set.
        test_theta_fake, test_theta = generate_test_set_for_calibration(prior, simulator, gen, n_test_samples=100,
                                                                        n_generator_simulations=1000,
                                                                        sample_seed=config.sample_seed)

        opt.logger.log(compute_calibration_metrics(test_theta_fake, test_theta, sbc_lines=True))

    wandb.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str)
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--group_name", type=str, default=None)
    parser.add_argument("--scoring_rule", type=str, default="energy_score", choices=["energy_score", "kernel_score"])
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--num_training_simulations", type=int, default=10000)
    parser.add_argument("--num_simulations_generator", type=int, default=3)
    parser.add_argument("--multi_gpu", type=bool, default=False)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--resume_dir", type=str, default=None,
                        help="Needs to be something as 'results/two_moons/wandb/run-20220325_172124-28i1s9ik/files', "
                             "where the part before 'files' is printed when training at the first round")
    parser.add_argument("--no_cuda", action="store_true")
    main(parser.parse_known_args())
