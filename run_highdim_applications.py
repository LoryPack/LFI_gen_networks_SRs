import argparse
import contextlib
import importlib
from argparse import Namespace as NSp
from os import makedirs
from os.path import join
from time import time

import torch
import yaml
from torch import nn
from typeguard.importhook import install_import_hook

import wandb

# comment these out when deploying:
install_import_hook('gatsbi.utils')
install_import_hook('gatsbi.optimize.utils')

from gatsbi.optimize import Base as Opt
from gatsbi.optimize import UnrolledOpt as UOpt
from gatsbi.task_utils.run_utils import _update_defaults
from gatsbi.utils import compute_calibration_metrics, generate_test_set_for_calibration, \
    generate_test_set_for_calibration_from_obs


def main(args):
    args, unknown_args = args
    print(args, unknown_args)

    # Get defaults
    with open(join("tasks", args.task_name, "defaults.yaml"), "r") as f:
        defaults = yaml.load(f, Loader=yaml.Loader)

    defaults["epochs"] = args.epochs
    use_wandb = not args.no_wandb

    # Update defaults
    if len(unknown_args) > 0:
        defaults = _update_defaults(defaults, unknown_args)

    # Get application module
    application = importlib.import_module("gatsbi.task_utils.%s" % args.task_name)

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
            name=args.task_name + "_GAN_" + str(defaults["num_simulations"])
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
        dis = application.Discriminator()

        # Make networks work across multiple GPUs
        if args.multi_gpu:
            gen = nn.DataParallel(gen)
            dis = nn.DataParallel(dis)

        if args.resume:
            assert args.resume_dir is not None
            if args.no_cuda:
                # kwargs for loading on cpu
                kwargs = {"map_location": torch.device('cpu')}
            chpt = torch.load(join(args.resume_dir, "checkpoint_models0.pt"), **kwargs)
            gen.load_state_dict(chpt["generator_state_dict"])
            dis.load_state_dict(chpt["dis_state_dict"])

        if not args.no_cuda:
            gen.cuda()
            dis.cuda()

        # Make optimiser
        print("Making optimiser")
        batch_size = min(1000, int(config.batch_size_perc * config.num_simulations))
        if args.task_name == "camera_model":
            path_to_data = "results/EMNIST_data"
            prior = application.Prior(path_to_data=path_to_data, few_samples=False)
        else:
            prior = application.Prior()
        simulator = application.Simulator()
        dataloader = {}
        if hasattr(application, "get_dataloader"):
            dataloader = application.get_dataloader(
                batch_size, config.hold_out, config.path_to_data
            )

        # Make optimizer
        if args.task_name == "shallow_water_model":
            opt = UOpt(
                generator=gen,
                discriminator=dis,
                prior=prior,
                simulator=simulator,
                optim_args=[config.gen_opt_args, config.dis_opt_args],
                dataloader=dataloader,
                loss=config.loss,
                round_number=0,
                training_opts={
                    "gen_iter": config.gen_iter,
                    "dis_iter": config.dis_iter,
                    "max_norm_gen": config.max_norm_gen,
                    "max_norm_dis": config.max_norm_dis,
                    "num_simulations": config.num_simulations,
                    "sample_seed": 42,
                    "hold_out": config.hold_out,
                    "batch_size": batch_size,
                    "unroll_steps": config.unroll_steps,
                    "log_dataloader": config.log_dataloader,
                },
                logger=run if use_wandb else None,
            )
        else:
            opt = Opt(
                generator=gen,
                discriminator=dis,
                prior=prior,
                simulator=simulator,
                optim_args=[config.gen_opt_args, config.dis_opt_args],
                dataloader=dataloader,
                loss=config.loss,
                round_number=0,
                training_opts={
                    "gen_iter": config.gen_iter,
                    "dis_iter": config.dis_iter,
                    "max_norm_gen": config.max_norm_gen,
                    "max_norm_dis": config.max_norm_dis,
                    "num_simulations": config.num_simulations,
                    "sample_seed": 42,
                    "hold_out": config.hold_out,
                    "batch_size": batch_size,
                    "log_dataloader": config.log_dataloader,
                },
                logger=run if use_wandb else None,
            )

        if args.resume:
            opt.epoch_ct = chpt["epoch"]

        # Train model
        print("Training")
        start = time()
        opt.train(args.epochs, 100)
        train_time = time() - start
        print("Training took %.2f seconds" % train_time)
        if use_wandb:
            opt.logger.log({"train_time": train_time})

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
                                                                                     batch_size=100,
                                                                                     data_is_image=args.task_name == "camera_model")
        else:
            test_theta_fake, test_theta = generate_test_set_for_calibration(prior, simulator, gen, n_test_samples=1000,
                                                                            n_generator_simulations=1000,
                                                                            sample_seed=config.sample_seed,
                                                                            data_is_image=args.task_name == "camera_model")
        fig_filename = join("results", args.task_name) + "/GAN_" + str(defaults["num_simulations"])

        res = compute_calibration_metrics(test_theta_fake, test_theta, sbc_lines=True,
                                          norm_rmse=args.task_name != "camera_model",
                                          sbc_lines_kwargs={"name": "GAN",
                                                            "filename": fig_filename + "_sbc_lines.pdf"},
                                          sbc_hist_kwargs={"filename": fig_filename + "_sbc_hist.pdf"})
        if use_wandb:
            opt.logger.log(res)

    if use_wandb:
        wandb.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str)
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--group_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--multi_gpu", type=bool, default=False)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--resume_dir", type=str, default=None,
                        help="Needs to be something as 'results/two_moons/wandb/run-20220325_172124-28i1s9ik/files', "
                             "where the part before 'files' is printed when training at the first round")
    parser.add_argument("--no_cuda", action="store_true")
    main(parser.parse_known_args())
