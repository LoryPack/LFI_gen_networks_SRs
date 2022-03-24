import argparse
from argparse import Namespace as NSp
from os import makedirs
from os.path import join

import torch
import wandb
import yaml

import sbibm

from gatsbi.networks.base import WrapGenMultipleSimulations
from gatsbi.optimize import BaseSR as Opt
from gatsbi.task_utils.benchmarks import (ProposalWrapper, load_generator,
                                          make_generator)
from gatsbi.task_utils.benchmarks.make_results import MakeResults
from gatsbi.task_utils.run_utils import _update_defaults


def main(args):
    args, unknown_args = args

    # Get defaults
    yamlname = "defaults_opt_sr.yaml" if args.opt else "defaults_sr.yaml"
    with open(join("tasks", args.task_name, yamlname), "r") as f:
        defaults = yaml.load(f, Loader=yaml.Loader)

    # Update defaults
    if len(unknown_args) > 0:
        defaults = _update_defaults(defaults)

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
        name=args.task_name + "_" + str(args.num_training_simulations) + "_" + str(args.num_simulations_generator) + (
            "_opt" if args.opt else "")
    )
    config = NSp(**wandb.config)

    run = wandb.run
    with run:
        # Make task and simulator
        print("Making networks")
        task_name = args.task_name
        task = sbibm.get_task(task_name)
        simulator = task.get_simulator()
        prior = task.get_prior()

        start = 0
        epochs_per_round = [args.epochs]
        budget_per_round = [args.num_training_simulations]

        for rnd, (epochs, budget) in enumerate(
                zip(epochs_per_round[start:], budget_per_round[start:]), start=start
        ):

            # Make proposal, generator and discriminator
            gen = make_generator(
                gen_seed=config.gen_seed,
                **config.gen_network_kwargs
            )

            dataloader = {}

            # Resume from previous state
            if args.resume:
                assert args.resume_dir is not None
                res_chpt = torch.load(
                    join(args.resume_dir, "checkpoint_models%d.pt" % rnd)
                )
                gen = load_generator(res_chpt["generator_state_dict"],
                                     gen)

                dataloader = torch.load(
                    join(args.resume_dir, "checkpoint_dataloader%d.pt" % rnd)
                )
                print("classifier dataloader")

            # Make proposal prior
            if rnd > 0:
                if args.resume:
                    classifier_theta = torch.load(
                        join(args.resume_dir, "classifier_theta%d.pt" % rnd)
                    )
                    classifier_obs = torch.load(
                        join(args.resume_dir, "classifier_obs%d.pt" % rnd)
                    )
                    directory = args.resume_dir
                else:
                    directory = run.dir
                # Update prior -> proposal
                prop = make_generator(
                    gen_seed=config.gen_seed,
                    **config.gen_network_kwargs
                )
                mdl_chpt = torch.load(
                    join(directory, "checkpoint_models%d.pt" % (rnd - 1))
                )
                prop.load_state_dict(mdl_chpt["generator_state_dict"])
                proposal = ProposalWrapper(
                    prop,
                    task.get_observation(config.obs_num),
                    lat_dist=None,
                    lat_dim=config.gen_network_kwargs["add_noise_kwargs"]["lat_dim"],
                ).prior

                prior = proposal

                if not args.resume:
                    # Get dataloader
                    dataloader = torch.load(
                        join(run.dir, "checkpoint_dataloader%d.pt" % (rnd - 1))
                    )
            # Move to GPU
            if not args.no_cuda:
                gen.cuda()

            # wrap the generator to use the SR method
            gen_wrapped = WrapGenMultipleSimulations(gen, n_simulations=args.num_simulations_generator)

            # Make optimiser
            print("Make optimiser")
            batch_size = min(1000, int(config.batch_size_perc * budget))
            lat_dim = config.gen_network_kwargs["add_noise_kwargs"]["lat_dim"]

            opt = Opt(
                gen_wrapped,
                prior,
                simulator,
                [config.gen_opt_args],
                scoring_rule=config.scoring_rule,
                training_opts={
                    "num_simulations": args.num_training_simulations,
                    "sample_seed": config.sample_seed,
                    "hold_out": int(config.hold_out_perc * args.num_training_simulations),
                    "batch_size": batch_size,
                    "log_dataloader": True,
                    "stop_thresh": config.stop_thresh,
                },
                logger=run,
            )
            setattr(opt, "lat_dist", None)
            if args.resume:
                opt.epoch_ct = res_chpt["epoch"]
                opt.dataloader = dataloader

            # Train model
            print("Training")
            opt.train(epochs, 100, start_early_stopping_after_epoch=1000)

            make_results = MakeResults(
                generator=gen,
                task=task,
                lat_dist=opt.lat_dist,
                save_dir=opt.logger.dir,
                cuda=not args.no_cuda
            )
            print("round", rnd)
            if rnd > 0:
                opt.logger.log(make_results.calc_c2st(config.obs_num))
            else:
                opt.logger.log(make_results.calc_c2st_all_obs())
        wandb.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str)
    parser.add_argument("--task_name", type=str)
    parser.add_argument("--group_name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=20000)
    parser.add_argument("--num_training_simulations", type=int, default=10000)
    parser.add_argument("--num_simulations_generator", type=int, default=3)
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument("--resume_dir", type=str, default=None)
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--opt", action="store_true")
    main(parser.parse_known_args())
