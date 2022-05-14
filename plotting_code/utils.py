import pandas as pd
import wandb
api = wandb.Api()

def obtain_wandb_data(projects_list):
    runs_list = []
    runs_list_GAN = []
    runs_list_SRs = []
    for project in projects_list:

        new_dict_list = []
        new_dict_list_GAN = []
        new_dict_list_SRs = []

        # Project is specified by <entity/project-name>
        runs = api.runs("lorypack/" + project)

        for run in runs:
            # .summary contains the output keys/values for metrics like accuracy.
            #  We call ._json_dict to omit large files
            summary_dict = run.summary._json_dict

            # .config contains the hyperparameters.
            #  We remove special values that start with _.
            config_dict = {k: v for k, v in run.config.items()
                           if not k.startswith('_')}

            # .name is the human-readable name of the run.
            new_dict = {"name": run.name, "ID": run.id, **config_dict, **summary_dict}
            new_dict_list.append(new_dict)

            if "GAN" in run.name:
                new_dict_list_GAN.append(new_dict)
            else:
                new_dict_list_SRs.append(new_dict)

        runs_list.append(pd.DataFrame(new_dict_list))
        runs_list_GAN.append(pd.DataFrame(new_dict_list_GAN))
        runs_list_SRs.append(pd.DataFrame(new_dict_list_SRs))

    return runs_list, runs_list_GAN, runs_list_SRs


if __name__ == "__main__":
    runs_list, runs_list_GAN, runs_list_SRs = obtain_wandb_data(["SLCP_2", "TwoMoons_2"])