import yaml
import os
import torch
import argparse

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from .envs.env_talos_deburring import EnvTalosDeburring

################
#  PARAMETERS  #
################
# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument(
    "-id",
    "--identication",
    default=None,
    help="Identification number for the training (usefull when launching several trainings in parallel)",
    type=int,
)
parser.add_argument(
    "-config",
    "--configurationFile",
    required=True,
    help="Path to file containg the configuration of the training",
)

args = parser.parse_args()

# Parsing configuration file
#   Configuration file name is provided in command line
config_filename = str(args.configurationFile)

with open(config_filename, "r") as config_file:
    params = yaml.safe_load(config_file)

params_designer = params["robot_designer"]
params_env = params["environment"]
params_training = params["training"]


train = True
display = False

tensorboard_log_dir = "./logs/"

training_id = args.identication
if training_id:
    training_name = params_training["name"] + "_" + str(training_id)
else:
    training_name = params_training["name"]

number_environments = params_training["environment_quantity"]
total_timesteps = params_training["total_timesteps"]
verbose = params_training["verbose"]

torch.set_num_threads(1)

##############
#  TRAINING  #
##############
if train:
    if number_environments == 1:
        envTrain = EnvTalosDeburring(params_designer, params_env, GUI=False)
    else:
        envTrain = SubprocVecEnv(
            number_environments
            * [
                lambda: Monitor(
                    EnvTalosDeburring(params_designer, params_env, GUI=False)
                )
            ]
        )

    model = SAC(
        "MlpPolicy", envTrain, verbose=verbose, tensorboard_log=tensorboard_log_dir
    )

    model.learn(
        total_timesteps=total_timesteps,
        tb_log_name=training_name,
    )

    envTrain.close()

    model.save(tensorboard_log_dir + training_name)

if display:
    model = SAC.load(tensorboard_log_dir + training_name)

    envDisplay = EnvTalosDeburring(params_designer, params_env, GUI=True)
    envDisplay.maxTime = 1000
    obs = envDisplay.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        _, _, done, _ = envDisplay.step(action)
        if done:
            input("Press to any key restart")
            envDisplay.reset()
    envDisplay.close()
