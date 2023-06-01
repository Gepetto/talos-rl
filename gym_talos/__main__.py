import yaml
import os
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from .envs.env_talos_deburring import EnvTalosDeburring

################
#  PARAMETERS  #
################
torch.set_num_threads(1)

train = True
display = False

tensorboard_log_dir = "./logs/"
tensorboard_log_name = "test"

config_filename = "/config_RL.yaml"

# Parameters filename
config_dir_path = os.path.dirname(os.path.realpath(__file__))

with open(config_dir_path + config_filename, "r") as paramFile:
    params = yaml.safe_load(paramFile)
params_designer = params["designer"]
params_env = params["env"]
params_training = params["training"]

number_environments = params_training["numEnv"]

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

    model = SAC("MlpPolicy", envTrain, verbose=0, tensorboard_log=tensorboard_log_dir)

    model.learn(
        total_timesteps=params_training["totalTimesteps"],
        tb_log_name=tensorboard_log_name,
    )

    envTrain.close()

    model.save(tensorboard_log_dir + tensorboard_log_name)

if display:
    model = SAC.load(tensorboard_log_dir + tensorboard_log_name)

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
