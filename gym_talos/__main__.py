import yaml
import os

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.env_util import SubprocVecEnv
from stable_baselines3.common.env_checker import check_env

from .envs.env_talos_deburring import EnvTalosDeburring

################
#  PARAMETERS  #
################
train = False
display = True

tensorboard_log_dir = "./logs/"
tensorboard_log_name = "test"

# Parameters filename
dir_path = os.path.dirname(os.path.realpath(__file__))
filename = "/config_RL.yaml"
with open(dir_path + filename, "r") as paramFile:
    params = yaml.safe_load(paramFile)
params_designer = params["designer"]
params_env = params["env"]
params_training = params["training"]

##############
#  TRAINING  #
##############
if train:
    # envTrain = EnvTalosDeburring(params_designer, params_env, GUI=False)
    envTrain = SubprocVecEnv(
        6 * [lambda: EnvTalosDeburring(params_designer, params_env, GUI=False)]
    )
    # env = DummyVecEnv([lambda: EnvTalosBase(targetPos, designer_conf)])
    # # Automatically normalize the input features and reward
    # env = VecNormalize(env, norm_obs=True, norm_reward=False,
    #                    clip_obs=10.)
    model = SAC("MlpPolicy", envTrain, verbose=1, tensorboard_log=tensorboard_log_dir)

    model.learn(
        total_timesteps=params_training["totalTimesteps"], tb_log_name=tensorboard_log_name
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
            input("Press to restart")
            envDisplay.reset()
    envDisplay.close()
