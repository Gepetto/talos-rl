import yaml

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from gym_talos.envs.env_talos_base import EnvTalosBase

################
#  PARAMETERS  #
################
display = True

# Parameters filename
filename = "./gym_talos/config_RL.yaml"

with open(filename, "r") as paramFile:
    params = yaml.safe_load(paramFile)
designer_conf = params["designer"]

##############
#  TRAINING  #
##############


envTrain = EnvTalosBase(designer_conf, GUI=False)
# env = DummyVecEnv([lambda: EnvTalosBase(targetPos, designer_conf)])
# # Automatically normalize the input features and reward
# env = VecNormalize(env, norm_obs=True, norm_reward=False,
#                    clip_obs=10.)
model = PPO("MlpPolicy", envTrain, verbose=1, tensorboard_log="./logs")
model.learn(total_timesteps=1000000)
envTrain.close()

if display:
    envDisplay = EnvTalosBase(designer_conf, GUI=True)
    obs = envDisplay.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        _, _, done, _ = envDisplay.step(action)
        if done:
            input("Press to restart")
            envDisplay.reset()
    envDisplay.close()
