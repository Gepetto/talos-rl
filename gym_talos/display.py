from pathlib import Path

import yaml
from stable_baselines3 import SAC

from .envs.env_talos_deburring import EnvTalosDeburring

training_name = "2023-06-01_test_1"

log_dir = Path("logs")
model_path = log_dir / training_name / f"{training_name[:-2]}.zip"
config_path = log_dir / training_name / f"{training_name[:-2]}.yaml"

model = SAC.load(model_path)

with config_path.open() as config_file:
    params = yaml.safe_load(config_file)

envDisplay = EnvTalosDeburring(
    params["robot_designer"],
    params["environment"],
    GUI=True,
)
envDisplay.maxTime = 1000
obs, info = envDisplay.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    _, _, terminated, truncated, _ = envDisplay.step(action)
    if terminated or truncated:
        input("Press to any key restart")
        envDisplay.reset()
envDisplay.close()
