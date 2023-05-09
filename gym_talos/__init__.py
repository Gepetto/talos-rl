from gym.envs.registration import register

register(
    id="talos-deburring-v0",
    entry_point="gym_talos.envs:EnvTalosDeburring",
)
