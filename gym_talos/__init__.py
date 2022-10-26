from gym.envs.registration import register

register(
    id="talos-base-v0",
    entry_point="gym_talos.envs:EnvTalosBase",
)
