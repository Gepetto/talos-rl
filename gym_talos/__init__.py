from gym.envs.registration import register

register(
    id="talos-base",
    entry_point="gym_talos.envs:EnvTalosBase",
)
