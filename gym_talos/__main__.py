from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_checker import check_env

from gym_talos.envs.env_talos_base import EnvTalosBase


modelPath = "/opt/openrobots/share/example-robot-data/robots/talos_data/"
URDF = modelPath + "robots/talos_reduced.urdf"
SRDF = modelPath + "srdf/talos.srdf"

controlledJoints = [
    "root_joint",
    "leg_left_1_joint",
    "leg_left_2_joint",
    "leg_left_3_joint",
    "leg_left_4_joint",
    "leg_left_5_joint",
    "leg_left_6_joint",
    "leg_right_1_joint",
    "leg_right_2_joint",
    "leg_right_3_joint",
    "leg_right_4_joint",
    "leg_right_5_joint",
    "leg_right_6_joint",
    "torso_1_joint",
    "torso_2_joint",
    "arm_left_1_joint",
    "arm_left_2_joint",
    "arm_left_3_joint",
    "arm_left_4_joint",
    "arm_right_1_joint",
    "arm_right_2_joint",
    "arm_right_3_joint",
    "arm_right_4_joint",
]

designer_conf = dict(
    urdfPath=URDF,
    srdfPath=SRDF,
    leftFootName="right_sole_link",
    rightFootName="left_sole_link",
    robotDescription="",
    controlledJointsNames=controlledJoints,
    toolFramePos=[0, -0.02, -0.0825],
)
targetPos = [0.6, 0.4, 1.1]


def play(model, targetPos, designer_conf):
    env = EnvTalosBase(targetPos, designer_conf, GUI=True)
    print("Null model")
    obs = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        # print(counter," - action: ",[round(a,4) for a in action])
        # input("...")
        obs, reward, done, _ = env.step(action)
        if done:
            # print("Max torques : ",env.robot.max_torques)
            input("Press to restart ...")
            env.reset()
    return None


env = EnvTalosBase(targetPos, designer_conf)
# env = DummyVecEnv([lambda: EnvTalosBase(targetPos, designer_conf)])
# # Automatically normalize the input features and reward
# env = VecNormalize(env, norm_obs=True, norm_reward=False,
#                    clip_obs=10.)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs")
model.learn(total_timesteps=10000)

play(model, targetPos, designer_conf)
