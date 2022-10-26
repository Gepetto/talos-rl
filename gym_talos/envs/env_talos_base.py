import gym
import numpy as np
import pinocchio as pin

from sobec import RobotDesigner

from gym_talos.simulator.bullet_Talos import TalosDeburringSimulator


class EnvTalosBase(gym.Env):
    def __init__(self, targetPos, designer_conf) -> None:

        # Robot wrapper
        self.pinWrapper = RobotDesigner()
        self.pinWrapper.initialize(designer_conf)

        gripper_SE3_tool = pin.SE3.Identity()
        gripper_SE3_tool.translation[0] = designer_conf["toolFramePos"][0]
        gripper_SE3_tool.translation[1] = designer_conf["toolFramePos"][1]
        gripper_SE3_tool.translation[2] = designer_conf["toolFramePos"][2]
        self.pinWrapper.addEndEffectorFrame(
            "deburring_tool", "gripper_left_fingertip_3_link", gripper_SE3_tool
        )

        # Simulator
        self.simulator = TalosDeburringSimulator(
            URDF=designer_conf["urdfPath"],
            targetPos=targetPos,
            rmodelComplete=self.pinWrapper.get_rModelComplete(),
            controlledJointsIDs=self.pinWrapper.get_controlledJointsIDs(),
            enableGUI=False,
        )

        # Parameters
        self.maxTime = 1000
        self.weight_posture = 1
        self.weight_command = 0
        self.desired_state = self.pinWrapper.get_x0()

        # Environment variables
        action_dim = len(designer_conf["controlledJointsNames"]) - 1
        self.action_space = gym.spaces.Box(
            low=-100, high=100, shape=(action_dim,), dtype=np.float64
        )

        observation_dim = self.desired_state.size
        self.observation_space = gym.spaces.Box(
            low=-100, high=100, shape=(observation_dim,), dtype=np.float64
        )

        self.timer = 0

    def reset(self, *, seed=None, options=None):
        self.timer = 0
        self.simulator.reset()

        x_measured = self.simulator.getRobotState()
        self.pinWrapper.updateReducedModel(x_measured)

        # for i in self.pinWrapper.get_x0():
        #     # np.can_cast(i.dtype,)
        #     print(type(i))
        return np.float32(self.pinWrapper.get_x0())

    def step(self, action):
        self.timer += 1

        self.simulator.step(action)

        x_measured = self.simulator.getRobotState()

        self.pinWrapper.updateReducedModel(x_measured)

        observation = self._getObservation()
        reward = self._getReward(action, observation)
        terminated = self._checkTermination()
        # truncated = self._checkTruncation()

        return observation, reward, terminated, {}

    def _getObservation(self):
        return np.float32(self.pinWrapper.get_x0())

    def _getReward(self, action, observation):
        reward = 0
        # Posture reward + command regularization
        reward_posture = -np.linalg.norm(observation - self.desired_state)
        reward_command = -np.linalg.norm(action)

        reward = (
            self.weight_posture * reward_posture + self.weight_command * reward_command
        )

        return reward

    def _checkTermination(self):
        return self.timer > (self.maxTime - 1)

    def _checkTruncation(self):
        raise NotImplementedError
