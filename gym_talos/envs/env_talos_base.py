import gym
import numpy as np
import pinocchio as pin

from sobec import RobotDesigner

from gym_talos.simulator.bullet_Talos import TalosDeburringSimulator


class EnvTalosBase(gym.Env):
    def __init__(self, targetPos, designer_conf, GUI=False) -> None:

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
            enableGUI=GUI,
            dt=1e-4,
        )

        # Parameters
        self.maxTime = 500
        self.weight_posture = 10
        self.weight_command = 1
        self.desired_state = self.pinWrapper.get_x0()
        self.torqueScale = np.array(
            [
                99,
                159,
                159,
                299,
                159,
                99,
                99,
                159,
                159,
                299,
                159,
                99,
                144,
                144,
                144,
                144,
                90,
                90,
                144,
                144,
                90,
                90,
            ]
        )

        self.lowerObsLim = np.concatenate(
            (
                self.pinWrapper.get_rModel().lowerPositionLimit,
                -self.pinWrapper.get_rModel().velocityLimit,
            ),
        )
        self.lowerObsLim[:7] = -5
        self.lowerObsLim[
            self.pinWrapper.get_rModel().nq : self.pinWrapper.get_rModel().nq + 6
        ] = -5

        self.upperObsLim = np.concatenate(
            (
                self.pinWrapper.get_rModel().upperPositionLimit,
                self.pinWrapper.get_rModel().velocityLimit,
            ),
        )
        self.upperObsLim[:7] = 5
        self.upperObsLim[
            self.pinWrapper.get_rModel().nq : self.pinWrapper.get_rModel().nq + 6
        ] = 5

        self.avgObs = (self.upperObsLim + self.lowerObsLim) / 2
        self.diffObs = self.upperObsLim - self.lowerObsLim

        # Environment variables
        action_dim = len(designer_conf["controlledJointsNames"]) - 1
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(action_dim,), dtype=np.float32
        )

        observation_dim = self.desired_state.size
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(observation_dim,), dtype=np.float64
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

        for i in range(10):
            self.simulator.step(self._scaleAction(action))

        x_measured = self.simulator.getRobotState()

        observation = self._getObservation(x_measured)
        reward = self._getReward(action, observation)
        terminated = self._checkTermination(x_measured)
        # truncated = self._checkTruncation()

        return observation, reward, terminated, {}

    def _scaleAction(self, action):
        return self.torqueScale * action

    def _getObservation(self, x_measured):
        return (x_measured - self.avgObs) / self.diffObs

    def _getReward(self, action, observation):
        reward = 0
        # Posture reward + command regularization
        reward_posture = -np.linalg.norm(observation - self.desired_state)
        reward_command = -np.linalg.norm(action)

        reward = (
            self.weight_posture * reward_posture + self.weight_command * reward_command
        )

        return reward

    def _checkTermination(self, x_measured):
        return (self.timer > (self.maxTime - 1)) or (x_measured[2] < 0.7)

    def _checkTruncation(self):
        raise NotImplementedError
