import gym
import numpy as np
import pinocchio as pin

from ..utils.modelLoader import TalosDesigner

from gym_talos.simulator.bullet_Talos import TalosDeburringSimulator


class EnvTalosBase(gym.Env):
    def __init__(self, params_designer, params_env, GUI=False) -> None:
        # Parameters
        self._init_parameters(params_env)

        # Robot model loader
        self.pinWrapper = TalosDesigner(
            URDF=params_designer["URDF"],
            SRDF=params_designer["SRDF"],
            toolPosition=params_designer["controlledJoints"],
            controlledJoints=params_designer["toolPosition"],
        )

        self.rmodel = self.pinWrapper.rmodel

        # Simulator
        self.simulator = TalosDeburringSimulator(
            URDF=params_designer["URDF"],
            rmodelComplete=self.pinWrapper.rmodelComplete,
            controlledJointsIDs=self.pinWrapper.controlledJointsID,
            enableGUI=GUI,
            dt=1e-4,
        )

        # Environment variables
        self.timer = 0
        if self.normalizeObs:
            self._init_obsNormalizer()
        self.desired_state = self.pinWrapper.get_x0()

        action_dim = len(params_designer["controlled_joints_names"]) - 1
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(action_dim,), dtype=np.float32
        )

        if self.normalizeObs:
            observation_dim = self.desired_state.size
            self.observation_space = gym.spaces.Box(
                low=-1, high=1, shape=(observation_dim,), dtype=np.float64
            )
        else:
            observation_dim = self.desired_state.size
            self.observation_space = gym.spaces.Box(
                low=-5, high=5, shape=(observation_dim,), dtype=np.float64
            )

    def _init_parameters(self, params_env):
        self.numSimulationSteps = params_env["numSimulationSteps"]
        self.normalizeObs = params_env["normalizeObs"]
        #   Action normalization parameter
        self.torqueScale = np.array(params_env["torqueScale"])

        #   Stop conditions
        self.maxTime = params_env["maxTime"]
        self.minHeight = params_env["minHeight"]

        #   Target
        self.targetPos = params_env["targetPosition"]

        #   Reward parameters
        self.weight_posture = params_env["weightPosture"]
        self.weight_command = params_env["weightCommand"]
        self.weight_alive = params_env["weightAlive"]

    def reset(self, *, seed=None, options=None):
        self.timer = 0
        self.simulator.reset()

        x_measured = self.simulator.getRobotState()
        self.pinWrapper.update_reduced_model(x_measured)

        return np.float32(x_measured)

    def step(self, action):
        self.timer += 1

        for _ in range(self.numSimulationSteps):
            self.simulator.step(self._scaleAction(action))

        x_measured = self.simulator.getRobotState()

        self.pinWrapper.update_reduced_model(x_measured)

        observation = self._getObservation(x_measured)
        reward = self._getReward(action, observation)
        terminated = self._checkTermination(x_measured)

        return observation, reward, terminated, {}

    def close(self):
        self.simulator.end()

    def _getObservation(self, x_measured):
        if self.normalizeObs:
            return self._obsNormalizer(x_measured)
        else:
            return x_measured

    def _getReward(self, action, observation):
        # target distance
        reward_toolPosition = np.linalg.norm(
            self.pinWrapper.oMtool.translation - self.targetPos
        )

        reward = reward_toolPosition
        return reward

    def _checkTermination(self, x_measured):
        stop_time = self.timer > (self.maxTime - 1)
        if self.minHeight > 0:
            stop_height = x_measured[2] < self.minHeight
        else:
            stop_height = False
        return stop_time or stop_height

    def _checkTruncation(self):
        raise NotImplementedError

    def _scaleAction(self, action):
        return self.torqueScale * action

    def _init_obsNormalizer(self):
        self.lowerObsLim = np.concatenate(
            (
                self.pinWrapper.get_rmodel().lowerPositionLimit,
                -self.pinWrapper.get_rmodel().velocityLimit,
            ),
        )
        self.lowerObsLim[:7] = -5
        self.lowerObsLim[
            self.pinWrapper.get_rmodel().nq : self.pinWrapper.get_rmodel().nq + 6
        ] = -5

        self.upperObsLim = np.concatenate(
            (
                self.pinWrapper.get_rmodel().upperPositionLimit,
                self.pinWrapper.get_rmodel().velocityLimit,
            ),
        )
        self.upperObsLim[:7] = 5
        self.upperObsLim[
            self.pinWrapper.get_rmodel().nq : self.pinWrapper.get_rmodel().nq + 6
        ] = 5

        self.avgObs = (self.upperObsLim + self.lowerObsLim) / 2
        self.diffObs = self.upperObsLim - self.lowerObsLim

    def _obsNormalizer(self, x_measured):
        return (x_measured - self.avgObs) / self.diffObs
