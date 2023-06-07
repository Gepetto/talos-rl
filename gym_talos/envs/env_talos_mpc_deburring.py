import gymnasium as gym
import numpy as np
import pinocchio as pin
from deburring_mpc import RobotDesigner

from gym_talos.simulator.bullet_Talos import TalosDeburringSimulator


class EnvTalosBase(gym.Env):
    def __init__(self, params_env, params_designer, param_ocp, GUI=False) -> None:
        self._init_parameters(params_env)

        # Robot wrapper
        self.pinWrapper = RobotDesigner()
        self.pinWrapper.initialize(params_designer)

        gripper_SE3_tool = pin.SE3.Identity()
        gripper_SE3_tool.translation[0] = params_designer["toolFramePos"][0]
        gripper_SE3_tool.translation[1] = params_designer["toolFramePos"][1]
        gripper_SE3_tool.translation[2] = params_designer["toolFramePos"][2]
        self.pinWrapper.add_end_effector_frame(
            "deburring_tool",
            "gripper_left_fingertip_3_link",
            gripper_SE3_tool,
        )

        self.rmodel = self.pinWrapper.get_rmodel()

        # OCP

        self._init_ocp(param_ocp)

        # Simulator
        self.simulator = TalosDeburringSimulator(
            URDF=params_designer["urdf_path"],
            rmodelComplete=self.pinWrapper.get_rmodel_complete(),
            controlledJointsIDs=self.pinWrapper.get_controlled_joints_ids(),
            enableGUI=GUI,
            dt=self.timeStepSimulation,
        )

        action_dimension = self.rmodel.nq
        observation_dimension = len(self.simulator.getRobotState())
        self._init_env_variables(action_dimension, observation_dimension)

    def _init_ocp(self, param_ocp):
        oMtarget = pin.SE3.Identity()
        oMtarget.translation[0] = self.targetPos[0]
        oMtarget.translation[1] = self.targetPos[1]
        oMtarget.translation[2] = self.targetPos[2]

        oMtarget.rotation = np.array([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])

        self.crocoWrapper = self.OCP(param_ocp, self.pinWrapper)
        self.crocoWrapper.initialize(self.pinWrapper.get_x0(), oMtarget)

        self.ddp = self.crocoWrapper.solver

        # Problem can be modified here to fit the needs of the RL

    def _init_parameters(self, params_env):
        """Load environment parameters from provided dictionnary

        Args:
            params_env: kwargs for the environment
        """
        # Simumlation timings
        self.timeStepSimulation = float(params_env["timeStepSimulation"])
        self.numSimulationSteps = params_env["numSimulationSteps"]

        self.normalizeObs = params_env["normalizeObs"]

        #   Stop conditions
        self.maxTime = params_env["maxTime"]
        self.minHeight = params_env["minHeight"]

        #   Target
        self.targetPos = params_env["targetPosition"]

        #   Reward parameters
        self.weight_target = params_env["w_target_pos"]
        self.weight_command = params_env["w_control_reg"]
        self.weight_truncation = params_env["w_penalization_truncation"]

    def _init_env_variables(self, action_dimension, observation_dimension):
        """Initialize internal variables of the environment

        Args:
            action_dimension: Dimension of the action space
            observation_dimension: Dimension of the observation space
        """
        self.timer = 0

        self.maxStep = int(
            self.maxTime / (self.timeStepSimulation * self.numSimulationSteps),
        )

        if self.normalizeObs:
            self._init_obsNormalizer()

        self.torqueScale = np.array(self.rmodel.effortLimit)

        action_dim = action_dimension
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(action_dim,),
            dtype=np.float32,
        )

        observation_dim = observation_dimension
        if self.normalizeObs:
            observation_dim = len(self.simulator.getRobotState())
            self.observation_space = gym.spaces.Box(
                low=-1,
                high=1,
                shape=(observation_dim,),
                dtype=np.float64,
            )
        else:
            observation_dim = len(self.simulator.getRobotState())
            self.observation_space = gym.spaces.Box(
                low=-5,
                high=5,
                shape=(observation_dim,),
                dtype=np.float64,
            )

    def close(self):
        """Properly shuts down the environment.

        Closes the simulator windows.
        """
        self.simulator.end()

    def reset(self, *, seed=None, options=None):
        """Reset the environment

        Brings the robot back to its half-sitting position

        Args:
            seed: seed that is used to initialize the environment's PRNG.
                Defaults to None.
            options: Additional information can be specified to reset the environment.
                Defaults to None.

        Returns:
            Observation of the initial state.
        """
        self.timer = 0
        self.simulator.reset()

        x_measured = self.simulator.getRobotState()
        self.pinWrapper.update_reduced_model(x_measured)

        return self._getObservation(x_measured)

    def step(self, action):
        """Execute a step of the environment

        One step of the environment is numSimulationSteps of the simulator with the same
        command.
        The model of the robot is updated using the observation taken from the
        environment.
        The termination and condition are checked and the reward is computed.

        Args:
            action: Normalized action vector

        Returns:
            Formatted observations
            Reward
            Boolean indicating this rollout is done
        """
        self.timer += 1
        torques = self._scaleAction(action)

        for _ in range(self.numSimulationSteps):
            self.simulator.step(torques)

        x_measured = self.simulator.getRobotState()

        self.pinWrapper.update_reduced_model(x_measured)

        observation = self._getObservation(x_measured)
        terminated = self._checkTermination(x_measured)
        truncated = self._checkTruncation(x_measured)
        reward = self._getReward(torques, x_measured, terminated, truncated)

        # No difference between termination and truncation in this version of Gym
        done = terminated or truncated

        return observation, reward, done, {}

    def _getObservation(self, x_measured):
        """Formats observations

        Normalizes the observation obtained from the simulator if nomalizeObs = True

        Args:
            x_measured: observation array obtained from the simulator

        Returns:
            Fromated observations
        """
        if self.normalizeObs:
            observation = self._obsNormalizer(x_measured)
        else:
            observation = x_measured
        return np.float32(observation)

    def _getReward(self, torques, x_measured, terminated, truncated):
        """Compute step reward

        The reward is composed of:
            - A bonus when the environment is still alive (no constraint has been
              infriged)
            - A cost proportional to the norm of the torques
            - A cost proportional to the distance of the end-effector to the target

        Args:
            torques: torque vector
            x_measured: observation array obtained from the simulator
            terminated: termination bool
            truncated: truncation bool

        Returns:
            Scalar reward
        """
        if truncated:
            reward_alive = 0
        else:
            reward_alive = 1

        # command regularization
        reward_command = -np.linalg.norm(torques)
        # target distance
        reward_toolPosition = -np.linalg.norm(
            self.pinWrapper.get_end_effector_pos() - self.targetPos,
        )

        return (
            self.weight_target * reward_toolPosition
            + self.weight_command * reward_command
            + self.weight_truncation * reward_alive
        )

    def _checkTermination(self, x_measured):
        """Check the termination conditions.

        Environment is terminated when the task has been successfully carried out.
        In our case it means that maxTime has been reached.

        Args:
            x_measured: observation array obtained from the simulator

        Returns:
            True if the environment has been terminated, False otherwise
        """
        return self.timer > (self.maxStep - 1)

    def _checkTruncation(self, x_measured):
        """Checks the truncation conditions.

        Environment is truncated when a constraint is infriged.
        There are two possible reasons for truncations:
         - Loss of balance of the robot:
            Rollout is stopped if position of CoM is under threshold
            No check is carried out if threshold is set to 0
         - Infrigement of the kinematic constraints of the robot
            Rollout is stopped if configuration exceeds model limits


        Args:
            x_measured: observation array obtained from the simulator

        Returns:
            True if the environment has been truncated, False otherwise.
        """
        # Balance
        truncation_balance = (not (self.minHeight == 0)) and (
            self.pinWrapper.CoM[2] < self.minHeight
        )

        # Limits
        truncation_limits_position = (
            x_measured[: self.rmodel.nq] > self.rmodel.upperPositionLimit
        ).any() or (x_measured[: self.rmodel.nq] < self.rmodel.lowerPositionLimit).any()
        truncation_limits_speed = (
            x_measured[-self.rmodel.nv :] > self.rmodel.velocityLimit
        ).any()
        truncation_limits = truncation_limits_position or truncation_limits_speed

        # Explicitely casting from numpy.bool_ to bool
        return bool(truncation_balance or truncation_limits)

    def _scaleAction(self, action):
        """Scales normalized actions to obtain robot torques

        Args:
            action: normalized action array

        Returns:
            torque array
        """
        return self.torqueScale * action

    def _init_obsNormalizer(self):
        """Initializes the observation normalizer using robot model limits"""
        self.lowerObsLim = np.concatenate(
            (
                self.rmodel.lowerPositionLimit,
                -self.rmodel.velocityLimit,
            ),
        )

        self.upperObsLim = np.concatenate(
            (
                self.rmodel.upperPositionLimit,
                self.rmodel.velocityLimit,
            ),
        )

        self.avgObs = (self.upperObsLim + self.lowerObsLim) / 2
        self.diffObs = self.upperObsLim - self.lowerObsLim

    def _obsNormalizer(self, x_measured):
        """Normalizes the observation taken from the simulator

        Args:
            x_measured: observation array obtained from the simulator

        Returns:
            normalized observation
        """
        return (x_measured - self.avgObs) / self.diffObs
