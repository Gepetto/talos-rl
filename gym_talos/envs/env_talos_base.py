import gym
import pinocchio as pin
import yaml

from sobec import RobotDesigner
from mpc_pointing import MPC_Point


from gym_talos.simulator.bullet_Talos import TalosDeburringSimulator

class EnvTalosBase(gym.Env):
    def __init__(self, targetPos, designer_conf) -> None:
        self.pinWrapper = RobotDesigner()
        self.pinWrapper.initialize(designer_conf)

        gripper_SE3_tool = pin.SE3.Identity()
        gripper_SE3_tool.translation[0] = designer_conf.toolFramePos[0]
        gripper_SE3_tool.translation[1] = designer_conf.toolFramePos[1]
        gripper_SE3_tool.translation[2] = designer_conf.toolFramePos[2]
        self.pinWrapper.addEndEffectorFrame(
            "deburring_tool", "gripper_left_fingertip_3_link", gripper_SE3_tool
        )

        self.simulator = TalosDeburringSimulator(
            URDF=designer_conf.URDF,
            targetPos=targetPos,
            rmodelComplete=self.pinWrapper.get_rModelComplete(),
            controlledJointsIDs=self.pinWrapper.get_controlledJointsIDs(),
            enableGUI=False,
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[ObsType, dict]:
        return super().reset(seed=seed, options)

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        self.simulator.step(action)

        x_measured = self.simulator.getRobotState()

        self.pinWrapper.updateReducedModel(x_measured)

        observation = self._getObservation()
        reward = self._getReward()
        terminated = self._checkTermination()
        truncated = self._checkTruncation()

        return observation, reward, terminated, truncated, {}

    def _getObservation(self):
        raise NotImplementedError

    def _getReward(self):
        raise NotImplementedError

    def _checkTermination(self):
        raise NotImplementedError

    def _checkTruncation(self):
        raise NotImplementedError