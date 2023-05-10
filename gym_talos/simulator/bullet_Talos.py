import numpy as np
import pybullet as p  # PyBullet simulator
import pybullet_data


class TalosDeburringSimulator:
    def __init__(
        self,
        URDF,
        rmodelComplete,
        controlledJointsIDs,
        enableGUI=False,
        enableGravity=True,
        dt=1e-3,
    ):

        self._setupBullet(enableGUI, enableGravity, dt)

        self._setupRobot(URDF, rmodelComplete, controlledJointsIDs)

    def _setupBullet(self, enableGUI, enableGravity, dt):
        # Start the client for PyBullet
        if enableGUI:
            self.physicsClient = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        # Set gravity (enabled by default)
        if enableGravity:
            p.setGravity(0, 0, -9.81)
        else:
            p.setGravity(0, 0, 0)

        # Set time step of the simulation
        p.setTimeStep(dt)

        # Load horizontal plane for PyBullet
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")

    def _setupRobot(self, URDF, rmodelComplete, controlledJointsIDs):
        rmodelComplete.q0 = rmodelComplete.referenceConfigurations["half_sitting"]
        self.q0 = rmodelComplete.q0
        self.initial_base_position = list(self.q0[:3])
        self.initial_base_orientation = list(self.q0[3:7])
        self.initial_joint_positions = list(self.q0[7:])

        rmodelComplete.armature = (
            rmodelComplete.rotorInertia * rmodelComplete.rotorGearRatio**2
        )

        # Load robot
        self.robotId = p.loadURDF(
            URDF,
            self.initial_base_position,
            self.initial_base_orientation,
            useFixedBase=False,
        )

        # Fetching the position of the center of mass
        # (which is different from the origin of the root link)
        self.localInertiaPos = p.getDynamicsInfo(self.robotId, -1)[3]

        # Expressing initial position wrt the CoM
        for i in range(3):
            self.initial_base_position[i] += self.localInertiaPos[i]

        self.names2bulletIndices = {
            p.getJointInfo(1, i)[1].decode(): i for i in range(p.getNumJoints(1))
        }

        self.bulletJointsIdInPinOrder = [
            self.names2bulletIndices[n] for n in rmodelComplete.names[2:]
        ]

        # Joints controlled with crocoddyl
        self.bullet_controlledJoints = [
            self.names2bulletIndices[rmodelComplete.names[i]]
            for i in controlledJointsIDs
        ]

        self._setInitialConfig()
        self._changeFriction(["leg_left_6_joint", "leg_right_6_joint"], 100, 30)
        self._setControlledJoints()

    def _setInitialConfig(self):
        """Initialize robot configuration in pyBullet

        :param q0 Intial robot configuration
        """
        for i in range(len(self.initial_joint_positions)):
            p.enableJointForceTorqueSensor(self.robotId, i, True)
            p.resetJointState(
                self.robotId,
                self.bulletJointsIdInPinOrder[i],
                self.initial_joint_positions[i],
            )

    def _changeFriction(self, names, lateralFriction=100, spinningFriction=30):
        for n in names:
            idx = self.names2bulletIndices[n]
            p.changeDynamics(
                self.robotId,
                idx,
                lateralFriction=lateralFriction,
                spinningFriction=spinningFriction,
            )

    def _setControlledJoints(self):
        """Define joints controlled by pyBullet

        :param rmodelComplete Complete model of the robot
        :param ControlledJoints List of ControlledJoints
        """
        # Disable default position controler in torque controlled joints
        # Default controller will take care of other joints
        p.setJointMotorControlArray(
            self.robotId,
            jointIndices=self.bullet_controlledJoints,
            controlMode=p.VELOCITY_CONTROL,
            forces=[0.0 for m in self.bullet_controlledJoints],
        )

    def _createTargetVisual(self, target):
        """Create visual representation of the target to track

        The visual will not appear unless the physics client is set to
        SHARED_MEMMORY
        :param target Position of the target in the world
        """
        RADIUS = 0.01
        LENGTH = 0.02
        blueBox = p.createVisualShape(
            shapeType=p.GEOM_CAPSULE,
            rgbaColor=[0, 0, 1, 1.0],
            visualFramePosition=[0.0, 0.0, 0.0],
            radius=RADIUS,
            length=LENGTH,
            halfExtents=[0.0, 0.0, 0.0],
        )

        self.target_MPC = p.createMultiBody(
            baseMass=0.0,
            baseInertialFramePosition=[0, 0, 0],
            baseVisualShapeIndex=blueBox,
            basePosition=[target[0], target[1], target[2]],
            useMaximalCoordinates=True,
        )

    def getRobotState(self):
        """Get current state of the robot from pyBullet"""
        # Get articulated joint pos and vel
        xbullet = p.getJointStates(self.robotId, self.bullet_controlledJoints)
        q = [x[0] for x in xbullet]
        vq = [x[1] for x in xbullet]

        # Get basis pose
        pos, quat = p.getBasePositionAndOrientation(self.robotId)
        # Get basis vel
        v, w = p.getBaseVelocity(self.robotId)

        # Concatenate into a single x vector
        x = np.concatenate([q, vq])

        # Magic transformation of the basis translation, as classical in Bullet.
        # x[:3] -= self.localInertiaPos

        return x

    def step(self, torques):
        """Do one step of simulation"""
        self._applyTorques(torques)
        p.stepSimulation()

    def _applyTorques(self, torques):
        """Apply computed torques to the robot"""
        p.setJointMotorControlArray(
            self.robotId,
            self.bullet_controlledJoints,
            controlMode=p.TORQUE_CONTROL,
            forces=torques,
        )

    def reset(self):
        # Reset base
        p.resetBasePositionAndOrientation(
            self.robotId,
            self.initial_base_position,
            self.initial_base_orientation,
            self.physicsClient,
        )
        p.resetBaseVelocity(
            self.robotId, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], self.physicsClient
        )
        # Reset joints
        for i in range(len(self.initial_joint_positions)):
            p.resetJointState(
                self.robotId,
                self.bulletJointsIdInPinOrder[i],
                self.initial_joint_positions[i],
            )

    def end(self):
        """Ends connection with pybullet."""
        p.disconnect()
