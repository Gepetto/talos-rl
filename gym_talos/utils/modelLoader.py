import numpy as np
import example_robot_data
import pinocchio as pin


class TalosDesigner:
    def __init__(self, URDF, SRDF, toolPosition, controlledJoints, **kwargs):
        modelPath = example_robot_data.getModelPath(URDF)
        self.URDF_path = modelPath + URDF

        if True:
            self.rmodelComplete = pin.buildModelFromUrdf(
                self.URDF_path, pin.JointModelFreeFlyer()
            )
        else:
            self.rmodelComplete = pin.buildModelFromUrdf(self.URDF_path)

        self._refineModel(self.rmodelComplete, SRDF)
        self._addLimits()

        self._addTool(toolPosition)

        self._buildReducedModel(controlledJoints)

    def _refineModel(self, model, SRDF):
        """Load additional information from SRDF file
        rotor inertia and gear ratio

        :param model Model of the robot to refine
        :param SRDF Path to SRDF file containing data to add to model
        """
        modelPath = example_robot_data.getModelPath(SRDF)

        pin.loadRotorParameters(model, modelPath + SRDF, False)
        model.armature = np.multiply(
            model.rotorInertia.flat, np.square(model.rotorGearRatio.flat)
        )

        pin.loadReferenceConfigurations(model, modelPath + SRDF, False)

    def _addLimits(self):
        """Add free flyers joint limits"""
        self.rmodelComplete.upperPositionLimit[:7] = 1
        self.rmodelComplete.lowerPositionLimit[:7] = -1

    def _addTool(self, toolPosition):
        """Add frame corresponding to the tool

        :param toolPosition Position of the tool frame in parent frame
        """
        placement_tool = pin.SE3.Identity()
        placement_tool.translation[0] = toolPosition[0]
        placement_tool.translation[1] = toolPosition[1]
        placement_tool.translation[2] = toolPosition[2]

        self.rmodelComplete.addBodyFrame(
            "driller",
            self.rmodelComplete.getJointId("gripper_left_joint"),
            placement_tool,
            self.rmodelComplete.getFrameId("gripper_left_fingertip_3_link"),
        )

        self.endEffectorId = self.rmodelComplete.getFrameId("driller")

    def _buildReducedModel(self, controlledJointsName):
        """Build a reduce model for which only selected joints are controlled

        :param controlledJoints List of the joints to control
        """
        self.q0Complete = self.rmodelComplete.referenceConfigurations["half_sitting"]

        # Check that controlled joints belong to model
        for joint in controlledJointsName:
            if joint not in self.rmodelComplete.names:
                print("ERROR")

        self.controlledJointsID = [
            i
            for (i, n) in enumerate(self.rmodelComplete.names)
            if n in controlledJointsName
        ]

        # Make list of blocked joints
        lockedJointsID = [
            self.rmodelComplete.getJointId(joint)
            for joint in self.rmodelComplete.names[1:]
            if joint not in controlledJointsName
        ]

        self.rmodel = pin.buildReducedModel(
            self.rmodelComplete, lockedJointsID, self.q0Complete
        )
        self.rdata = self.rmodel.createData()

        # Define a default State
        self.q0 = self.rmodel.referenceConfigurations["half_sitting"]
        self.rmodel.defaultState = np.concatenate([self.q0, np.zeros(self.rmodel.nv)])

    def update_reduced_model(self, x_measured):
        pin.forwardKinematics(self.rmodel, self.rdata, x_measured[: self.rmodel.nq])
        pin.updateFramePlacements(self.rmodel, self.rdata)

        self.oMtool = self.rdata.oMf[self.endEffectorId]
