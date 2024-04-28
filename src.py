import pinocchio
import crocoddyl
import copy
import numpy as np
from numpy.linalg import norm, solve
import os

class damped_IK():

    def __init__(self, urdf_path, constraints, eps = 1e-3, max_iteration=1000, dt=0.1, damp=1e-12, verbose=False) -> None:
        
        self.model = pinocchio.buildModelFromUrdf(urdf_path)
        self.data  = self.model.createData()

        self.constraints = copy.deepcopy(constraints)

        # Solver parameters
        self.eps = eps
        self.max_iteration = max_iteration
        self.dt = dt
        self.damp = damp

        self.verbose = verbose

        # Preprocess constraints
        for i in range(len(constraints)):
            self.constraints[i]['frame'] = self.model.getJointId(self.constraints[i]["frame"])
            self.constraints[i]['weigth'] = np.ones(6)*self.constraints[i]['weigth']
            
            if self.constraints[i]['type'] == 'position':
                self.constraints[i]['weigth'][-3:] = 0
            elif self.constraints[i]['type'] == 'orientation':
                self.constraints[i]['weigth'][:3] = 0
            else:
                print("Error in constraint type")

    def run(self, q, targets, return_path=False):

        self.targets = copy.deepcopy(targets)

        assert len(self.targets) == len(self.constraints), "Should have same number of constraints as target"

        # Preprocess constraints
        for i in range(len(self.targets)):
            if self.constraints[i]['type'] == 'position':
                self.targets[i] = pinocchio.SE3(np.eye(3), self.targets[i])
            else:
                self.targets[i] = pinocchio.SE3(self.targets[i], np.zeros(3))
        
        # Run gradient descent
        i=0
        q_path = [q]
        while True:
            pinocchio.forwardKinematics(self.model, self.data, q)

            # Compute error of each target
            errors = []
            for constraint, target in zip(self.constraints, self.targets):
                iMd = self.data.oMi[constraint['frame']].actInv(target)
                errors.append(constraint["weigth"]*pinocchio.log(iMd).vector)  # in joint frame

            # Early stop if close enough
            all_errors = np.concatenate(errors)
            if np.mean(np.abs(all_errors[all_errors!=0])) < self.eps:
                success = True
                break
            if i >= self.max_iteration:
                success = False
                break
            
            # Compute Jacobian of each target joint to get error of gradient
            joint_velocity = np.zeros(self.model.nv)
            for constraint, error, target in zip(self.constraints, errors, self.targets):
                J = pinocchio.computeJointJacobian(self.model, self.data, q, constraint["frame"])  # in joint frame
                iMd = self.data.oMi[constraint['frame']].actInv(target)
                J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)

                # Joint position update is basically the joint velocity
                joint_velocity += - J.T.dot(solve(J.dot(J.T) + self.damp * np.eye(6), error))
            joint_velocity /= len(errors)

            # Fancy update of state q using pinocchio function
            q = pinocchio.integrate(self.model, q, joint_velocity*self.dt)
            q[q<self.model.lowerPositionLimit] = self.model.lowerPositionLimit[q<self.model.lowerPositionLimit]
            q[q>self.model.upperPositionLimit] = self.model.upperPositionLimit[q>self.model.upperPositionLimit]
            q_path.append(q)

            i += 1

        if self.verbose:
            pinocchio.forwardKinematics(self.model, self.data, q)
            for constraint, target in zip(self.constraints, self.targets):
                if constraint['type'] == 'position': 
                    print("-----\nPosition constraint of joint ", constraint['frame'])
                    print('Reached: ', self.data.oMi[constraint['frame']].translation)
                    print('Target:', target.translation)

                if constraint['type'] == 'orientation': 
                    print("-----\nOrientation constraint of joint ", constraint['frame'])
                    print('Reached:\n', self.data.oMi[constraint['frame']].rotation)
                    print('Target:\n', target.rotation)


        return (q_path if return_path else q, 
                success)
    

class croco_IK():

    def __init__(self, urdf_path, constraints, eps = 1e-3, max_iteration=1000, dt=0.1, damp=1e-12, verbose=False) -> None:
        
        
        # robot = pinocchio.robot_wrapper.RobotWrapper.BuildFromURDF(urdf_path, 
        #                                                            os.path.join(*urdf_path.split('/')[:-1], '..', 'meshes'))

        robot = pinocchio.robot_wrapper.RobotWrapper.BuildFromURDF('/home/albericlajarte/Desktop/example-robot-data/robots/panda_description/urdf/panda.urdf',
                                                           '/home/albericlajarte/Desktop/example-robot-data/robots/panda_description/meshes')
        
        self.model = robot.model
        # self.data  = self.model.createData()
        self.state = crocoddyl.StateMultibody(self.model)

        self.constraints = copy.deepcopy(constraints)

        # Solver parameters
        self.eps = eps
        self.max_iteration = max_iteration
        self.dt = dt
        self.damp = damp

        self.verbose = verbose

        # # Preprocess constraints
        # for i in range(len(constraints)):
        #     self.constraints[i]['frame'] = self.model.getJointId(self.constraints[i]["frame"])
        #     self.constraints[i]['weigth'] = np.ones(6)*self.constraints[i]['weigth']
            
        #     if self.constraints[i]['type'] == 'position':
        #         self.constraints[i]['weigth'][-3:] = 0
        #     elif self.constraints[i]['type'] == 'orientation':
        #         self.constraints[i]['weigth'][:3] = 0
        #     else:
        #         print("Error in constraint type")

    def run(self, q, targets, return_path=False):

        state = crocoddyl.StateMultibody(self.model)

        ## Create the cost functions ##
        runningCostModel = crocoddyl.CostModelSum(state)
        terminalCostModel = crocoddyl.CostModelSum(state)

        # State constraint to reach target poses
        for i, (constraint, target) in enumerate(zip(self.constraints, targets)):

            if constraint['type'] == 'position':
                new_cost = crocoddyl.CostModelResidual(
                            state,
                            crocoddyl.ResidualModelFrameTranslation(state, 
                                                                    self.model.getFrameId(constraint['frame']), 
                                                                    target))
                
                    
            elif constraint['type'] == 'orientation':
                new_cost = crocoddyl.CostModelResidual(
                            state,
                            crocoddyl.ResidualModelFrameRotation(state, 
                                                                self.model.getFrameId(constraint['frame']), 
                                                                target))
                
            runningCostModel.addCost(f"Pose{i}", new_cost, constraint['weigth'])
            terminalCostModel.addCost(f"Pose{i}", new_cost, 100*constraint['weigth'])

        # State and Control regularization and limits
        activation_xbounds = crocoddyl.ActivationModelQuadraticBarrier(
            crocoddyl.ActivationBounds( np.concatenate((self.model.lowerPositionLimit, -np.ones(state.nv)*100)), 
                                        np.concatenate((self.model.upperPositionLimit, np.ones(state.nv)*100))
                                        )
        )
    
        xRegCost = crocoddyl.CostModelResidual(state, activation_xbounds, crocoddyl.ResidualModelState(state))
        uRegCost = crocoddyl.CostModelResidual(state, crocoddyl.ResidualModelControl(state))

        # Then let's added the running and terminal cost functions
        runningCostModel.addCost("stateReg", xRegCost, 1e1)
        runningCostModel.addCost("ctrlReg", uRegCost, 1e-7)
        terminalCostModel.addCost("stateReg", xRegCost, 1e1)
        terminalCostModel.addCost("ctrlReg", uRegCost, 1e-7)

        ## Create the actuation model ##
        actuationModel = crocoddyl.ActuationModelFull(state)

        # Create the action model
        runningModel = crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(
                state, actuationModel, runningCostModel
            ),
            self.dt,
        )
        terminalModel = crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(
                state, actuationModel, terminalCostModel
            )
        )

        ## Solve problem ##
        x0 = np.concatenate([q, np.zeros_like(q)])
        problem = crocoddyl.ShootingProblem(x0, [runningModel] * self.max_iteration, terminalModel)

        # Creating the DDP solver for this OC problem, defining a logger
        ddp = crocoddyl.SolverDDP(problem)

        # Solving it with the DDP algorithm
        ddp.solve()

        if self.verbose:
            data = self.model.createData()
            pinocchio.forwardKinematics(self.model, data, np.array(ddp.xs)[-1, :len(q)])
            pinocchio.updateFramePlacements(self.model, data)

            for constraint, target in zip(self.constraints, targets):
                if constraint['type'] == 'position': 
                    print("-----\nPosition constraint of joint ", constraint['frame'])
                    print('Reached: ', data.oMf[self.model.getFrameId(constraint['frame'])].translation)
                    print('Target:', target)

                if constraint['type'] == 'orientation': 
                    print("-----\nOrientation constraint of joint ", constraint['frame'])
                    print('Reached:\n', data.oMf[self.model.getFrameId(constraint['frame'])].rotation)
                    print('Target:\n', target)


        return (np.array(ddp.xs)[:, :len(q)] if return_path else ddp.xs[-1],
                ddp.isFeasible)


        