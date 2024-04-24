import pinocchio
import copy
import numpy as np
from numpy.linalg import norm, solve

class damped_IK():

    def __init__(self, urdf_path, constraints, eps = 1e-3, max_iteration=1000, dt=0.1, damp=1e-12) -> None:
        
        self.model = pinocchio.buildModelFromUrdf(urdf_path)
        self.data  = self.model.createData()

        self.constraints = copy.deepcopy(constraints)

        # Solver parameters
        self.eps = eps
        self.max_iteration = max_iteration
        self.dt = dt
        self.damp = damp

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
            q_path.append(q)

            i += 1

        return (q_path if return_path else q, 
                success)