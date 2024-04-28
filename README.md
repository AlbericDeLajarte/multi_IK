# Multi_IK

## Overview

Run inverse kinematic algorithm to reach multiple task space targets.

The solver is initalized with a urdf file describing the robot, and a list of constraints specifying the kind of target you need.
Here is an example defining a position and orientation constraint on joint1 of your robot:

``` python
constraints = [{'frame': 'joint1', 'type': 'position', 'weigth':1},
               {'frame': 'joint1', 'type': 'orientation', 'weigth':1},
               ]
IK_solver = damped_IK('my_robot.urdf', constraints)
```

The targets are given as a list of numpy arrays, either a (3,) array for position, or a (3,3) rotation matrix for orientation.

Check the `test_IK.ipynb` notebook for a full example

## Features
- [x] Fast using pinocchio
- [x] Multiple targets
- [x] Closest solution from initial guess
- [x] Stable around singularity
- [ ] Joint limits
- [ ] Self collision
- [ ] Joint velocity target
