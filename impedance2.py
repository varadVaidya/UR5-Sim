import numpy as np
import pybullet as p
import pybullet_data
from UR5 import UR5manipulator


### init. the robot arm
arm = UR5manipulator()
# initJointAngles = [0.7, -0.707,0.5, 0., 0, 0]
# arm.setJointAngles(initJointAngles)

xDes = np.array(arm.getForwardKinematics()).reshape(6)
currentX = np.array(arm.getForwardKinematics()).reshape(6)
print (xDes - currentX)

Kp = 0*np.eye(6)
Kd = 0*np.eye(6)

### set the damping to zero
for i in range(arm.numJoints):
    p.changeDynamics(arm.robotID, i, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)

## find the initial torque to apply:

arm.turnOFFJointControl()
while True:
    ## Apply torque and sense the forces
    #
    forceExperienced = np.array([0,0,0,0,0,0])
    ## Forward Dynamics
    #
    
    ## calculate the parameters again
    jointAngles,jointVelocities,jointTorques = arm.getJointStates()    
    massMatrix,gravityMatrix,coriolisMatrix = arm.getDynamicMatrices()
    jacobian = arm.calculateJacobian(jointAngles)
    analyticJacobian = np.dot(  np.linalg.inv(  arm.calculateAnalyticalVelocityMap()    ) ,jacobian)
    
    
    currentX = np.array(arm.getForwardKinematics()).reshape(6)
    deltaX = xDes - currentX
    
    currentVel = np.dot( analyticJacobian , jointVelocities)

        
    ## implement impedance control
    forceImpedance = np.dot( Kp , deltaX) - np.dot ( Kd , currentVel )
    
    jointImpedance = np.dot(analyticJacobian.T,forceImpedance)
    
    torque = jointImpedance + gravityMatrix
    
    p.setJointMotorControlArray(arm.robotID,arm.controlableJoints,controlMode = p.TORQUE_CONTROL,forces = torque)
    p.stepSimulation()
      
    
    
    
    
    
    

