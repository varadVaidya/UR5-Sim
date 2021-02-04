import pybullet as p
import pybullet_data
import numpy as np
from math import sqrt
from UR5 import UR5manipulator

###################
#########
### WRONG IMPLEMENTAION.###########

#### The Gui forces are the forces experienced by the end effector

arm = UR5manipulator()          # setting up the manipulator
p.setRealTimeSimulation(0)
## set init joint anglees and positon robot their
initJointAngles = [0.7, -0.707,0.5, 0.1, 0, 0]
arm.setJointAngles(initJointAngles)

## find the gravity matrix for the start. 
massMatrix,gravityMatrix,coriolisMatrix = arm.getDynamicMatrices()
torque = gravityMatrix      ## apply the gravty torque at the start.
## find desired position to apply impedance control their
xDes = np.array(arm.getForwardKinematics()).reshape(6)
print("Xdes:",xDes)
#input("check")

## turn of the internal damping...

for i in range(arm.numJoints):
    p.changeDynamics(arm.robotID, i, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)

Kp = np.diag(5*[0.1],k=-1)+np.diag(6*[1.5],k=0)+np.diag(5*[0.1],k=1)
Kd = np.diag(5*[0.1],k=-1)+np.diag(6*[1.2],k=0)+np.diag(5*[0.1],k=1)
externalForce = np.array([0,0,0,0,0,0])
GUIForce = arm.ForceGUI(externalForce)
arm.turnOFFJointControl()       # turn off joint control to apply torque

while True:
    
    externalForce = arm.readGUIparams(GUIForce)
  
    #forward dymanics
    jointAngles,jointVelocities,jointTorques = arm.getJointStates()
    #print("jointAngles are:",jointAngles)
    if len(jointAngles) != 6:
        raise ValueError 

    #### calculate the dynamic matrices in joint space
    massMatrix,gravityMatrix,coriolisMatrix = arm.getDynamicMatrices()
    #print("massMatrix",massMatrix)
    jacobian = arm.calculateJacobian(jointAngles)
    jacobianInv = np.linalg.inv(jacobian)
    massMatrixInv = np.linalg.inv(massMatrix)
    ## calculate the dynamic mstirces in cartesian space
    spatialInertia = np.linalg.inv( np.linalg.multi_dot(( jacobian, massMatrixInv, jacobian.T)) )
    
    spatialGravity = np.dot(jacobianInv.T,gravityMatrix)
    
    
    ## calculate current end effector position
    currentX = np.array(arm.getForwardKinematics()).reshape(6)
    deltaX = xDes - currentX    
    
    aMatrix = arm.calculateAnalyticalVelocityMap()
    appliedForce = np.linalg.multi_dot(( np.linalg.inv(aMatrix).T , Kp ,deltaX)) - \
        np.linalg.multi_dot((Kd,jacobian,jointVelocities)) + spatialGravity
    #appliedForce = np.linalg.multi_dot((Kp,deltaX)) - np.linalg.multi_dot((Kd,np.linalg.inv(aMatrix),jacobian,jointVelocities)) + gravityMatrix
    torque = np.dot(jacobian.T,appliedForce) 
    
    p.applyExternalForce(arm.robotID,arm.endEffectorLink,externalForce[0:3],[0,0,0],flags=p.LINK_FRAME)
    p.applyExternalTorque(arm.robotID,arm.endEffectorLink,externalForce[3:6],p.LINK_FRAME)  
    p.setJointMotorControlArray(arm.robotID,arm.controlableJoints,controlMode = p.TORQUE_CONTROL,forces=torque) 
    p.stepSimulation()

    #p.setJointMotorControlArray(arm.robotID,arm.controlableJoints,controlMode = p.TORQUE_CONTROL,forces=torque)




