import pybullet as p
import pybullet_data
import numpy as np
import time

'''
Define a class for the UR5 pybullet simulator

NOTE: https://gramaziokohler.github.io/compas_fab/latest/examples/03_backends_ros/07_ros_create_urdf_ur5_with_measurement_tool.html

the above link has a way to mfg custuom UDRF file format
'''

UR5PATH = "ur_e_description/urdf/ur5e.urdf"
UR5_BASE_POSITION = [0,0,0]             ## Base position in x,y,z coordinates
UR5_BASE_ORIENTATION = [0,0,0,1]        ## Base orientation in unit quaternions



class UR5manipulator():
    
    def __init__(self):
        
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0,0,-9.81)
        #self.plane = p.loadURDF("plane.urdf")
        self.robotID = p.loadURDF(UR5PATH, basePosition = UR5_BASE_POSITION, baseOrientation = UR5_BASE_ORIENTATION,
                                  useFixedBase = False)
        self.controlableJoints = []
        
        self.numJoints = p.getNumJoints(self.robotID)
        print("NUmber of Joints: ", self.numJoints)
        
        for jointIndex in range(self.numJoints):

            if p.getJointInfo(self.robotID,jointIndex)[2] == 0:
                self.controlableJoints.append(jointIndex)
                
        print(self.controlableJoints)
        self.endEffectorLink = self.controlableJoints[-1]
        print("The end effector link is:",self.endEffectorLink)
        
    def turnOFFJointControl(self):
        p.setJointMotorControlArray(self.robotID,self.controlableJoints,p.VELOCITY_CONTROL,forces = [0]*6)
    
    def setJointAngles(self,jointAngles):
        zero = [0]*6 
        
        
        for __ in range(200):
            p.setJointMotorControlArray(self.robotID,self.controlableJoints,controlMode = p.POSITION_CONTROL ,targetPositions = jointAngles,targetVelocities = zero,
                                    positionGains = [1]*6,velocityGains = [1]*6)
            p.stepSimulation()
            
    def getJointStates(self):
        joint_states = p.getJointStates(self.robotID, self.controlableJoints)
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        joint_torques = [state[3] for state in joint_states]
        
        return joint_positions, joint_velocities, joint_torques
               
                
    def getForwardKinematics(self):
        
        endEffector = p.getLinkState(self.robotID,self.endEffectorLink)
        pos,ori = endEffector[0],p.getEulerFromQuaternion(endEffector[1])
        
        return pos,ori
    
    def getInverseKinematics(self,endEffector):
        '''
        End Effector pos : to be in pos, ori
        '''
        jointAngles = p.calculateInverseKinematics(ur5.robotID,self.endEffectorLink,endEffector[0],
                                                   p.getQuaternionFromEuler(endEffector[1]))
        
        return jointAngles
        
    def calculateJacobian(self,jointNagles):
        endEffectorState = p.getLinkState(self.robotID,self.endEffectorLink,computeForwardKinematics=True)
        
        zero = [0]*6
        
        jacobianLinear,jacobianRotational = p.calculateJacobian(self.robotID,self.endEffectorLink,endEffectorState[2],jointNagles,zero,zero)
        
        jacobianLinear = np.array(jacobianLinear)

        jacobianRotational = np.array(jacobianRotational)
        
        return np.vstack((jacobianLinear,jacobianRotational))
    
    def calculateAnalyticalVelocityMap(self):
        '''
        will retuen A matrix as required by the HandBook of Robotics:
        
        '''
        endEffector = p.getLinkState(self.robotID,self.endEffectorLink)
        quat = endEffector[1]
        r,pit,y = p.getEulerFromQuaternion(quat)
        
        AMatrix3 = np.array([
            [1,0,np.sin(pit)],
            [0,np.cos(r),-np.cos(pit)*np.sin(r)],
            [0,np.sin(r),np.cos(pit)*np.cos(r)]
        ])
        #print(AMatrix3)
        AMatrix6 = np.block([
            [np.eye(3),       np.zeros((3,3))],
            [np.zeros((3,3)),   AMatrix3]
        ])
        
        return AMatrix6
        
    
    def getDynamicMatrices(self):
        
        jointInfo = p.getJointStates(self.robotID,self.controlableJoints)

        jointAngles ,jointVel = [],[]
        
        for i in range(len(self.controlableJoints)):
            jointAngles.append(jointInfo[i][0])
            jointVel.append(jointInfo[i][1])
        zero = [0]*6
        
        massMatrix = np.array(p.calculateMassMatrix(self.robotID,jointAngles))

        gravityMatrix = np.array(p.calculateInverseDynamics(self.robotID,jointAngles,zero,zero))
        
        coriolisMatrix = np.array(p.calculateInverseDynamics(self.robotID,jointAngles,jointVel,zero)) - gravityMatrix
                
        return massMatrix,gravityMatrix,coriolisMatrix
    
    
    def ForceGUI(self, forces, max_limit = 10, min_limit = -10):
        fxId = p.addUserDebugParameter("fx", min_limit, max_limit, forces[0]) #force along x
        fyId = p.addUserDebugParameter("fy", min_limit, max_limit, forces[1]) #force along y
        fzId = p.addUserDebugParameter("fz", min_limit, max_limit, forces[2]) #force along z
        mxId = p.addUserDebugParameter("mx", min_limit, max_limit, forces[3]) #moment along x
        myId = p.addUserDebugParameter("my", min_limit, max_limit, forces[4]) #moment along y
        mzId = p.addUserDebugParameter("mz", min_limit, max_limit, forces[5]) #moment along z
        return [fxId, fyId, fzId, mxId, myId, mzId]
    
    def readGUIparams(self, ids):
        val1 = p.readUserDebugParameter(ids[0])
        val2 = p.readUserDebugParameter(ids[1])
        val3 = p.readUserDebugParameter(ids[2])
        val4 = p.readUserDebugParameter(ids[3])
        val5 = p.readUserDebugParameter(ids[4])
        val6 = p.readUserDebugParameter(ids[5])
        return np.array([val1, val2, val3, val4, val5, val6])   
        
    
        
        
if __name__ == "__main__":
    
    '''
    free fall
    '''
    
    # ur5 = UR5manipulator()
    
    # p.setJointMotorControlArray(ur5.robotID,ur5.controlableJoints,p.VELOCITY_CONTROL,forces = [0]*6)
    
    # while p.isConnected():
    #     time.sleep(0.01)
    #      ## should do free fall
    #     p.setJointMotorControlArray(ur5.robotID,ur5.controlableJoints,p.TORQUE_CONTROL,forces = np.zeros(6))
    #     p.stepSimulation()
    
    # '''
    # velocity control
    # '''
    # ur5 = UR5manipulator()
    # while p.isConnected:
    #     #p.setJointMotorControlArray(ur5.robotID,ur5.controlableJoints,p.VELOCITY_CONTROL, targetVelocities = [-0.1]*6)
    #     pos,ori = ur5.getForwardKinematics()
    #     jacobian = ur5.calculateJacobian(ur5.getInverseKinematics([pos,ori]))
    #     ## asume vel = [0,0,0,-1,0,0]
    #     vel = [0,0,0,-0.01,0.00,0.01]
    #     vel = np.array(vel)
    #     jointVel = np.dot( np.linalg.inv(jacobian), vel ).tolist()
    #     p.setJointMotorControlArray(ur5.robotID,ur5.controlableJoints,p.VELOCITY_CONTROL, targetVelocities = jointVel)
    #     p.stepSimulation()
    
    '''
    check forward kinematics
    
    '''
    ur5 = UR5manipulator()
    initJointAngles = [0.5, -0.707, 1.0, 0, -1, 0.2]
    print("init Joint Angles are:",initJointAngles)
    ur5.setJointAngles(initJointAngles)
    imverseKin = ur5.getInverseKinematics(ur5.getForwardKinematics())
    print("angles from inverse kinematics",imverseKin)
    
    
    '''
    apply counter gravity torque
    '''
    # ur5 = UR5manipulator()
    
    
    # initJointAngles = [0.5, -0.707, 1.0, -1.57, -1.57, -1.57]
    # ur5.setJointAngles(initJointAngles)
    # ur5.calculateJacobian(initJointAngles)
    # p.setJointMotorControlArray(ur5.robotID,ur5.controlableJoints,p.VELOCITY_CONTROL,forces = [0]*6)
    
    # while True:
    #     massMatrix,gravityMatrix,coriolisMatrix = ur5.getDynamicMatrices()
    #     torque = gravityMatrix
    #     p.setJointMotorControlArray(ur5.robotID,ur5.controlableJoints,p.TORQUE_CONTROL,forces = torque)
    #     p.stepSimulation()
    
    '''
    # ### apply check quaternion map.
    '''
    # pi = np.pi
    # givenEuler = [-pi/4,pi/6,pi/4]
    # r,pit,y = givenEuler
    # AMatrix3 = np.array([
    #         [1,0,np.sin(pit)],
    #         [0,np.cos(r),-np.cos(pit)*np.sin(r)],
    #         [0,np.sin(r),np.cos(pit)*np.cos(r)]
    #     ])
    # print(AMatrix3)
    # quat = p.getQuaternionFromEuler(givenEuler) 
    # euler = p.getEulerFromQuaternion(quat)
    # matrix = np.array([
    #         [1,0,np.sin(euler[1])],
    #         [0,np.cos(euler[0]),-np.cos(euler[1])*np.sin(euler[0])],
    #         [0,np.sin(r),np.cos(euler[1])*np.cos(euler[0])]
    #     ])
    
    # print("the quat is:",quat)
    # print("the euler angles for the quaternion are:",euler)
    # print("the matrixc formed is",matrix)
    
    
    
    
    ####
    ##  IMPEDANCE CONTROL DIDN'T WORK
    
    # '''
    # impedance control. 
    # '''       
    # ur5 = UR5manipulator()
    
    
    # initJointAngles = [0.5, -0.707, 1.0, -1.57, -1.57, -1.57]
    # ur5.setJointAngles(initJointAngles)
    # #time.sleep(5)

    # desiredEndEffector = ur5.getForwardKinematics()
    # desiredEndEffector = np.hstack((desiredEndEffector[1],desiredEndEffector[0]))
    # p.setJointMotorControlArray(ur5.robotID,ur5.controlableJoints,p.VELOCITY_CONTROL,forces = [0]*6)
    
    # for i in range(ur5.numJoints):
    #     p.changeDynamics(ur5.robotID, i, linearDamping=0.0, angularDamping=0.0, jointDamping=0.0)
        
    # kMatrix = np.diag([10,10,10,10,10,10])
    # cMatrix = np.diag([100,100,100,100,100,100])
        
    # while True:
    #     jointInfo = p.getJointStates(ur5.robotID,ur5.controlableJoints)
    #     jointAngles ,jointVel = [],[]
        
    #     for i in range(len(ur5.controlableJoints)):
    #         jointAngles.append(jointInfo[i][0])
    #         jointVel.append(jointInfo[i][1])
        
    #     pos,Quat = ur5.getForwardKinematics()
    #     euler = Quat
    #     endEffector = np.hstack((euler,pos))
    #     endEffectorVel = np.dot(ur5.calculateJacobian(jointAngles) , jointVel) 
        
    #     deltaX = -endEffector + desiredEndEffector
        
    #     threshHold = 1e-6
    #     indices = deltaX < threshHold
    #     deltaX[indices] = 0
        
        
    #     print("deltaX",deltaX)
        
    #     deltaV = -endEffectorVel
    #     indices = deltaV < threshHold
    #     deltaV[indices] = 0
    #     print("DeltaV",deltaV)
        
    #     # springTorque = np.linalg.multi_dot((ur5.calculateJacobian(jointAngles).T,kMatrix,deltaX))
    #     # dampingTorque = np.linalg.multi_dot((ur5.calculateJacobian(jointAngles).T,cMatrix,deltaV))
        
    #     springForce = np.dot(kMatrix,deltaX)
    #     print("springForce:",springForce)
        
    #     dampingForce = np.dot(cMatrix,deltaV)
    #     print("dampingForce",dampingForce)
    #     spatialImpedance = springForce + dampingForce
    #     jointImpedance = np.dot(ur5.calculateJacobian(jointAngles).T, spatialImpedance)
    #     print("jointImpedance:",jointImpedance)
        
    #     massMatrix,gravityMatrix,coriolisMatrix = ur5.getDynamicMatrices()
    #     torque = gravityMatrix + jointImpedance
    #     p.setJointMotorControlArray(ur5.robotID,ur5.controlableJoints,p.TORQUE_CONTROL,forces = torque)
    #     p.stepSimulation()
        
    
    
    pass


