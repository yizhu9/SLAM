#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from utils import *
from transforms3d.euler import mat2euler
import scipy.linalg


# In[2]:


def upper(x):
    output = np.zeros([3,3])
    output[0,1] = -x[2]
    output[1,0] = x[2]
    output[0,2] = x[1]
    output[2,0] = -x[1]
    output[1,2] = -x[0]
    output[2,1] = x[0]
    return output
def derri(q):
    output = np.eye(4)
    output[2,2]=0
    output[0,2] = -q[0]/q[2]
    output[1,2] = -q[1]/q[2]
    output[3,2] = -q[3]/q[2]
    return output/q[2]
def circle(r):
    output = np.zeros([4,6])
    output[0:3,0:3] = r[3]*np.eye(3)
    output[0:3,3:6] = -upper(r[0:3])
    return output
def pif(x):
    return x/x[2]
def upper_pose(x):
    output = np.zeros([4,4])
    output[0:3,0:3] = upper(x[3:6])
    output[0:3,3] = x[0:3,0]
    
    return output


# In[3]:

#change the file name and you can select which file to read.
if __name__ == '__main__':
	filename = "./data/0027.npz"
	t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)


# In[4]:


print(t.shape)
print(features.shape)
print(linear_velocity.shape)
print(rotational_velocity.shape)
print(K.shape)
print(b)
print(cam_T_imu.shape)
# (a) IMU Localization via EKF Prediction

	# (b) Landmark Mapping via EKF Update

	# (c) Visual-Inertial SLAM (Extra Credit)

	# You can use the function below to visualize the robot pose over time
	#visualize_trajectory_2d(world_T_imu,show_ori=True)


# In[5]:


# In[21]:


mu = np.zeros([4,4])
mu[0:3,0:3] = np.eye(3)
mu[3,3] =1
M = features.shape[1]
cov = 0.5*np.eye(3*features.shape[1]+6)
cov[3*M:3*M+6,3*M:3*M+6] = 0.000*np.eye(6)
world_T_imu = np.zeros([4,4,t.shape[1]-1])
world_T_imu_up = np.zeros([4,4,t.shape[1]-1])

K_4 = np.zeros([4,4])
K_4[0:2,0:3] = K[0:2,0:3]
K_4[2:4,0:3] = K[0:2,0:3]
K_4[2,3] = -K[0,0]*b
K_34 = np.zeros([3,4])
K_34[0:2,0:3] = K[0:2,0:3]
K_34[2,3] = K[0,0]*b

D = np.zeros([4,3])
D[0:3,0:3] = np.eye(3)


flag = np.zeros(features.shape[1])
m_mu = np.zeros([4,features.shape[1]])

for i in range(t.shape[1]-1):
    V = 8000 *np.eye(4)
    W = np.eye(6)*0.05
    # (a) IMU Localization via EKF Prediction
    dt = t[:,i+1]-t[:,i]
    transform1 = np.zeros([4,4])
    transform1[0:3,0:3] = upper(rotational_velocity[:,i])
    transform1[0:3,3] = linear_velocity[:,i]
    mu = scipy.linalg.expm(-dt*transform1).dot(mu)
    world_T_imu_up[:,:,i] = np.linalg.inv(mu)
    #print(transform1)
    transform2 = np.zeros([6,6])
    transform2[0:3,0:3] = upper(rotational_velocity[:,i])
    transform2[3:6,3:6] = upper(rotational_velocity[:,i])
    transform2[0:3,3:6] = upper(linear_velocity[:,i])
    cov[3*M:3*M+6,3*M:3*M+6] = scipy.linalg.expm(-dt*transform2).dot(cov[3*M:3*M+6,3*M:3*M+6].dot(scipy.linalg.expm(-dt*transform2).T))+dt*dt*W
    # (b) Landmark Mapping via EKF Update

    # (c) Visual-Inertial SLAM (Extra Credit)
    #mapping
    count=0
    for k in range(features.shape[1]):
        if sum(features[:,k,i])!=-4 and flag[k]!=0:
            count+=1
    if count==0:
        pass
    else:
        
        
        z_hat = np.zeros([4,count])
        I_V_map = np.zeros([4*count,4*count])
        z = np.zeros([4,count])
        
        H = np.zeros([4*count,3*features.shape[1]+6])
        count=0
        for j in range(features.shape[1]):
            if sum(features[:,j,i])!=-4 and flag[j]!=0:
                q = cam_T_imu.dot(mu.dot(m_mu[:,j]))
                dpi_dq = derri(q)
                
                z_hat[:,count] = K_4.dot(pif(cam_T_imu.dot(mu.dot(m_mu[:,j]))))
                z[:,count] = features[:,j,i]
                #print((z_hat[:, count] - z[:, count]).T.dot(z_hat[:, count] - z[:, count]))
                if ((z_hat[:, count] - z[:, count]).T.dot(z_hat[:, count] - z[:, count])) > 100000:
                    z[:, count] = z_hat[:, count]
                #H[i] = K.dot(dpi_dq).dot(cam_T_imu).dot(circle(mu.dot(features[:,j,i])))
                H[4*count:4*(count+1),3*M:3*M+6] = K_4.dot(dpi_dq).dot(cam_T_imu).dot(circle(mu.dot(m_mu[:,j])))
                H_i_j = K_4.dot(derri(cam_T_imu.dot(mu.dot(m_mu[:,j])))).dot(cam_T_imu.dot(mu)).dot(D)
                H[count*4:(count+1)*4,3*j:3*(j+1)] = H_i_j
                I_V_map[4*count:4*(count+1),4*count:4*(count+1)] = V
                count = count+1
                
        Kt1_t = cov.dot(H.T).dot(np.linalg.inv(H.dot(cov.dot(H.T))+I_V_map))
        
        Kt_pose = Kt1_t[3*M:3*M+6,:]
        Kt_map = Kt1_t[0:3*M,:]
        mu = scipy.linalg.expm(upper_pose(Kt_pose.dot((z-z_hat).reshape(-1,1,order='F')))).dot(mu)
        temp = (Kt_map.dot((z-z_hat).reshape(-1,1,order='F'))).reshape(3,-1,order='F')
        m_mu = m_mu + D.dot(temp)
        cov = cov - (Kt1_t.dot(H).dot(cov))

    world_T_imu[:,:,i] = np.linalg.inv(mu)
    
    for k in range(features.shape[1]):
        if sum(features[:, k, i]) != -4:
            if flag[k] == 0:
                d = features[0, k, i] - features[2, k, i]
                z = K[0, 0] * b / d
                optical_bp = np.array(
                    [(features[0, k, i] - K[0, 2]) * z / K[0, 0], (features[1, k, i] - K[1, 2]) * z / K[1, 1], z,
                     1])

                m_mu[:, k] = np.linalg.inv(mu).dot(np.linalg.inv(cam_T_imu)).dot(optical_bp.T)
                #print((m_mu[:, k]-world_T_imu[:,3,i]).T.dot(m_mu[:, k]-world_T_imu[:,3,i]))
                flag[k] = 1
                if((m_mu[:, k]-world_T_imu[:,3,i]).T.dot(m_mu[:, k]-world_T_imu[:,3,i]))>200000:
                    m_mu[:, k]= np.array([0,0,0,1]).T
                    flag[k] = 0



visualize_trajectory_2d(world_T_imu,m_mu,path_name="Path",show_ori=True)
visualize_trajectory_2d(world_T_imu_up,m_mu,path_name="Path",show_ori=True)

# In[ ]:






