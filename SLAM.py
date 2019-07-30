#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from localization import *
import matplotlib.pyplot as plt; plt.ion()
from map import *
import cv2
from map_utils import *
dataset = 20
from PIL import Image
from texture import *
from scipy import signal


# In[2]:


with np.load("Encoders%d.npz" % dataset) as data:
    encoder_counts = data["counts"]  # 4 x n encoder counts
    encoder_stamps = data["time_stamps"]  # encoder time stamps

with np.load("Hokuyo%d.npz" % dataset) as data:
    lidar_angle_min = data["angle_min"]  # start angle of the scan [rad]
    lidar_angle_max = data["angle_max"]  # end angle of the scan [rad]
    lidar_angle_increment = data["angle_increment"]  # angular distance between measurements [rad]
    lidar_range_min = data["range_min"]  # minimum range value [m]
    lidar_range_max = data["range_max"]  # maximum range value [m]
    lidar_ranges = data["ranges"]  # range data [m] (Note: values < range_min or > range_max should be discarded)
    lidar_stamps = data["time_stamps"]  # acquisition times of the lidar scans

with np.load("Imu%d.npz" % dataset) as data:
    imu_angular_velocity = data["angular_velocity"]  # angular velocity in rad/sec
    imu_linear_acceleration = data["linear_acceleration"]  # Accelerations in gs (gravity acceleration scaling)
    imu_stamps = data["time_stamps"]  # acquisition times of the imu measurements
if dataset!= 23:
    with np.load("Kinect%d.npz" % dataset) as data:
        disp_stamps = data["disparity_time_stamps"]  # acquisition times of the disparity images
        rgb_stamps = data["rgb_time_stamps"]  # acquisition times of the rgb images

N = 1
X = np.zeros((1,N)) #arange(MAP['xmin'], MAP['xmax'], 0.4)
X = np.random.uniform(-0.4,0.4,N).reshape(1,N)
Y = np.zeros((1,N)) #arange(MAP['ymin'], MAP['ymax'], 0.4)
Y = np.random.uniform(-0.4,0.4,N).reshape(1,N)
theta = np.zeros((1,N)) #np.arange(-180, 177, 3.6) / 180 * math.pi
Xt = np.array([X, Y, theta]).reshape(3, N)  # where N should be 100, particles, Xt0
weight = (np.array([1/N] * N).reshape(1, N))


# init MAP
MAP = {}
MAP['res'] = 0.1  # meters
MAP['xmin'] = -30  # meters
MAP['ymin'] = -30
MAP['xmax'] = 30
MAP['ymax'] = 30
MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))  # cells
MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.int8)  # DATA TYPE: char or int8
tm = np.zeros((MAP['sizex'], MAP['sizey'], 3), dtype=np.uint8)

x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map

x_range = np.arange(-0.4,0.4+0.1,0.1)
y_range = np.arange(-0.4,0.4+0.1,0.1)
idxs = []
idx2 = []
imu_fit = []
j = 0
trajectory = []
log_odd = np.zeros((MAP['sizex'],MAP['sizey']))
cm = MAP['map']
print(encoder_stamps.shape[0])
for i in range(encoder_stamps.shape[0]):
    checking_value = encoder_stamps[i].squeeze()
    idx = np.where(abs(imu_stamps - checking_value) == min(abs(imu_stamps - checking_value)))
    if idx[0][0]==imu_stamps.shape[0]-1:
        idx2.append(idx[0][0])
        idxs.append(idx[0][0]-1)
    else:
        idxs.append(idx[0][0])
        idx2.append(idx[0][0]+1)
        
b, a = signal.butter(1, 0.1, 'low')
tra = []
#imu_stamps_fit = (imu_stamps[idxs]+imu_stamps[idxs+1])/2

imu_angular_velocity_fit = (imu_angular_velocity[:,idxs]+imu_angular_velocity[:,idx2])/2
#imu_angular_velocity_fit = imu_angular_velocity[:,idxs]
yaw_data = signal.filtfilt(b, a, imu_angular_velocity_fit[2,:])
#yaw_data = imu_angular_velocity_fit[2,:]
for i in range(lidar_stamps.shape[0]-1):
    print(i)
    
    angles = np.arange(-135 / 180 * math.pi, 135 / 180 * math.pi, 0.00436332)
    indx_l = np.argmin(abs(lidar_stamps-encoder_stamps[i]))
    ranges = lidar_ranges[:, indx_l]
    indValid = np.logical_and((ranges < 30), (ranges > 0.1))
    ranges = ranges[indValid]
    angles_2 = angles[indValid]
    xs0 = np.array([ranges * np.cos(angles_2)+0.1])
    ys0 = np.array([ranges * np.sin(angles_2)])

    b_3 = np.concatenate([np.concatenate([xs0, ys0], axis=0), np.zeros(xs0.shape)], axis=0)
    b_4 = np.concatenate([b_3, np.ones(xs0.shape)], axis=0)
    #b_4 = np.array([xs0,ys0,0,1]).reshape(4,-1)
    step = (encoder_stamps[i+1]-encoder_stamps[i])
    Xt, weight, w_T_b_best, best_particle ,log_odd,cm= particle_filter_v2(N, Xt, weight, b_4, cm, MAP, x_range, y_range, x_im, y_im, encoder_counts[:,i], yaw_data[i],step,log_odd)
    tx = np.ceil((best_particle[0] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    ty = np.ceil((best_particle[1] - MAP['ymin']) / MAP['res']).astype(np.int16) - 1
    tra.append([tx,ty])
    trajectory.append(best_particle)
    '''
    texture--not finished
    if (encoder_stamps[i].squeeze() <= rgb_stamps[-1].squeeze()) and (encoder_stamps[i].squeeze() >= rgb_stamps[0].squeeze()):

        indx_r = np.argmin(rgb_stamps-encoder_stamps[i])
        indx_d = np.argmin(disp_stamps-encoder_stamps[i])
        img = Image.open('../dataRGBD/Disparity20/disparity20_'+str(indx_d+1)+'.png')
        disparity_img = np.array(img.getdata(),  np.uint16).reshape(img.size[1], img.size[0])
        RGB = cv2.imread('../dataRGBD/RGB20/rgb20_'+str(indx_r+1)+'.png')
        tm = texture_mapping(w_T_b_best, RGB,MAP, disparity_img, tm)
        plt.imshow(tm)
        plt.pause(0.0001)
    '''
    #show_lidar(angles_2,ranges)
    #plt.scatter(trajectory, ty)
    #plt.imshow(cm,cmap = 'hot')
    dr = np.array(trajectory)
    #plt.scatter(dr[:,1],dr[:,0],marker = '.',linewidth = 0.1)
    if i%20 ==0:
        #plt.scatter(trajectory, ty)
        plt.imshow(cm,cmap = 'hot')
        dr = np.array(tra)
        plt.scatter(dr[:,1],dr[:,0],marker = '.',linewidth = 0.1)
        #plt.savefig("23_p_"+str(i)+".png")
        plt.show()
        plt.pause(0.0001)


#plt.imshow(cm,cmap = 'hot')
#plt.show()


# In[ ]:


plt.imshow(cm,cmap = 'hot')
plt.show()


# In[ ]:


#np.save('best_p_20',trajectory)
#np.save('map20',cm)
#print(trajectory)

