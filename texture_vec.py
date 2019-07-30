#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math
from transforms3d.euler import euler2mat, mat2euler
from matplotlib import pyplot as plt
import os, cv2
import time
from PIL import Image

dataset = 20

K_cal = np.array([[585.05108211, 0, 242.94140713],
                    [0, 585.05108211, 315.83800193],
                      [0, 0, 1]])
K_cal_inv = np.linalg.inv(K_cal)

R_oc = np.zeros([4, 4])
R_oc[0:3, 0:3] = [[0, 0, 1],
                [-1, 0, 0],
                [0, -1, 0]]
R_oc[3, 3] = 1




def transformation(best_particle):
    '''
    create a the transformation matrix from body to world frame
    '''
    Rz = np.zeros([3,3])
    w_T_b = np.zeros((4, 4))
    theta = best_particle[2]
    Rz[0,0] = math.cos(theta)
    Rz[1,1] = math.cos(theta)
    Rz[2,2] = 1
    Rz[0,1] = -math.sin(theta)
    Rz[1,0] = math.sin(theta)
    w_T_b[0:3, 0:3] = Rz
    w_T_b[:, 3] = [best_particle[0], best_particle[1], 0, 1]
    return w_T_b


def sensor2body(pitch, yaw):
    R = euler2mat(0, pitch, yaw, axes='sxyz')
    b_T_c = np.zeros([4,4])
    b_T_c[0:3, 0:3] = R
    b_T_c[:, 3] = [0.18, 0.005, 0.36, 1]
    return b_T_c


def texturing(best_particle, RGB, MAP, depth_dt, tm):
    w_T_b = transformation(best_particle)
    w_T_b[2, 3] = 0.127  # z-axis height of center of gravity
    b_T_rgb = sensor2body(0.36, 0.021)
    w_T_c_rgb = np.dot(w_T_b, b_T_rgb)  # where w_T_b is from the particle filter
    deprgb = np.zeros([480, 640, 3])

    res_x = 6; res_y =8
    xsize = int(480/res_x); ysize=int(640/res_y)
    depth_dt = depth_dt[0:480:res_x, 0:640:res_y]
    i = np.arange(0, 480, res_x)
    j = np.arange(0, 640, res_y)
    dd = -0.00304*depth_dt + 3.31
    indvalid = np.where(dd > 0)
    depth = 1.03/dd
    
    rgbi = np.ceil((i.repeat(ysize).reshape(xsize, ysize) * 526.37 + dd*(-4.5*1750.46) + 19276.0)/585.051)
    rgbj = np.tile(np.ceil((j * 526.37 + 16662.0) / 585.051), xsize).reshape(xsize, ysize)

    deprgb[i[indvalid[0]], j[indvalid[1]], :] = RGB[np.int16(rgbi[indvalid]), np.int16(rgbj[indvalid]), :]
    i = i.repeat(ysize).reshape(xsize, ysize)
    j = j.repeat(xsize).reshape(xsize, ysize)
    u_v = np.stack((rgbi-240, rgbj-320, np.ones((xsize, ysize))), axis=2)
    temp = np.array(u_v).dot(K_cal_inv[None, None, :]).squeeze()
    sl = np.stack((depth, temp[:, :, 0] * depth, temp[:, :, 1] * depth, np.ones((xsize, ysize))), axis=2)
    temp2 = R_oc[None, None, :].dot(sl.reshape((xsize, ysize, 4, 1))).squeeze().swapaxes(2, 0).swapaxes(1, 0)
    sw = w_T_c_rgb[None, None, :].dot(temp2.reshape((xsize, ysize, 4, 1))).squeeze()
    ind_sw = np.where(np.logical_and(sw[2, :, :] < 1,sw[2, :, :]>-1))

    s_x = np.ceil((sw[0, ind_sw[0], ind_sw[1]] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    s_y = np.ceil((sw[1, ind_sw[0], ind_sw[1]] - MAP['xmin']) / MAP['res']).astype(np.int16) - 1
    tm[s_x, s_y, :] = RGB[np.int16(rgbi[indvalid][ind_sw[0]]), np.int16(rgbj[indvalid][ind_sw[1]]), :]

    return tm




# texture

with np.load("Encoders%d.npz" % dataset) as data:
    encoder_counts = data["counts"]  # 4 x n encoder counts
    encoder_stamps = data["time_stamps"]  # encoder time stamps
with np.load("Kinect%d.npz" % dataset) as data:
    disp_stamps = data["disparity_time_stamps"]  # acquisition times of the disparity images
    rgb_stamps = data["rgb_time_stamps"]  # acquisition times of the rgb images
#with np.load('trajectory%d.npz' % dataset) as data:
 #   tract_xy = data['xy']
  #  tract_c = data['cell']
tract_xy = np.load("lyu/best_p_%d.npy" % dataset)
map_cm = np.load("lyu/map%d.npy"%dataset) 
stamps = encoder_stamps
MAP = {}
MAP['res'] = 0.1  # meters
MAP['xmin'] = -30  # meters
MAP['ymin'] = -30
MAP['ymax'] = 30
MAP['xmax'] = 30
MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))  # cells
MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
tex_map = np.zeros((MAP['sizex'], MAP['sizey'], 3), dtype=np.uint8)

ts = time.time()


# In[2]:


tract_xy = np.transpose(tract_xy)


# In[3]:


print(tract_xy.shape)


# In[ ]:


for i in range(stamps.shape[0] - 1):
    tb = time.time()
    if (stamps[i].squeeze() <= rgb_stamps[-1].squeeze()) and (stamps[i].squeeze() >= rgb_stamps[0].squeeze()):
        indx_r = np.argmin(abs(rgb_stamps - stamps[i]))
        indx_d = np.argmin(abs(disp_stamps - stamps[i]))
        img = Image.open('../dataRGBD/Disparity' + str(dataset) + '/disparity' + str(dataset) + '_' + str(indx_d + 1) + '.png')
        disparity_img = np.array(img.getdata(), np.uint16).reshape(img.size[1], img.size[0])
        RGB = cv2.imread('../dataRGBD/RGB' + str(dataset) + '/rgb' + str(dataset) + '_' + str(indx_d + 1) + '.png')
        RGB = cv2.cvtColor(RGB, cv2.COLOR_RGB2BGR)
        tex_map = texturing(tract_xy[:, i], RGB, MAP, disparity_img, tex_map)
        #plt.imshow(tex_map)
        #plt.pause(0.0001)
        #plt.title('Current iteration: %d' % i)
    te = time.time()
    print('Current iteration: %d, Total update took: %s sec, Expected time : %d min.\n' % (i, (te-ts), (te-tb)*stamps.shape[0]/60))

plt.figure()
plt.imshow(tex_map)


# In[ ]:


plt.figure()
boolcm = (map_cm<0).reshape(map_cm.shape[0],map_cm.shape[1],1)

plt.imshow(tex_map*boolcm)
plt.savefig('lyu/texture20_1.png')

