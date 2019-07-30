import numpy as np
from transforms3d.euler import euler2mat, mat2euler
from numpy import unravel_index
from rotation import *
import math
from map import *
def mapCorrelation(im, x_im, y_im, vp, xs, ys):
    '''
    INPUT
    im              the map
    x_im,y_im       physical x,y positions of the grid map cells
    vp[0:2,:]       occupied x,y positions from range sensor (in physical unit)
    xs,ys           physical x,y,positions you want to evaluate "correlation"

    OUTPUT
    c               sum of the cell values of all the positions hit by range sensor
    '''
    nx = im.shape[0]
    ny = im.shape[1]
    xmin = x_im[0]
    xmax = x_im[-1]
    xresolution = (xmax-xmin)/(nx-1)
    ymin = y_im[0]
    ymax = y_im[-1]
    yresolution = (ymax-ymin)/(ny-1)
    nxs = xs.size
    nys = ys.size
    cpr = np.zeros((nxs, nys))
    for jy in range(0,nys):
        y1 = vp[1,:] + ys[jy]
        #iy = np.int16(np.round((y1-ymin)/yresolution))
        iy = np.ceil((y1 - ymin) / yresolution).astype(np.int16) - 1
        for jx in range(0,nxs):
            x1 = vp[0,:] + xs[jx]
            #ix = np.int16(np.round((x1-xmin)/xresolution))
            ix = np.ceil((x1 - xmin) / xresolution).astype(np.int16) - 1
            valid = np.logical_and( np.logical_and((iy >=0), (iy < ny)),np.logical_and((ix >=0), (ix < nx)))
            cpr[jx,jy] = np.sum(im[ix[valid],iy[valid]])
    return cpr

def softmax(x):

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def preditction(N, encoder, imu, Xt,step):
    speed_right = (encoder[0]+encoder[2])*0.0022/2
    speed_left = (encoder[1]+encoder[3])*0.0022/2
    speed = (speed_right+speed_left)/2
    omega = imu
    x1 = speed*math.cos(Xt[2,0]+omega*step/2)*math.sin(omega*step/2)/(omega*step/2)
    x2 = speed*math.sin(Xt[2,0]+omega*step/2)*math.sin(omega*step/2)/(omega*step/2)
    x3 = omega*step
    delta = np.array([x1,x2,x3]).reshape(3,1)
    Xt1 = Xt + np.tile(delta, N)+step*np.array([0*np.random.normal(0, 0.095, N),0*np.random.normal(0, 0.095, N),0*np.random.normal(0, 0.095, N)])
    return Xt1

def update(weight, mt, N, b_4, Xt1, MAP, x_rangew, y_rangew, x_im,y_im,log_odd):
    
    test_line = b_4
    
    corrs = [[],[],[],[],[],[]]
    max_corr = np.zeros([5])
    for i in range(5):
        for j in range(N):
            theta = Xt1[2, j]+(i-2)*0.001
            w_T_b= np.zeros((4, 4))
            Rz_particle = euler2mat(0, 0, theta, axes='sxyz')
            w_T_b[0:3, 0:3] = Rz_particle
            w_T_b[:, 3] = [Xt1[0, j], Xt1[1, j], 0, 1]  # 4*4 matrix
            Y = np.dot(w_T_b, test_line) # the measurement converted to one particle state 4*1081
            #Y_3 = np.dot(Rz_particle, test_line[0:3])
            #bool_position = Y[2, :] >= 0.1
            #Y = Y[:, bool_position]
            #x_range = np.arange(-0.2+Xt1[0, j],0.2+Xt1[0, j]+0.1,0.1)
            #y_range = np.arange(-0.2+Xt1[1, j],0.2+Xt1[1, j]+0.1,0.1)
            x_range = np.arange(-0.3,0.3+0.1,0.1)
            y_range = np.arange(-0.3,0.3+0.1,0.1)
            corr_matrix = mapCorrelation(mt, x_im,y_im, Y[0:3, :], x_range, y_range) # a matrix data
            #corr_matrix = mapCorrelation(mt, x_im,y_im, Y_3, x_range, y_range)
            corr = np.max(corr_matrix)
            corrs[i].append(corr)
            
        max_corr[i] = np.max(corrs[i])
    Xt1[2, j] = (max_corr.argmax()-2)*0.001+Xt1[2, j]
    
    corrs = []
    for j in range(N):
        theta = Xt1[2, j]
        w_T_b= np.zeros((4, 4))
        Rz_particle = rotationz(theta)
        w_T_b[0:3, 0:3] = Rz_particle
        w_T_b[:, 3] = [Xt1[0, j], Xt1[1, j], 0, 1]  # 4*4 matrix
        Y = np.dot(w_T_b, test_line) # the measurement converted to one particle state 4*1081
        Y_3 = np.dot(Rz_particle, test_line[0:3])
        #bool_position = Y[2, :] >= 0.1
        #Y = Y[:, bool_position]
        x_range = np.arange(-0.3,0.3+0.1,0.1)
        y_range = np.arange(-0.3,0.3+0.1,0.1)
        #x_range = np.arange(-0.2,0.2+0.1,0.1)
        #y_range = np.arange(-0.2,0.2+0.1,0.1)
        corr_matrix = mapCorrelation(mt, x_im,y_im, Y[0:3, :], x_range, y_range) # a matrix data
        #corr_matrix = mapCorrelation(mt, x_im,y_im, Y_3, x_range, y_range)
        corr = np.max(corr_matrix)
        #print(corr_matrix)
        location = unravel_index(corr_matrix.argmax(), corr_matrix.shape)

        change_x = (location[0]-3)*MAP['res']
        change_y = (location[1]-3)*MAP['res']

        Xt1[0, j] = Xt1[0, j] + change_x
        Xt1[1, j] = Xt1[1, j] + change_y
        corrs.append(corr)
    ph = softmax(np.array(corrs)).reshape(1,N)
    weight = weight * ph / np.sum(weight * ph)
    return weight, Xt1


def stratified_resampling(Xt1, weight_old, N):
    resample_X = np.zeros([3, N])
    new_weight = np.tile(1/N, N).reshape(1, N)
    j = 0
    c = weight_old[0, 0]
    for k in range(N):
        u = np.random.uniform(0, 1/N)
        beta = u + k/N
        while beta > c :
            j = j +1
            c = c + weight_old[0, j]
        resample_X[:, k] = Xt1[:, j]
    return new_weight, resample_X

def particle_filter_preonly(N,  Xt, weight, b_4, mt, MAP, x_range, y_range, x_im,y_im, encoder, imu,step,log_odd):
    Xt1 = preditction(N, encoder, imu, Xt,step)
    #weight_new, Xt1= update(weight, mt, N, b_4, Xt1, MAP, x_range, y_range, x_im, y_im,log_odd)

    #best_index = np.argmax(weight_new)
    #best_particle = Xt1[:, best_index]

    #w_T_b_best = np.zeros((4, 4))
    #Rz_particle_best = rotationz(best_particle[2])
    #w_T_b_best[0:3, 0:3] = Rz_particle_best
    #w_T_b_best[:, 3] = [best_particle[0], best_particle[1], 0, 1]
    #Neff = 1/np.dot(weight_new.reshape(1,N), weight_new.reshape(N,1))

    #if Neff < 5:
        #weight_new, Xt1 = stratified_resampling(Xt1, weight_new, N)

    return Xt1


def particle_filter(N,  Xt, weight, b_4, mt, MAP, x_range, y_range, x_im,y_im, encoder, imu,step,log_odd):
    Xt1 = preditction(N, encoder, imu, Xt,step)
    weight_new, Xt1= update(weight, mt, N, b_4, Xt1, MAP, x_range, y_range, x_im, y_im,log_odd)

    best_index = np.argmax(weight_new)
    best_particle = Xt1[:, best_index]

    w_T_b_best = np.zeros((4, 4))
    Rz_particle_best = rotationz(best_particle[2])
    w_T_b_best[0:3, 0:3] = Rz_particle_best
    w_T_b_best[:, 3] = [best_particle[0], best_particle[1], 0, 1]
    Neff = 1/np.dot(weight_new.reshape(1,N), weight_new.reshape(N,1))

    if Neff < 5:
        weight_new, Xt1 = stratified_resampling(Xt1, weight_new, N)

    return Xt1, weight_new, w_T_b_best, best_particle

def particle_filter_v2(N,  Xt, weight, b_4, cm, MAP, x_range, y_range, x_im,y_im, encoder, imu,step,log_odd):
    Xt1 = preditction(N, encoder, imu, Xt,step)
    

    best_index = np.argmax(weight)
    best_particle = Xt1[:, best_index]

    w_T_b_best = np.zeros((4, 4))
    Rz_particle_best = rotationz(best_particle[2])
    w_T_b_best[0:3, 0:3] = Rz_particle_best
    w_T_b_best[:, 3] = [best_particle[0], best_particle[1], 0, 1]
    
    log_odd, cm = mapping(log_odd, b_4, best_particle, w_T_b_best, MAP)
    weight_new, Xt1= update(weight, cm, N, b_4, Xt1, MAP, x_range, y_range, x_im, y_im,log_odd)
    
    Neff = 1/np.dot(weight_new.reshape(1,N), weight_new.reshape(N,1))

    if Neff < 5:
        weight_new, Xt1 = stratified_resampling(Xt1, weight_new, N)

    return Xt1, weight_new, w_T_b_best, best_particle,log_odd,cm