import numpy as np

import math

import map_utils

from scipy.special import expit

def mapping(log_odd,b_4,  best_particle, w_T_b_best, MAP, b = 4):
    
	occu_max = 30
	occu_min = -30
	body_point = b_4
	world_point = np.dot(w_T_b_best, body_point)
	#bool_position = world_point[2,:] >0 #>= 0.1 + 0.93
	#world_point = world_point[:, bool_position]
	x_particle = np.ceil((best_particle[0]-MAP['xmin'])/MAP['res']).astype(np.int16)-1
	y_particle = np.ceil((best_particle[1]-MAP['ymin'])/MAP['res']).astype(np.int16)-1


	for i in range(world_point.shape[1]):
		ex = np.ceil((world_point[0,i] - MAP['xmin'])/MAP['res']).astype(np.int16)-1 #in map coordinate to the pixel
		ey = np.ceil((world_point[1,i] - MAP['ymin'])/MAP['res']).astype(np.int16)-1 #in map coordinate to the pixel
		if ex > 1 and ey > 1 and ex < MAP['sizex'] and ey < MAP['sizey']:
			pass
		else:
			continue
		scan_section = map_utils.bresenham2D(x_particle.item(), y_particle.item(), ex.item(), ey.item()) #a beam section for just one end
		scan_section = scan_section.astype(int)

		log_odd[scan_section[0][-1], scan_section[1][-1:]] = \
			log_odd[scan_section[0][-1], scan_section[1][-1]] + math.log(b)

		log_odd[scan_section[0][1:-1], scan_section[1][1:-1]] = \
			log_odd[scan_section[0][1:-1], scan_section[1][1:-1]] + math.log(1 / b)
	log_odd[log_odd>occu_max] = occu_max
	log_odd[log_odd<occu_min] = occu_min
	P_occupied = 1- expit(-log_odd)
	bool_occupied_cells = P_occupied > 0.5
	bool_free_cells = P_occupied < 0.5
	mt = bool_free_cells *(-1) + bool_occupied_cells * 1

	return log_odd,mt