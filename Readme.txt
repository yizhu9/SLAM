The top level module for this project is SLAM.py and Texture_vec.py.

Run SLAM.py to get grid map(also log_odds),trajectory(position and angular) and Texture_vec.py to get texture map. 

Basically, you should run SLAM.py first to get the trajectory of the best particle and the map and save them(be careful about the file name). 

Then use the trajectory and the map to run the texture map.

The mapping.py, localization.py, rotation.py, map_util.py and the transformation folder are the files i write for my top level module.

The localization.py file is my particle filter which contain the predict, update and resampling step.

The .ipynb file is to make change to the origin file and see the difference.
