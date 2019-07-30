#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import math
import numpy as np
def rotationz(raw):
    R = np.zeros([3,3])
    R[0,0] = math.cos(raw)
    R[1,1] = math.cos(raw)
    R[2,2] = 1
    R[0,1] = -math.sin(raw)
    R[1,0] = math.sin(raw)
    return R

