"""
Created on Wed Feb 12 11:19:33 2020

@author: Anil
"""
# %load testAlg.py
import numpy as np
import sys
from DA import DA

data, max_c, gamma, interactive = sys.argv[1:]
data = np.load(data)


obj = DA(data=data, max_m=int(max_c), gamma=float(gamma),  interactive = int(interactive))
obj.train()