import numpy as np
from self_functions import *

x = np.array([1, 2, 3, 4, 5])
x = np.insert(x, 0, 4)
print(x.shape)