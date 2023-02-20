# Import libraries
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt


a = np.random.randint(0, 5, (3, 4, 3))
b = np.random.randint(0, 5, (3, 4, 3))

print(a)
print(b)
c = np.absolute(a-b)
print(c)
print(np.sum(c))