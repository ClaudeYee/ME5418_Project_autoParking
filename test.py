from matplotlib.animation import FuncAnimation

import Auto_Parking
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from parameter import *

A = np.zeros([10, 10])
B = np.full([10, 10], -1)

for i in range(PARKLOT_SIZE[0]):
    for j in range(PARKLOT_SIZE[1]):
        A[i, j] = 1

A[9, 9] = -1

B[4, 4] = 1

S = Auto_Parking.State(A, B)

a = [6, 0]

print(S.shape1)

print(S.moveAgent(a))

print(S.hitbox)
print(S.state)
