from matplotlib.animation import FuncAnimation

import Auto_Parking
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from parameter import *

print(1)

A = np.zeros([10, 10])
B = np.full([10, 10], -1)

for i in range(PARKLOT_SIZE[0]):
    for j in range(PARKLOT_SIZE[1]):
        A[i, j] = 0

A[9, 9] = 1

B[4, 4] = 1

S = Auto_Parking.State(A, B)

a = [0, 1]

print(S.shape1)

for i in range(8):
    print(S.moveValidity(a))
    a = [0, i]
    if S.moveValidity(a) == 0:
        print("dir", a[1])
        S.moveAgent(a)
        print(S.parking_complete())
        print(S.next_hitbox_index)
        print(S.next_hitbox)
# print(S.state.shape)

print(S.current_pos)
