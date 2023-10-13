import Auto_Parking
import numpy as np

A = np.zeros([5, 5])
B = np.full((5, 5), -1)

for i in range(3):
    for j in range(3):
        A[i, j] = 1

A[4, 4] = -1

B[2, 2] = 0

S = Auto_Parking.State(A, B, [1, 1, 1])

a = [6, 0]

print(S.shape1)

print(S.moveAgent(a))

print(S.hitbox)
print(S.state)