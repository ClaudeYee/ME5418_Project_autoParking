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

# print(S.shape1)
#
print(S.moveValidity(a))
S.moveAgent(a)
# print(S.hitbox)
# print(S.state)

rotated_center=[0, 0]
angle=0

fig, ax = plt.subplots()
rectangle = patches.Rectangle(rotated_center, ROBOT_SIZE[1], ROBOT_SIZE[0], angle=-angle, color='r')
ax.add_patch(rectangle)


def init():
    env = plt.imshow(S.state)
    return env,


# 更新新一帧的数据
def update(frame):
    n = frame % 8
    angle = 360 * n / 8
    angle_rad = np.radians(angle)
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    rotateMatrix = np.array([[cos_angle, -sin_angle], [sin_angle, cos_angle]])

    l = np.array([ROBOT_SIZE[1], ROBOT_SIZE[0]]) / 2
    rotated_center = np.array(S.robot_next_state[0]) - np.array(l).dot(rotateMatrix)

    # rectangle = patches.Rectangle(rotated_center, ROBOT_SIZE[1], ROBOT_SIZE[0], angle=-angle, color='r')
    angle = -angle + 360
    if angle <= 0:
        angle += 1
    elif angle == 360:
        angle -= 1

    rectangle.angle = angle
    rectangle.set(x=rotated_center[0], y=rotated_center[1], height=ROBOT_SIZE[0], width=ROBOT_SIZE[1])

    return rectangle,

print("started")

# 调用 FuncAnimation
ani = FuncAnimation(fig
                    , update
                    , init_func=init
                    , frames=8
                    , interval=100
                    , blit=True
                    )

ani.save("testAnimation.gif", fps=10, writer="imagemagick")

print("Done")