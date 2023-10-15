import numpy as np


def getState(pos):
    size = [np.size(pos, 0), np.size(pos, 1)]
    for i in range(size[0]):
        for j in range(size[0]):
            if pos[i, j] != -1:
                return [[i, j], pos[i, j]]


if __name__ == "__main__":
    position = [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
    state = getState(position)