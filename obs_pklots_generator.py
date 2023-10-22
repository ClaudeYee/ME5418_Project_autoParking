import os.path

import numpy as np
import random
import skimage
import matplotlib.pyplot as plt


def generate_obs(size, obstacle_ratio=10, remove_edge_ratio=3):
    world_size = size[0]
    world = np.zeros((world_size, world_size))
    all_x = range(2, world_size - 2)
    all_y = range(2, world_size - 2)
    obs_edge = []
    obs_corner_x = []
    while len(obs_corner_x) < world_size // obstacle_ratio:
        corn_x = random.sample(all_x, 1)
        near_flag = False
        for i in obs_corner_x:
            if abs(i - corn_x[0]) == 1:
                near_flag = True
        if not near_flag:
            obs_corner_x.append(corn_x[0])
    obs_corner_y = []
    while len(obs_corner_y) < world_size // obstacle_ratio:
        corn_y = random.sample(all_y, 1)
        near_flag = False
        for i in obs_corner_y:
            if abs(i - corn_y[0]) == 1:
                near_flag = True
        if not near_flag:
            obs_corner_y.append(corn_y[0])
    obs_corner_x.append(0)
    obs_corner_x.append(world_size - 1)
    obs_corner_y.append(0)
    obs_corner_y.append(world_size - 1)
    for i in obs_corner_x:
        edge = []
        for j in range(world_size):
            world[i][j] = 1
            if j not in obs_corner_y:
                edge.append([i, j])
            if j in obs_corner_y and edge != []:
                obs_edge.append(edge)
                edge = []
    for i in obs_corner_y:
        edge = []
        for j in range(world_size):
            world[j][i] = 1
            if j not in obs_corner_x:
                edge.append([j, i])
            if j in obs_corner_x and edge != []:
                obs_edge.append(edge)
                edge = []
    all_edge_list = range(len(obs_edge))
    remove_edge = random.sample(all_edge_list, len(obs_edge) // remove_edge_ratio)
    for edge_number in remove_edge:
        for current_edge in obs_edge[edge_number]:
            world[current_edge[0]][current_edge[1]] = 0
    for edges in obs_edge:
        if len(edges) == 1 or len(edges) <= world_size // 20:
            for coordinates in edges:
                world[coordinates[0]][coordinates[1]] = 0
    _, count = skimage.measure.label(world, background=-1, connectivity=1, return_num=True)

    door_list = []
    # door_occu_list = []
    while count != 1 and len(obs_edge) > 0:
        door_edge_index = random.sample(range(len(obs_edge)), 1)[0]
        door_edge = obs_edge[door_edge_index]
        door_index = random.sample(range(len(door_edge)), 1)[0]
        door = door_edge[door_index]
        world[door[0]][door[1]] = 0
        door_list.append(door)
        door_occu = find_neighboring_one(world, door[0], door[1])
        # door_occu_list.append(door_occu)
        _, count = skimage.measure.label(world, background=-1, connectivity=1, return_num=True)
        # if new_count == count:
        #     world[door[0]][door[1]] = -1
        #     obs_edge.remove(door_edge)
        # else:
        obs_edge.remove(door_edge)
        #     count = new_count
    # world = np.zeros((world_size, world_size))
    '''
    door_width = 4
    while count != 1 and len(obs_edge) > 0:
        door_edge_index = random.choice(range(len(obs_edge)))
        door_edge = obs_edge[door_edge_index]

        # Calculate the maximum valid starting index for the door
        max_door_index = len(door_edge) - door_width

        # Check if the range for selecting the starting index is non-negative
        if max_door_index >= 0:
            door_index = random.choice(range(max_door_index + 1))
        else:
            # If the range is negative, skip this iteration
            continue

        # Modify the grid points to place a door of specified width
        door = door_edge[door_index:door_index + door_width]
        for point in door:
            world[point[0]][point[1]] = 0

        _, count = skimage.measure.label(world, background=-1, connectivity=1, return_num=True)
        obs_edge.remove(door_edge)
    '''

    for door in door_list:
        if 0 < door[0] < size[0] - 1 and (world[door[0] - 1, door[1]] == 1 and world[door[0] + 1, door[1]] == 1):
            world[door[0] - 1, door[1]] = 0
            world[door[0] + 1, door[1]] = 0
        if 0 < door[1] < size[1] - 1 and (world[door[0], door[1] - 1] == 1 and world[door[0], door[1] + 1] == 1):
            world[door[0], door[1] - 1] = 0
            world[door[0], door[1] + 1] = 0
        # if door[0] < size[0] - 1 and world[door[0] + 1, door[1]] == 1:
        #     world[door[0] + 1, door[1]] = 0
        # if door[1] > 0 and world[door[0], door[1] - 1] == 1:
        #     world[door[0], door[1] - 1] = 0
        # if door[1] < size[0] - 1 and world[door[0], door[1] + 1] == 1:
        #     world[door[0], door[1] + 1] = 0
        # if 0 < door[0] < size[0] - 1 and 0 < door[1] < size[0] - 1 and (world[door[0] - 1, door[1]] == 1 and world[door[0] + 1, door[1]] == 1 and world[door[0], door[1] - 1] == 1 and world[door[0], door[1] + 1] == 1):
        #     world[door[0], door[1]] = 0

    world[:, -1] = world[:, 0] = 1
    world[-1, :] = world[0, :] = 1
    # nodes_obs = get_map_nodes(world)
    # return world, nodes_obs
    # world = np.ones_like(world) - world
    return world

def find_neighboring_one(matrix, i, j):
    rows, cols = len(matrix), len(matrix[0])
    neighbors = []

    # Check the top neighbor
    if i > 0 and matrix[i - 1][j] == 1:
        neighbors.append([i - 1, j])

    # Check the bottom neighbor
    if i < rows - 1 and matrix[i + 1][j] == 1:
        neighbors.append([i + 1, j])

    # Check the left neighbor
    if j > 0 and matrix[i][j - 1] == 1:
        neighbors.append([i, j - 1])

    # Check the right neighbor
    if j < cols - 1 and matrix[i][j + 1] == 1:
        neighbors.append([i, j + 1])

    # Return the list of indices of neighboring elements with value 0
    return neighbors


def generate_pklots(size, parklot_size, obs_world):
    # Create the parking world
    world_size = size
    pklot_world = np.zeros(world_size)

    # Generate the two parking lots directions
    # 0: horizontal parking lot and 1: vertical parking lot
    pklot1_dir = random.choice([0, 1])
    pklot2_dir = random.choice([0, 1])

    # From the parking lots size and direction generate two parking lot that will be added to the parking world
    # via arrays of 1 if the parking lot is horizontal or 2 if it is vertical
    def lots_generate(dir, num, pklot_size):
        if dir == 0:
            return num * np.ones(pklot_size)
        else:
            return num * np.transpose(np.ones(pklot_size))

    pklot1 = lots_generate(pklot1_dir, 1, parklot_size)
    pklot2 = lots_generate(pklot2_dir, 2, parklot_size)

    # Add pklot1 to the world
    x1, y1 = np.random.randint(0, pklot_world.shape[0] - pklot1.shape[0] + 1), np.random.randint(0, pklot_world.shape[1] - pklot1.shape[1] + 1)
    while np.all(pklot_world[x1:x1 + pklot1.shape[0], y1:y1 + pklot1.shape[1]]) != 0:
        x1, y1 = np.random.randint(0, pklot_world.shape[0] - pklot1.shape[0] + 1), np.random.randint(0, pklot_world.shape[1] - pklot1.shape[1] + 1)
    pklot_world[x1:x1 + pklot1.shape[0], y1:y1 + pklot1.shape[1]] = pklot1

    # Randomly choose positions for matrix2, ensuring no overlap with matrix1
    while True:
        x2, y2 = np.random.randint(0, pklot_world.shape[0] - pklot2.shape[0] + 1), np.random.randint(0, pklot_world.shape[1] - pklot2.shape[1] + 1)
        if np.all(pklot_world[x2:x2 + pklot2.shape[0], y2:y2 + pklot2.shape[1]] == 0):
            pklot_world[x2:x2 + pklot2.shape[0], y2:y2 + pklot2.shape[1]] = pklot2
            break
    # Check whether pklot_world has any conflict with obs_world
    #TODO: --------------------------Remember to fix this function--------------------------------------
    def place_available(pklot_world, obs_world):
        obs = np.argwhere(obs_world > 0)
        pklot = np.argwhere(pklot_world > 0)
        comparison = np.equal(pklot[:, np.newaxis, :], obs)
        if np.any(np.all(comparison, axis=2)):
            return False
        else:
            return True

    if place_available(pklot_world, obs_world):
        pklot1_coord = [(2 * x1 + pklot1.shape[0]) / 2, (2 * y1 + pklot1.shape[1]) / 2]
        pklot2_coord = [(2 * x2 + pklot2.shape[0]) / 2, (2 * y2 + pklot2.shape[1]) / 2]
        return pklot_world, pklot1_coord, pklot2_coord

    else:
        print("Not available, retry to place two parkinglots")
        pklot_world = generate_pklots(world_size, parklot_size, obs_world)
        return pklot_world


if __name__ == "__main__":
    image_save_path = "test_pictures/"
    map_size = [60, 60]
    obs_world0 = generate_obs(size=map_size, obstacle_ratio=10, remove_edge_ratio=3)
    plt.imshow(obs_world0, cmap="gray")
    plt.axis((0, map_size[1], map_size[0], 0))

    plt.suptitle('Test Environment')
    plt.tight_layout()

    if not os.path.exists(image_save_path):
        os.makedirs(image_save_path)
    plt.savefig('{}/test.png'.format(image_save_path))
