ROBOT_SIZE = [3, 5]
PARKLOT_SIZE = [4, 6]
WORLD_SIZE = [30, 30]
PROB = [0.3, 0.5]

DIST_REWARD_PARAM = 60

# reward design
TRANSLATE_COST = -0.1
ROTATE_COST = -0.1
FINAL_REWARD = 50

MAX_EPISODE_LENGTH = 64       # The maximum steps that an episode can hold
TIMESTEPS_ROLLOUT = 256
BATCH_SIZE = 64
UPDATES_PER_ITERATION = 8       # how many times pi in numerator will update

IN_CHANNEL = 3
LSTM_LAYERS = 2
OUTPUT_DIM = 512

# BATCH_SIZE = 128
GAMMA = 0.95

LR_CRITIC = 0.001
LR_ACTOR = 0.0003
CLIP = 0.2

IMG_SAVE_PATH = "process_visual_episodes_22_Nov_11"

USE_GPU = False
TRAIN = False           # if Train = True, save the model
