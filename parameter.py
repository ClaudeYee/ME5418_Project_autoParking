ROBOT_SIZE = [3, 5]
PARKLOT_SIZE = [4, 6]
WORLD_SIZE = [60, 60]
PROB = [0.3, 0.5]

DIST_REWARD_PARAM = 30

# reward design
TRANSLATE_COST = -0.1
ROTATE_COST = -0.2
# DISTANCE_REWARD = 0.3
FINAL_REWARD = 40

MAX_EPISODE_LENGTH = 128        # The maximum steps that an episode can hold
TIMESTEPS_ROLLOUT = 512
BATCH_SIZE = 32
UPDATES_PER_ITERATION = 1       # how many times pi in numerator will update

IN_CHANNEL = 3
LSTM_LAYERS = 2
OUTPUT_DIM = 512

# BATCH_SIZE = 128
GAMMA = 0.98

LR = 0.005
CLIP = 0.2

IMG_SAVE_PATH = "process_visual_episodes_18_Nov"

USE_GPU = False
TRAIN = False