# Controlling randomization
SEED = 500

# Epsilon Greedy policy constants
EPSILON_MAX = 1.0
EPSILON_MIN = 0.1
EPSILON_INTERVAL = EPSILON_MAX - EPSILON_MIN
EPSILON_GREEDY_FRAMES = 10 ** 6

# Hyperparameters
GAMMA = 0.99
MAX_STEPS_PER_EPISODE = 10000
LEARNING_RATE = 10 ** -4
TRAIN_STEPS = 2 * (10 ** 6)