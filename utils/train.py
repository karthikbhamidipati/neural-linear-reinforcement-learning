import numpy as np
from baselines.common.atari_wrappers import *
from keras.optimizers import *

from models.deep_q_network import get_dqn_extractor
from models.neural_linear_wrapper import NeuralLinearWrapper
from models.sarsa_model import get_sarsa_model
from utils.config import SEED, LEARNING_RATE, MAX_STEPS_PER_EPISODE, TRAIN_STEPS
from utils.epsilon_greedy_policy import get_policy


def train():
    # Setting Breakout env
    env = make_atari("BreakoutNoFrameskip-v4")

    # Warp the frames, grey scale, stake four frame and scale to smaller ratio
    env = wrap_deepmind(env, frame_stack=True, scale=True)
    env = NeuralLinearWrapper(env, get_dqn_extractor())
    env.seed(SEED)

    num_actions = env.action_space.n
    num_features = env.observation_space.shape

    sarsa_agent = get_sarsa_model(num_actions, num_features, get_policy())

    optimizer = Adam(learning_rate=LEARNING_RATE)
    sarsa_agent.compile(optimizer=optimizer, metrics=['mae'])

    test(env, sarsa_agent)

    # Performing warm up
    sarsa_agent.fit(env, nb_steps=300, visualize=False, nb_max_episode_steps=MAX_STEPS_PER_EPISODE, verbose=0)

    # training the model
    train_history = sarsa_agent.fit(env, nb_steps=TRAIN_STEPS, visualize=False, nb_max_episode_steps=max_steps_per_episode, verbose=1)
    print(np.mean(train_history.history['episode_reward'][-10000:]))

    test(env, sarsa_agent)


def test(env, sarsa_agent, num_steps=200):
    test_history = sarsa_agent.test(env, nb_episodes=num_steps, visualize=False, verbose=0)
    print(np.mean(test_history.history['episode_reward']))
