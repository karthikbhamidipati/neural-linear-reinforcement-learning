import gym
import numpy as np
from keras.models import load_model

from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from models.model_utils import get_extractor_model
from models.neural_linear_wrapper import NeuralLinearWrapper


def get_env(extractor_model, save_dir, weights_dir=None):
    model = get_extractor_model(extractor_model, weights_dir)

    env = make_atari("BreakoutNoFrameskip-v4")
    env = wrap_deepmind(env, frame_stack=True, scale=True)
    env = NeuralLinearWrapper(env, model)
    env.seed(500)

    return gym.wrappers.Monitor(env, save_dir, video_callable=lambda episode_id: True, force=True)


def save_video(env, sarsa_model, n_episodes=10):
    rewards = np.zeros(n_episodes, dtype=float)

    for i in range(n_episodes):
        # Resetting the state for each episode
        state = np.array(env.reset())
        done = False

        while not done:
            # Choosing an action based on greedy policy
            action_values = sarsa_model.predict(state)
            action = np.argmax(action_values)

            # Perform action and get next state, reward and done
            state_next, reward, done, _ = env.step(action)
            state = np.array(state_next)

            # Update the reward observed at episode i
            rewards[i] += reward

    env.close()
    return rewards


def evaluate(save_dir, model_dir, extractor_model, weights_dir=None):
    env = get_env(extractor_model, save_dir + '/' + extractor_model, weights_dir)
    sarsa_model = load_model(model_dir)
    rewards = save_video(env, sarsa_model)

    print(rewards, np.mean(rewards))
