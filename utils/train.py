import shutil

from keras.optimizers import *
from matplotlib import pyplot as plt

from baselines.common.atari_wrappers import *
from models.model_utils import get_extractor_model
from models.neural_linear_wrapper import NeuralLinearWrapper
from models.sarsa_model import get_sarsa_model
from utils.config import SEED, LEARNING_RATE, MAX_STEPS_PER_EPISODE, TRAIN_STEPS
from utils.epsilon_greedy_policy import get_policy


def train(save_dir, extractor_model, weights_dir=None):
    model = get_extractor_model(extractor_model, weights_dir)

    # Setting Breakout env
    env = make_atari("BreakoutNoFrameskip-v4")

    # Warp the frames, grey scale, stake four frame and scale to smaller ratio
    env = wrap_deepmind(env, frame_stack=True, scale=True)
    env = NeuralLinearWrapper(env, model)
    env.seed(SEED)

    num_actions = env.action_space.n
    num_features = env.observation_space.shape

    save_dir = os.path.join(save_dir, extractor_model, '')

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir, ignore_errors=True)

    os.makedirs(save_dir, exist_ok=True)

    sarsa_agent = get_sarsa_model(num_features, num_actions, get_policy())

    optimizer = Adam(learning_rate=LEARNING_RATE)
    sarsa_agent.compile(optimizer=optimizer, metrics=['mae'])

    test(save_dir, env, sarsa_agent)

    # Performing warm up
    sarsa_agent.fit(env, nb_steps=300, visualize=False, nb_max_episode_steps=MAX_STEPS_PER_EPISODE, verbose=0)

    # training the model
    train_history = sarsa_agent.fit(env, nb_steps=TRAIN_STEPS, visualize=False,
                                    nb_max_episode_steps=MAX_STEPS_PER_EPISODE, verbose=1)
    save_trained_model(env, optimizer, sarsa_agent, save_dir, train_history)

    test(save_dir, env, sarsa_agent)


def save_trained_model(env, optimizer, sarsa_agent, save_dir, train_history):
    print('Average rewards of last 10000 steps during training',
          np.mean(train_history.history['episode_reward'][-10000:]))

    sarsa_agent.save_weights(save_dir + 'agent/')
    np.save(save_dir + 'agent_config.npy', sarsa_agent.get_config())
    sarsa_agent.model.save(save_dir + 'model/')
    np.save(save_dir + 'optimizer.npy', optimizer.get_weights())
    np.save(save_dir + 'optimizer_config.npy', optimizer.get_config())
    env.feature_extractor.save(save_dir + 'feature_extractor')

    plot_history(save_dir, train_history)


def plot_history(save_dir, history):
    plt.plot(history.history['episode_reward'])
    plt.title('Rewards vs epochs before training')
    plt.xlabel('epochs')
    plt.ylabel('reward')
    plt.savefig(save_dir + 'pre_train_rewards.png')


def test(save_dir, env, sarsa_agent, num_steps=200):
    test_history = sarsa_agent.test(env, nb_episodes=num_steps, visualize=False, verbose=0)
    print('Average rewards during testing: ', np.mean(test_history.history['episode_reward']))
    plot_history(save_dir, test_history)
