from keras import Sequential
from keras.layers import *
from rl.agents.sarsa import SarsaAgent

from utils.config import GAMMA


def get_sarsa_model(num_features, num_actions, policy):
    model = Sequential(
        [
            Input(shape=num_features),
            Flatten(),
            Dense(num_actions, activation='linear', kernel_initializer='glorot_normal', name='output_layer')
        ]
    )

    return SarsaAgent(model=model, nb_actions=num_actions, policy=policy, gamma=GAMMA)
