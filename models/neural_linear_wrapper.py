import gym
import numpy as np


class NeuralLinearWrapper(gym.Wrapper):
    def __init__(self, env, feature_extractor):
        gym.Wrapper.__init__(self, env)

        self.feature_extractor = feature_extractor
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(1, 1, self.feature_extractor.output_shape[1]),
                                                dtype=np.uint8)

    def reset(self, **kwargs):
        return self.extract_linear_features(self.env.reset(**kwargs))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.extract_linear_features(obs), reward, done, info

    def extract_linear_features(self, obs):
        obs = np.expand_dims(obs, axis=0)
        neural_linear_features = self.feature_extractor(obs, training=False)
        return neural_linear_features
