from os.path import exists

from keras import Sequential
from keras.initializers import GlorotUniform, GlorotNormal, HeUniform, HeNormal
from keras.layers import Conv2D, Input, Flatten
from keras.models import Model, load_model

_init_strategy_dict = {
    "glorot_uniform": GlorotUniform,
    "glorot_normal": GlorotNormal,
    "he_uniform": HeUniform,
    "he_normal": HeNormal
}


def _random_init_dqn(init_strategy):
    initializer = _init_strategy_dict[init_strategy]
    return Sequential(
        [
            Input(shape=(84, 84, 4), name="input"),
            Conv2D(32, 8, strides=4, activation="relu", name="conv_1", kernel_initializer=initializer),
            Conv2D(64, 4, strides=2, activation="relu", name="conv_2", kernel_initializer=initializer),
            Conv2D(64, 3, strides=1, activation="relu", name="conv_3", kernel_initializer=initializer),
            Flatten()
        ]
    )


def _pretrained_dqn(weights_dir):
    if not exists(weights_dir):
        raise ValueError("Invalid path for pretrained weights")

    model = load_model(weights_dir)
    return Model(inputs=model.input, outputs=model.layers[-3].output)


def get_dqn_extractor(init_strategy: str = 'glorot_uniform', weights_dir: str = None) -> Model:
    if init_strategy == 'pretrained':
        return _pretrained_dqn(weights_dir)
    elif init_strategy in _init_strategy_dict:
        return _random_init_dqn(init_strategy)
    else:
        raise NotImplementedError("Invalid argument for init_strategy: {}\nPossible values: {}"
                                  .format(init_strategy,
                                          list(_init_strategy_dict.keys()).append('pretrained')))
