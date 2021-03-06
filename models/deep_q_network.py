from os.path import exists

from keras import Sequential
from keras.initializers import glorot_uniform, glorot_normal, he_uniform, he_normal
from keras.layers import Conv2D, Input, Flatten, MaxPooling2D
from keras.models import Model, load_model

_init_strategy_dict = {
    "glorot_uniform": glorot_uniform,
    "glorot_normal": glorot_normal,
    "he_uniform": he_uniform,
    "he_normal": he_normal
}


def _random_init_dqn(init_strategy):
    initializer = _init_strategy_dict[init_strategy]()
    dqn_model = Sequential(
        [
            Input(shape=(84, 84, 4), name="input"),
            Conv2D(32, 8, strides=4, activation="relu", name="conv_1", kernel_initializer=initializer),
            Conv2D(64, 4, strides=2, activation="relu", name="conv_2", kernel_initializer=initializer),
            Conv2D(64, 3, strides=1, activation="relu", name="conv_3", kernel_initializer=initializer),
            Flatten()
        ]
    )

    return dqn_model


def _pretrained_dqn(weights_dir):
    if not exists(weights_dir):
        raise ValueError("Invalid path for pretrained weights")

    model = load_model(weights_dir)
    return Model(inputs=model.input, outputs=model.layers[-3].output)


def _dqn_pooling():
    dqn_pooling_model = Sequential(
        [
            Input(shape=(84, 84, 4), name="input"),
            Conv2D(32, 8, activation="relu", name="conv_1", kernel_initializer=glorot_normal()),
            MaxPooling2D(8, 4),
            Conv2D(64, 4, activation="relu", name="conv_2", kernel_initializer=glorot_normal()),
            MaxPooling2D(4, 2),
            Conv2D(64, 3, activation="relu", name="conv_3", kernel_initializer=glorot_normal()),
            MaxPooling2D(3, 1),
            Flatten()
        ]
    )

    return dqn_pooling_model


def get_dqn_extractor(init_strategy: str = 'glorot_uniform', weights_dir: str = None) -> Model:
    if init_strategy == 'pretrained':
        return _pretrained_dqn(weights_dir)
    elif init_strategy == 'pooling':
        return _dqn_pooling()
    elif init_strategy in _init_strategy_dict:
        return _random_init_dqn(init_strategy)
    else:
        raise NotImplementedError("Invalid argument for init_strategy: {}\nPossible values: {}"
                                  .format(init_strategy,
                                          list(_init_strategy_dict.keys()).append('pretrained')))
