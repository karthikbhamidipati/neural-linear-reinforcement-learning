from keras.applications import ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2
from keras.layers import Conv2D, Input
from keras.models import Sequential


def get_resnet_extractor(model_name, init_strategy=None):
    input_shape = (84, 84, 3)
    weights = 'imagenet' if init_strategy is not None else None
    resnet_model = _get_resnet_model(model_name, input_shape, weights)

    return Sequential(name='channel_sampler',
                      layers=[
                          Input(shape=(84, 84, 4)),
                          Conv2D(3, kernel_size=1, strides=1, padding='same', kernel_initializer='glorot_normal'),
                          resnet_model
                      ])


def _get_resnet_model(model_name, input_shape, weights):
    if model_name == 'resnet50':
        return ResNet50(include_top=False, input_shape=input_shape,
                        weights=weights, pooling='avg')
    elif model_name == 'resnet101':
        return ResNet101(include_top=False, input_shape=input_shape,
                         weights=weights, pooling='avg')
    elif model_name == 'resnet152':
        return ResNet152(include_top=False, input_shape=input_shape,
                         weights=weights, pooling='avg')
    elif model_name == 'resnet50v2':
        return ResNet50V2(include_top=False, input_shape=input_shape,
                          weights=weights, pooling='avg')
    elif model_name == 'resnet101v2':
        return ResNet101V2(include_top=False, input_shape=input_shape,
                           weights=weights, pooling='avg')
    elif model_name == 'resnet152v2':
        return ResNet152V2(include_top=False, input_shape=input_shape,
                           weights=weights, pooling='avg')
    else:
        raise ValueError("Invalid Model Name")
