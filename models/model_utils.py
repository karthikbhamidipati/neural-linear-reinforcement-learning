from models.deep_q_network import get_dqn_extractor
from models.resnet_model import get_resnet_extractor


def get_extractor_model(extractor_model, weights_dir=None):
    model_name, init_strategy = extractor_model.lower().split('_', 1)
    if model_name == 'dqn':
        return get_dqn_extractor(init_strategy, weights_dir)
    elif model_name.startswith('resnet'):
        return get_resnet_extractor(model_name, init_strategy)
    else:
        raise NotImplementedError("Model not defined")