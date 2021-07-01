from models.deep_q_network import get_dqn_extractor


def get_extractor_model(extractor_model, weights_dir=None):
    model_name, init_strategy = extractor_model.split('_', 1)
    if model_name == 'dqn':
        get_dqn_extractor(init_strategy, weights_dir)