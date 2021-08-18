from argparse import ArgumentParser

from utils.evaluate import evaluate
from utils.train import train


def is_sub_arg(arg):
    key, value = arg
    return value is not None and key != 'action'


def clean_args(args):
    action = args.action
    cleaned_args = dict(filter(is_sub_arg, args._get_kwargs()))
    return action, cleaned_args


def main():
    parser = ArgumentParser()
    action_parser = parser.add_subparsers(title="actions", dest="action", required=True,
                                          help="select action to execute")

    # args for training
    train_parser = action_parser.add_parser("train", help="train the model")

    # args for training
    train_parser.add_argument("-s", "--save-dir", dest="save_dir", required=True,
                              help="directory for saving model artefacts")
    train_parser.add_argument("-e", "--extractor-model", dest="extractor_model", required=True,
                              help="Name of the extractor model")
    train_parser.add_argument("-w", "--weights-dir", dest="weights_dir", required=False,
                              help="Extractor Model weights directory")

    # args for evaluation
    eval_parser = action_parser.add_parser("evaluate", help="evaluate the model")

    # args for evaluation
    eval_parser.add_argument("-s", "--save-dir", dest="save_dir", required=True,
                             help="directory for saving gameplay videos")
    eval_parser.add_argument("-m", "--model-dir", dest="model_dir", required=True,
                             help="linear SARSA model weights directory")
    eval_parser.add_argument("-e", "--extractor-model", dest="extractor_model", required=True,
                             help="Name of the extractor model")
    eval_parser.add_argument("-w", "--weights-dir", dest="weights_dir", required=False,
                             help="Extractor Model weights directory")

    action, args = clean_args(parser.parse_args())

    if action == 'train':
        train(**args)
    elif action == 'evaluate':
        evaluate(**args)


if __name__ == '__main__':
    main()
