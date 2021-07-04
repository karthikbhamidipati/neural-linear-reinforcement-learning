import os
from argparse import ArgumentParser

from utils.train import train


def main():
    parser = ArgumentParser()

    # args for training
    parser.add_argument("-s", "--save-dir", dest="save_dir", required=True,
                        help="directory for saving model artefacts")
    parser.add_argument("-e", "--extractor-model", dest="extractor_model", required=True,
                        help="Name of the extractor model")
    parser.add_argument("-w", "--weights-dir", dest="weights_dir", required=False,
                        help="Extractor Model weights directory")

    args_dict = vars(parser.parse_args())

    print('args:', args_dict)

    train(**args_dict)


if __name__ == '__main__':
    main()
