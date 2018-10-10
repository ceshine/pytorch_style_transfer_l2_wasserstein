import os
import logging
from argparse import ArgumentParser

import PIL
import numpy as np

from styletransfer import StyleTransfer

LOGGER = logging.getLogger()


def init_logger():
    formatter = logging.Formatter(
        '[[%(asctime)s]] %(message)s',
        datefmt='%m/%d/%Y %I:%M:%S %p'
    )
    # Remove all existing handlers
    LOGGER.handlers = []
    # Initialize handlers
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    LOGGER.addHandler(sh)
    LOGGER.setLevel(logging.DEBUG)


def main():
    init_logger()
    parser = make_parser()
    args = parser.parse_args()

    style_image = np.array(PIL.Image.open(
        args.style)).astype("float32") / 255
    subject_image = np.array(PIL.Image.open(
        args.subject)).astype("float32") / 255

    transfer = StyleTransfer(args.log_interval)
    transfer.synthesize(
        subject_image, style_image, args.output, steps=args.steps)


def make_parser():
    parser = ArgumentParser()
    parser.add_argument('--subject', dest='subject',
                        help='subject image, to be transformed', default='content_images/wave_small.jpg')
    parser.add_argument('--style', dest='style',
                        help='image portraying style to be transferred', default='style_images/kngwa_small.jpg')
    parser.add_argument('--output', dest='output',
                        help='path for output', default='output.jpg')
    parser.add_argument('--steps', dest='steps', type=int,
                        help='# of steps optimizer can run', default=20)
    parser.add_argument('--log_interval', dest='log_interval', type=int,
                        help='will save the sythesized image per n steps (set 0 to disable)', default=0)
    return parser


if __name__ == '__main__':
    main()
