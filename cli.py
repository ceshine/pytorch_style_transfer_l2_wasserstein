import os
from argparse import ArgumentParser

import PIL
import numpy as np

from styletransfer import StyleTransfer


def main():
    parser = make_parser()
    args = parser.parse_args()

    style_image = np.array(PIL.Image.open(
        args.style)).astype("float32") / 255
    subject_image = np.array(PIL.Image.open(
        args.subject)).astype("float32") / 255

    transfer = StyleTransfer()
    transfer.synthesize(
        subject_image, style_image, args.output, steps=args.steps)


def make_parser():
    parser = ArgumentParser()
    parser.add_argument('--subject', dest='subject',
                        help='subject image, to be transformed', default='media/wave_small.jpg')
    parser.add_argument('--style', dest='style',
                        help='image portraying style to be transferred', default='media/wave_kngwa.jpg')
    parser.add_argument('--output', dest='output',
                        help='path for output', default='output.jpg')
    parser.add_argument('--steps', dest='steps', type=int,
                        help='# of steps optimizer can run', default=20)
    return parser


if __name__ == '__main__':
    main()
