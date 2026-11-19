#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import sys
sys.path.append(os.getcwd())
import numpy as np
import matplotlib.pyplot as plt
import importlib.util
from pathlib import Path
from pcr_framework.src.constants import Modes
from pcr_framework.src.composite import Composite
from src.bmvc26_recognition import BMVC26Recognition


def parse_options():
    """
    Parse options from command line.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', '-d', dest='input_data', type=int, default=176977,  # Figure 2 (a) in the main paper
                        help='Input latent identifier.')
    parser.add_argument('--show-viewer', '-v', dest='show_viewer', action="store_true",
                        help='Show results visually.')
    parser.add_argument('--save-image', '-i', dest='save_image', action="store_true",
                        help='Save processed images.')
    args, unknown = parser.parse_known_args()
    print(parser.format_usage())
    input_data = args.input_data
    show_viewer = args.show_viewer
    save_image = args.save_image
    return unknown, input_data, show_viewer, save_image


def main():
    """
    On Structured Disentangled Representation and the Limits of Global Scores conditional generation script.
    """
    # =========================
    # Load computer vision components
    # =========================
    unknown, input_data, show_viewer, save_image = parse_options()
    composite = Composite()
    sr = BMVC26Recognition('')
    composite.add(sr)
    composite.parse_options(unknown)
    composite.load(Modes.TEST)
    spec = importlib.util.find_spec('pcr_framework')
    output_path = os.path.join('pcr_framework' if spec is None else os.path.dirname(spec.origin), 'output')
    dirname = os.path.join(output_path, 'images/')
    Path(dirname).mkdir(parents=True, exist_ok=True)
    # =========================
    # Load full-dataset latents
    # =========================
    path = sr.path + 'data/' + sr.database + '/lat' + str(sr.nlats) + '/dim' + str(sr.ndims)
    filename = os.path.join(path, ('epoch=000000.npz' if sr.nlats == 6 and sr.ndims == 32 else 'epoch=000010.npz') if sr.database == 'shapes3d' else sr.database + '_representations.npz')
    print(f"Latents .npz file: {filename}")
    data = np.load(filename, allow_pickle=True)
    print(f"NPZ keys: {data.files}")
    latents = data['latents' if 'latents' in data else list(data.keys())[0]]
    print('Latents shape:', latents.shape)
    # =========================
    # Save predictions
    # =========================
    pred = []
    composite.process(latents, pred)
    grid_size = 5
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(2.8*grid_size, 2.8*grid_size), facecolor='none')
    axes = np.atleast_2d(axes)
    for ax_row in axes:
        for ax in np.atleast_1d(ax_row):
            ax.axis('off')
            ax.set_facecolor('none')
    axes[0, 0].imshow(pred[0])
    axes[0, 0].axis('off')
    for idx in range(len(pred)-1):
        row = 1+(idx//grid_size)
        col = idx%grid_size
        if row >= grid_size:
            break
        axes[row, col].imshow(pred[idx+1])
        axes[row, col].axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
    if show_viewer:
        plt.show()
    if save_image:
        filename = os.path.join(dirname, f'disdiff_{sr.database}_lat{sr.nlats}_dim{sr.ndims}_slots_{str(sr.slots)}.png')
        plt.savefig(filename, dpi=180, bbox_inches='tight', pad_inches=0, transparent=True)
    plt.close(fig)
    print('End of bmvc26_recognition_generation')


if __name__ == '__main__':
    main()
