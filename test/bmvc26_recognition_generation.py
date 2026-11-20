#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import sys
sys.path.append(os.getcwd())
import copy
import numpy as np
import importlib.util
from pathlib import Path
from pcr_framework.src.constants import Modes
from pcr_framework.src.composite import Composite
from pcr_framework.src.annotations import GenericVideo, GenericImage
from pcr_framework.src.viewer import Viewer
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
    input_data = int(args.input_data)
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
    viewer = Viewer('bmvc_recognition_generation')
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
    # Conditional generation on fixed latent units
    # =========================
    ann, pred = GenericVideo(), GenericVideo()
    img_ann = GenericImage(os.path.join(dirname, 'example.png'))
    img_ann.id = input_data
    img_ann.tile = np.array([0, 0, 64, 64])
    ann.add_image(img_ann)
    pred.add_image(copy.deepcopy(img_ann))
    sr.process(None, pred, latents)
    if show_viewer:
        for img_pred in pred.images:
            viewer.set_image(img_pred)
        composite.show(viewer, ann, pred)
        viewer.show(0, as_video=False)
    if save_image:
        for img_pred in pred.images:
            viewer.set_image(img_pred)
        composite.show(viewer, ann, pred)
        viewer.save(dirname, as_video=False, format='tif')
        composite.save(dirname, pred)
    print('End of bmvc26_recognition_generation')


if __name__ == '__main__':
    main()
