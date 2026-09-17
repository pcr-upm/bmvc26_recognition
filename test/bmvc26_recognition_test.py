#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import sys
sys.path.append(os.getcwd())
import cv2
import numpy as np
import importlib.util
from pathlib import Path
from pcr_framework.src.constants import Modes
from pcr_framework.src.composite import Composite
from pcr_framework.src.annotations import GenericVideo
from pcr_framework.src.viewer import Viewer
from src.bmvc26_recognition import BMVC26Recognition


def parse_options():
    """
    Parse options from command line.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', '-d', dest='input_data', required=True, default='',
                        help='Input as image video file.')
    parser.add_argument('--save-video', '-v', dest='save_video', action="store_true",
                        help='Save processed video.')
    args, unknown = parser.parse_known_args()
    print(parser.format_usage())
    input_data = args.input_data
    save_video = args.save_video
    return unknown, input_data, save_video


def load_annotations(filename):
    """
    Read the annotations of a single video directly from the matching JSON file.
    """
    import json
    with open(filename, 'r', encoding='utf-8') as ifs:
        payload = json.load(ifs)
    annotations = payload.get('annotations', [])
    gt = []
    for anno in annotations:
        gt.append(dict(segment=anno['segment'], label=anno['label']))
    gt.sort(key=lambda x: x['segment'][0])
    return gt


def main():
    """
    On Structured Disentangled Representation and the Limits of Global Scores test image script.
    """
    print('OpenCV ' + cv2.__version__)
    unknown, input_data, save_video = parse_options()

    # Load vision components
    composite = Composite()
    sr = BMVC26Recognition('')
    composite.add(sr)
    composite.parse_options(unknown)
    composite.load(Modes.TEST)
    if save_video:
        viewer = Viewer('bmvc26_recognition_test')
        spec = importlib.util.find_spec('pcr_framework')
        output_path = os.path.join('pcr_framework' if spec is None else os.path.dirname(spec.origin), 'output')
        dirname = os.path.join(output_path, 'images/')
        Path(dirname).mkdir(parents=True, exist_ok=True)

    # Load annotations
    ann, pred = GenericVideo(), GenericVideo()

    # Process video
    ticks = cv2.getTickCount()
    composite.process(ann, pred)
    ticks = cv2.getTickCount() - ticks
    if save_video:
        for img_pred in pred.images:
            viewer.set_image(img_pred)
        composite.show(viewer, ann, pred)
        fps = 'FPS = ' + "{0:.3f}".format(pred.frames / (ticks / cv2.getTickFrequency()))
        viewer.text(pred.images[0], fps, (20, np.shape(viewer.get_image(pred.images[0]))[0] - 20), 0.5, (0, 255, 0))
        viewer.save(dirname, as_video=True, format='avi', fps=30, codec='XVID')
    print('End of bmvc26_recognition_test')


if __name__ == '__main__':
    main()
