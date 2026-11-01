#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Roberto Valle'
__email__ = 'roberto.valle@upm.es'

import os
import sys
sys.path.append(os.getcwd())
import numpy as np
from pcr_framework.src.datasets import Database, Sources
from pcr_framework.src.constants import Modes
from pcr_framework.src.composite import Composite
from src.bmvc26_recognition import BMVC26Recognition


def parse_options():
    """
    Parse options from command line.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--show-viewer', '-v', dest='show_viewer', action="store_true",
                        help='Show results visually.')
    args, unknown = parser.parse_known_args()
    print(parser.format_usage())
    show_viewer = args.show_viewer
    return unknown, show_viewer


def load_annotations(source, ref):
    """
    Load ground-truth annotations from the specified database source.
    """
    import tempfile
    from tqdm import tqdm
    from datasets import load_dataset
    print(f"Loading dataset from {source.name}: {ref}")
    datasets = [subclass().get_names() for subclass in Database.__subclasses__()]
    idx = next((idx for idx, subset in enumerate(datasets) if ref in subset), None)
    if idx is None:
        raise ValueError(f'Database is not implemented: {ref}')
    # Load all lines (one image per line)
    anns = []
    db = Database.__subclasses__()[idx]()
    namespace, _, _ = db.get_namespace(Modes.TRAIN, source, ref)
    path = tempfile.gettempdir()
    splits = ['train'] if ref == 'shapes3d' else ['train', 'validation', 'test']
    for split in splits:
        kwargs = {'path': namespace, 'cache_dir': path, 'split': split, 'download_mode': 'force_redownload'}
        ds = load_dataset(**kwargs)
        for line in tqdm(ds, file=sys.stdout):
            seq = db.load_line(source, ref, path, line)
            if len(seq.images) == 0:
                continue
            anns.append(seq)
    return anns


def main():
    """
    On Structured Disentangled Representation and the Limits of Global Scores latent-unit probing script.
    """
    # =========================
    # Load computer vision components
    # =========================
    unknown, show_viewer = parse_options()
    composite = Composite()
    sr = BMVC26Recognition('')
    composite.add(sr)
    composite.parse_options(unknown)
    # =========================
    # Load full-dataset latents and labels
    # =========================
    composite.load(Modes.TRAIN)
    anns = load_annotations(Sources.HUGFACE, sr.database)
    labels = np.asarray([[int(cat.label.name) for cat in ann.images[0].objects[0].categories] if sr.database == 'shapes3d' else [int(ann.images[0].objects[0].has_category(factor)) for factor in sr.factor_names] for ann in anns], dtype=np.int64)
    if sr.database == 'shapes3d':
        # Sort rows to the 3dshapes.h5 order (floor, object, orientation, scale, shape, wall)
        labels = labels[:, [0, 5, 1, 3, 4, 2]]
        labels = labels[np.lexsort(labels.T[::-1])]
    nsamples = sr.latents.shape[0]
    if len(labels) != nsamples:
        raise ValueError(f"Number of global labels ({len(labels)}) does not match number of latent samples ({nsamples}).")
    # =========================
    # Build train/test sets from the full representation
    # =========================
    if sr.database == 'shapes3d':
        x_train = sr.latents[sr.train_indices]
        x_valid = sr.latents[sr.valid_indices]
        y_train = labels[sr.train_indices]
        y_valid = labels[sr.valid_indices]
    elif sr.database == 'celeba':
        x_train = sr.latents
        x_valid = sr.latents
        y_train = labels
        y_valid = labels
    composite.train((x_train, y_train), (x_valid, y_valid))
    # =========================
    # Conditional image generation (Figure 2 subfigures)
    # =========================
    if sr.database == 'shapes3d' and sr.model is not None:
        if sr.nlats == 6 and sr.ndims == 4:
            # Subfigure A: Model (6, 4), fixed latent units [1, 2], reference image 176977
            idx_sub1 = 176977
            sr.process(idx_sub1, anns[idx_sub1] if len(anns) > idx_sub1 else None, fixed_slots=[1, 2])

            # Subfigura B
            idx_sub2 = 177432
            sr.process(idx_sub2, anns[idx_sub2] if len(anns) > idx_sub2 else None, fixed_slots=[2, 4])

        elif sr.nlats == 10 and sr.ndims == 32:
            # Subfigura C
            idx_sub3 = 176977
            sr.process(idx_sub3, anns[idx_sub3] if len(anns) > idx_sub3 else None, fixed_slots=[5, 9])

    print('End of bmvc26_recognition_latent')
    # if sr.database == 'shapes3d' and sr.model is not None:
    #     if sr.nlats == 6 and sr.ndims == 4:
    #         # Subfigure A: (6, 4), fixed latent units [1, 2], ref. imagen 176977
    #         idx_sub1 = 176977
    #         sr.process(idx_sub1, anns[idx_sub1] if len(anns) > idx_sub1 else None, fixed_slots=[1, 2])

    #         # Subfigura B
    #         idx_sub2 = 177432
    #         sr.process(idx_sub2, anns[idx_sub2] if len(anns) > idx_sub2 else None, fixed_slots=[2, 4])

    #         # Subfigure C
    #         if os.path.isdir('data/shapes3d/lat10/dim32'):
    #             sr_10_32 = BMVC26Recognition('')
    #             sr_10_32.parse_options(['--database', 'shapes3d', '--nlats', '10', '--ndims', '32', '--fixed-slots', '5', '9'])
    #             sr_10_32.load(Modes.TRAIN)
    #             if sr_10_32.model is not None:
    #                 idx_sub3 = 176977
    #                 sr_10_32.process(idx_sub3, anns[idx_sub3] if len(anns) > idx_sub3 else None, fixed_slots=[5, 9])

    #     elif sr.nlats == 10 and sr.ndims == 32:
    #         idx_sub3 = 176977
    #         sr.process(idx_sub3, anns[idx_sub3] if len(anns) > idx_sub3 else None, fixed_slots=[5, 9])
    #     else:
    #         sr.process(176977, anns[176977] if len(anns) > 176977 else None)

    print('End of bmvc26_recognition_latent')

if __name__ == '__main__':
    main()
