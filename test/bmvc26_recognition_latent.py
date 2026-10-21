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


def convert_labels_to_class_ids(all_labels, selected_labels):
    value_to_id = []
    for j in range(all_labels.shape[1]):
        vals = np.unique(all_labels[:, j])
        value_to_id.append({float(v): i for i, v in enumerate(vals)})
    class_ids = np.zeros_like(selected_labels, dtype=np.int64)
    for j in range(selected_labels.shape[1]):
        class_ids[:, j] = np.array([value_to_id[j][float(v)] for v in selected_labels[:, j]], dtype=np.int64)
    return class_ids


def validate_indices(train_indices, test_indices, num_samples):
    if np.any(train_indices < 0) or np.any(train_indices >= num_samples):
        raise IndexError('Some train indices are out of range for the latent array.')
    if np.any(test_indices < 0) or np.any(test_indices >= num_samples):
        raise IndexError('Some test indices are out of range for the latent array.')
    overlap = np.intersect1d(train_indices, test_indices)
    print(f'Train instances: {len(train_indices)}')
    print(f'Test instances: {len(test_indices)}')
    print(f'Train/test overlap: {len(overlap)}')
    if len(overlap) > 0:
        raise ValueError(f'Leakage detected: train and test splits share {len(overlap)} samples. First overlaps: {overlap[:10]}')

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
    labels = []
    for seq in anns:
        attributes = {attr.label.name for attr in seq.images[0].objects[0].attributes}
        labels.append([int(factor in attributes) for factor in sr.factor_names])
    labels =  np.asarray(labels, dtype=np.int64)
    nsamples = sr.latents.shape[0]
    if len(labels) != nsamples:
        raise ValueError(f"Number of global labels ({len(labels)}) does not match number of latent samples ({nsamples}).")
    if sr.database == 'shapes3d':
        import h5py
        with h5py.File("/home/database/shapes3d/3dshapes.h5", 'r') as f:
            labels = f['labels'][:]
        if sr.latents.shape[0] != labels.shape[0]:
            raise ValueError(f'The latent file must follow the original Shapes3D order. Latents have {sr.latents.shape[0]} samples, but 3dshapes.h5 has {labels.shape[0]} labels.')
        labels = convert_labels_to_class_ids(labels, labels)
    # =========================
    # Build train/test sets from the full representation
    # =========================
    if sr.database == 'shapes3d':
        validate_indices(train_indices=sr.train_indices, test_indices=sr.valid_indices, num_samples=nsamples)
        x_train = sr.latents[sr.train_indices]
        x_valid = sr.latents[sr.valid_indices]
        y_train = labels[sr.train_indices]
        y_valid = labels[sr.valid_indices]
    if sr.database == 'celeba':
        x_train = sr.latents
        x_valid = sr.latents
        y_train = labels
        y_valid = labels
    print('Train latents shape:', x_train.shape)
    print('Test latents shape:', x_valid.shape)
    print('Train labels shape:', y_train.shape)
    print('Test labels shape:', y_valid.shape)
    composite.train((x_train, y_train), (x_valid, y_valid))
    print('End of bmvc26_recognition_latent')


if __name__ == '__main__':
    main()
