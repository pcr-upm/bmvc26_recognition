#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Jorge Elías'
__email__ = 'jorge.elias@alumnos.upm.es'

import os
import sys
sys.path.append(os.getcwd())
import h5py
import numpy as np
from pcr_framework.src.constants import Modes
from pcr_framework.src.composite import Composite
from src.bmvc26_recognition import BMVC26Recognition


def parse_options():
    """
    Parse options from command line.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--h5_path', type=str, default='/home/database/shapes3d/3dshapes.h5',
                        help='The original 3dshapes.h5 file path.')
    args, unknown = parser.parse_known_args()
    print(parser.format_usage())
    h5_path = args.h5_path
    return unknown, h5_path

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
    unknown, h5_path = parse_options()
    # Load computer vision components
    composite = Composite()
    sr = BMVC26Recognition('')
    composite.add(sr)

    composite.parse_options(unknown)
    composite.load(Modes.TRAIN)
    # =========================
    # Load full-dataset latents and labels
    # =========================
    num_samples = sr.latents.shape[0]
    if sr.database == 'shapes3d':
        with h5py.File(h5_path, 'r') as f:
            all_labels = f['labels'][:]
        if sr.latents.shape[0] != all_labels.shape[0]:
            raise ValueError(f'The latent file must follow the original Shapes3D order. Latents have {sr.latents.shape[0]} samples, but 3dshapes.h5 has {all_labels.shape[0]} labels.')
    if sr.database == 'celeba':
        with open("/home/database/celeba/list_attr_celeba.txt", "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        declared_num = int(lines[0])
        all_attr_names = lines[1].split()
        attr_to_idx = {a: i for i, a in enumerate(all_attr_names)}
        missing = [a for a in sr.factor_names if a not in attr_to_idx]
        if missing:
            raise ValueError(f"Missing attributes in CelebA annotations: {missing}")
        selected_cols = [attr_to_idx[a] for a in sr.factor_names]
        filenames = []
        labels = []
        for line in lines[2:]:
            parts = line.split()
            filename = parts[0]
            raw_attrs = [int(x) for x in parts[1:]]
            y = [raw_attrs[c] for c in selected_cols]
            y = ((np.array(y, dtype=np.int64) + 1) // 2).astype(np.int64)
            filenames.append(filename)
            labels.append(y)
        filenames = np.asarray(filenames)
        all_labels = np.asarray(labels, dtype=np.int64)
        if len(filenames) != declared_num:
            raise ValueError(f"Annotation file declares {declared_num} samples, but {len(filenames)} were read.")
        if len(all_labels) != num_samples:
            raise ValueError(f"Number of global labels ({len(all_labels)}) does not match number of latent samples ({num_samples}).")
    # =========================
    # Build train/test sets from the full representation
    # =========================
    if sr.database == 'shapes3d':
        validate_indices(train_indices=sr.train_indices, test_indices=sr.valid_indices, num_samples=num_samples)
        x_train = sr.latents[sr.train_indices]
        x_valid = sr.latents[sr.valid_indices]
        y_train = convert_labels_to_class_ids(all_labels, all_labels[sr.train_indices])
        y_valid = convert_labels_to_class_ids(all_labels, all_labels[sr.valid_indices])
    if sr.database == 'celeba':
        x_train = sr.latents
        x_valid = sr.latents
        y_train = all_labels
        y_valid = all_labels
    print('Train latents shape:', x_train.shape)
    print('Test latents shape:', x_valid.shape)
    print('Train labels shape:', y_train.shape)
    print('Test labels shape:', y_valid.shape)
    composite.train((x_train, y_train), (x_valid, y_valid))
    print('End of bmvc26_recognition_latent')


if __name__ == '__main__':
    main()
