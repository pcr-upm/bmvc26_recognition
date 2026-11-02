#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Jorge Elías'
__email__ = 'jorgeeliasgarcia2@gmail.com'

import os
import sys
sys.path.append(os.getcwd())
import h5py
import argparse
import numpy as np
import pandas as pd
from pcr_framework.src.composite import Composite
from src.bmvc26_recognition import BMVC26Recognition


def parse_options():
    parser = argparse.ArgumentParser(description='Reproduce Tables 1 and 2 (full-representation linear SVM probing).')
    parser.add_argument('--backbone', type=str, default='all', choices=['resnet', 'simclr', 'disdiff', 'all'],
                        help='Backbone to evaluate (default: all).')
    parser.add_argument('--regime', type=str, default='all', choices=['unbiased', 'biased', 'all'],
                        help='Training regime to evaluate (default: all).')
    parser.add_argument('--samples-per-class', type=int, default=25,
                        help='Balanced training samples per class for each SVM probe (default: 25).')
    parser.add_argument('--seed', type=int, default=23,
                        help='Random seed for balanced subsampling and LinearSVC (default: 23).')
    args, unknown = parser.parse_known_args()
    return args, unknown


def load_labels_local(database, factor_names):
    if database == 'shapes3d':
        h5_candidates = [
            os.path.join('data', 'shapes3d', '3dshapes.h5'),
            os.path.expanduser('~/data/disentanglement_lib/shapes3d/3dshapes.h5'),
            '/home/jelias/data/disentanglement_lib/shapes3d/3dshapes.h5',
        ]
        h5_path = next((p for p in h5_candidates if os.path.isfile(p)), None)
        if h5_path is None:
            raise FileNotFoundError("No se encontró '3dshapes.h5' en 'data/shapes3d/3dshapes.h5'.")
        with h5py.File(h5_path, 'r') as f:
            raw_labels = f['labels'][:]
        class_ids = np.zeros_like(raw_labels, dtype=np.int64)
        for j in range(raw_labels.shape[1]):
            vals = np.unique(raw_labels[:, j])
            val_to_id = {float(v): i for i, v in enumerate(vals)}
            class_ids[:, j] = np.asarray([val_to_id[float(v)] for v in raw_labels[:, j]], dtype=np.int64)
        return class_ids

    elif database == 'celeba':
        attr_candidates = [
            os.path.join('data', 'celeba', 'list_attr_celeba.txt'),
            '/home/database/recognition/faces/celeba/list_attr_celeba.txt',
        ]
        attr_path = next((p for p in attr_candidates if os.path.isfile(p)), None)
        if attr_path is None:
            raise FileNotFoundError("No se encontró 'list_attr_celeba.txt' en 'data/celeba/'.")
        with open(attr_path, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        all_attr_names = lines[1].split()
        attr_to_idx = {a: i for i, a in enumerate(all_attr_names)}
        selected_cols = [attr_to_idx[a] for a in factor_names]
        labels = []
        for line in lines[2:]:
            parts = line.split()
            raw_attrs = [int(x) for x in parts[1:]]
            y = [raw_attrs[c] for c in selected_cols]
            labels.append(((np.array(y, dtype=np.int64) + 1) // 2).astype(np.int64))
        return np.asarray(labels, dtype=np.int64)


def main():
    args, unknown = parse_options()
    composite = Composite()
    sr = BMVC26Recognition('')
    composite.add(sr)
    composite.parse_options(unknown)

    databases = ['shapes3d', 'celeba'] if sr.database == 'all' else [sr.database]
    backbones = ['resnet', 'simclr', 'disdiff'] if args.backbone == 'all' else [args.backbone]
    regimes = ['unbiased', 'biased'] if args.regime == 'all' else [args.regime]

    backbone_labels = {'resnet': 'ResNet', 'simclr': 'SimCLR', 'disdiff': 'DisDiff'}
    regime_labels = {'unbiased': 'Unbiased', 'biased': 'Biased'}

    table1_rows = {}
    table2_cols = {}

    for db in databases:
        sr.parse_options(['--database', db, '--nlats', '10', '--ndims', '32'])
        all_labels = load_labels_local(db, sr.factor_names)

        train_idx_path = os.path.join('splits', db, 'train_unbiased.npy')
        test_idx_path = os.path.join('splits', db, 'test_fixed.npy')
        if not os.path.isfile(test_idx_path):
            test_idx_path = os.path.join('splits', db, 'valid.npy')

        train_idx = np.load(train_idx_path).astype(np.int64)
        test_idx = np.load(test_idx_path).astype(np.int64)
        y_train_full = all_labels[train_idx]
        y_test_full = all_labels[test_idx]

        for b_key in backbones:
            for r_key in regimes:
                row_id = (backbone_labels[b_key], regime_labels[r_key])
                print(f'\nEvaluating {row_id[0]} ({row_id[1]}) on {db}...')
                x_train_full, x_test_full = sr.load_full_representation(b_key, r_key)
                factor_accs, shape_cls = sr.evaluate_full_representation(
                    x_train_full, y_train_full, x_test_full, y_test_full,
                    samples_per_class=args.samples_per_class, seed=args.seed
                )
                if row_id not in table1_rows:
                    table1_rows[row_id] = {}
                table1_rows[row_id].update(factor_accs)

                if db == 'shapes3d' and r_key == 'biased' and shape_cls:
                    for metric_name in ['Prec.', 'Rec.', 'F1']:
                        table2_cols[(backbone_labels[b_key], metric_name)] = {
                            cls_name: shape_cls[cls_name][metric_name]
                            for cls_name in ['Cube', 'Cylinder', 'Sphere', 'Capsule']
                        }

    if table1_rows:
        print('\n' + '=' * 80)
        print('TABLE 1: Factor-wise classification accuracy (%) on test splits')
        print('=' * 80)
        df_t1 = pd.DataFrame.from_dict(table1_rows, orient='index')
        df_t1.index = pd.MultiIndex.from_tuples(df_t1.index, names=['Backbone', 'Training'])
        print(df_t1.round(2))

    if table2_cols:
        print('\n' + '=' * 80)
        print('TABLE 2: Class-wise Precision, Recall, and F1 (%) for shape (Shapes3D Biased)')
        print('=' * 80)
        df_t2 = pd.DataFrame(table2_cols)
        df_t2.columns = pd.MultiIndex.from_tuples(df_t2.columns)
        print(df_t2.round(2))

    print('\nEnd of bmvc26_recognition_representation')


if __name__ == '__main__':
    main()