#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Jorge Elías'
__email__ = 'jorge.elias@alumnos.upm.es'

import os
import sys
sys.path.append(os.getcwd())
import cv2
import json
import h5py
import torch
import numpy as np
import pandas as pd
import importlib.util
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from ground_truth.shapes3d import Dataset
from src.evaluate import eval_func
np.random.seed(42)


# FACTOR_NAMES = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
FACTOR_NAMES = ['Eyeglasses', 'Male', 'Wearing_Hat', 'Smiling', 'Mustache', 'Wearing_Earrings']


def parse_options():
    """
    Parse options from command line.
    """
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--npz', type=str, required=True,
                        help='The .npz file containing latents path.',)
    parser.add_argument('--h5_path', type=str, default='/home/database/shapes3d/3dshapes.h5',
                        help='The original 3dshapes.h5 file path.')
    parser.add_argument('--train_idx_path', type=str, default='/home/database/shapes3d/splits/train_unbiased.npy',
                        help='Path to train indices .npy')
    parser.add_argument('--test_idx_path', type=str, default='/home/database/shapes3d/splits/test_fixed.npy',
                        help='Path to test indices .npy')
    parser.add_argument('--attr_path', type=str, default='/home/database/celeba/list_attr_celeba.txt',
                        help='Path to CelebA list_attr_celeba.txt')
    parser.add_argument('--splits_dir', type=str, default='/home/database/celeba/splits_per_factor', 
                        help='Directory containing balanced per-attribute CelebA splits')
    parser.add_argument('--latent_key', type=str, default=None,
                        help='Optional latent key inside the .npz file. If omitted, the script tries to infer it.')
    args, unknown = parser.parse_known_args()
    print(parser.format_usage())
    return args, unknown

def numpy_to_python(obj):
    if isinstance(obj, dict):
        return {k: numpy_to_python(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [numpy_to_python(v) for v in obj]
    if isinstance(obj, tuple):
        return [numpy_to_python(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj

def get_latent_key(npz_data, requested_key=None):
    if requested_key is not None:
        if requested_key not in npz_data:
            raise KeyError(f"Requested latent key '{requested_key}' not found. "
                           f"Available keys: {list(npz_data.keys())}")
        return requested_key
    candidate_keys = ['latents', 'z', 'features', 'representations']
    for key in candidate_keys:
        if key in npz_data:
            return key
    raise KeyError(f'No latent key found in .npz. Available keys: {list(npz_data.keys())}. '
                   f'Expected one of: {candidate_keys}')

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
        raise ValueError(f'Leakage detected: train and test splits share {len(overlap)} samples. '
                         f'First overlaps: {overlap[:10]}')

def main():
    """
    On Structured Disentangled Representation and the Limits of Global Scores latent-unit probing script.
    """
    print('OpenCV ' + cv2.__version__)
    print('NumPy ' + np.__version__)
    args, unknown = parse_options()

    # =========================
    # Load full-dataset latents
    # =========================
    data = np.load(args.npz, allow_pickle=True)
    print('NPZ keys:', data.files)

    latent_key = get_latent_key(data, args.latent_key)
    all_latents = data[latent_key]
    print('Latent key:', latent_key)
    print('All latents shape:', all_latents.shape)
    if all_latents.ndim != 3:
        raise ValueError(f'Expected latents with shape (N, K, D), got {all_latents.shape}')

    num_samples = all_latents.shape[0]
    num_slots = all_latents.shape[1]
    input_dim = all_latents.shape[2]

    # =========================================================
    # Load CelebA labels in the same global order
    # =========================================================
    factor_names = FACTOR_NAMES
    with open(args.attr_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    declared_num = int(lines[0])
    all_attr_names = lines[1].split()
    attr_to_idx = {a: i for i, a in enumerate(all_attr_names)}
    missing = [a for a in factor_names if a not in attr_to_idx]
    if missing:
        raise ValueError(f"Missing attributes in CelebA annotations: {missing}")
    selected_cols = [attr_to_idx[a] for a in factor_names]
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
        raise ValueError(f"Number of global labels ({len(all_labels)}) does not match "
                         f"number of latent samples ({num_samples}).")

    num_classes_per_factor = [2 for j in range(all_labels.shape[1])]
    
    # =========================
    # Load Shapes3D labels
    # =========================
    # with h5py.File(args.h5_path, 'r') as f:
    #     all_labels = f['labels'][:]

    # if all_latents.shape[0] != all_labels.shape[0]:
    #     raise ValueError(f'The latent file must follow the original Shapes3D order. '
    #                      f'Latents have {all_latents.shape[0]} samples, '
    #                      f'but 3dshapes.h5 has {all_labels.shape[0]} labels.')

    # factor_names = FACTOR_NAMES
    # if all_labels.shape[1] != len(factor_names):
    #     factor_names = [f'factor_{i}' for i in range(all_labels.shape[1])]

    # num_classes_per_factor = [int(len(np.unique(all_labels[:, j]))) for j in range(all_labels.shape[1])]

    # =========================
    # Load train/test indices
    # =========================
    # train_indices = np.load(args.train_idx_path).astype(np.int64)
    # test_indices = np.load(args.test_idx_path).astype(np.int64)

    # validate_indices(train_indices=train_indices, test_indices=test_indices, num_samples=num_samples)

    # =========================
    # Build train/test sets from the full representation
    # =========================
    # train_latents = all_latents[train_indices]
    # test_latents = all_latents[test_indices]

    # train_label_values = all_labels[train_indices]
    # test_label_values = all_labels[test_indices]

    # Y_train_all = convert_labels_to_class_ids(all_labels, train_label_values)
    # Y_test_all = convert_labels_to_class_ids(all_labels, test_label_values)

    # print('Train latents shape:', train_latents.shape)
    # print('Test latents shape:', test_latents.shape)
    # print('Train labels shape:', Y_train_all.shape)
    # print('Test labels shape:', Y_test_all.shape)

    # =========================
    # Metadata
    # =========================
    results = {
        'metadata': {
            'all_npz': args.npz,
            'latent_key': latent_key,
            # 'h5_path': args.h5_path,
            # 'train_idx_path': args.train_idx_path,
            # 'test_idx_path': args.test_idx_path,
            'all_latents_shape': list(all_latents.shape),
            # 'train_latents_shape': list(train_latents.shape),
            # 'test_latents_shape': list(test_latents.shape),
            'num_slots_evaluated': int(num_slots),
            'model': {
                'type': 'LinearSVC',
                'random_state': 42,
                'dual': False,
                'max_iter': 5000,
                'input_representation': 'single_slot_raw_vector',
                'input_dim': int(input_dim),
            },
            'factors': [
                {
                    'factor_name': factor_names[j],
                    'factor_index': j,
                    'num_classes': num_classes_per_factor[j],
                }
                for j in range(len(factor_names))
            ],
        },
        'results': {},
    }

    # =========================
    # Train/evaluate one SVM per (factor, slot)
    # =========================
    for factor_index, factor_name in enumerate(factor_names):
        print(f'\n========== Factor: {factor_name} (index={factor_index}) ==========')
        # y_train = Y_train_all[:, factor_index]
        # y_test = Y_test_all[:, factor_index]
        safe_name = factor_name.lower()
        train_idx_path = os.path.join(args.splits_dir, f'{safe_name}_train.npy')
        test_idx_path = os.path.join(args.splits_dir, f'{safe_name}_test.npy')
        train_idx = np.load(train_idx_path).astype(np.int64)
        test_idx = np.load(test_idx_path).astype(np.int64)
        if np.any(train_idx < 0) or np.any(train_idx >= num_samples):
            raise IndexError(f"Train indices out of range for factor {factor_name}")
        if np.any(test_idx < 0) or np.any(test_idx >= num_samples):
            raise IndexError(f"Test indices out of range for factor {factor_name}")
        y_train = all_labels[train_idx, factor_index]
        y_test = all_labels[test_idx, factor_index]

        print("Train size:", len(train_idx))
        print("Test size:", len(test_idx))
        print('y_train shape:', y_train.shape)
        print('y_test shape:', y_test.shape)
        print('Unique train classes:', np.unique(y_train))
        print('Unique test classes:', np.unique(y_test))

        results['results'][factor_name] = {}

        for slot_index in range(num_slots):
            print(f'Training LinearSVC for slot {slot_index} -> predict {factor_name}')

            # X_train = train_latents[:, slot_index, :]
            # X_test = test_latents[:, slot_index, :]
            X_train = all_latents[train_idx, slot_index, :]
            X_test = all_latents[test_idx, slot_index, :]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            clf = LinearSVC(random_state=42, dual=False, max_iter=5000)
            clf.fit(X_train_scaled, y_train)

            y_train_pred = clf.predict(X_train_scaled)
            y_test_pred = clf.predict(X_test_scaled)

            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)

            print(f'Factor={factor_name} | Slot={slot_index} | '
                  f'train_acc={train_acc:.4f} | test_acc={test_acc:.4f}')

            results['results'][factor_name][str(slot_index)] = {'train_accuracy': float(train_acc), 'test_accuracy': float(test_acc)}

    print('=' * 80)
    table_data = {name: [results['results'][name][str(slot)]['test_accuracy'] for slot in range(num_slots)] for name in factor_names}
    df_test = pd.DataFrame(table_data, index=[f'Slot {i}' for i in range(num_slots)])
    # df_test.columns = ['Flo', 'Wal', 'Obj', 'Scl', 'Shp', 'Ori']
    df_test.columns = ['Eye', 'Mal', 'Hat', 'Smi', 'Mus', 'Ear']
    df_test = df_test.mul(100).round(2)
    print(df_test)

    spec = importlib.util.find_spec('pcr_framework')
    output_path = os.path.join('pcr_framework' if spec is None else os.path.dirname(spec.origin), 'output')
    Path(output_path).mkdir(parents=True, exist_ok=True)
    output_json = os.path.join(output_path, 'SVM.json')
    results = numpy_to_python(results)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'\nAll results saved to: {output_json}')

    print('=' * 80)
    label_dataset = Dataset(np.arange(0, 480000))
    print('Computing metrics...')
    latents = torch.from_numpy(all_latents).float()
    results = eval_func(eval_dataset=label_dataset, data_tensor=latents, metric_folder=output_path, it=num_samples, seed=42)
    print('=' * 80)

    if 'factor_VAE' in results:
        fv = results['factor_VAE']
        print('FactorVAE summary:')
        print(f"  eval_accuracy:  {fv.get('eval_accuracy')}")
    if 'dci' in results:
        dci = results['dci']
        print('DCI summary:')
        print(f"  disentanglement: {dci.get('disentanglement')}")
    print('End of bmvc26_recognition_latent')


if __name__ == '__main__':
    main()
