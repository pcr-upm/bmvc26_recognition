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
import numpy as np
import importlib.util
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
np.random.seed(42)


FACTOR_NAMES = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']


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

    # =========================
    # Load Shapes3D labels
    # =========================
    with h5py.File(args.h5_path, 'r') as f:
        all_labels = f['labels'][:]

    if all_latents.shape[0] != all_labels.shape[0]:
        raise ValueError(f'The latent file must follow the original Shapes3D order. '
                         f'Latents have {all_latents.shape[0]} samples, '
                         f'but 3dshapes.h5 has {all_labels.shape[0]} labels.')

    factor_names = FACTOR_NAMES
    if all_labels.shape[1] != len(factor_names):
        factor_names = [f'factor_{i}' for i in range(all_labels.shape[1])]

    num_classes_per_factor = [int(len(np.unique(all_labels[:, j]))) for j in range(all_labels.shape[1])]

    # =========================
    # Load train/test indices
    # =========================
    train_indices = np.load(args.train_idx_path).astype(np.int64)
    test_indices = np.load(args.test_idx_path).astype(np.int64)

    validate_indices(train_indices=train_indices, test_indices=test_indices, num_samples=num_samples)

    # =========================
    # Build train/test sets from the full representation
    # =========================
    train_latents = all_latents[train_indices]
    test_latents = all_latents[test_indices]

    train_label_values = all_labels[train_indices]
    test_label_values = all_labels[test_indices]

    Y_train_all = convert_labels_to_class_ids(all_labels, train_label_values)
    Y_test_all = convert_labels_to_class_ids(all_labels, test_label_values)

    print('Train latents shape:', train_latents.shape)
    print('Test latents shape:', test_latents.shape)
    print('Train labels shape:', Y_train_all.shape)
    print('Test labels shape:', Y_test_all.shape)

    # =========================
    # Metadata
    # =========================
    results = {
        'metadata': {
            'all_npz': args.npz,
            'latent_key': latent_key,
            'h5_path': args.h5_path,
            'train_idx_path': args.train_idx_path,
            'test_idx_path': args.test_idx_path,
            'all_latents_shape': list(all_latents.shape),
            'train_latents_shape': list(train_latents.shape),
            'test_latents_shape': list(test_latents.shape),
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
        y_train = Y_train_all[:, factor_index]
        y_test = Y_test_all[:, factor_index]

        print('y_train shape:', y_train.shape)
        print('y_test shape:', y_test.shape)
        print('Unique train classes:', np.unique(y_train))
        print('Unique test classes:', np.unique(y_test))

        results['results'][factor_name] = {}

        for slot_index in range(num_slots):
            print(f'Training LinearSVC for slot {slot_index} -> predict {factor_name}')

            X_train = train_latents[:, slot_index, :]
            X_test = test_latents[:, slot_index, :]

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

    spec = importlib.util.find_spec('pcr_framework')
    output_path = os.path.join('pcr_framework' if spec is None else os.path.dirname(spec.origin), 'output')
    Path(output_path).mkdir(parents=True, exist_ok=True)
    output_json = os.path.join(output_path, 'SVM.json')
    results = numpy_to_python(results)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print(f'\nAll results saved to: {output_json}')

    print("=" * 80)
    print("DisDiff metric evaluation from .npz")
    print("=" * 80)
    print(f"NPZ:     {args.npz}")
    print(f"Outdir:  {output_path}")
    print(f"Dataset: {'shapes3d'}")
    print(f"Seed:    {42}")
    print("=" * 80)

    print("[1/3] Loading latents...")
    latents, num_samples = load_latents_npz(args.npz)

    print(f"Loaded latents shape: {tuple(latents.shape)}")
    print(f"Output id / num_samples: {num_samples}")

    expected_json = os.path.join(output_path, f"{num_samples}.json")

    if os.path.exists(expected_json) and not conf.overwrite:
        print()
        print(f"[SKIP] Output JSON already exists: {expected_json}")
        print("Use --overwrite to recompute it.")
        return 0

    print("[2/3] Loading ground-truth dataset...")
    label_dataset = make_dis_dataset('shapes3d')

    if not conf.no_metadata:
        metadata_path = maybe_write_run_metadata(outdir=output_path, npz_path=args.npz, dataset='shapes3d', seed=42, latents=latents, num_samples=num_samples)
        print(f"Run metadata written to: {metadata_path}")

    print("[3/3] Computing metrics...")
    results = eval_func(eval_dataset=label_dataset, data_tensor=latents, metric_folder=output_path, it=num_samples, seed=42, preflix=conf.prefix)

    print()
    print("=" * 80)
    print("Finished.")
    print(f"Metric JSON written to: {expected_json}")
    print("=" * 80)

    if "factor_VAE" + conf.prefix in results:
        fv = results["factor_VAE" + conf.prefix]
        if isinstance(fv, dict):
            print("FactorVAE summary:")
            print(f"  train_accuracy: {fv.get('train_accuracy')}")
            print(f"  eval_accuracy:  {fv.get('eval_accuracy')}")
            print(f"  num_active_dims: {fv.get('num_active_dims')}")
    print('End of bmvc26_recognition_latent')


if __name__ == '__main__':
    main()
