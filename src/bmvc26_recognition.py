#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Jorge Elías'
__email__ = 'jorge.elias@alumnos.upm.es'

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from ground_truth.shapes3d import Dataset
from pcr_framework.src.recognition import Recognition
from src.evaluate import eval_func
np.random.seed(42)


class BMVC26Recognition(Recognition):
    """
    On Structured Disentangled Representation and the Limits of Global Scores
    """
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.nlats = None
        self.ndims = None
        self.latents = None

    def parse_options(self, params):
        super().parse_options(params)
        import argparse
        parser = argparse.ArgumentParser(prog='BMVC26Recognition', add_help=False)
        parser.add_argument('--nlats', type=int, default=10, 
                            help='The number of latent units.')
        parser.add_argument('--ndims', type=int, default=32, 
                            help='The dimensionality of the latent space.')
        args, unknown = parser.parse_known_args(params)
        print(parser.format_usage())
        self.nlats = int(args.nlats)
        self.ndims = int(args.ndims)
        if self.database == 'shapes3d':
            self.factor_names = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
            self.columns = ['Flo', 'Wal', 'Obj', 'Scl', 'Shp', 'Ori']
            self.train_indices = np.load('/home/database/shapes3d/splits/train_unbiased.npy').astype(np.int64)
            self.valid_indices = np.load('/home/database/shapes3d/splits/test_fixed.npy').astype(np.int64)
        elif self.database == 'celeba':
            self.factor_names = ['Eyeglasses', 'Male', 'Wearing_Hat', 'Smiling', 'Mustache', 'Wearing_Earrings']
            self.columns = ['Eye', 'Mal', 'Hat', 'Smi', 'Mus', 'Ear']
            self.train_indices = [np.load(os.path.join('/home/database/celeba/splits_per_factor', f'{factor_name.lower()}_train.npy')).astype(np.int64) for factor_name in self.factor_names]
            self.valid_indices = [np.load(os.path.join('/home/database/celeba/splits_per_factor', f'{factor_name.lower()}_test.npy')).astype(np.int64) for factor_name in self.factor_names]
        else:
            raise ValueError('Database is not implemented')

    def train(self, anns_train, anns_valid):
        (x_train, y_train) = anns_train
        (x_valid, y_valid) = anns_valid
        results = {'metadata': {'database': self.database, 'latents_train': list(x_train.shape), 'num_slots': int(self.nlats), 'model': {'type': 'LinearSVC', 'random_state': 42, 'dual': False, 'max_iter': 5000, 'input_representation': 'single_slot_raw_vector', 'input_dim': int(self.ndims)}, 'factors': [{'factor_name': self.factor_names[j], 'factor_index': j, 'num_classes': int(len(np.unique(y_train[:, j])))} for j in range(len(self.factor_names))]}, 'results': {}}
        print('Train one SVM per factor/slot:')
        for factor_index, factor_name in enumerate(self.factor_names):
            print(f'\n========== Factor: {factor_name} (index={factor_index}) ==========')
            y_train_factor = y_train[:, factor_index] if self.database == 'shapes3d' else y_train[self.train_indices[factor_index], factor_index]
            y_valid_factor = y_valid[:, factor_index] if self.database == 'shapes3d' else y_valid[self.valid_indices[factor_index], factor_index]
            print('y_train_factor shape:', y_train_factor.shape, ' classes:', np.unique(y_train_factor))
            print('y_valid_factor shape:', y_valid_factor.shape, ' classes:', np.unique(y_valid_factor))
            results['results'][factor_name] = {}
            for slot_index in range(self.nlats):
                print(f'Training LinearSVC for slot {slot_index} -> predict {factor_name}')
                scaler = StandardScaler()
                x_train_factor = scaler.fit_transform(x_train[:, slot_index, :] if self.database == 'shapes3d' else x_train[self.train_indices[factor_index], slot_index, :])
                x_valid_factor = scaler.transform(x_valid[:, slot_index, :] if self.database == 'shapes3d' else x_valid[self.valid_indices[factor_index], slot_index, :])
                clf = LinearSVC(random_state=42, dual=False, max_iter=5000)
                clf.fit(x_train_factor, y_train_factor)
                pred_train_factor = clf.predict(x_train_factor)
                pred_valid_factor = clf.predict(x_valid_factor)
                train_acc = accuracy_score(y_train_factor, pred_train_factor)
                valid_acc = accuracy_score(y_valid_factor, pred_valid_factor)
                print(f'Factor={factor_name} | Slot={slot_index} | '
                      f'train_acc={train_acc:.4f} | valid_acc={valid_acc:.4f}')
                results['results'][factor_name][str(slot_index)] = {'train_acc': float(train_acc), 'valid_acc': float(valid_acc)}
        # Draw Table 3/6 in the main paper
        print('=' * 80)
        table_data = {name: [results['results'][name][str(slot)]['valid_acc'] for slot in range(self.nlats)] for name in self.factor_names}
        df_test = pd.DataFrame(table_data, index=[f'Slot {i}' for i in range(self.nlats)])
        df_test.columns = self.columns
        df_test = df_test.mul(100).round(2)
        print(df_test)
        # Compute disentanglement metrics in Shapes3D
        if self.database == 'shapes3d':
            print('=' * 80)
            print('Computing metrics...')
            all_latents = np.concatenate([x_train, x_valid])[np.argsort(np.concatenate([self.train_indices, self.valid_indices]))]
            latents = torch.from_numpy(all_latents).float()
            results = eval_func(eval_dataset=Dataset(np.arange(0, 480000)), data_tensor=latents, seed=42)
            print('=' * 80)
            if 'factor_VAE' in results:
                fv = results['factor_VAE']
                print('FactorVAE summary:')
                print(f"  eval_accuracy:  {fv.get('eval_accuracy')}")
            if 'dci' in results:
                dci = results['dci']
                print('DCI summary:')
                print(f"  disentanglement: {dci.get('disentanglement')}")

    def load(self, mode):
        def get_latent_key(npz_data, requested_key=None):
            if requested_key is not None:
                if requested_key not in npz_data:
                    raise KeyError(f"Requested latent key '{requested_key}' not found. Available keys: {list(npz_data.keys())}.")
                return requested_key
            candidate_keys = ['latents', 'z', 'features', 'representations']
            for key in candidate_keys:
                if key in npz_data:
                    return key
            raise KeyError(f"No latent key found in .npz. Expected one of: {candidate_keys}. Available keys: {list(npz_data.keys())}.")

        print('Load model')
        path = 'data/' + self.database + '/lat' + str(self.nlats) + '/dim' + str(self.ndims)
        filename = ('epoch=000000.npz' if self.nlats == 6 and self.ndims == 32 else 'epoch=000010.npz') if self.database == 'shapes3d' else self.database + '_representations.npz'
        data = np.load(os.path.join(path, filename), allow_pickle=True)
        print(f"Model: {os.path.join(path, filename)} | NPZ keys: {data.files}")
        latent_key = get_latent_key(data, 'latents')
        self.latents = data[latent_key]
        print('Latents shape:', self.latents.shape)
        if self.latents.ndim != 3:
            raise ValueError(f'Expected latents with shape (N, K, D), got {self.latents.shape}')

    def process(self, ann, pred):
        print('Process image')
