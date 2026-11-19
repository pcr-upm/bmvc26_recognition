#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Jorge Elías'
__email__ = 'jorge.elias@alumnos.upm.es'

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import torch
import torchinfo
import importlib
import contextlib
import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from ground_truth.shapes3d import Dataset
from pcr_framework.src.recognition import Recognition
from src.evaluate import eval_func
from ldm.modules.diffusionmodules.openaimodel_shift import UNetModel
np.random.seed(42)


class BMVC26Recognition(Recognition):
    """
    On Structured Disentangled Representation and the Limits of Global Scores
    """
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.model = None
        self.gpus = None
        self.device = None
        self.nlats = None
        self.ndims = None
        self.slots = None

    def parse_options(self, params):
        super().parse_options(params)
        import argparse
        parser = argparse.ArgumentParser(prog='BMVC26Recognition', add_help=False)
        parser.add_argument('--gpu', dest='gpu', type=int, action='append',
                            help='GPU ID (negative value indicates CPU).')
        parser.add_argument('--nlats', type=int, default=10, 
                            help='The number of latent units.')
        parser.add_argument('--ndims', type=int, default=32, 
                            help='The dimensionality of the latent space.')
        parser.add_argument('--slot', dest='slot', type=int, action='append', default=[1, 2],
                            help='Latent unit indices to keep fixed.')
        args, unknown = parser.parse_known_args(params)
        print(parser.format_usage())
        mode_gpu = torch.cuda.is_available() and -1 not in args.gpu
        self.gpus = args.gpu
        self.device = torch.device('cuda' if mode_gpu else 'cpu')
        self.nlats = int(args.nlats)
        self.ndims = int(args.ndims)
        self.slots = args.slot
        if self.database == 'shapes3d':
            self.factor_names = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
            self.columns = ['Flo', 'Wal', 'Obj', 'Scl', 'Shp', 'Ori']
            self.train_indices = np.load('splits/shapes3d/train_unbiased.npy').astype(np.int64)
            self.valid_indices = np.load('splits/shapes3d/valid.npy').astype(np.int64)
        elif self.database == 'celeba':
            self.factor_names = ['Eyeglasses', 'Male', 'Wearing_Hat', 'Smiling', 'Mustache', 'Wearing_Earrings']
            self.columns = ['Eye', 'Mal', 'Hat', 'Smi', 'Mus', 'Ear']
            self.train_indices = [np.load(os.path.join('splits/celeba', f'{factor_name.lower()}_train.npy')).astype(np.int64) for factor_name in self.factor_names]
            self.valid_indices = [np.load(os.path.join('splits/celeba', f'{factor_name.lower()}_valid.npy')).astype(np.int64) for factor_name in self.factor_names]
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
                print('x_train_factor shape:', x_train_factor.shape)
                print('x_valid_factor shape:', x_valid_factor.shape)
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
        print('Load model')
        path = self.path + 'data/' + self.database + '/lat' + str(self.nlats) + '/dim' + str(self.ndims)
        filename = os.path.join(path, 'lt' + str(self.nlats) + '_' + str(self.ndims) + '_disdiff_train' + ('' if self.database == 'shapes3d' else '_celeba') + '.yaml')
        print(f"Config .yaml file: {filename}")
        config = OmegaConf.load(filename)
        _orig_unet_init_model = UNetModel.init_model
        UNetModel.init_model = lambda self: (_orig_unet_init_model(self) if getattr(self, 'ckpt_path', None) is not None and os.path.isfile(str(self.ckpt_path)) else None)
        module, cls = config.model["target"].rsplit(".", 1)
        self.model = getattr(importlib.import_module(module), cls)(**config.model.get("params", {}))
        filename = os.path.join(path, 'last.ckpt')
        print(f"Model .ckpt file: {filename}")
        ckpt = torch.load(filename, map_location='cpu', weights_only=False)
        self.ckpt_global_step = int(ckpt.get('global_step', 0)) if isinstance(ckpt, dict) else 0
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        missing, _ = self.model.load_state_dict(state_dict, strict=False)
        if hasattr(self.model, 'model_ema') and hasattr(self.model.model_ema, 'm_name2s_name'):
            missing_set = set(missing)
            with torch.no_grad():
                for name, p in self.model.model.named_parameters():
                    s_name = self.model.model_ema.m_name2s_name.get(name)
                    if s_name and f'model_ema.{s_name}' in missing_set and hasattr(self.model.model_ema, s_name):
                        getattr(self.model.model_ema, s_name).copy_(p.data)
        self.model = self.model.to(self.device)
        self.model.eval()
        latent_channels = config.model.params.channels
        latent_size = config.model.params.image_size
        image_channels = config.model.params.first_stage_config.params.ddconfig.in_channels
        image_size = config.model.params.first_stage_config.params.ddconfig.resolution
        torchinfo.summary(self.model, input_size=[(1, latent_channels, latent_size, latent_size), (1, image_channels, image_size, image_size)], depth=3, device=self.device.type, col_names=['input_size', 'output_size', 'num_params', 'kernel_size'])

    def process(self, ann, pred):
        def denorm_img(x):
            x = x.detach().cpu()
            x = (x + 1.0) / 2.0
            x = x.clamp(0, 1)
            x = x.permute(1, 2, 0).numpy()
            return x

        def get_ema_context(model):
            if hasattr(model, "ema_scope"):
                return model.ema_scope("custom-grid")
            return contextlib.nullcontext()

        @torch.no_grad()
        def sample_in_batches(cond_flat):
            outputs = []
            sample_batch_size = 8
            ddim_steps = 200
            ddim_eta = 1.0
            for start in range(0, cond_flat.shape[0], sample_batch_size):
                chunk = cond_flat[start:start + sample_batch_size]
                samples, _ = self.model.sample_log(cond=chunk, batch_size=chunk.shape[0], ddim=True, ddim_steps=ddim_steps, eta=ddim_eta)
                x_samples = self.model.decode_first_stage(samples)
                outputs.append(x_samples)
            return torch.cat(outputs, dim=0)

        print('Process image')
        ref_idx = 176977 
        c_orig_split = torch.from_numpy(ann[ref_idx:ref_idx+1]).float().to(self.device)
        c_orig_split = c_orig_split.reshape(1, self.model.model.diffusion_model.latent_unit, self.ndims)
        with get_ema_context(self.model):
            x_ref = sample_in_batches(c_orig_split.reshape(1, -1))
        pred.append(denorm_img(x_ref[0]))
        n_generations = 20
        available = [i for i in range(len(ann)) if i not in set({int(ref_idx)})]
        if not available:
            raise ValueError("No available dataset indices after applying forbidden set.")
        rng = np.random.default_rng(42)
        rand_indices = rng.choice(available, size=n_generations, replace=n_generations>len(available)).tolist()
        random_splits = torch.from_numpy(ann[rand_indices]).float().to(self.device)
        random_splits = random_splits.reshape(-1, self.model.model.diffusion_model.latent_unit, self.ndims)
        mixed = random_splits.clone()        
        for slot in self.slots:
            if slot < 0 or slot >= self.model.model.diffusion_model.latent_unit:
                raise ValueError(f"Slot {slot} is invalid. Valid range is [0, {self.model.model.diffusion_model.latent_unit-1}].")
            mixed[:, slot, :] = c_orig_split[0, slot, :].unsqueeze(0).expand(mixed.shape[0], -1)
        mixed_cond_flat = mixed.reshape(mixed.shape[0], -1)
        with get_ema_context(self.model):
            x_gen = sample_in_batches(mixed_cond_flat)
        for idx in range(x_gen.shape[0]):
            pred.append(denorm_img(x_gen[idx]))
