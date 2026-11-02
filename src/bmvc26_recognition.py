#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Jorge Elías'
__email__ = 'jorgeeliasgarcia2@gmail.com'

import os
import sys
import json
import glob
import types
import random
import inspect
import h5py
from contextlib import nullcontext

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from omegaconf import OmegaConf
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# For compatibilites and optional modules
import pytorch_lightning.utilities
if 'pytorch_lightning.utilities.distributed' not in sys.modules:
    dist_mod = types.ModuleType('pytorch_lightning.utilities.distributed')
    try:
        from pytorch_lightning.utilities.rank_zero import rank_zero_only
    except ImportError:
        from pytorch_lightning.utilities import rank_zero_only
    dist_mod.rank_zero_only = rank_zero_only
    sys.modules['pytorch_lightning.utilities.distributed'] = dist_mod

for _opt_mod in ('clip', 'kornia'):
    if _opt_mod not in sys.modules:
        try:
            __import__(_opt_mod)
        except ImportError:
            sys.modules[_opt_mod] = types.ModuleType(_opt_mod)

from ground_truth.shapes3d import Dataset
from pcr_framework.src.recognition import Recognition
from src.evaluate import eval_func

np.random.seed(42)


def get_obj_from_str(string, reload=False):
    import importlib
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config):
    if "target" not in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def set_seed(seed=None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def denorm_img(x):
    x = x.detach().cpu()
    x = (x + 1.0) / 2.0
    x = x.clamp(0, 1)
    x = x.permute(1, 2, 0).numpy()
    return x


def get_ema_context(model):
    if hasattr(model, "ema_scope"):
        return model.ema_scope("custom-grid")
    return nullcontext()


def sample_random_indices(rng, dataset_len, n, forbidden=None):
    forbidden = set() if forbidden is None else set(forbidden)
    available = [i for i in range(dataset_len) if i not in forbidden]
    if not available:
        raise ValueError("No available dataset indices after applying forbidden set.")
    replace = n > len(available)
    return rng.choice(available, size=n, replace=replace).tolist()


def mix_fixed_slots(orig_split, random_splits, fixed_slots):
    mixed = random_splits.clone()
    for slot in fixed_slots:
        mixed[:, slot, :] = orig_split[0, slot, :].unsqueeze(0).expand(mixed.shape[0], -1)
    return mixed.reshape(mixed.shape[0], -1)


def validate_fixed_slots(fixed_slots, latent_unit):
    if not fixed_slots:
        raise ValueError("You must provide at least 1 fixed slot.")
    if len(set(fixed_slots)) != len(fixed_slots):
        raise ValueError(f"Fixed slots must be unique, got: {fixed_slots}")
    for slot in fixed_slots:
        if slot < 0 or slot >= latent_unit:
            raise ValueError(f"Slot {slot} is invalid. Valid range is [0, {latent_unit - 1}].")


import torch.nn as nn
import torchvision.models as models
from sklearn.metrics import precision_score, recall_score, f1_score


class ResNetMultiTask(nn.Module):
    def __init__(self, backbone_name="resnet18", pretrained=False, repr_dim=320, task_names=None, num_classes=None):
        super().__init__()
        if task_names is None:
            task_names = ["floor_hue", "wall_hue", "object_hue", "scale", "shape", "orientation"]
        if num_classes is None:
            num_classes = [10, 10, 10, 8, 4, 15]
        self.task_names = task_names
        self.num_classes = num_classes
        try:
            backbone = models.resnet18(weights=None)
        except TypeError:
            backbone = models.resnet18(pretrained=pretrained)
        backbone_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Identity() if repr_dim == backbone_dim else nn.Linear(backbone_dim, repr_dim)
        self.feat_dim = repr_dim
        self.heads = nn.ModuleDict({t: nn.Linear(self.feat_dim, c) for t, c in zip(task_names, num_classes)})

    def load_state_dict(self, state_dict, strict=True):
        # Mapeo automático de claves antiguas/alternativas del state_dict de ResNet
        new_state_dict = {}
        mapping = {
            'head_floor': 'heads.floor_hue',
            'head_wall': 'heads.wall_hue',
            'head_object': 'heads.object_hue',
            'head_scale': 'heads.scale',
            'head_shape': 'heads.shape',
            'head_orientation': 'heads.orientation'
        }
        for k, v in state_dict.items():
            new_k = k
            for old_p, new_p in mapping.items():
                if old_p in k:
                    new_k = k.replace(old_p, new_p)
                    break
            new_state_dict[new_k] = v
        return super().load_state_dict(new_state_dict, strict=False)

    def forward(self, x):
        features = self.proj(self.backbone(x))
        logits = {t: self.heads[t](features) for t in self.task_names}
        return logits, features


class SimCLRResNet18(nn.Module):
    def __init__(self, repr_dim=320, proj_dim=128, hidden_dim=320, small_resnet_stem=True):
        super().__init__()
        try:
            backbone = models.resnet18(weights=None)
        except TypeError:
            backbone = models.resnet18(pretrained=False)
        if small_resnet_stem:
            backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            backbone.maxpool = nn.Identity()
        backbone_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.repr_proj = nn.Identity() if repr_dim == backbone_dim else nn.Linear(backbone_dim, repr_dim)
        self.projection_head = nn.Sequential(
            nn.Linear(repr_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim, bias=False),
            nn.BatchNorm1d(proj_dim, affine=False),
        )

    def forward(self, x, return_features=False):
        h = self.repr_proj(self.backbone(x))
        if return_features:
            return h
        return F.normalize(self.projection_head(h), dim=1)

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
        self.model = None
        self.device = None
        self.fixed_slots = [1, 2]
        self.grid_size = 5
        self.ddim_steps = 200
        self.ddim_eta = 1.0
        self.sample_batch_size = 8
        self.seed = 42
        self.config_path = None
        self.ckpt_path = None
        self.outdir = None
        self._process_counter = 0

    def parse_options(self, params):
        super().parse_options(params)
        import argparse
        parser = argparse.ArgumentParser(prog='BMVC26Recognition', add_help=False)
        parser.add_argument('--nlats', type=int, default=10, help='The number of latent units.')
        parser.add_argument('--ndims', type=int, default=32, help='The dimensionality of the latent space.')
        parser.add_argument('--fixed-slots', '--fixed_slots', dest='fixed_slots',
                            type=str, nargs='+', default=['1', '2'],
                            help='Latent unit indices to keep fixed (e.g. --fixed-slots 2 4).')
        parser.add_argument('--grid-size', '--grid_size', dest='grid_size', type=int, default=5)
        parser.add_argument('--ddim-steps', '--ddim_steps', dest='ddim_steps', type=int, default=200)
        parser.add_argument('--ddim-eta', '--ddim_eta', dest='ddim_eta', type=float, default=0.0)
        parser.add_argument('--sample-batch-size', '--sample_batch_size', dest='sample_batch_size', type=int, default=8)
        parser.add_argument('--seed', type=int, default=42)
        parser.add_argument('--config', type=str, default=None)
        parser.add_argument('--ckpt', type=str, default=None)
        parser.add_argument('--outdir', type=str, default=None)
        args, unknown = parser.parse_known_args(params)
        print(parser.format_usage())
        self.nlats = int(args.nlats)
        self.ndims = int(args.ndims)

        parsed_slots = []
        for item in args.fixed_slots:
            for part in str(item).split(','):
                part = part.strip()
                if part != '':
                    parsed_slots.append(int(part))
        self.fixed_slots = parsed_slots
        self.grid_size = int(args.grid_size)
        self.ddim_steps = int(args.ddim_steps)
        self.ddim_eta = float(args.ddim_eta)
        self.sample_batch_size = int(args.sample_batch_size)
        self.seed = args.seed
        self.config_path = args.config
        self.ckpt_path = args.ckpt
        self.outdir = args.outdir

        if self.database == 'shapes3d':
            self.factor_names = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation']
            self.columns = ['Flo', 'Wal', 'Obj', 'Scl', 'Shp', 'Ori']
            self.train_indices = np.load('splits/shapes3d/train_unbiased.npy').astype(np.int64)
            self.valid_indices = np.load('splits/shapes3d/valid.npy').astype(np.int64)
        elif self.database == 'celeba':
            self.factor_names = ['Eyeglasses', 'Male', 'Wearing_Hat', 'Smiling', 'Mustache', 'Wearing_Earrings']
            self.columns = ['Eye', 'Mal', 'Hat', 'Smi', 'Mus', 'Ear']
            self.train_indices = [
                np.load(os.path.join('splits/celeba', f'{fn.lower()}_train.npy')).astype(np.int64)
                for fn in self.factor_names
            ]
            self.valid_indices = [
                np.load(
                    os.path.join('splits/celeba', f'{fn.lower()}_test.npy')
                    if os.path.isfile(os.path.join('splits/celeba', f'{fn.lower()}_test.npy'))
                    else os.path.join('splits/celeba', f'{fn.lower()}_valid.npy')
                ).astype(np.int64)
                for fn in self.factor_names
            ]
        else:
            raise ValueError('Database is not implemented')

    def train(self, anns_train, anns_valid):
        (x_train, y_train) = anns_train
        (x_valid, y_valid) = anns_valid
        svm_seed = 23 if self.database == 'celeba' else 42
        np.random.seed(svm_seed)
        results = {'metadata': {'database': self.database, 'latents_train': list(x_train.shape), 'num_slots': int(self.nlats), 'model': {'type': 'LinearSVC', 'random_state': svm_seed, 'dual': False, 'max_iter': 5000, 'input_representation': 'single_slot_raw_vector', 'input_dim': int(self.ndims)}, 'factors': [{'factor_name': self.factor_names[j], 'factor_index': j, 'num_classes': int(len(np.unique(y_train[:, j])))} for j in range(len(self.factor_names))]}, 'results': {}}
        print('Train one SVM per factor/slot:')
        for factor_index, factor_name in enumerate(self.factor_names):
            print(f'\n========== Factor: {factor_name} (index={factor_index}) ==========')
            y_train_factor = y_train[:, factor_index] if self.database == 'shapes3d' else y_train[self.train_indices[factor_index], factor_index]
            y_valid_factor = y_valid[:, factor_index] if self.database == 'shapes3d' else y_valid[self.valid_indices[factor_index], factor_index]
            results['results'][factor_name] = {}
            for slot_index in range(self.nlats):
                scaler = StandardScaler()
                x_train_factor = scaler.fit_transform(x_train[:, slot_index, :] if self.database == 'shapes3d' else x_train[self.train_indices[factor_index], slot_index, :])
                x_valid_factor = scaler.transform(x_valid[:, slot_index, :] if self.database == 'shapes3d' else x_valid[self.valid_indices[factor_index], slot_index, :])
                clf = LinearSVC(random_state=svm_seed, dual=False, max_iter=5000)
                clf.fit(x_train_factor, y_train_factor)
                train_acc = accuracy_score(y_train_factor, clf.predict(x_train_factor))
                valid_acc = accuracy_score(y_valid_factor, clf.predict(x_valid_factor))
                results['results'][factor_name][str(slot_index)] = {'train_acc': float(train_acc), 'valid_acc': float(valid_acc)}
        print('=' * 80)
        table_data = {name: [results['results'][name][str(slot)]['valid_acc'] for slot in range(self.nlats)] for name in self.factor_names}
        df_test = pd.DataFrame(table_data, index=[f'Slot {i}' for i in range(self.nlats)])
        df_test.columns = self.columns
        print(df_test.mul(100).round(2))
        if self.database == 'shapes3d':
            print('=' * 80)
            if len(x_train) + len(x_valid) == 480000:
                all_latents = np.concatenate([x_train, x_valid])[np.argsort(np.concatenate([self.train_indices, self.valid_indices]))]
            else:
                all_latents = self.latents
            latents = torch.from_numpy(all_latents).float()
            metrics = eval_func(eval_dataset=Dataset(np.arange(0, len(all_latents))), data_tensor=latents, seed=42)
            print(metrics)

    def _get_default_disdiff_config(self):
        return OmegaConf.create({
            'model': {
                'base_learning_rate': 1e-4,
                'target': 'ldm.models.diffusion.ddpm.LatentDiffusion',
                'params': {
                    'linear_start': 0.0015, 'linear_end': 0.0155, 'num_timesteps_cond': 1,
                    'log_every_t': 200, 'timesteps': 1000, 'loss_type': 'l2',
                    'first_stage_key': 'image', 'cond_stage_key': 'image',
                    'image_size': 16, 'channels': 3, 'cond_stage_trainable': True,
                    'concat_mode': False, 'scale_by_std': True, 'conditioning_key': 'adm',
                    'monitor': 'train/loss_step', 'dis_loss_flag': True, 'dis_loss_type': 'Z', 'dis_weight': 0.05,
                    'unet_config': {
                        'target': 'ldm.modules.diffusionmodules.openaimodel_shift.UNetModel',
                        'params': {
                            'image_size': 16, 'in_channels': 3, 'out_channels': 3, 'model_channels': 64,
                            'attention_resolutions': [1, 2, 4], 'num_res_blocks': 2, 'channel_mult': [1, 2, 4, 4],
                            'num_heads': 8, 'repre_emb_channels': int(self.ndims), 'use_scale_shift_norm': True,
                            'resblock_updown': True, 'seprate_decoder': True, 'orth_emb': True,
                            'ckpt_path': None, 'load_ema_ckpt': False, 'latent_unit': int(self.nlats),
                        },
                    },
                    'first_stage_config': {
                        'target': 'ldm.models.autoencoder.VQModelInterface',
                        'params': {
                            'embed_dim': 3, 'n_embed': 2048, 'monitor': 'val/rec_loss', 'ckpt_path': None,
                            'ddconfig': {'double_z': False, 'z_channels': 3, 'resolution': 64, 'in_channels': 3, 'out_ch': 3, 'ch': 32, 'ch_mult': [1, 2, 4], 'num_res_blocks': 2, 'attn_resolutions': [], 'dropout': 0.0},
                            'lossconfig': {'target': 'torch.nn.Identity'},
                        },
                    },
                    'cond_stage_config': {
                        'target': 'ldm.modules.encoders.modules.Encoder4',
                        'params': {'d': 128, 'latent_dim': int(self.nlats * self.ndims)},
                    },
                },
            }
        })

    def _resolve_config_and_ckpt(self, base_path):
        config_path = self.config_path
        if config_path is None:
            candidates = [
                os.path.join(base_path, f'lt{self.nlats}_{self.ndims}_disdiff_train.yaml'),
                os.path.join(base_path, f'disdiff_train_{self.database}.yaml'),
                os.path.join(base_path, 'disdiff_train_shapes3D.yaml'),
            ]
            config_path = next((p for p in candidates if os.path.isfile(p)), None)
            if config_path is None:
                yamls = glob.glob(os.path.join(base_path, '*.yaml'))
                if yamls:
                    config_path = yamls[0]

        ckpt_path = self.ckpt_path
        if ckpt_path is None:
            default_ckpt = ('epoch=000000.ckpt' if self.nlats == 6 and self.ndims == 32 else 'epoch=000010.ckpt') if self.database == 'shapes3d' else f'{self.database}_representations.ckpt'
            candidates = [os.path.join(base_path, default_ckpt), os.path.join(base_path, 'last.ckpt')]
            ckpt_path = next((p for p in candidates if os.path.isfile(p)), None)
            if ckpt_path is None:
                ckpts = glob.glob(os.path.join(base_path, '*.ckpt'))
                if ckpts:
                    ckpt_path = ckpts[0]

        return config_path, ckpt_path

    def _sanitize_config_ckpt_paths(self, config):
        if 'model' in config and 'params' in config.model:
            params = config.model.params
            if 'ckpt_path' in params:
                if not params.ckpt_path or not os.path.isfile(str(params.ckpt_path)):
                    params.pop('ckpt_path', None)

            if 'unet_config' in params and 'params' in params.unet_config:
                u_params = params.unet_config.params
                if 'ckpt_path' in u_params:
                    if not u_params.ckpt_path or not os.path.isfile(str(u_params.ckpt_path)):
                        u_params.pop('ckpt_path', None)
                u_params.latent_unit = int(self.nlats)
                u_params.repre_emb_channels = int(self.ndims)

            if 'first_stage_config' in params and 'params' in params.first_stage_config:
                fs_params = params.first_stage_config.params
                if 'ckpt_path' in fs_params:
                    if not fs_params.ckpt_path or not os.path.isfile(str(fs_params.ckpt_path)):
                        fs_params.pop('ckpt_path', None)

            if 'cond_stage_config' in params and 'params' in params.cond_stage_config:
                cs_params = params.cond_stage_config.params
                if 'ckpt_path' in cs_params:
                    if not cs_params.ckpt_path or not os.path.isfile(str(cs_params.ckpt_path)):
                        cs_params.pop('ckpt_path', None)
                cs_params.latent_dim = int(self.nlats * self.ndims)
        return config

    def _load_generative_model(self, config_path, ckpt_path):
        if self.device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print(f'Loading DisDiff generative model on {self.device}...')

        if config_path is not None and os.path.isfile(config_path):
            config = OmegaConf.load(config_path)
        else:
            config = self._get_default_disdiff_config()
        config = self._sanitize_config_ckpt_paths(config)

        from ldm.modules.diffusionmodules.openaimodel_shift import UNetModel
        _orig_unet_init_model = UNetModel.init_model

        def _safe_unet_init_model(unet_self):
            ckpt_p = getattr(unet_self, 'ckpt_path', None)
            if ckpt_p is not None and os.path.isfile(str(ckpt_p)):
                return _orig_unet_init_model(unet_self)

        UNetModel.init_model = _safe_unet_init_model

        model = instantiate_from_config(config.model)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        self.ckpt_global_step = int(ckpt.get('global_step', 0)) if isinstance(ckpt, dict) else 0
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        missing, _ = model.load_state_dict(state_dict, strict=False)

        if hasattr(model, 'model_ema') and hasattr(model.model_ema, 'm_name2s_name'):
            missing_set = set(missing)
            with torch.no_grad():
                for name, p in model.model.named_parameters():
                    s_name = model.model_ema.m_name2s_name.get(name)
                    if s_name and f'model_ema.{s_name}' in missing_set and hasattr(model.model_ema, s_name):
                        getattr(model.model_ema, s_name).copy_(p.data)

        model = model.to(self.device)
        model.eval()
        self.model = model

    @torch.no_grad()
    def _generate_representations_npz(self, filename, batch_size=64):
        if self.model is None:
            raise RuntimeError("Generative DisDiff model (.ckpt) must be loaded before generating representations (.npz).")

        from tqdm import tqdm
        latent_unit = int(self.model.model.diffusion_model.latent_unit)

        if self.database == 'shapes3d':
            import h5py

            h5_candidates = [
                os.path.join('data', 'shapes3d', '3dshapes.h5'),
                os.path.expanduser('~/data/disentanglement_lib/shapes3d/3dshapes.h5'),
                '/home/jelias/data/disentanglement_lib/shapes3d/3dshapes.h5',
            ]
            h5_path = next((p for p in h5_candidates if os.path.isfile(p)), None)
            if h5_path is None:
                raise FileNotFoundError("No se encontró '3dshapes.h5'. Colócalo en 'data/shapes3d/3dshapes.h5'.")

            print(f'Generating {self.database} representations (.npz) from checkpoint using {h5_path}...')

            with h5py.File(h5_path, 'r') as f:
                images_ds = f['images']
                num_samples = len(images_ds)
                latents_arr = np.empty((num_samples, latent_unit, int(self.ndims)), dtype=np.float32)

                chunk_size = batch_size * 100
                for chunk_start in tqdm(range(0, num_samples, chunk_size), desc='Extracting latents', file=sys.stdout):
                    chunk_end = min(chunk_start + chunk_size, num_samples)
                    imgs_chunk = images_ds[chunk_start:chunk_end]

                    for b_start in range(0, len(imgs_chunk), batch_size):
                        b_end = min(b_start + batch_size, len(imgs_chunk))
                        imgs = imgs_chunk[b_start:b_end]
                        x = torch.from_numpy(imgs).to(self.device, dtype=torch.float32)
                        x = x.permute(0, 3, 1, 2).contiguous()
                        x = (x / 255.0 - 0.5) / 0.5

                        cond = self.model.cond_stage_model(x)
                        if isinstance(cond, (tuple, list)):
                            cond = cond[0]
                        cond = torch.stack(cond.chunk(latent_unit, dim=1), dim=1)
                        latents_arr[chunk_start + b_start:chunk_start + b_end] = cond.detach().cpu().numpy()

        elif self.database == 'celeba':
            from torchvision import transforms

            celeba_candidates = [
                os.path.join('data', 'celeba'),
                '/home/database/recognition/faces/celeba',
            ]
            celeba_root = next(
                (p for p in celeba_candidates if os.path.isfile(os.path.join(p, 'list_attr_celeba.txt')) and os.path.isdir(os.path.join(p, 'images'))),
                None
            )
            if celeba_root is None:
                raise FileNotFoundError(
                    " Complete CelebA not found. Ensure having 'list_attr_celeba.txt', "
                    "'list_bbox_celeba.txt' in the directory 'images/' inside 'data/celeba/'."
                )

            image_dir = os.path.join(celeba_root, 'images')
            attr_path = os.path.join(celeba_root, 'list_attr_celeba.txt')
            bbox_path = os.path.join(celeba_root, 'list_bbox_celeba.txt')

            with open(attr_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            declared_num = int(lines[0])
            filenames = [line.split()[0] for line in lines[2:]]
            if len(filenames) != declared_num:
                raise ValueError(f"Cabecera: {declared_num} imágenes, leídas: {len(filenames)}")

            with open(bbox_path, 'r') as f:
                bbox_lines = [line.strip() for line in f if line.strip()]
            declared_bbox_num = int(bbox_lines[0])
            if declared_bbox_num != declared_num:
                raise ValueError(f"El número de muestras en bbox ({declared_bbox_num}) no coincide con attr ({declared_num}).")

            bboxes = {}
            for line in bbox_lines[2:]:
                parts = line.split()
                bboxes[parts[0]] = (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))

            bbox_margin = 0.30
            image_size = 64
            resize_center = transforms.Resize(int(round(image_size * 1.2)))
            center_crop = transforms.CenterCrop(image_size)
            final_transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])

            def _load_and_crop_celeba(fname):
                img_path = os.path.join(image_dir, fname)
                img = Image.open(img_path).convert('RGB')
                try:
                    x1, y1, w, h = bboxes[fname]
                    if w <= 0 or h <= 0:
                        raise ValueError("Invalid bbox")
                    margin_x = int(round(w * bbox_margin))
                    margin_y = int(round(h * bbox_margin))
                    img_w, img_h = img.size
                    crop_x1 = max(0, x1 - margin_x)
                    crop_y1 = max(0, y1 - margin_y)
                    crop_x2 = min(img_w, x1 + w + margin_x)
                    crop_y2 = min(img_h, y1 + h + margin_y)
                    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                        raise ValueError("Invalid cropped bbox")
                    img = img.crop((crop_x1, crop_y1, crop_x2, crop_y2))
                except Exception:
                    img = center_crop(resize_center(img))
                return final_transform(img)

            print(f'Generating {self.database} representations (.npz) from checkpoint using {image_dir} (online bbox crop)...')
            num_samples = len(filenames)
            latents_arr = np.empty((num_samples, latent_unit, int(self.ndims)), dtype=np.float32)

            for b_start in tqdm(range(0, num_samples, batch_size), desc='Extracting latents (CelebA)', file=sys.stdout):
                b_end = min(b_start + batch_size, num_samples)
                batch_files = filenames[b_start:b_end]
                x = torch.stack([_load_and_crop_celeba(fn) for fn in batch_files], dim=0).to(self.device, dtype=torch.float32)

                cond = self.model.cond_stage_model(x)
                if isinstance(cond, (tuple, list)):
                    cond = cond[0]
                cond = torch.stack(cond.chunk(latent_unit, dim=1), dim=1)
                latents_arr[b_start:b_end] = cond.detach().cpu().numpy()

        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        global_step = getattr(self, 'ckpt_global_step', 0)
        np.savez(filename, latents=latents_arr, num_samples=np.array(global_step))
        print(f'Saved generated representations to: {filename} | shape={latents_arr.shape}')

    def load(self, mode):
        print('Load model representations')
        path = 'data/' + self.database + '/lat' + str(self.nlats) + '/dim' + str(self.ndims)
        os.makedirs(path, exist_ok=True)

        config_path, ckpt_path = self._resolve_config_and_ckpt(path)
        if ckpt_path is None:
            raise FileNotFoundError(f"No checkpoint (.ckpt) found in {path}")

        # 1. Load the model from .ckpt
        self._load_generative_model(config_path, ckpt_path)

        # 2. Generate the .npz from .ckpt
        default_npz = ('epoch=000000.npz' if self.nlats == 6 and self.ndims == 32 else 'epoch=000010.npz') if self.database == 'shapes3d' else self.database + '_representations.npz'
        filename = os.path.join(path, default_npz)
        if not os.path.isfile(filename) and self.database in ('shapes3d', 'celeba'):
            self._generate_representations_npz(filename)
        elif not os.path.isfile(filename):
            npzs = glob.glob(os.path.join(path, '*.npz'))
            if npzs:
                filename = npzs[0]

        # Load representations
        data = np.load(filename, allow_pickle=True)
        key = 'latents' if 'latents' in data else list(data.keys())[0]
        self.latents = data[key]

    def _try_extract_reference_image(self, obj):
        if obj is None:
            return None
        raw_img = None
        if hasattr(obj, 'images') and len(obj.images) > 0:
            img_obj = obj.images[0]
            if hasattr(img_obj, 'data') and img_obj.data is not None:
                raw_img = img_obj.data[..., ::-1].copy()
            elif hasattr(img_obj, 'filename') and img_obj.filename and os.path.isfile(str(img_obj.filename)):
                raw_img = np.asarray(Image.open(img_obj.filename).convert('RGB'))
        elif isinstance(obj, dict) and 'image' in obj and obj['image'] is not None:
            raw_img = obj['image']
        elif isinstance(obj, (str, os.PathLike)) and os.path.isfile(str(obj)):
            raw_img = np.asarray(Image.open(obj).convert('RGB'))
        elif isinstance(obj, Image.Image):
            raw_img = np.asarray(obj.convert('RGB'))
        elif isinstance(obj, np.ndarray) and obj.dtype != object and obj.ndim in (2, 3):
            raw_img = obj
        elif torch.is_tensor(obj) and obj.ndim in (3, 4):
            raw_img = obj.detach().cpu().numpy()

        if raw_img is None:
            return None

        x = torch.from_numpy(np.array(raw_img))
        if x.ndim == 3:
            if x.shape[-1] in (1, 3):
                x = x.unsqueeze(0)
            elif x.shape[0] in (1, 3):
                x = x.permute(1, 2, 0).unsqueeze(0)
        elif x.ndim == 4:
            if x.shape[1] in (1, 3) and x.shape[-1] not in (1, 3):
                x = x.permute(0, 2, 3, 1)
        
        x = x.permute(0, 3, 1, 2).contiguous().float().to(self.device)
        if x.max() > 1.0:
            x = x / 255.0
        if x.min() >= 0.0:
            x = x * 2.0 - 1.0
        if x.shape[-2:] != (64, 64):
            x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=False)
        return x

    @torch.no_grad()
    def _encode_condition(self, x_bchw):
        cond_flat = self.model.cond_stage_model(x_bchw)
        if isinstance(cond_flat, (tuple, list)):
            cond_flat = cond_flat[0]
        latent_unit = self.model.model.diffusion_model.latent_unit
        cond_split = torch.stack(cond_flat.chunk(latent_unit, dim=1), dim=1)
        return cond_flat, cond_split

    @torch.no_grad()
    def _sample_in_batches(self, cond_flat):
        outputs = []
        for start in range(0, cond_flat.shape[0], self.sample_batch_size):
            chunk = cond_flat[start:start + self.sample_batch_size]
            samples, _ = self.model.sample_log(cond=chunk, batch_size=chunk.shape[0], ddim=True, ddim_steps=self.ddim_steps, eta=self.ddim_eta)
            x_samples = self.model.decode_first_stage(samples)
            outputs.append(x_samples)
        return torch.cat(outputs, dim=0)

    def _get_output_dirs(self):
        pcr_root = os.path.dirname(os.path.dirname(os.path.abspath(inspect.getfile(Recognition))))
        pcr_images_dir = os.path.join(pcr_root, 'output', 'images')
        os.makedirs(pcr_images_dir, exist_ok=True)
        dirs = [pcr_images_dir]
        if self.outdir is not None:
            os.makedirs(self.outdir, exist_ok=True)
            if os.path.abspath(self.outdir) != os.path.abspath(pcr_images_dir):
                dirs.append(self.outdir)
        return dirs

    def process(self, ann, pred, fixed_slots=None):
        print('Process image (Conditional generation)')
        if self.model is None:
            raise RuntimeError("Generative DisDiff model checkpoint (.ckpt) not loaded.")

        set_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        slots_to_fix = self.fixed_slots if fixed_slots is None else [int(s) for s in fixed_slots]
        latent_unit = self.model.model.diffusion_model.latent_unit
        validate_fixed_slots(slots_to_fix, latent_unit)

        orig_idx = ann if isinstance(ann, (int, np.integer)) else None
        x_bchw = self._try_extract_reference_image(pred)
        if x_bchw is None:
            x_bchw = self._try_extract_reference_image(ann)

        if x_bchw is not None:
            x_orig_plot = denorm_img(x_bchw[0])
            _, c_orig_split = self._encode_condition(x_bchw)
        else:
            ref_idx = int(orig_idx) if orig_idx is not None else 176977 # Ref. image used in the fisrt subfigure of Figure 2. 
            orig_idx = ref_idx
            c_orig_split = torch.from_numpy(self.latents[ref_idx:ref_idx + 1]).float().to(self.device)
            c_orig_split = c_orig_split.reshape(1, latent_unit, self.ndims)
            with get_ema_context(self.model):
                x_ref = self._sample_in_batches(c_orig_split.reshape(1, -1))
            x_orig_plot = denorm_img(x_ref[0])

        forbidden = {int(orig_idx)} if orig_idx is not None else set()
        n_generations = max(0, (self.grid_size - 1) * self.grid_size)

        if n_generations > 0:
            rand_indices = sample_random_indices(rng, dataset_len=len(self.latents), n=n_generations, forbidden=forbidden)
            random_splits = torch.from_numpy(self.latents[rand_indices]).float().to(self.device)
            random_splits = random_splits.reshape(-1, latent_unit, self.ndims)
            mixed_cond_flat = mix_fixed_slots(orig_split=c_orig_split, random_splits=random_splits, fixed_slots=slots_to_fix)
            with get_ema_context(self.model):
                x_gen = self._sample_in_batches(mixed_cond_flat)
        else:
            x_gen = None

        fig, axes = plt.subplots(self.grid_size, self.grid_size, figsize=(2.8 * self.grid_size, 2.8 * self.grid_size), facecolor='none')
        if self.grid_size == 1:
            axes = np.array([[axes]])
        else:
            axes = np.atleast_2d(axes)

        for ax_row in axes:
            for ax in np.atleast_1d(ax_row):
                ax.axis('off')
                ax.set_facecolor('none')

        axes[0, 0].imshow(x_orig_plot)
        axes[0, 0].axis('off')

        if x_gen is not None:
            for idx in range(x_gen.shape[0]):
                row = 1 + (idx // self.grid_size)
                col = idx % self.grid_size
                if row >= self.grid_size:
                    break
                axes[row, col].imshow(denorm_img(x_gen[idx]))
                axes[row, col].axis('off')

        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        slot_str = '-'.join(str(s) for s in slots_to_fix)
        tag = f'idx{orig_idx}' if orig_idx is not None else f'sample{self._process_counter}'
        self._process_counter += 1

        for out_dir in self._get_output_dirs():
            outpath = os.path.join(
                out_dir,
                f'disdiff_{self.database}_lat{self.nlats}_dim{self.ndims}_fixed_slots_{slot_str}_{tag}.png'
            )
            plt.savefig(outpath, dpi=180, bbox_inches='tight', pad_inches=0, transparent=True)
            print(f'Saved generated grid to: {outpath}')

        plt.close(fig)
        return x_gen
    
    def _resolve_rep_dir_and_files(self, backbone, regime):
        base_dir = os.path.join('data', self.database, backbone, regime)
        os.makedirs(base_dir, exist_ok=True)
        return base_dir

    @torch.no_grad()
    def _generate_discriminative_npz(self, backbone, ckpt_path, filename, batch_size=128):
        from tqdm import tqdm
        from torchvision import transforms

        if self.device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        state_dict = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt)) if isinstance(ckpt, dict) else ckpt

        if backbone == 'resnet':
            t_names = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape', 'orientation'] if self.database == 'shapes3d' else ['eyeglasses', 'male', 'wearing_hat', 'smiling', 'mustache', 'wearing_earrings']
            n_cls = [10, 10, 10, 8, 4, 15] if self.database == 'shapes3d' else [2, 2, 2, 2, 2, 2]
            model = ResNetMultiTask(backbone_name='resnet18', pretrained=False, repr_dim=320, task_names=t_names, num_classes=n_cls).to(self.device)
        elif backbone == 'simclr':
            repr_dim = ckpt.get('repr_dim', 320) if isinstance(ckpt, dict) else 320
            proj_dim = ckpt.get('proj_dim', 128) if isinstance(ckpt, dict) else 128
            stem = ckpt.get('small_resnet_stem', True) if isinstance(ckpt, dict) else True
            model = SimCLRResNet18(repr_dim=repr_dim, proj_dim=proj_dim, hidden_dim=repr_dim, small_resnet_stem=stem).to(self.device)
        else:
            raise ValueError(f'Unsupported discriminative backbone: {backbone}')

        model.load_state_dict(state_dict)
        model.eval()

        if self.database == 'shapes3d':
            h5_candidates = [
                os.path.join('data', 'shapes3d', '3dshapes.h5'),
                os.path.expanduser('~/data/disentanglement_lib/shapes3d/3dshapes.h5'),
                '/home/jelias/data/disentanglement_lib/shapes3d/3dshapes.h5',
            ]
            h5_path = next((p for p in h5_candidates if os.path.isfile(p)), None)
            if h5_path is None:
                raise FileNotFoundError("No se encontró '3dshapes.h5'. Colócalo en 'data/shapes3d/3dshapes.h5'.")

            print(f'Generating {backbone} ({self.database}) representations (.npz) from {ckpt_path}...')
            with h5py.File(h5_path, 'r') as f:
                images_ds = f['images']
                num_samples = len(images_ds)
                feats_arr = np.empty((num_samples, 320), dtype=np.float32)
                chunk_size = batch_size * 100
                for chunk_start in tqdm(range(0, num_samples, chunk_size), desc=f'Extracting {backbone}', file=sys.stdout):
                    chunk_end = min(chunk_start + chunk_size, num_samples)
                    imgs_chunk = images_ds[chunk_start:chunk_end]
                    for b_start in range(0, len(imgs_chunk), batch_size):
                        b_end = min(b_start + batch_size, len(imgs_chunk))
                        x = torch.from_numpy(imgs_chunk[b_start:b_end]).to(self.device, dtype=torch.float32)
                        x = x.permute(0, 3, 1, 2).contiguous()
                        x = (x / 255.0 - 0.5) / 0.5
                        feats = model(x)[1] if backbone == 'resnet' else model(x, return_features=True)
                        feats_arr[chunk_start + b_start:chunk_start + b_end] = feats.detach().cpu().numpy()

        elif self.database == 'celeba':
            celeba_candidates = [os.path.join('data', 'celeba'), '/home/database/recognition/faces/celeba']
            celeba_root = next((p for p in celeba_candidates if os.path.isfile(os.path.join(p, 'list_attr_celeba.txt'))), None)
            if celeba_root is None:
                raise FileNotFoundError("No se encontró 'list_attr_celeba.txt' en 'data/celeba/'.")

            attr_path = os.path.join(celeba_root, 'list_attr_celeba.txt')
            with open(attr_path, 'r') as f:
                lines = [line.strip() for line in f if line.strip()]
            filenames = [line.split()[0] for line in lines[2:]]

            cropped_dir = os.path.join(celeba_root, 'images_cropped_6464')
            use_precrop = os.path.isdir(cropped_dir)
            image_dir = cropped_dir if use_precrop else os.path.join(celeba_root, 'images')

            bboxes = {}
            if not use_precrop:
                bbox_path = os.path.join(celeba_root, 'list_bbox_celeba.txt')
                with open(bbox_path, 'r') as f:
                    bbox_lines = [line.strip() for line in f if line.strip()]
                for line in bbox_lines[2:]:
                    parts = line.split()
                    bboxes[parts[0]] = (int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4]))

            bbox_margin = 0.30
            image_size = 64
            resize_center = transforms.Resize(int(round(image_size * 1.2)))
            center_crop = transforms.CenterCrop(image_size)
            tf_Pre = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            tf_Online = transforms.Compose([transforms.Resize((image_size, image_size)), transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            def _load_celeba_img(fname):
                img = Image.open(os.path.join(image_dir, fname)).convert('RGB')
                if use_precrop:
                    return tf_Pre(img)
                try:
                    x1, y1, w, h = bboxes[fname]
                    if w <= 0 or h <= 0:
                        raise ValueError()
                    mx, my = int(round(w * bbox_margin)), int(round(h * bbox_margin))
                    iw, ih = img.size
                    cx1, cy1, cx2, cy2 = max(0, x1 - mx), max(0, y1 - my), min(iw, x1 + w + mx), min(ih, y1 + h + my)
                    if cx2 <= cx1 or cy2 <= cy1:
                        raise ValueError()
                    img = img.crop((cx1, cy1, cx2, cy2))
                except Exception:
                    img = center_crop(resize_center(img))
                return tf_Online(img)

            print(f'Generating {backbone} (celeba) representations (.npz) from {ckpt_path} using {image_dir}...')
            num_samples = len(filenames)
            feats_arr = np.empty((num_samples, 320), dtype=np.float32)
            for b_start in tqdm(range(0, num_samples, batch_size), desc=f'Extracting {backbone} (CelebA)', file=sys.stdout):
                b_end = min(b_start + batch_size, num_samples)
                x = torch.stack([_load_celeba_img(fn) for fn in filenames[b_start:b_end]], dim=0).to(self.device, dtype=torch.float32)
                feats = model(x)[1] if backbone == 'resnet' else model(x, return_features=True)
                feats_arr[b_start:b_end] = feats.detach().cpu().numpy()

        os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        np.savez(filename, features=feats_arr)
        print(f'Saved generated representations to: {filename} | shape={feats_arr.shape}')

    def load_full_representation(self, backbone, regime):
        base_dir = self._resolve_rep_dir_and_files(backbone, regime)

        # Soporta .npz pre-extraídos por split si existen en la carpeta
        split_train_npz = os.path.join(base_dir, 'train_unbiased_representations.npz')
        split_test_npz = os.path.join(base_dir, 'test_fixed_representations.npz')
        if os.path.isfile(split_train_npz) and os.path.isfile(split_test_npz):
            d_tr = np.load(split_train_npz, allow_pickle=True)
            d_te = np.load(split_test_npz, allow_pickle=True)
            k_tr = 'features' if 'features' in d_tr else ('latents' if 'latents' in d_tr else list(d_tr.keys())[0])
            k_te = 'features' if 'features' in d_te else ('latents' if 'latents' in d_te else list(d_te.keys())[0])
            x_tr = d_tr[k_tr].reshape(d_tr[k_tr].shape[0], -1)
            x_te = d_te[k_te].reshape(d_te[k_te].shape[0], -1)
            return x_tr, x_te

        npz_path = os.path.join(base_dir, f'{backbone}_{self.database}_{regime}_representations.npz')
        if not os.path.isfile(npz_path):
            existing_npzs = glob.glob(os.path.join(base_dir, '*.npz'))
            if existing_npzs:
                npz_path = existing_npzs[0]
            elif backbone == 'disdiff':
                self.nlats, self.ndims = 10, 32
                config_path, ckpt_path = self._resolve_config_and_ckpt(base_dir)
                if config_path is None:
                    fallback_dir = os.path.join('data', self.database, 'lat10', 'dim32')
                    config_path, _ = self._resolve_config_and_ckpt(fallback_dir)
                if ckpt_path is None:
                    raise FileNotFoundError(f"No se encontró checkpoint (.ckpt) de DisDiff ({regime}) en {base_dir}")
                self._load_generative_model(config_path, ckpt_path)
                self._generate_representations_npz(npz_path)
            else:
                ckpts = glob.glob(os.path.join(base_dir, '*.pt')) + glob.glob(os.path.join(base_dir, '*.ckpt'))
                if not ckpts:
                    raise FileNotFoundError(f"No se encontró checkpoint (.pt/.ckpt) de {backbone} ({regime}) en {base_dir}")
                self._generate_discriminative_npz(backbone, ckpts[0], npz_path)

        data = np.load(npz_path, allow_pickle=True)
        key = next((k for k in ('latents', 'features', 'z', 'representations') if k in data.files), data.files[0])
        all_feats = data[key]
        all_feats = all_feats.reshape(all_feats.shape[0], -1)

        train_idx_path = os.path.join('splits', self.database, 'train_unbiased.npy')
        test_idx_path = os.path.join('splits', self.database, 'test_fixed.npy')
        if not os.path.isfile(test_idx_path):
            test_idx_path = os.path.join('splits', self.database, 'valid.npy')

        train_idx = np.load(train_idx_path).astype(np.int64)
        test_idx = np.load(test_idx_path).astype(np.int64)
        return all_feats[train_idx], all_feats[test_idx]

    def evaluate_full_representation(self, x_train_full, y_train_full, x_test_full, y_test_full, samples_per_class=25, seed=23):
        num_classes_list = [10, 10, 10, 8, 4, 15] if self.database == 'shapes3d' else [2, 2, 2, 2, 2, 2]
        factor_accs = {}
        shape_class_metrics = {}

        for j, factor_name in enumerate(self.factor_names):
            num_classes = num_classes_list[j]
            y_tr_f = y_train_full[:, j]
            y_te_f = y_test_full[:, j]

            rng = np.random.default_rng(seed)
            selected = []
            for cid in range(num_classes):
                pos = np.flatnonzero(y_tr_f == cid)
                selected.append(rng.choice(pos, size=samples_per_class, replace=False))
            selected = np.concatenate(selected)
            rng.shuffle(selected)

            scaler = StandardScaler()
            x_tr_sub = scaler.fit_transform(x_train_full[selected])
            x_te_eval = scaler.transform(x_test_full)
            y_tr_sub = y_tr_f[selected]

            clf = LinearSVC(C=1.0, random_state=seed, dual=False, max_iter=5000)
            clf.fit(x_tr_sub, y_tr_sub)
            y_pred = clf.predict(x_te_eval)

            factor_accs[self.columns[j]] = float(accuracy_score(y_te_f, y_pred)) * 100.0

            if self.database == 'shapes3d' and factor_name == 'shape':
                labels_ids = [0, 1, 2, 3]
                prec = precision_score(y_te_f, y_pred, labels=labels_ids, average=None, zero_division=0)
                rec = recall_score(y_te_f, y_pred, labels=labels_ids, average=None, zero_division=0)
                f1 = f1_score(y_te_f, y_pred, labels=labels_ids, average=None, zero_division=0)
                for cid, cname in enumerate(['Cube', 'Cylinder', 'Sphere', 'Capsule']):
                    shape_class_metrics[cname] = {
                        'Prec.': float(prec[cid]) * 100.0,
                        'Rec.': float(rec[cid]) * 100.0,
                        'F1': float(f1[cid]) * 100.0,
                    }

        return factor_accs, shape_class_metrics