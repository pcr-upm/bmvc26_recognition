#!/usr/bin/python
# -*- coding: UTF-8 -*-
__author__ = 'Jorge Elías'
__email__ = 'jorge.elias@alumnos.upm.es'

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')))
import numpy as np
from pcr_framework.src.recognition import Recognition
np.random.seed(42)


class BMVC26Recognition(Recognition):
    """
    On Structured Disentangled Representation and the Limits of Global Scores
    """
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.model = None
        self.gpu = None
        self.ckpt = None

    def parse_options(self, params):
        super().parse_options(params)
        import argparse
        parser = argparse.ArgumentParser(prog='BMVC26Recognition', add_help=False)
        parser.add_argument('--gpu', dest='gpu', type=int, default=-1,
                            help='GPU ID (negative value indicates CPU).')
        parser.add_argument('--ckpt', type=str, default=None, 
                            help='The checkpoint path.')
        args, unknown = parser.parse_known_args(params)
        print(parser.format_usage())
        self.gpu = args.gpu
        self.ckpt = args.ckpt

    def train(self, anns_train, anns_valid):
        print('Train model')

    def load(self, mode):
        print('Load model')

    def process(self, ann, pred):
        print('Process image')
