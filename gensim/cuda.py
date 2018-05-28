#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2011 Radim Rehurek <radimrehurek@seznam.cz>
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html

import numpy as np
from contextlib import contextmanager

try:
    import cupy as cp

    def get_array_module(x):
        return cp.get_array_module(x)

    def is_cupy(x):
        return get_array_module(x) == cp

    def asnumpy(x):
        return cp.asnumpy(x)

    cupy_gammaln = cp.core.create_ufunc(
        'gammaln', ('e->e', 'f->f', 'd->d'),
        'out0 = lgamma(in0)')

    cupy_digamma = cp.ElementwiseKernel(
        'T x',
        'T y',
        '''
        const T c = 8.5;
        const T euler_mascheroni = 0.57721566490153286060;
        T r;
        T value;
        T x2;
    
        if (x <= 0.000001) {
            value = - euler_mascheroni - 1.0 / x + 1.6449340668482264365 * x;
        } else {
            value = 0.0;
            x2 = x;
            while (x2 < c) {
                value = value - 1.0 / x2;
                x2 = x2 + 1.0;
            }
            // Use Stirling's (actually de Moivre's) expansion.
            r = 1.0 / x2;
            value = value + log(x2) - 0.5 * r;
            r = r * r;
            value = value 
                - r * (1.0 / 12.0 
                       - r * (1.0 / 120.0 
                              - r * (1.0 / 252.0 
                                     - r * (1.0 / 240.0 
                                            - r * (1.0 / 132.0)))));
        }
        y = value
        ''',
        'digamma')

#def atomicAdd(x, y):
#    """x += y as an atomic operation"""

    #atomicAdd = cp.core.create_ufunc(
    #    'atomicAdd', ('e->e', 'f->f', 'd->d', 'F->F', 'D->D'),
    #    'atomicAdd(&oaut0, in0)')

    def atomicAdd(x, y):
        """x += y as an atomic operation"""
        cp.ElementwiseKernel(
            'S in, raw S out',
            '',
            '''
            atomicAdd(&out[i], in);
            ''',
            'atomicAdd')(y, x)
        return x

    @contextmanager
    def prof_range(label):
        try:
            cp.cuda.nvtx.RangePush(label)
            yield
        finally:
            cp.cuda.nvtx.RangePop()


except ImportError:

    def get_array_module(x):
        return np

    def is_cupy(x):
        return False

    def asnumpy(x):
        return x

    @contextmanager
    def prof_range(label):
        yield
