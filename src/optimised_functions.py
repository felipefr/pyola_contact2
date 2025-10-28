#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 13:48:41 2025

@author: frocha
"""

import numba as nb
import numpy as np
from utils import *
from scipy.spatial import cKDTree
import scipy.sparse as sp
# from sklearn.neighbors import KDTree
import copy
from fem_lib import GetStiffnessAndForceElemental


def GetStiffnessAndForce(X, cells, Disp, Dtan, rhs):
    rhs.fill(0.0)
    rows, cols, vals, rhs = assemble_triplets(X, cells, Disp, Dtan, rhs)
    # print(rows)
    K = sp.coo_matrix((vals, (rows, cols)), shape=(rhs.size, rhs.size)).tolil()
    
    return rhs, K

# @nb.jit(parallel = True)
def assemble_triplets(X, cells, Disp, Dtan, rhs):
    n_threads = nb.get_num_threads()
    n_elem = cells.shape[0]
    nnpe = cells.shape[1]*3
    triplet_size = nnpe * nnpe
    rows = np.empty(n_elem * triplet_size, dtype=np.int64)
    cols = np.empty(n_elem * triplet_size, dtype=np.int64)
    vals = np.empty(n_elem * triplet_size, dtype=np.float64)
    rhs_local = np.zeros((n_threads, Disp.shape[0]), dtype=np.float64)

    for e in nb.prange(n_elem):
        thread_id = nb.get_thread_id() 
        Elxy = X[cells[e, :], :]
        IDOF = get_dofs_given_nodes_ids(cells[e, :])
        EleDisp = Disp[IDOF].reshape((8,3)).T
        ResL, GKL = GetStiffnessAndForceElemental(Elxy, EleDisp, Dtan)
        base = e * triplet_size
        k = 0
        for a in range(nnpe):
            rhs_local[thread_id, IDOF[a]] += ResL[a]
            for b in range(nnpe):
                rows[base + k] = IDOF[a]
                cols[base + k] = IDOF[b]
                vals[base + k] = GKL[a, b]
                k += 1
                
        
    for t in range(n_threads):
        rhs += rhs_local[t]

    return rows, cols, vals, rhs