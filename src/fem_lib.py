#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 19:04:33 2025

@author: frocha
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import numba as nb
from numba import njit, float64, int64
from utils import *


def GetStiffnessAndForce(X, cells, Disp, Residual, GKF, Dtan):    
    for IE in range(cells.shape[0]):
        Elxy = X[cells[IE, :], :]  # zero-based
        IDOF = get_dofs_given_nodes_ids(cells[IE, :])
        EleDisp = Disp[IDOF].reshape((8,3)).T # before was fortran order

        ResL, GKL = GetStiffnessAndForceElemental(Elxy, EleDisp, Dtan)
        Residual[IDOF] += ResL 
        GKF[np.ix_(IDOF, IDOF)] += GKL
                    
    return Residual, GKF


# def GetStiffnessAndForce_opt(X, cells, Disp, Dtan, rhs):
#     rhs.fill(0.0)
#     rows, cols, vals, rhs = assemble_triplets(X, cells, Disp, Dtan, rhs)
#     # print(rows)
#     K = sp.coo_matrix((vals, (rows, cols)), shape=(rhs.size, rhs.size)).tolil()
    
#     return rhs, K


def GetStiffnessAndForce_opt2(FEMod, Disp, Dtan, rhs):
    rhs.fill(0.0)
    rows, cols, vals, rhs = assemble_triplets2(FEMod.X, FEMod.cells, FEMod.DSF, FEMod.WIP, Dtan, Disp, rhs)
    K = sp.coo_matrix((vals, (rows, cols)), shape=(rhs.size, rhs.size)).tolil()
    
    return rhs, K

# @njit(cache=True)
# def GetStiffnessAndForceElemental(XL, DispL, Dtan):
#     XG = np.array([-0.57735026918963, 0.57735026918963])
#     WGT = np.array([1.0, 1.0])
    
#     ResL = np.zeros(24, dtype = np.float64) 
#     GKL = np.zeros((24,24), dtype = np.float64) 
#     for LX in range(2):
#         for LY in range(2):
#             for LZ in range(2):
#                 E1, E2, E3 = XG[LX], XG[LY], XG[LZ]
#                 Shpd, Det = GetShapeFunction([E1, E2, E3], XL)
#                 FAC = WGT[LX] * WGT[LY] * WGT[LZ] * Det                
                
#                 F = DispL @ Shpd.T + np.eye(3)
                
#                 Strain = 0.5 * (F.T @ F - np.eye(3))
                
#                 StrainVoigt = ten2voigt(Strain, 2.) # fac 2 to strain
#                 StressVoigt = Dtan @ StrainVoigt
#                 BN, BG = getBmatrices(Shpd, F)
            
#                 # Assemble internal force vector
#                 ResL -= FAC * (BN.T @ StressVoigt)
                
#                 Stress = voigt2ten(StressVoigt, 1.) # fac 1 to strain
                
#                 # Build SHEAD (block diagonal stress tensor)
#                 SHEAD = np.zeros((9, 9))
#                 SHEAD[0:3, 0:3] = Stress
#                 SHEAD[3:6, 3:6] = Stress
#                 SHEAD[6:9, 6:9] = Stress
                
#                 # Element stiffness matrix
#                 GKL += FAC * (BN.T @ Dtan @ BN + BG.T @ SHEAD @ BG)

#     return ResL, GKL


@njit(cache=True)
def GetStiffnessAndForceElemental2(DSFIP, WIP, XL, DispL, Dtan):
    ResL = np.zeros(24, dtype = np.float64) 
    GKL = np.zeros((24,24), dtype = np.float64)
    nip = len(WIP)
    for ip in range(nip):
        Shpd, Det = get_global_shape_derivative(DSFIP[ip], XL)
        FAC = WIP[ip] * Det                
        
        F = DispL @ Shpd.T + np.eye(3)
        
        Strain = 0.5 * (F.T @ F - np.eye(3))
        
        StrainVoigt = ten2voigt(Strain, 2.) # fac 2 to strain
        StressVoigt = Dtan @ StrainVoigt
        BN, BG = getBmatrices(Shpd, F)
    
        # Assemble internal force vector
        ResL -= FAC * (BN.T @ StressVoigt)
        
        Stress = voigt2ten(StressVoigt, 1.) # fac 1 to strain
        
        # Build SHEAD (block diagonal stress tensor)
        SHEAD = np.zeros((9, 9))
        SHEAD[0:3, 0:3] = Stress
        SHEAD[3:6, 3:6] = Stress
        SHEAD[6:9, 6:9] = Stress
        
        # Element stiffness matrix
        GKL += FAC * (BN.T @ Dtan @ BN + BG.T @ SHEAD @ BG)

    return ResL, GKL



@njit(cache=True)
def getBmatrices(Shpd, F):
    """
    Compute BN (strain-displacement) and BG (geometric) matrices
    for an 8-node hexahedral element.

    Parameters
    ----------
    Shpd : (3, 8) array
        Derivatives of shape functions w.r.t local coordinates.
        Shpd = [dN/dX; dN/dY; dN/dZ]
    F : (3, 3) array
        Deformation gradient.

    Returns
    -------
    BN : (6, 24) array
        Strain-displacement matrix for material stiffness term.
    BG : (9, 24) array
        Geometric stiffness matrix term.
    """
    BN = np.zeros((6, 24))
    BG = np.zeros((9, 24))

    for I in range(8):
        col = slice(I * 3, I * 3 + 3)

        # Compute BN block (6x3)
        BN[:, col] = np.array([
            [Shpd[0, I] * F[0, 0], Shpd[0, I] * F[1, 0], Shpd[0, I] * F[2, 0]],
            [Shpd[1, I] * F[0, 1], Shpd[1, I] * F[1, 1], Shpd[1, I] * F[2, 1]],
            [Shpd[2, I] * F[0, 2], Shpd[2, I] * F[1, 2], Shpd[2, I] * F[2, 2]],
            [Shpd[0, I] * F[0, 1] + Shpd[1, I] * F[0, 0],
              Shpd[0, I] * F[1, 1] + Shpd[1, I] * F[1, 0],
              Shpd[0, I] * F[2, 1] + Shpd[1, I] * F[2, 0]],
            [Shpd[1, I] * F[0, 2] + Shpd[2, I] * F[0, 1],
              Shpd[1, I] * F[1, 2] + Shpd[2, I] * F[1, 1],
              Shpd[1, I] * F[2, 2] + Shpd[2, I] * F[2, 1]],
            [Shpd[0, I] * F[0, 2] + Shpd[2, I] * F[0, 0],
              Shpd[0, I] * F[1, 2] + Shpd[2, I] * F[1, 0],
              Shpd[0, I] * F[2, 2] + Shpd[2, I] * F[2, 0]]
        ])

        # Compute BG block (9x3)
        BG[:, col] = np.array([
            [Shpd[0, I], 0, 0],
            [Shpd[1, I], 0, 0],
            [Shpd[2, I], 0, 0],
            [0, Shpd[0, I], 0],
            [0, Shpd[1, I], 0],
            [0, Shpd[2, I], 0],
            [0, 0, Shpd[0, I]],
            [0, 0, Shpd[1, I]],
            [0, 0, Shpd[2, I]]
        ])

    return BN, BG

@njit(cache=True)
def GetShapeFunction(XI, Elxy):
    XNode = np.array([
        [-1, 1, 1, -1, -1, 1, 1, -1],
        [-1, -1, 1, 1, -1, -1, 1, 1],
        [-1, -1, -1, -1, 1, 1, 1, 1]
    ])
    DSF = np.zeros((3, 8))
    for I in range(8):
        XP, YP, ZP = XNode[:, I]
        XI0 = [1 + XI[0]*XP, 1 + XI[1]*YP, 1 + XI[2]*ZP]
        DSF[0, I] = 0.125 * XP * XI0[1] * XI0[2]
        DSF[1, I] = 0.125 * YP * XI0[0] * XI0[2]
        DSF[2, I] = 0.125 * ZP * XI0[0] * XI0[1]

    GJ = DSF @ Elxy
    Det = np.linalg.det(GJ)
    ShpD = np.linalg.inv(GJ) @ DSF
    return ShpD, Det

# @njit(cache=True)
def get_global_shape_derivative(DSF, XL):
    Jac = DSF @ XL
    Det = np.linalg.det(Jac)
    GDSF = np.linalg.inv(Jac) @ DSF
    return GDSF, Det
    

# @njit(cache=True)
def get_local_shape_derivative(XI):
    XNode = np.array([
        [-1, 1, 1, -1, -1, 1, 1, -1],
        [-1, -1, 1, 1, -1, -1, 1, 1],
        [-1, -1, -1, -1, 1, 1, 1, 1]
    ])
    DSF = np.zeros((3, 8))
    for I in range(8):
        XP, YP, ZP = XNode[:, I]
        XI0 = [1 + XI[0]*XP, 1 + XI[1]*YP, 1 + XI[2]*ZP]
        DSF[0, I] = 0.125 * XP * XI0[1] * XI0[2]
        DSF[1, I] = 0.125 * YP * XI0[0] * XI0[2]
        DSF[2, I] = 0.125 * ZP * XI0[0] * XI0[1]

    return DSF


# sig = "Tuple((int64[:], int64[:], float64[:], float64[:]))(float64[:,:], int64[:,:], float64[:], float64[:,:], float64[:])"
# @njit([sig], parallel = True, cache = True)
# def assemble_triplets(X, cells, Disp, Dtan, rhs):
#     n_threads = nb.get_num_threads()
#     n_elem = cells.shape[0]
#     nnpe = cells.shape[1]*3
#     triplet_size = nnpe * nnpe
#     rows = np.empty(n_elem * triplet_size, dtype=np.int64)
#     cols = np.empty(n_elem * triplet_size, dtype=np.int64)
#     vals = np.empty(n_elem * triplet_size, dtype=np.float64)
#     rhs_local = np.zeros((n_threads, Disp.shape[0]), dtype=np.float64)

#     for e in nb.prange(n_elem):
#         thread_id = nb.get_thread_id() 
#         Elxy = X[cells[e, :], :]
#         IDOF = get_dofs_given_nodes_ids(cells[e, :])
#         EleDisp = Disp[IDOF].reshape((8,3)).T
#         ResL, GKL = GetStiffnessAndForceElemental(Elxy, EleDisp, Dtan)
#         base = e * triplet_size
#         k = 0
#         for a in range(nnpe):
#             rhs_local[thread_id, IDOF[a]] += ResL[a]
#             for b in range(nnpe):
#                 rows[base + k] = IDOF[a]
#                 cols[base + k] = IDOF[b]
#                 vals[base + k] = GKL[a, b]
#                 k += 1
                
        
#     for t in range(n_threads):
#         rhs += rhs_local[t]

#     return rows, cols, vals, rhs


# sig = "Tuple((int64[:], int64[:], float64[:], float64[:]))(float64[:,:], int64[:,:], float64[:,:], float64[:]," 
# sig += "float64[:,:], float64[:], float64[:])"
# @njit([sig], parallel = True, cache = True)
def assemble_triplets2(X, cells, DSFIP, WIP, Dtan, Disp, rhs):
    n_threads = nb.get_num_threads()
    n_elem = cells.shape[0]
    nnpe = cells.shape[1]*3
    triplet_size = nnpe * nnpe
    rows = np.empty(n_elem * triplet_size, dtype=np.int64)
    cols = np.empty(n_elem * triplet_size, dtype=np.int64)
    vals = np.empty(n_elem * triplet_size, dtype=np.float64)
    rhs_local = np.zeros((n_threads, Disp.shape[0]), dtype=np.float64)

    # for e in nb.prange(n_elem):
    for e in range(n_elem):
        thread_id = nb.get_thread_id() 
        Elxy = X[cells[e, :], :]
        IDOF = get_dofs_given_nodes_ids(cells[e, :])
        EleDisp = Disp[IDOF].reshape((8,3)).T
        ResL, GKL = GetStiffnessAndForceElemental2(DSFIP, WIP, Elxy, EleDisp, Dtan)
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
