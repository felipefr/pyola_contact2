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
import numba
# from oct2py import octave
# octave.addpath(octave.genpath("/home/felipe/UPEC/Bichon/codes/ContactFEA/"))  # doctest: +SKIP


@numba.jit(nopython=True)
def ten2voigt(T, fac):
    return np.array([T[0,0], T[1,1], T[2,2],
                     fac*T[0,1], fac*T[1,2], fac*T[0,2]], dtype = np.float64)

@numba.jit(nopython=True)
def voigt2ten(v, fac):
    T = np.array([[v[0], v[3]/fac, v[5]/fac],
                  [v[3]/fac, v[1], v[4]/fac],
                  [v[5]/fac, v[4]/fac, v[2]]], dtype = np.float64)
    return T



@numba.jit(nopython=True)
def get_dofs_given_nodes_ids(nodes_ids):
    DOFs = np.empty(nodes_ids.shape[0] * 3, dtype=np.int64)
    for m, node in enumerate(nodes_ids):
        DOFs[3*m:3*m+3] = np.arange(3*node, 3*(node+1)) # python convention

    return DOFs

# @numba.jit(nopython=True)
def GetStiffnessAndForce(Nodes, Eles, Disp, Residual, GKF, Dtan):
    XG = np.array([-0.57735026918963, 0.57735026918963])
    WGT = np.array([1.0, 1.0])

    
    for IE in range(Eles.shape[0]):
        Elxy = Nodes[Eles[IE, :], :]  # zero-based
        
        IDOF = get_dofs_given_nodes_ids(Eles[IE, :])
        EleDisp = Disp[IDOF].reshape((8,3)).T # before was fortran order
        for LX in range(2):
            for LY in range(2):
                for LZ in range(2):
                    E1, E2, E3 = XG[LX], XG[LY], XG[LZ]
                    Shpd, Det = GetShapeFunction([E1, E2, E3], Elxy)
                    FAC = WGT[LX] * WGT[LY] * WGT[LZ] * Det                
                    
                    F = EleDisp @ Shpd.T + np.eye(3)
                    
                    Strain = 0.5 * (F.T @ F - np.eye(3))
                    
                    # ten2voigt
                    # f = 2.0 if kind == 'strain' else 1.0
                    StrainVoigt = ten2voigt(Strain, 2.)
                    StressVoigt = Dtan @ StrainVoigt
                    BN, BG = getBmatrices(Shpd, F)
                
                    # Assemble internal force vector
                    Residual[IDOF] -= FAC * (BN.T @ StressVoigt)
                    
                    # Convert stress to tensor form
                    
                    # # ten2voigt
                    # f = 2.0 if kind == 'strain' else 1.0
                    Stress = voigt2ten(StressVoigt, 1.)
                    
                    # Build SHEAD (block diagonal stress tensor)
                    SHEAD = np.zeros((9, 9))
                    SHEAD[0:3, 0:3] = Stress
                    SHEAD[3:6, 3:6] = Stress
                    SHEAD[6:9, 6:9] = Stress
                    
                    # Element stiffness matrix
                    EKF = BN.T @ Dtan @ BN + BG.T @ SHEAD @ BG

                    # Assemble global tangent stiffness matrix
                    GKF[np.ix_(IDOF, IDOF)] += FAC * EKF
                    
    return Residual, GKF

@numba.jit(nopython=True)
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

@numba.jit(nopython=True)
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


# === Placeholders for missing routines ===
def ModelInformation_Beam():
    return {
        "Prop": np.array([210e9, 0.3]),
        "Nodes": np.zeros((8, 3)),
        "Eles": np.zeros((1, 8), dtype=int),
        "Cons": np.zeros((0, 3)),
        "ExtF": np.zeros((0, 3)),
        "SlaveSurf": np.zeros((3, 0)),
        "MasterSurf": np.zeros((2, 0)),
        "FricFac": 0.0
    }

def InitializeContactPairs(FEMod):
    return []

def DetermineContactState(FEMod, ContactPairs, Dt, PreDisp, GKF, Residual, Disp):
    return ContactPairs, GKF, Residual