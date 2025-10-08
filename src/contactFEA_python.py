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
# from oct2py import octave
# octave.addpath(octave.genpath("/home/felipe/UPEC/Bichon/codes/ContactFEA/"))  # doctest: +SKIP

def ten2voigt(T, kind='strain'):
    f = 2.0 if kind == 'strain' else 1.0
    return np.array([T[0,0], T[1,1], T[2,2],
                     f*T[0,1], f*T[1,2], f*T[0,2]])

def voigt2ten(v, kind='strain'):
    f = 0.5 if kind == 'strain' else 1.0
    T = np.array([[v[0], f*v[3], f*v[5]],
                  [f*v[3], v[1], f*v[4]],
                  [f*v[5], f*v[4], v[2]]])
    return T

def GetStiffnessAndForce(Nodes, Eles, Disp, Residual, GKF, Dtan):
    XG = np.array([-0.57735026918963, 0.57735026918963])
    WGT = np.array([1.0, 1.0])

    for IE in range(Eles.shape[0]):
        Elxy = Nodes[Eles[IE, :] - 1, :]  # zero-based
        IDOF = np.zeros(24, dtype=int)
        for I in range(8):
            II = 3 * I # zero-based
            IDOF[II:II+3] = np.arange(3 * (Eles[IE, I] - 1),
                                      3 * (Eles[IE, I] - 1) + 3)
        
        EleDisp = Disp[IDOF].reshape(3, 8)
        
        for LX in range(2):
            for LY in range(2):
                for LZ in range(2):
                    E1, E2, E3 = XG[LX], XG[LY], XG[LZ]
                    Shpd, Det = GetShapeFunction([E1, E2, E3], Elxy)
                    FAC = WGT[LX] * WGT[LY] * WGT[LZ] * Det
                    
                    F = EleDisp @ Shpd.T + np.eye(3)
                    Strain = 0.5 * (F.T @ F - np.eye(3))
                    StrainVoigt = ten2voigt(Strain, 'strain')
                    StressVoigt = Dtan @ StrainVoigt
                    BN, BG = getBmatrices(Shpd, F)
                
                    # Assemble internal force vector
                    Residual[IDOF] -= FAC * (BN.T @ StressVoigt)
                    
                    # Convert stress to tensor form
                    Stress = voigt2ten(StressVoigt, 'stress')
                    
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