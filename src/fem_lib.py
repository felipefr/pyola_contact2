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
from utils import *


def modify_FEMod(FEMod):
    FEMod.cells = FEMod.Eles.astype(np.int64)-1
    FEMod.X = FEMod.Nodes.astype(np.float64)
    FEMod.Cons = FEMod.Cons.astype(np.int64)
    del FEMod.Nodes
    del FEMod.Eles
    FEMod.SlaveSurf = FEMod.SlaveSurf.astype(np.int64)
    FEMod.MasterSurf = FEMod.MasterSurf.astype(np.int64)
    
    nMasterSurf = FEMod.MasterSurf.shape[1]
    FEMod.master_surf_cells = np.array([ GetSurfaceNode(FEMod.cells[FEMod.MasterSurf[0, i] - 1, :], 
                                                        FEMod.MasterSurf[1, i] - 1) 
                                        for i in range(nMasterSurf)], dtype = np.int64) 

    nSlaveSurf = FEMod.SlaveSurf.shape[1]
    FEMod.slave_surf_cells = np.array([ GetSurfaceNode(FEMod.cells[FEMod.SlaveSurf[0, i] - 1, :], 
                                                        FEMod.SlaveSurf[1, i] - 1) 
                                        for i in range(nSlaveSurf)], dtype = np.int64)   
    
    FEMod.master_surf_nodes = np.setdiff1d(FEMod.master_surf_cells.flatten(), [])
    
    gp = 1.0 / np.sqrt(3.0)
    FEMod.SurfIPs = np.array([ [-gp, -gp], [ gp, -gp], [ gp,  gp], [-gp,  gp]], dtype = np.float64) 
    
    FEMod.ShpfSurf = [ GetSurfaceShapeFunction(ip) for ip in FEMod.SurfIPs ]

# @numba.jit(nopython=True)
def GetStiffnessAndForce(X, cells, Disp, Residual, GKF, Dtan):    
    for IE in range(cells.shape[0]):
        Elxy = X[cells[IE, :], :]  # zero-based
        IDOF = get_dofs_given_nodes_ids(cells[IE, :])
        EleDisp = Disp[IDOF].reshape((8,3)).T # before was fortran order

        ResL, GKL = GetStiffnessAndForceElemental(Elxy, EleDisp, Dtan)
        Residual[IDOF] += ResL 
        GKF[np.ix_(IDOF, IDOF)] += GKL
                    
    return Residual, GKF


@numba.jit(nopython=True)
def GetStiffnessAndForceElemental(XL, DispL, Dtan):
    XG = np.array([-0.57735026918963, 0.57735026918963])
    WGT = np.array([1.0, 1.0])
    
    ResL = np.zeros(24, dtype = np.float64) 
    GKL = np.zeros((24,24), dtype = np.float64) 
    for LX in range(2):
        for LY in range(2):
            for LZ in range(2):
                E1, E2, E3 = XG[LX], XG[LY], XG[LZ]
                Shpd, Det = GetShapeFunction([E1, E2, E3], XL)
                FAC = WGT[LX] * WGT[LY] * WGT[LZ] * Det                
                
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
