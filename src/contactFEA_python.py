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
from oct2py import octave

octave.addpath(octave.genpath("/home/felipe/UPEC/Bichon/codes/ContactFEA/"))  # doctest: +SKIP

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

def ContactFEA():
    FEMod = ModelInformation_Beam()
    ContactPairs = InitializeContactPairs(FEMod)

    E, nu = FEMod["Prop"]
    Dtan = (E * (1 - nu)) / ((1 + nu) * (1 - 2 * nu)) * np.array([
        [1, nu / (1 - nu), nu / (1 - nu), 0, 0, 0],
        [nu / (1 - nu), 1, nu / (1 - nu), 0, 0, 0],
        [nu / (1 - nu), nu / (1 - nu), 1, 0, 0, 0],
        [0, 0, 0, (1 - 2 * nu) / (2 * (1 - nu)), 0, 0],
        [0, 0, 0, 0, (1 - 2 * nu) / (2 * (1 - nu)), 0],
        [0, 0, 0, 0, 0, (1 - 2 * nu) / (2 * (1 - nu))]
    ])

    # NR parameters
    Dt, MinDt, IterMax, GivenIter, MaxDt = 0.01, 1e-7, 16, 8, 0.1
    Time = 0.0

    NodeNum, Dim = FEMod["Nodes"].shape
    AllDOF = Dim * NodeNum
    Disp = np.zeros(AllDOF)

    IterOld = GivenIter + 1
    NRConvergeNum = 0
    Istep = -1
    Flag10 = 1

    while Flag10 == 1:  # Incremental loop
        Flag10 = 0
        Flag11 = 1
        Flag20 = 1
        DispSave = Disp.copy()
        tempContactPairs = ContactPairs.copy()
        Time0 = Time
        Istep += 1
        Time += Dt

        while Flag11 == 1:  # Reduction loop
            NRConvergeNum += 1
            Flag11 = 0

            if Time - 1 > 1e-10:  # Completed
                if 1 + Dt - Time > 1e-10:
                    Dt = 1 + Dt - Time
                    Time = 1
                else:
                    break

            Factor = Time
            SDisp = Dt * FEMod["Cons"][:, 2]
            Iter = 0
            PreDisp = Disp.copy()

            while Flag20 == 1:  # Newton–Raphson loop
                Flag20 = 0
                Iter += 1
                Residual = np.zeros(AllDOF)
                GKF = sp.lil_matrix((AllDOF, AllDOF))
                ExtFVect = np.zeros(AllDOF)
                NCon = FEMod["Cons"].shape[0]

                Residual, GKF = GetStiffnessAndForce(FEMod, Disp, Residual, GKF, Dtan)
                # Contact handling:
                ContactPairs, GKF, Residual = DetermineContactState(
                    FEMod, ContactPairs, Dt, PreDisp, GKF, Residual, Disp
                )

                # External forces
                if FEMod["ExtF"].shape[0] > 0:
                    LOC = Dim * (FEMod["ExtF"][:, 0].astype(int) - 1) + FEMod["ExtF"][:, 1].astype(int)
                    ExtFVect[LOC] += Factor * FEMod["ExtF"][:, 2]
                Residual += ExtFVect

                # Displacement BC
                if NCon != 0:
                    FixDOF = Dim * (FEMod["Cons"][:, 0].astype(int) - 1) + FEMod["Cons"][:, 1].astype(int)
                    GKF[FixDOF, :] = 0
                    for i, dof in enumerate(FixDOF):
                        GKF[dof, dof] = 1
                        Residual[dof] = 0
                        if Iter == 1:
                            Residual[dof] = SDisp[i]

                # Convergence check
                if Iter > 1:
                    FixDOF = Dim * (FEMod["Cons"][:, 0].astype(int) - 1) + FEMod["Cons"][:, 1].astype(int)
                    FreeDOF = np.setdiff1d(np.arange(AllDOF), FixDOF)
                    Resid = np.max(np.abs(Residual[FreeDOF]))

                    if Iter == 2:
                        print("\nTime   Time step   Iter   Residual")
                    print(f"{Time:10.5f} {Dt:10.3e} {Iter:5d} {Resid:14.5e}")

                    if Resid < 1e-7:  # Converged
                        # Update contact pairs
                        for cp in ContactPairs:
                            if cp["CurContactState"] == 0:
                                cp.update({
                                    "PreMasterSurf": np.zeros(2),
                                    "rp": 0, "sp": 0,
                                    "PreContactState": 0,
                                    "Pre_g": 0,
                                    "Pressure": 0,
                                    "Traction": 0
                                })
                            else:
                                cp["PreMasterSurf"] = cp["CurMasterSurf"].copy()
                                cp["rp"] = cp["rc"]
                                cp["sp"] = cp["sc"]
                                cp["PreContactState"] = cp["CurContactState"]
                                cp["Pre_g"] = cp["Cur_g"]

                            cp["rc"] = 0
                            cp["sc"] = 0
                            cp["Cur_g"] = 0
                            cp["CurMasterSurf"] = np.zeros(2)
                            cp["CurContactState"] = 0

                        if NRConvergeNum > 1 and Iter < GivenIter and IterOld < GivenIter:
                            Dt = min(1.5 * Dt, MaxDt)
                        IterOld = Iter
                        Flag10 = 1
                        break

                    if Iter + 1 > IterMax:
                        Dt *= 0.25
                        Time = Time0 + Dt
                        if Dt < MinDt:
                            raise RuntimeError("Incremental step too small")
                        Disp = DispSave.copy()
                        ContactPairs = tempContactPairs.copy()
                        print("Not converged. Reducing load increment.")
                        NRConvergeNum = 0
                        Flag11 = 1
                        Flag20 = 1
                        break

                # Solve system
                IncreDisp = spla.spsolve(GKF.tocsr(), Residual)
                Disp += IncreDisp
                Flag20 = 1

    # TODO: plotting routines and stress recovery here






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