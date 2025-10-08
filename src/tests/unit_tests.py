#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 18:13:31 2025

@author: frocha
"""

from oct2py import octave
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from contactFEA_python import *

# octave.addpath(octave.genpath("/home/felipe/UPEC/Bichon/codes/ContactFEA/"))  # doctest: +SKIP
octave.addpath(octave.genpath("/home/felipe/sources/pyola_contact2/src/"))  # doctest: +SKIP

# ====== Test GKF (Python) and GKF2 (Matlab) ================================
# --- Parameters ---
# Tmax = 0.1
# Nit = 2
# NNRmax = 2
# tolNR = 1e-7
# TimeList = np.linspace(0.0, Tmax, Nit)

# # --- Mesh and model ---
# FEMod = octave.ModelInformation_Beam()

# # --- Material ---
# E=FEMod.Prop[0,0]; nu=FEMod.Prop[0,1];
# Dtan= octave.getIsotropicCelas(E,nu);

# # --- Contact ---
# contactPairs = octave.InitializeContactPairs(FEMod)

# NodeNum, Dim = FEMod.Nodes.shape
# AllDOF = Dim * NodeNum

# FixDOF = Dim * (FEMod.Cons[:, 0] - 1) + FEMod.Cons[:, 1]
# FixDOF = FixDOF.astype(int)
# FreeDOF = np.setdiff1d(np.arange(AllDOF), FixDOF)

# # Disp=np.zeros((AllDOF,1));
# Disp = np.random.rand(AllDOF).reshape((AllDOF,1))

# # Global stiffness and residual
# # GKF = sp.lil_matrix((AllDOF, AllDOF))
# GKF = np.zeros((AllDOF, AllDOF))

# Residual = np.zeros((AllDOF,1))
# ExtFVect = np.zeros((AllDOF,1))
# NCon = FEMod.Cons.shape[0]

# # Internal force and tangent stiffness
# Residual = Residual.flatten()
# Disp = Disp.flatten()
# Residual, GKF = GetStiffnessAndForce(FEMod.Nodes, FEMod.Eles.astype('int'), Disp, Residual, GKF, Dtan)
# Residual = Residual.reshape((-1,1)) 
# Disp = Disp.reshape((-1,1)) 

# GKF2 = GKF.copy()
# Residual2 = Residual.copy()

# # GKF = sp.lil_matrix((AllDOF, AllDOF))
# GKF = np.zeros((AllDOF, AllDOF))
# Residual.fill(0.0)

# Residual, GKF = octave.GetStiffnessAndForce(FEMod, Disp, Residual, GKF, Dtan, nout = 2)

# # print(np.allclose(GKF.todense(), GKF2.todense()))
# print(np.allclose(GKF, GKF2))

# ====== Test copy of structures ================================

def ensure_list(x):
    """Ensure consistent list of dicts, even for single structs."""
    if isinstance(x, dict):
        return [x]
    elif isinstance(x, (list, np.ndarray)):
        return list(x)
    else:
        raise TypeError(f"Unexpected type {type(x)}")


A = octave.InitializeFakeStruct(7)
A = ensure_list(A)



# B = octave.ChangeFakeStruct(np.array(A, dtype=object),8)
# B = ensure_list(B)
