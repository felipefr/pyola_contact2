#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 17:12:15 2025

@author: frocha
"""

from oct2py import octave
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from contactFEA_python import *

# octave.addpath(octave.genpath("/home/felipe/UPEC/Bichon/codes/ContactFEA/"))  # doctest: +SKIP
octave.addpath(octave.genpath("/home/felipe/sources/pyola_contact2/src/"))  # doctest: +SKIP

# --- Parameters ---
Tmax = 0.1
Nit = 3
NNRmax = 20
tolNR = 1e-7
TimeList = np.linspace(0.0, Tmax, 10)

# --- Mesh and model ---
FEMod = octave.ModelInformation_Beam()

# --- Material ---
E=FEMod.Prop[0,0]; nu=FEMod.Prop[0,1];
Dtan= octave.getIsotropicCelas(E,nu);

# --- Contact ---
contactPairs = octave.InitializeContactPairs(FEMod, nout = 1)
# contactPairs = ensure_list(contactPairs) # expected to work , but useless

NodeNum, Dim = FEMod.Nodes.shape
AllDOF = Dim * NodeNum

FixDOF = Dim * (FEMod.Cons[:, 0] - 1) + FEMod.Cons[:, 1] - 1 
FixDOF = FixDOF.astype(int)
FreeDOF = np.setdiff1d(np.arange(AllDOF), FixDOF)

Disp=np.zeros((AllDOF,1), order = 'F');

# --- Main loop ---
for i in range(Nit - 1):
    Time = TimeList[i + 1]
    Dt = TimeList[i + 1] - TimeList[i]
    LoadFac = Time
    SDisp = Dt * FEMod.Cons[:, 2]
    PreDisp = Disp.copy()
    normRes = 9999.9

    print(f"\n\tTime = {Time:10.5f}")

    for k in range(NNRmax):
        # Global stiffness and residual
        GKF = sp.lil_matrix((AllDOF, AllDOF))
        # GKF = np.zeros((AllDOF, AllDOF))
        
        Residual = np.zeros((AllDOF,1), order = 'F')
        ExtFVect = np.zeros((AllDOF,1), order = 'F')
        NCon = FEMod.Cons.shape[0]

        # Internal force and tangent stiffness
        Residual = Residual.flatten()
        Disp = Disp.flatten()
        Residual, GKF = GetStiffnessAndForce(FEMod.Nodes, FEMod.Eles.astype('int'), Disp, Residual, GKF, Dtan)
        Residual = Residual.reshape((-1,1), order = 'F') 
        Disp = Disp.reshape((-1,1), order = 'F') 
    
        # Contact update
        contactPairs, GKF, Residual = octave.DetermineContactState(
            FEMod, contactPairs, Dt, PreDisp, GKF, Residual, Disp, nout = 3)

        # External load boundary
        if FEMod.ExtF.shape[0] > 0:
            LOC = Dim * (FEMod.ExtF[:, 0].astype(int) - 1) + FEMod.ExtF[:, 1].astype(int) - 1  # convert to 0-based
            ExtFVect[LOC,0] += LoadFac * FEMod.ExtF[:, 2]
        Residual[:,0] += ExtFVect[:,0]

        # Displacement boundary conditions
        GKF[FixDOF, :] = 0.0
        for i, dof in enumerate(FixDOF):
            GKF[dof, dof] = 1.0
        Residual[FixDOF, 0] = 0.0

        if k == 0:
            Residual[FixDOF, 0] = SDisp.flatten()
        else:
            normRes = np.linalg.norm(Residual)
            print(f"{k+1:27d} {normRes:14.5e}")

        # Check convergence
        if normRes < tolNR:
            contactPairs = octave.updateContact(contactPairs, nout = 1)
            break

        # Newton–Raphson update
        IncreDisp = spla.spsolve(GKF.tocsr(), Residual)
        Disp[:,0] += IncreDisp.flatten()
        
    print("norm disp = ", np.linalg.norm(Disp))

UM = np.linalg.norm(Disp.reshape((-1,3)), axis = 1)
octave.PlotStructuralContours(FEMod.Nodes,FEMod.Eles,Disp,UM.reshape((-1,1)))

