#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 12:51:05 2025

@author: frocha
"""

from io_lib import *
import numba
numba.set_num_threads(32)
    
#sfrom oct2py import octave
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from fem_lib import *
from contact_lib import *
from utils import *
from timeit import default_timer as timer
from mesh import Mesh
from contact_pairs import ContactPairs


# # octave.addpath(octave.genpath("/home/felipe/UPEC/Bichon/codes/ContactFEA/"))  # doctest: +SKIP
# octave.addpath(octave.genpath("/home/frocha/sources/pyola_contact2/src/matlab/"))  # doctest: +SKIP

# --- Parameters ---
Tmax = 0.15
Nit = 4
NNRmax = 20
tolNR = 1e-7
FricFac = 0.0
TimeList = np.linspace(0.0, Tmax, 10)

# --- Mesh and model ---
FEMod = Mesh('Beam.inp', 
             facets_id = [('_MASTERSURF_S4',3), ('_SLAVESURF_S6',5)],
             force_bnd_id = 'SET-4',
             dirichlet_bnd_id ='CONNODE')
    

# --- Material ---
E=FEMod.Prop[0,0]; nu=FEMod.Prop[0,1];
Dtan= get_isotropic_celas(E, nu);

# --- Contact ---
# contactPairs = InitializeContactPairs(FEMod)
contactPairs = ContactPairs(FEMod, FricFac = FricFac, master_surf_id = 0, slave_surf_id = 1)

NodeNum, Dim = FEMod.X.shape
AllDOF = Dim * NodeNum

FixDOF = Dim * (FEMod.Cons[:, 0] - 1) + FEMod.Cons[:, 1] - 1 
FixDOF = FixDOF.astype(np.int64)
FreeDOF = np.setdiff1d(np.arange(AllDOF), FixDOF).astype(np.int64)

start = timer()
# --- Main loop ---
Disp=np.zeros(AllDOF, dtype = np.float64);
ExtFVect = np.zeros(AllDOF, dtype = np.float64)
Residual = np.zeros(AllDOF, dtype = np.float64)
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
        ExtFVect.fill(0.0)

        NCon = FEMod.Cons.shape[0]

        # Internal force and tangent stiffness
        # Residual, GKF = GetStiffnessAndForce(FEMod.X, FEMod.cells, Disp, Residual, GKF, Dtan)
        Residual, GKF = GetStiffnessAndForce_opt(FEMod.X, FEMod.cells, Disp, Dtan, Residual)
    
        contactPairs, GKF, Residual = CalculateContactKandF(
            FEMod, contactPairs, Dt, PreDisp, GKF, Residual, Disp)
        
        # External load boundary
        if FEMod.ExtF.shape[0] > 0:
            LOC = Dim * (FEMod.ExtF[:, 0].astype(int) - 1) + FEMod.ExtF[:, 1].astype(int) - 1  # convert to 0-based
            ExtFVect[LOC] += LoadFac * FEMod.ExtF[:, 2]
        Residual += ExtFVect

        # Displacement boundary conditions
        GKF[FixDOF, :] = 0.0
        # GKF.data[GKF.indptr[FixDOF[0]]:GKF.indptr[FixDOF[-1]+1]] = 0.0
        for i, dof in enumerate(FixDOF):
            GKF[dof, dof] = 1.0
        Residual[FixDOF] = 0.0

        if k == 0:
            Residual[FixDOF] = SDisp.flatten()
        else:
            normRes = np.linalg.norm(Residual)
            print(f"{k+1:27d} {normRes:14.5e}")

        # Check convergence
        if normRes < tolNR:
            # contactPairs.update_contact()
            contactPairs.update_history_slave_points()
            break
        
        # Newton–Raphson update
        IncreDisp = spla.spsolve(GKF.tocsr(), Residual)
        Disp += IncreDisp
        
    print("norm disp = ", np.linalg.norm(Disp))

end = timer()
print("time : ", end-start)
UM = np.linalg.norm(Disp.reshape((-1,3)), axis = 1)
# plot_structural_contours(FEMod, {'UMag' : UM}, U = Disp.reshape(-1,3))
