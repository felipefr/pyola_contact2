#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 29 14:43:27 2025

@author: felipe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  8 17:12:15 2025

@author: frocha
"""

import numba
numba.set_num_threads(8)
    
from oct2py import octave
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from fem_lib import *
from contact_lib import *
from friction_contact_lib import *
from utils import *
from timeit import default_timer as timer



# octave.addpath(octave.genpath("/home/felipe/UPEC/Bichon/codes/ContactFEA/"))  # doctest: +SKIP
octave.addpath(octave.genpath("/home/frocha/sources/pyola_contact2/src/matlab/"))  # doctest: +SKIP

# --- Parameters ---
Tmax = 0.15
Nit = 4
NNRmax = 20
tolNR = 1e-7
TimeList = np.linspace(0.0, Tmax, 10)

# --- Mesh and model ---
FEMod = octave.ModelInformation_Beam()
modify_FEMod(FEMod)

FEMod.FricFac = 0.1

# --- Material ---
E=FEMod.Prop[0,0]; nu=FEMod.Prop[0,1];
Dtan= get_isotropic_celas(E, nu);

# --- Contact ---
contactPairs = InitializeContactPairs(FEMod)

NodeNum, Dim = FEMod.X.shape
AllDOF = Dim * NodeNum

FixDOF = Dim * (FEMod.Cons[:, 0] - 1) + FEMod.Cons[:, 1] - 1 
FreeDOF = np.setdiff1d(np.arange(AllDOF), FixDOF)

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
    
        contactPairs, GKF, Residual = DetermineContactState(
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
            # contactPairs = octave.updateContact(contactPairs, nout = 1)
            updateContact(contactPairs)
            break
        
        # Newton–Raphson update
        IncreDisp = spla.spsolve(GKF.tocsr(), Residual)
        Disp += IncreDisp
        
    print("norm disp = ", np.linalg.norm(Disp))

end = timer()
print("time : ", end-start)
print("Using", numba.get_num_threads(), "threads")
# UM = np.linalg.norm(Disp.reshape((-1,3)), axis = 1)
# octave.PlotStructuralContours(FEMod.X,FEMod.cells,Disp,UM.reshape((-1,1)))

