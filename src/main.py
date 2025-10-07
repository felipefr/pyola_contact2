#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 10:59:53 2025

@author: frocha
"""

from oct2py import octave
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from contactFEA_python import *

octave.addpath(octave.genpath("/home/felipe/UPEC/Bichon/codes/ContactFEA/"))  # doctest: +SKIP


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


# octave.ContactFEA_refac()

FEMod = octave.ModelInformation_Beam()
Dt=0.01; MinDt=1.0E-7; IterMax=16; GivenIter=8; MaxDt=0.1; Time = 0; # N-R parameters

# Material
E=FEMod.Prop[0,0]; nu=FEMod.Prop[0,1];
Dtan= octave.getIsotropicCelas(E,nu);

contactPairs=octave.InitializeContactPairs(FEMod);

NodeNum, Dim = np.shape(FEMod.Nodes); 
AllDOF = Dim*NodeNum;
Disp=np.zeros((AllDOF,1));
# Disp=np.zeros(AllDOF);

IterOld=GivenIter+1; NRConvergeNum=0; Istep = -1; Flag10 = 1;

op = "pytho"

count = 0
while count<1:  # Incremental loop
    count += 1
# while Flag10 == 1:  # Incremental loop
    Flag10 = 0
    Flag11 = 1
    Flag20 = 1
    ReductionNumber = 0

    DispSave = Disp.copy()
    tempContactPairs = contactPairs.copy()
    Time0 = Time

    Istep += 1
    Time += Dt

    while Flag11 == 1:  # Reduction loop
        NRConvergeNum += 1
        Flag11 = 0

        # Check whether the calculation is completed
        if (Time - 1) > 1e-10:
            if (1 + Dt - Time) > 1e-10:
                Dt = 1 + Dt - Time
                Time = 1
            else:
                break

        Factor = Time
        SDisp = Dt * FEMod.Cons[:, 2]   # MATLAB 1-based -> Python 0-based
        Iter = 0
        PreDisp = Disp.copy()

        while Flag20 == 1:  # Newton-Raphson loop
            Flag20 = 0
            Iter += 1

            GKF = sp.lil_matrix((AllDOF, AllDOF))  # sparse stiffness
            GKF2 = sp.lil_matrix((AllDOF, AllDOF))  # sparse stiffness
            Residual = np.zeros((AllDOF,1))
            Residual2 = np.zeros((AllDOF,1))
            ExtFVect = np.zeros((AllDOF,1))
            # Residual = np.zeros(AllDOF)
            # ExtFVect = np.zeros(AllDOF)
            NCon = FEMod.Cons.shape[0]

            # Internal force and tangent stiffness
            
            # if(op == "python"):
            Residual2, GKF2 = GetStiffnessAndForce(FEMod.Nodes, FEMod.Eles.astype('int'), Disp, Residual2.flatten(), GKF2, Dtan)
            Residual2 = Residual2.reshape((len(Residual2),1))
        
            Residual, GKF = octave.GetStiffnessAndForce_simpler(FEMod.Nodes, FEMod.Eles.astype('int'), Disp, Residual, GKF, Dtan, nout = 2)
            
            print(np.allclose(GKF, GKF2))
            
            
            # Contact state
            contactPairs, GKF, Residual = octave.DetermineContactState(
                FEMod, contactPairs, Dt, PreDisp, GKF, Residual, Disp, nout = 3
            )

            # External load boundary
            if FEMod.ExtF.shape[0] > 0:
                LOC = Dim * (FEMod.ExtF[:, 0].astype(int) - 1) + FEMod.ExtF[:, 1].astype(int) - 1  # convert to 0-based
                ExtFVect[LOC,0] += Factor * FEMod.ExtF[:, 2]
            Residual += ExtFVect

            # Displacement boundary conditions
            if NCon != 0:
                FixDOF = Dim * (FEMod.Cons[:, 0].astype(int) - 1) + FEMod.Cons[:, 1].astype(int) - 1
                GKF[FixDOF, :] = 0
                for i, dof in enumerate(FixDOF):
                    GKF[dof, dof] = 1.0
                Residual[FixDOF] = 0.0
                if Iter == 1:
                    Residual[FixDOF,0] = SDisp
                    


            if Iter > 1:
                FixDOF = Dim * (FEMod.Cons[:, 0] - 1) + FEMod.Cons[:, 1] - 1
                FreeDOF = np.setdiff1d(np.arange(AllDOF), FixDOF)

                Resid = np.max(np.abs(Residual[FreeDOF]))

                if Iter > 2:
                    print(f"{Iter:27d} {Resid:14.5e}")
                else:
                    print("\n \t Time  Time step   Iter \t  Residual")
                    print(f"{Time:10.5f} {Dt:10.3e} {Iter:5d} {Resid:14.5e}")

                if Resid < 1e-7:  # Convergence
                    octave.updateContact(contactPairs)
                    if NRConvergeNum > 1 and Iter < GivenIter and IterOld < GivenIter:
                        Enlarge = 1.5
                        Dt = min(Enlarge * Dt, MaxDt)  # Increase step
                    IterOld = Iter
                    Flag10 = 1
                    break

                if Iter + 1 > IterMax:  # Too many NR iterations
                    Reduce = 0.25
                    Dt = Reduce * Dt
                    Time = Time0 + Dt
                    if Dt < MinDt:
                        raise RuntimeError("Incremental step too small")
                    Disp = DispSave.copy()
                    contactPairs = tempContactPairs.copy()
                    print(f"Not converged or reached MaxIteration. Reducing load increment {ReductionNumber:3d}")
                    NRConvergeNum = 0
                    Flag11 = 1
                    Flag20 = 1
                    break

            
            # Solve linear system: GKF \ Residual
            print(spla.norm(GKF))
            print(np.linalg.norm(Residual)) 
            IncreDisp = spla.spsolve(GKF.tocsr(), Residual)
            print(np.linalg.norm(IncreDisp))  
            Disp[:,0] += IncreDisp
            Flag20 = 1