#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 16:33:54 2025

@author: frocha
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 08:57:08 2025

@author: frocha
"""
import numpy as np
from oct2py import Struct
from oct2py import octave
from utils import (GetSurfaceNode, get_dofs_given_nodes_ids, get_deformed_position_given_dofs, 
                   GetSurfaceShapeFunction, TransVect2SkewSym, GetSurfaceXYZ, get_surface_geometry, get_deformed_position)
from ray_tracing import GetContactPointbyRayTracing
from scipy.spatial import cKDTree
from contact_assemblers import *
from contact_lib import get_Ac_tilde, get_frictionless_aux_operators

octave.addpath(octave.genpath("/home/felipe/sources/pyola_contact2/src/"))  # doctest: +SKIP

normtau = lambda s: np.sqrt(s**2 + tau1)
Ntau = lambda s: s/normtau(s) # aproximates sign
dNtau = lambda s: (1-Ntau(s)**2)/normtau(s)

FB = lambda a,b: np.sqrt(a**2 + b**2 + 2*tau2) - a - b  # ab=tau2 at convergence
d1FB = lambda a,b : a/(FB(a,b)+a+b) - 1
d2FB = lambda a,b : b/(FB(a,b)+a+b) - 1


def ContactSearch(FEMod, ContactPairs, Disp):
    method = "newton"
    
    ContactPairs.update_master_slave_XYZ(FEMod, Disp)
    MasterSurfNodeXYZ = get_deformed_position(ContactPairs.master_surf_nodes, FEMod.X, Disp) # redudant computations
    tree = cKDTree(MasterSurfNodeXYZ)
    
    idx_active = []
    
    for i, sp in enumerate(ContactPairs.slave_points):  
        sp.update_slave(FEMod, Disp)
        Master_idx, rr, ss, gg, Exist = GetContactPointbyRayTracing(
            FEMod, ContactPairs, Disp, sp.point, sp.frame, 
            ContactPairs.master_surf_XYZ, tree, method)
        
        if Exist == 1:
            sp.is_active = True
            sp.master_surf_idx = Master_idx
            sp.Xi = np.array([rr, ss])
            sp.gap = gg
            sp.update_master(FEMod, Disp)
            idx_active.append(i)
        else:
            # print("contact not found at ", i)
            sp.is_active = False
            sp.master_surf_idx = -1
            sp.Xi.fill(0.)
            sp.gap = 0.

    return idx_active

def CalculateContactKandF(FEMod, ContactPairs, Dt, PreDisp, GKF, Residual, Disp):
    """
    Conservative, minimal-fix translation of the MATLAB DetermineContactState.
    Only small indexing/shape fixes applied (int casts, no reshape).
    """
    
    idx_active = ContactSearch(FEMod, ContactPairs, Disp.reshape((-1,1)))

    FricFac = ContactPairs.FricFac
    
    # --- Loop over contact pairs
    for i in idx_active:
        sp = ContactPairs.slave_points[i]

        KL, ResL, ContactPairs = get_KF_FB(sp, FEMod, ContactPairs)

        dofs = sp.get_contact_dofs()
        Residual[dofs] += ResL
        GKF[np.ix_(dofs, dofs)] += KL
            
    return ContactPairs, GKF, Residual 

def get_KF_FB(sp, FEMod, ContactPairs):
    gap = sp.gap
    tn = sp.pressure
    
    Na, dNa = FEMod.ShpfSurf[sp.idxIP]
    tau1 = sp.frame[1:3]
    Cur_n = sp.frame[0]
    J1 = sp.J 
    
    # --- master geometry at current geo, with current and previous steps ---
    Nb = sp.Nb
    dNb = sp.dNb
    tau2 = sp.master_tangent
    
    tv = tn * Cur_n
    
    Ngap, Ngap_star, Ac_tilde, Deta, B2tilde, PN = get_frictionless_aux_operators(Na, Nb, dNa, dNb, tau1, tau2, Cur_n, J1, gap)

    # Dtn_n = eps*np.outer(Cur_n,Cur_n) @ ( Ngap_star + gap*Dn) # Dtn[n] = eps*Dgn[n]
    Dgn_n = eps*np.outer(Cur_n,Cur_n) @ ( Ngap_star + gap*Dn) # Dtn[n] = eps*Dgn[n]
    
    Dn = PN@Ac_tilde # [PN@Ac, 0]  

    Dtv = tn*Dn # only Dn term appears as tn in "independent"
    DJ1 = J1*np.outer(tv,Cur_n)@Ac_tilde
    Kuu = Ngap.T@(J1*Dtv + DJ1)
    Kuu += np.einsum('i, jik -> kj', J1*tv, B2tilde)@Deta # virtual gap tangent term
        
    Fu = -J1*Ngap.T@tv # the residual is perfect
    
    
    return KL, ContactNodeForce, ContactPairs