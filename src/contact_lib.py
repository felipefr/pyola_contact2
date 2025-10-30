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

octave.addpath(octave.genpath("/home/felipe/sources/pyola_contact2/src/"))  # doctest: +SKIP


def CalculateContactKandF(FEMod, ContactPairs, Dt, PreDisp, GKF, Residual, Disp):
    """
    Conservative, minimal-fix translation of the MATLAB DetermineContactState.
    Only small indexing/shape fixes applied (int casts, no reshape).
    """
    # --- Integration points (2x2 Gauss)
    gp = 1.0 / np.sqrt(3.0)
    IntegralPoint = np.array([
        [-gp, -gp],
        [ gp, -gp],
        [ gp,  gp],
        [-gp,  gp]], dtype = np.float64)

    

    MasterSurfXYZ, SlaveSurfXYZ, SlavePoints, SlavePointsFrame = get_master_slave_XYZ(FEMod, ContactPairs, Disp)
    ContactPairs = ContactSearch(FEMod, ContactPairs, Disp.reshape((-1,1)), 
                                 MasterSurfXYZ, SlaveSurfXYZ, SlavePoints, SlavePointsFrame)
    # support both dict-like and attribute-style FEMod
    FricFac = FEMod['FricFac'] if isinstance(FEMod, dict) else FEMod.FricFac
    
    nPairs = ContactPairs.SlaveSurf.shape[1]
    
    # --- Loop over contact pairs
    for i in range(nPairs):

        # Check for active contact
        if ContactPairs.CurMasterSurf[0, i] == 0:
            continue  # No contact

        # Case 1: first contact or frictionless contact
        if (FricFac == 0) or (ContactPairs.PreMasterSurf[0, i] == 0):
            ContactPairs.CurContactState[i] = 2  # Slip
        else:
            ContactPairs.CurContactState[i] = decide_stick_slip(FEMod, ContactPairs, Disp, PreDisp, i, IntegralPoint, FricFac, 
                              SlaveSurfXYZ[int(i/4)], SlavePoints[i], SlavePointsFrame[i])
            
        if(ContactPairs.CurContactState[i] == 2): # slip
            KL, ResL, ContactPairs = CalculateContactKandF_slip(FEMod, ContactPairs, Dt, PreDisp, i, 
                                                                        Disp, IntegralPoint)

        elif(ContactPairs.CurContactState[i] == 1): # stick
            KL, ResL, ContactPairs = CalculateContactKandF_stick(FEMod, ContactPairs, Dt, PreDisp, i, 
                                                                        Disp, IntegralPoint)

        MasterSurfNodes = GetSurfaceNode(FEMod.cells[ContactPairs.CurMasterSurf[0,i] - 1, :], 
                                                     ContactPairs.CurMasterSurf[1,i] - 1 )
        MasterSurfDOF = get_dofs_given_nodes_ids(MasterSurfNodes)

        SlaveSurfNodes = GetSurfaceNode(FEMod.cells[ContactPairs.SlaveSurf[0,i]-1, :], 
                                                    ContactPairs.SlaveSurf[1,i]-1) 
        
        SlaveSurfDOF = get_dofs_given_nodes_ids(SlaveSurfNodes)

        ContactDOF = np.concatenate([np.asarray(SlaveSurfDOF), 
                                     np.asarray(MasterSurfDOF)])
        
        Residual[ContactDOF] += ResL
        GKF[np.ix_(ContactDOF, ContactDOF)] += KL
            
    return ContactPairs, GKF, Residual

def ContactSearch(FEMod, ContactPairs, Disp, MasterSurfXYZ, SlaveSurfXYZ, SlavePoints, SlavePointsFrame):
    """
    ContactSearch - conservative translation from MATLAB
    (GetContactPointbyRayTracing still in Octave, 1-based safe)
    """
    
    method = "newton"

    MasterSurfNodeXYZ = get_deformed_position(FEMod.master_surf_nodes, FEMod.X, Disp) # redudant computations
    tree = cKDTree(MasterSurfNodeXYZ)
    
    for i in range(FEMod.slave_surf_cells.shape[0]):        
        for j in range(4): # gauss point number
            ipair = 4*i + j
            rr, ss, MasterEle, MasterSign, gg, Exist = GetContactPointbyRayTracing(
                FEMod, Disp, SlavePoints[ipair,:], SlavePointsFrame[ipair,:,:], MasterSurfXYZ, tree, method)
            
            
            if Exist == 1:
                ContactPairs.CurMasterSurf[:, ipair] = np.array([MasterEle, MasterSign])
                ContactPairs.rc[ipair] = rr
                ContactPairs.sc[ipair] = ss
                ContactPairs.Cur_g[ipair] = gg
            else:
                # print("contact not found at ", i)
                ContactPairs.CurMasterSurf[:, ipair] = 0 
                ContactPairs.rc[ipair] = 0
                ContactPairs.sc[ipair] = 0
                ContactPairs.Cur_g[ipair] = 0
                ContactPairs.CurContactState[ipair] = 0

    return ContactPairs


def CalculateContactKandF_slip(FEMod, ContactPairs, Dt, PreDisp, i, Disp, IntegralPoint):
    """
    Python translation of the MATLAB CalculateFrictionlessContactKandF.
    - i is a zero-based index into ContactPairs (Python convention).
    - ContactPairs fields are numpy arrays (struct-like object from Oct2Py or dict-like).
    """
    FricFac = FEMod.FricFac
    I3 = np.eye(3)
    
    # --- current slave geometry & previous slave geometry ---
    ip_idx = ContactPairs.SlaveIntegralPoint[i] - 1             # MATLAB->Python index
    CurIP = IntegralPoint[ip_idx, :].astype(np.float64)                                 # shape (2,)

    Na, dNa = GetSurfaceShapeFunction(CurIP)
    CurSlaveSurfXYZ, _ = GetSurfaceXYZ(FEMod.cells, FEMod.X, Disp, ContactPairs.SlaveSurf[:, i] - 1)
    PreSlaveSurfXYZ, _ = GetSurfaceXYZ(FEMod.cells, FEMod.X, PreDisp, ContactPairs.SlaveSurf[:, i] - 1)
    
    # geometric quantities
    Cur_n, J1, Cur_N1Xa, Cur_N2Xa, Cur_x1 = get_surface_geometry(Na, dNa, CurSlaveSurfXYZ)
    _, _, Pre_N1Xa, Pre_N2Xa, Pre_x1 = get_surface_geometry(Na, dNa, PreSlaveSurfXYZ)

    # normal traction
    tn = ContactPairs.Cur_g[i] * ContactPairs.pc[i]

    dx1 = Cur_x1 - Pre_x1
    PN = I3 - np.outer(Cur_n, Cur_n)

    dg1_slave = Cur_N1Xa - Pre_N1Xa
    dg2_slave = Cur_N2Xa - Pre_N2Xa
    m1 = np.cross(dg1_slave, Cur_N2Xa) + np.cross(Cur_N1Xa, dg2_slave)
    c1 = (PN @ m1) / J1

    # --- master geometry at current and previous steps ---
    Nb, dNb = GetSurfaceShapeFunction(np.array((ContactPairs.rc[i], ContactPairs.sc[i]), dtype = np.float64))
    CurMasterSurfXYZ, _ = GetSurfaceXYZ(FEMod.cells, FEMod.X, Disp, ContactPairs.CurMasterSurf[:, i] - 1)
    PreMasterSurfXYZ, _ = GetSurfaceXYZ(FEMod.cells, FEMod.X, PreDisp, ContactPairs.CurMasterSurf[:, i] - 1)
    

    _, _, Cur_N1Xb, Cur_N2Xb, Cur_x2 = get_surface_geometry(Nb, dNb, CurMasterSurfXYZ)
    _, _, _, _, Pre_x2 = get_surface_geometry(Nb, dNb, PreMasterSurfXYZ)

    dx2 = Cur_x2 - Pre_x2

    # --- precompute projection matrices and related arrays ---    
    # Compute the 2x2 coupling matrix
    Cur_NXa = np.vstack((Cur_N1Xa, Cur_N2Xa))  # shape (2, n)
    Cur_NXb = np.vstack((Cur_N1Xb, Cur_N2Xb))  # shape (2, n)
    a_ab = np.linalg.inv(Cur_NXa @ Cur_NXb.T)  # inverse
    
    # Compute bar vectors (each row is g1/g2 vector)
    g_bar_slave  = a_ab @ Cur_NXb       # shape (2, n), rows: g1_bar_slave, g2_bar_slave
    g_bar_master = a_ab.T @ Cur_NXa     # shape (2, n), rows: g1_bar_master, g2_bar_master
    
    # Optional: extract individual vectors
    g1_bar_slave, g2_bar_slave   = g_bar_slave[0, :], g_bar_slave[1, :]
    g1_bar_master, g2_bar_master = g_bar_master[0, :], g_bar_master[1, :]
    
    # Projections
    N1 = np.outer(Cur_n, Cur_n)
    N1_bar = I3 - np.outer(Cur_N1Xa, g1_bar_slave) - np.outer(Cur_N2Xa, g2_bar_slave)

    # mc1_bar, mb2_bar: (3 x 4) each
    mc1_bar = np.kron(dNa[0].T, g1_bar_slave.reshape(-1,1)) + np.kron(dNa[1].T, g2_bar_slave.reshape(-1,1))
    mb2_bar = np.kron(dNb[0].T, g1_bar_master.reshape(-1,1)) + np.kron(dNb[1].T, g2_bar_master.reshape(-1,1))

    Cur_g1_hat_slave = TransVect2SkewSym(Cur_N1Xa)
    Cur_g2_hat_slave = TransVect2SkewSym(Cur_N2Xa)
    Ac = (np.kron(dNa[0].T, Cur_g2_hat_slave) - np.kron(dNa[1].T, Cur_g1_hat_slave)) / J1

    # N1_wave is a 3x3 matrix: outer(Cur_n, N1_bar @ Cur_n)
    N1_wave = np.outer(Cur_n, N1_bar.dot(Cur_n))

    # Mc1_bar & Mb2_bar arranged as in original code (3 x 12)
    Mc1_bar = np.hstack([np.outer(Cur_n, mc1_bar[:, k]) for k in range(4)])
    Mb2_bar = - np.hstack([np.outer(Cur_n, mb2_bar[:, k]) for k in range(4)])

    # Gbc: shape (4,4)
    Gbc = ContactPairs.Cur_g[i] * (dNb.T @ a_ab @ dNa)   # result 4x4

    # --- relative velocity and tangential direction ---
    r1 = ContactPairs.Cur_g[i] * c1 + dx1 - dx2
    vr = r1 / Dt
    s1_temp = PN.dot(vr)

    if np.linalg.norm(s1_temp) > 1e-8:
        s1 = s1_temp / np.linalg.norm(s1_temp)
    else:
        s1 = np.zeros(3)
        dh = 0.0  # not used further here (kept for parity)

    # --- contact nodal force (frictionless baseline uses tv = tn * Cur_n) --
    tv = tn * Cur_n
    
    # Assemble frictionless stiffness
    KL = -get_frictionless_K(Na, Nb, ContactPairs.pc[i], tn, Ac, Mc1_bar, Mb2_bar, Gbc, N1, N1_wave, J1)
    
    if FricFac != 0 and np.linalg.norm(s1) > 1e-8:        
        tv += tn* FricFac * s1
        FrictionalK = get_frictional_K_slip(Na, Nb, ContactPairs.pc[i], tn, Ac, Mc1_bar, 
                                       Mb2_bar, Gbc, N1, N1_wave, J1, Cur_n, m1, 
                                       dg1_slave, dg2_slave, dNa, PN, vr, Dt, s1, r1, 
                                       ContactPairs.Cur_g[i], FricFac, c1, N1_bar, mc1_bar, mb2_bar)
    
        
        KL -= FrictionalK   

    ContactNodeForce = assemble_contact_force(Na, Nb, J1, tv)   # 1D length 24
    ContactPairs.Pressure[i]  = abs(tn)
    ContactPairs.Traction[i] = np.linalg.norm(tv)
    
    return KL, ContactNodeForce, ContactPairs




def CalculateContactKandF_stick(FEMod, ContactPairs, Dt, PreDisp, i, Disp, IntegralPoint):    
    # --- slave geometry at current IP ---
    ip_idx = int(ContactPairs.SlaveIntegralPoint[i]) - 1
    CurIP = IntegralPoint[ip_idx, :]
    Na, dNa = GetSurfaceShapeFunction(CurIP)
    CurSlaveSurfXYZ, SlaveSurfDOF = GetSurfaceXYZ(FEMod.cells, FEMod.X, Disp, ContactPairs.SlaveSurf[:, i].astype(np.int64) - 1)

    Cur_x1 = CurSlaveSurfXYZ.T @ Na
    Cur_NXa = dNa @ CurSlaveSurfXYZ

    Cur_n = np.cross(Cur_NXa[0], Cur_NXa[1])
    Cur_n /= np.linalg.norm(Cur_n)
    J1 = np.linalg.norm(np.cross(Cur_NXa[0], Cur_NXa[1]))

    # --- master geometry ---
    Nb, _ = GetSurfaceShapeFunction(np.array([ContactPairs.rp[i], ContactPairs.sc[i]]))
    CurMasterSurfXYZ_rpsp, MasterSurfDOF = GetSurfaceXYZ(FEMod.cells, FEMod.X, Disp, ContactPairs.PreMasterSurf[:, i].astype(np.int64) - 1)
    Cur_x2_p = CurMasterSurfXYZ_rpsp.T @ Nb

    Cur_g1_hat_slave = TransVect2SkewSym(Cur_NXa[0])
    Cur_g2_hat_slave = TransVect2SkewSym(Cur_NXa[1])
    Ac = (np.kron(dNa[0].T, Cur_g2_hat_slave) - np.kron(dNa[1].T, Cur_g1_hat_slave)) / J1
       
    # --- relative sliding vector ---
    gs = Cur_x2_p - Cur_x1
    tv = ContactPairs.pc[i] * gs    
    ContactNodeForce = assemble_contact_force(Na, Nb, J1, tv)   # 1D length 24

    # # --- stiffness ---
    KL = -get_frictional_K_stick(Na, Nb, ContactPairs.pc[i], J1, tv, Ac, Cur_n)

    
    ContactPairs.Pressure[i] = abs(np.dot(tv, Cur_n))
    ContactPairs.Traction[i] = np.linalg.norm(tv)
    
    return KL, ContactNodeForce, ContactPairs 


def decide_stick_slip(FEMod, ContactPairs, Disp, PreDisp, i, IP, FricFac, SlaveSurfXYZ, SlavePoint, SlavePointsFrame):
    # --- Case 2: possible stick/slip contact ---
    # ensure integer index when indexing IntegralPoint (ContactPairs stores 1..4)
    ip_idx = ContactPairs.SlaveIntegralPoint[i].astype(np.int64) - 1
    CurIP = IP[ip_idx]        # shape (2,)
    Na, dNa = GetSurfaceShapeFunction(CurIP)
    
    Cur_x1 = SlavePoint

    # Master surface (previous) - Nb uses rp,sp which are already numeric
    Nb, _ = GetSurfaceShapeFunction(np.array([ContactPairs.rp[i], ContactPairs.sp[i]]))
    pre_master_surf = ContactPairs.PreMasterSurf[:, i].astype(np.int64) - 1 
    MasterSurfNodes = GetSurfaceNode(FEMod.cells[pre_master_surf[0],:], pre_master_surf[1])
    MasterSurfDOF = get_dofs_given_nodes_ids(MasterSurfNodes)
    CurMasterSurfXYZ_p = get_deformed_position_given_dofs(MasterSurfNodes, FEMod.X, Disp, MasterSurfDOF)
    Cur_x2_p = Nb @ CurMasterSurfXYZ_p

    # Relative motion and projection
    gs = Cur_x2_p - Cur_x1
    tv = ContactPairs.pc[i] * gs

    # Current normal
    Cur_NXa = dNa@SlaveSurfXYZ # (2,4),(4,3)-> (2,3)
    Cur_n = np.cross(Cur_NXa[0], Cur_NXa[1])
    Cur_n = Cur_n / np.linalg.norm(Cur_n)

    # Tangential/normal trial components
    tn_trial = abs(np.dot(tv, Cur_n))
    tt_trial = np.sqrt(np.linalg.norm(tv)**2 - tn_trial**2)

    # Slip/stick criterion
    fai = tt_trial - FricFac * tn_trial
    
    return 1 if fai < 0 else 2




def flatteningContactPairs(ContactPairs):
    # Preallocate arrays (using plain numpy first)
    ContactPairs.pc  = ContactPairs.pc.flatten()
    ContactPairs.SlaveIntegralPoint = ContactPairs.SlaveIntegralPoint.flatten().astype(np.int64)

    ContactPairs.rc = ContactPairs.rc.flatten()
    ContactPairs.sc = ContactPairs.sc.flatten()
    ContactPairs.Cur_g = ContactPairs.Cur_g.flatten()
    ContactPairs.Pre_g = ContactPairs.Pre_g.flatten()
    ContactPairs.rp = ContactPairs.rp.flatten()
    ContactPairs.sp = ContactPairs.sp.flatten()
    ContactPairs.CurContactState = ContactPairs.CurContactState.flatten().astype(np.int64)
    ContactPairs.PreContactState = ContactPairs.PreContactState.flatten().astype(np.int64)
    ContactPairs.Pressure = ContactPairs.Pressure.flatten()
    ContactPairs.Traction = ContactPairs.Traction.flatten()
    
    ContactPairs.SlaveSurf = ContactPairs.SlaveSurf.astype(np.int64)
    ContactPairs.CurMasterSurf = ContactPairs.CurMasterSurf.astype(np.int64)

    return ContactPairs



def get_master_slave_XYZ(FEMod, ContactPairs, Disp): # integral points already chosen
    nPairs = ContactPairs.SlaveSurf.shape[1]
    SlaveSurf = ContactPairs.SlaveSurf.astype(np.int64) - 1
    MasterSurfXYZ = np.array([ get_deformed_position(msc, FEMod.X, Disp) for msc in FEMod.master_surf_cells]).reshape((-1,4,3)) # redudant computations
    
    SlaveSurfXYZ =  np.array([ get_deformed_position(ssc, FEMod.X, Disp) for ssc in FEMod.slave_surf_cells]) # 120x4x3
    SlavePoints = np.empty((nPairs, 3))
    SlavePointsFrame = np.empty((nPairs, 3, 3)) # (:, [normal, t1, t2], ndim) not the best convention
    
    for i in range(FEMod.slave_surf_cells.shape[0]):
        for j in range(4):
            ipair = 4*i + j
            SlavePoints[ipair, :] = SlaveSurfXYZ[i].T@FEMod.ShpfSurf[j][0]
            SlavePointsFrame[ipair, 1:3, :] = FEMod.ShpfSurf[j][1] @ SlaveSurfXYZ[i]
            SlavePointsFrame[ipair, 0, :] = np.cross(SlavePointsFrame[ipair, 1, :], SlavePointsFrame[ipair, 2, :])
            SlavePointsFrame[ipair, 0, :] /= np.linalg.norm(SlavePointsFrame[ipair, 0, :])

    return MasterSurfXYZ, SlaveSurfXYZ, SlavePoints, SlavePointsFrame

def updateContact(contactPairs):
    """
    updateContact - Update contact history between time steps (struct-of-vectors version)
    
    Parameters
    ----------
    contactPairs : Struct
        Structure with contact pair data (fields from InitializeContactPairs)
    
    Returns
    -------
    contactPairs : Struct
        Updated contactPairs structure
        
    To do:
        - abadon matlab convention of row-vector (1xN)
    """

    nPairs = contactPairs.SlaveSurf.shape[1]

    for i in range(nPairs):
        if contactPairs.CurContactState[i] == 0:
            # --- No contact ---
            contactPairs.PreMasterSurf[:, i] = 0
            contactPairs.rp[i] = 0.0
            contactPairs.sp[i] = 0.0
            contactPairs.PreContactState[i] = 0
            contactPairs.Pre_g[i] = 0.0
            contactPairs.Pressure[i] = 0.0
            contactPairs.Traction[i] = 0.0
        else:
            # --- Slip or stick contact ---
            contactPairs.PreMasterSurf[:, i] = contactPairs.CurMasterSurf[:, i]
            contactPairs.rp[i] = contactPairs.rc[i]
            contactPairs.sp[i] = contactPairs.sc[i]
            contactPairs.PreContactState[i] = contactPairs.CurContactState[i]
            contactPairs.Pre_g[i] = contactPairs.Cur_g[i]

        # --- Reset current step quantities ---
        contactPairs.rc[i] = 0.0
        contactPairs.sc[i] = 0.0
        contactPairs.Cur_g[i] = 0.0
        contactPairs.CurMasterSurf[:, i] = 0 
        contactPairs.CurContactState[i] = 0

    return contactPairs

def InitializeContactPairs(FEMod):
    """
    InitializeContactPairs - Initialize contact data as a struct of arrays.

    Parameters
    ----------
    FEMod : Struct or dict
        FEM data structure, must include field 'SlaveSurf' (2×n)

    Returns
    -------
    ContactPairs : Struct
        Structure where each field is a NumPy array (not a list of structs)
    """

    # Extract data
    SlaveSurf = np.asarray(FEMod['SlaveSurf'])
    nSlave = SlaveSurf.shape[1]
    nGauss = 4
    nPairs = nSlave * nGauss

    # Preallocate arrays (using plain numpy first)
    pc  = 1e6 * np.ones(nPairs)
    SlaveSurf_arr = np.zeros((2, nPairs), dtype=np.int64)
    SlaveIntegralPoint = np.zeros(nPairs, dtype=np.int64)

    CurMasterSurf = np.zeros((2, nPairs), dtype=np.int64)
    rc = np.zeros(nPairs)
    sc = np.zeros(nPairs)
    Cur_g = np.zeros(nPairs)
    Pre_g = np.zeros(nPairs)
    PreMasterSurf = np.zeros((2, nPairs), dtype=np.int64)
    rp = np.zeros(nPairs)
    sp = np.zeros(nPairs)
    CurContactState = np.zeros(nPairs, dtype=np.int64)
    PreContactState = np.zeros(nPairs, dtype=np.int64)
    Pressure = np.zeros(nPairs)
    Traction = np.zeros(nPairs)

    # Fill fields
    for i in range(nSlave):
        for j in range(nGauss):
            k = i * nGauss + j
            SlaveSurf_arr[:, k] = SlaveSurf[:, i]
            SlaveIntegralPoint[k] = j + 1  # MATLAB 1-based indexing

    # === Convert to Struct explicitly (like MATLAB struct) ===
    ContactPairs = Struct(
        pc=pc,
        SlaveSurf=SlaveSurf_arr,
        SlaveIntegralPoint=SlaveIntegralPoint,
        CurMasterSurf=CurMasterSurf,
        rc=rc,
        sc=sc,
        Cur_g=Cur_g,
        Pre_g=Pre_g,
        PreMasterSurf=PreMasterSurf,
        rp=rp,
        sp=sp,
        CurContactState=CurContactState,
        PreContactState=PreContactState,
        Pressure=Pressure,
        Traction=Traction
    )

    return ContactPairs



# def ContactSearch(FEMod, ContactPairs, Disp, IntegralPoint):
#     """
#     ContactSearch - conservative translation from MATLAB
#     (GetContactPointbyRayTracing still in Octave, 1-based safe)
#     """
    
#     nPairs = ContactPairs.SlaveSurf.shape[1]
#     MasterSurf_ = FEMod.MasterSurf - 1
#     SlaveSurf = ContactPairs.SlaveSurf.astype(np.int64) - 1

#     for i in range(nPairs):
#         # --- Get current slave surface geometry ---
#         SlaveSurfNode = GetSurfaceNode(FEMod.cells[SlaveSurf[0,i], :], SlaveSurf[1,i])
#         SlaveSurfNodeXYZ = get_deformed_position(SlaveSurfNode, FEMod.X, Disp)

#         # Current integration point coordinates (MATLAB -> Python: subtract 1)
#         ip_idx = ContactPairs.SlaveIntegralPoint[i] - 1
#         CurIP = IntegralPoint[ip_idx, :].astype(np.float64)
        
#         N, dN = GetSurfaceShapeFunction(CurIP)
#         SlavePoint = SlaveSurfNodeXYZ.T@N
#         SlavePointTan = (dN @ SlaveSurfNodeXYZ).T # [(2,4)x(4,3)].T --> (3,2)

#         rr, ss, MasterEle, MasterSign, gg, Exist = opt.GetContactPointbyRayTracing(
#             FEMod.cells, FEMod.X, MasterSurf_, Disp, SlavePoint, SlavePointTan)
            
#         if Exist == 1:
#             ContactPairs.CurMasterSurf[:, i] = np.array([MasterEle, MasterSign])
#             ContactPairs.rc[i] = rr
#             ContactPairs.sc[i] = ss
#             ContactPairs.Cur_g[i] = gg
#         else:
#             # print("contact not found at ", i)
#             ContactPairs.CurMasterSurf[:, i] = 0 
#             ContactPairs.rc[i] = 0
#             ContactPairs.sc[i] = 0
#             ContactPairs.Cur_g[i] = 0
#             ContactPairs.CurContactState[i] = 0

#     return ContactPairs
