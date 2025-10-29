#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 08:57:08 2025

@author: frocha
"""
import numpy as np
from oct2py import Struct
from oct2py import octave
from utils import *
import optimised_functions as opt
from ray_tracing import GetContactPointbyRayTracing
from scipy.spatial import cKDTree

octave.addpath(octave.genpath("/home/felipe/sources/pyola_contact2/src/"))  # doctest: +SKIP

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


def ContactSearch(FEMod, ContactPairs, Disp, IntegralPoint):
    """
    ContactSearch - conservative translation from MATLAB
    (GetContactPointbyRayTracing still in Octave, 1-based safe)
    """
    
    method = "newton"
    nPairs = ContactPairs.SlaveSurf.shape[1]
    SlaveSurf = ContactPairs.SlaveSurf.astype(np.int64) - 1
    
    MasterSurfXYZ = np.array([ get_deformed_position(msc, FEMod.X, Disp) for msc in FEMod.master_surf_cells]).reshape((-1,4,3)) # redudant computations
    MasterSurfNodeXYZ = get_deformed_position(FEMod.master_surf_nodes, FEMod.X, Disp) # redudant computations
    tree = cKDTree(MasterSurfNodeXYZ)
    
    SlaveSurfNodeXYZ =  np.array([ get_deformed_position(ssc, FEMod.X, Disp) for ssc in FEMod.slave_surf_cells]) # 120x4x3
    SlavePoint = np.empty((nPairs, 3))
    SlavePointFrame = np.empty((nPairs, 3, 3)) # (:, [normal, t1, t2], ndim) not the best convention
    
    for i in range(FEMod.slave_surf_cells.shape[0]):
        for j in range(4):
            ipair = 4*i + j
            SlavePoint[ipair, :] = SlaveSurfNodeXYZ[i].T@FEMod.ShpfSurf[j][0]
            SlavePointFrame[ipair, 1:3, :] = FEMod.ShpfSurf[j][1] @ SlaveSurfNodeXYZ[i]
            SlavePointFrame[ipair, 0, :] = np.cross(SlavePointFrame[ipair, 1, :], SlavePointFrame[ipair, 2, :])
            SlavePointFrame[ipair, 0, :] /= np.linalg.norm(SlavePointFrame[ipair, 0, :])


    for i in range(FEMod.slave_surf_cells.shape[0]):        
        for j in range(4): # gauss point number
            ipair = 4*i + j
            rr, ss, MasterEle, MasterSign, gg, Exist = GetContactPointbyRayTracing(
                FEMod, Disp, SlavePoint[ipair,:], SlavePointFrame[ipair,:,:], MasterSurfXYZ, tree, method)
            
            
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


def CalculateFrictionlessContactKandF(FEMod, ContactPairs, Dt, PreDisp, i, GKF, Residual, Disp, IntegralPoint):
    """
    Python translation of the MATLAB CalculateFrictionlessContactKandF.
    - i is a zero-based index into ContactPairs (Python convention).
    - ContactPairs fields are numpy arrays (struct-like object from Oct2Py or dict-like).
    """
    
    # Only handle slip (state == 2)
    if ContactPairs.CurContactState[i] != 2:
        return GKF, Residual, ContactPairs

    # --- current slave geometry & previous slave geometry ---
    ip_idx = ContactPairs.SlaveIntegralPoint[i] - 1             # MATLAB->Python index
    CurIP = IntegralPoint[ip_idx, :].astype(np.float64)                                 # shape (2,)

    Na, [N1a, N2a] = GetSurfaceShapeFunction(CurIP)
    SlaveSurfNodes = GetSurfaceNode(FEMod.cells[ContactPairs.SlaveSurf[0,i]-1, :], 
                                                ContactPairs.SlaveSurf[1,i]-1)
    SlaveSurfDOF = get_dofs_given_nodes_ids(SlaveSurfNodes)
    CurSlaveSurfXYZ = get_deformed_position_given_dofs(SlaveSurfNodes, FEMod.X, Disp, SlaveSurfDOF)
    PreSlaveSurfXYZ = get_deformed_position_given_dofs(SlaveSurfNodes, FEMod.X, PreDisp, SlaveSurfDOF)
    
    # geometric quantities
    Cur_n, J1, Cur_N1Xa, Cur_N2Xa, Cur_x1 = get_surface_geometry(Na, N1a, N2a, CurSlaveSurfXYZ)
    _, _, Pre_N1Xa, Pre_N2Xa, Pre_x1 = get_surface_geometry(Na, N1a, N2a, PreSlaveSurfXYZ)

    # normal traction
    tn = ContactPairs.Cur_g[i] * ContactPairs.pc[i]

    dx1 = Cur_x1 - Pre_x1
    PN = np.eye(3) - np.outer(Cur_n, Cur_n)

    dg1_slave = Cur_N1Xa - Pre_N1Xa
    dg2_slave = Cur_N2Xa - Pre_N2Xa
    m1 = np.cross(dg1_slave, Cur_N2Xa) + np.cross(Cur_N1Xa, dg2_slave)
    c1 = PN.dot(m1) / J1

    # --- master geometry at current and previous steps ---
    Nb, [N1b, N2b] = GetSurfaceShapeFunction(np.array((ContactPairs.rc[i], ContactPairs.sc[i]), dtype = np.float64))
    MasterSurfNodes = GetSurfaceNode(FEMod.cells[ContactPairs.CurMasterSurf[0,i] - 1, :], 
                                                 ContactPairs.CurMasterSurf[1,i] - 1 )
    MasterSurfDOF = get_dofs_given_nodes_ids(MasterSurfNodes)
    CurMasterSurfXYZ = get_deformed_position_given_dofs(MasterSurfNodes, FEMod.X, Disp, MasterSurfDOF)
    PreMasterSurfXYZ = get_deformed_position_given_dofs(MasterSurfNodes, FEMod.X, PreDisp, MasterSurfDOF)

    _, _, Cur_N1Xb, Cur_N2Xb, Cur_x2 = get_surface_geometry(Nb, N1b, N2b, CurMasterSurfXYZ)
    _, _, _, _, Pre_x2 = get_surface_geometry(Nb, N1b, N2b, PreMasterSurfXYZ)

    dx2 = Cur_x2 - Pre_x2

    # --- relative velocity and tangential direction ---
    r1 = ContactPairs.Cur_g[i] * c1 + dx1 - dx2
    vr = r1 / Dt
    s1_temp = PN.dot(vr)

    if np.linalg.norm(s1_temp) > 1e-8:
        s1 = s1_temp / np.linalg.norm(s1_temp)
    else:
        s1 = np.zeros(3)
        dh = 0.0  # not used further here (kept for parity)

    # --- contact nodal force (frictionless baseline uses tv = tn * Cur_n) ---
    tv = tn * Cur_n
    ContactNodeForce = assemble_contact_force(Na, Nb, J1, tv)   # 1D length 24

    # ContactDOF: concatenate DOFs (ensure integer numpy array)
    ContactDOF = np.concatenate([np.asarray(SlaveSurfDOF), 
                                 np.asarray(MasterSurfDOF)])

    # Residual expected shape: (nDOF, 1) or (nDOF,) — handle both
    if Residual.ndim == 1:
        Residual[ContactDOF] += ContactNodeForce
    else:
        Residual[np.ix_(ContactDOF, [0])] += ContactNodeForce.reshape(-1, 1)

    ContactPairs.Pressure[i]  = abs(tn)
    ContactPairs.Traction[i] = np.linalg.norm(tv)

    # --- precompute projection matrices and related arrays ---
    A_ab = np.array([[Cur_N1Xa.dot(Cur_N1Xb), Cur_N1Xa.dot(Cur_N2Xb)],
                     [Cur_N2Xa.dot(Cur_N1Xb), Cur_N2Xa.dot(Cur_N2Xb)]])
    a_ab = np.linalg.inv(A_ab)

    g1_bar_slave  = a_ab[0,0] * Cur_N1Xb + a_ab[1,0] * Cur_N2Xb
    g2_bar_slave  = a_ab[0,1] * Cur_N1Xb + a_ab[1,1] * Cur_N2Xb
    g1_bar_master = a_ab[0,0] * Cur_N1Xa + a_ab[0,1] * Cur_N2Xa
    g2_bar_master = a_ab[1,0] * Cur_N1Xa + a_ab[1,1] * Cur_N2Xa

    N1 = np.outer(Cur_n, Cur_n)
    N1_bar = np.eye(3) - np.outer(Cur_N1Xa, g1_bar_slave) - np.outer(Cur_N2Xa, g2_bar_slave)

    # mc1_bar, mb2_bar: (3 x 4) each
    mc1_bar = np.kron(N1a.reshape(1,4), g1_bar_slave.reshape(3,1)) + np.kron(N2a.reshape(1,4), g2_bar_slave.reshape(3,1))
    mb2_bar = np.kron(N1b.reshape(1,4), g1_bar_master.reshape(3,1)) + np.kron(N2b.reshape(1,4), g2_bar_master.reshape(3,1))

    Cur_g1_hat_slave = TransVect2SkewSym(Cur_N1Xa)
    Cur_g2_hat_slave = TransVect2SkewSym(Cur_N2Xa)
    Ac = (np.kron(N1a.reshape(1,4), Cur_g2_hat_slave) - np.kron(N2a.reshape(1,4), Cur_g1_hat_slave)) / J1

    # N1_wave is a 3x3 matrix: outer(Cur_n, N1_bar @ Cur_n)
    N1_wave = np.outer(Cur_n, N1_bar.dot(Cur_n))

    # Mc1_bar & Mb2_bar arranged as in original code (3 x 12)
    Mc1_bar = np.hstack([np.outer(Cur_n, mc1_bar[:, k]) for k in range(4)])
    Mb2_bar = np.hstack([-np.outer(Cur_n, mb2_bar[:, k]) for k in range(4)])

    # N12 arrays for Gbc: shape (2,4)
    N12a = np.vstack((N1a.reshape(1,4), N2a.reshape(1,4)))
    N12b = np.vstack((N1b.reshape(1,4), N2b.reshape(1,4)))
    Gbc = ContactPairs.Cur_g[i] * (N12b.T.dot(a_ab).dot(N12a))   # result 4x4

    # Assemble frictionless stiffness
    FrictionlessK = get_frictionless_K(Na, Nb, ContactPairs.pc[i], tn, Ac, Mc1_bar, Mb2_bar, Gbc, N1, N1_wave, J1)

    # Subtract into global stiffness
    GKF[np.ix_(ContactDOF, ContactDOF)] -= FrictionlessK

    return GKF, Residual, ContactPairs


def DetermineFrictionlessContactState(FEMod, ContactPairs, Dt, PreDisp, GKF, Residual, Disp):
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

    # --- Contact search and friction factor

    ContactPairs = ContactSearch(FEMod, ContactPairs, Disp.reshape((-1,1)), IntegralPoint)
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

            # i should start in 1 to be compatible with octave
            # GKF, Residual, ContactPairs = octave.CalculateContactKandF(
            #     FEMod, ContactPairs, Dt, PreDisp, i+1, GKF, Residual, Disp, IntegralPoint, nout = 3) 
            # flattenising_struct(ContactPairs)
            
            GKF, Residual, ContactPairs = CalculateFrictionlessContactKandF(
                FEMod, ContactPairs, Dt, PreDisp, i, GKF, Residual, Disp.reshape((-1,1)), IntegralPoint) 
            continue

    return ContactPairs, GKF, Residual


def assemble_contact_force(Na, Nb, J1, tv):
    """
    Assemble the 24x1 contact nodal force vector for a 4-node slave and 4-node master surface pair.

    Parameters
    ----------
    Na, Nb : (4,) arrays
        Shape functions of slave and master surfaces at the integration point.
    J1 : float
        Surface Jacobian.
    tv : (3,) array
        Traction vector at the contact point.

    Returns
    -------
    ContactNodeForce : (24,) array
        Assembled nodal contact force vector.
    """
    ContactNodeForce = np.zeros(24)
    for a in range(4):
        idx = slice(3*a, 3*a + 3)
        ContactNodeForce[idx] = Na[a] * J1 * tv
        ContactNodeForce[idx.start + 12: idx.stop + 12] = -Nb[a] * J1 * tv
    
    
    return ContactNodeForce

def get_frictionless_K(Na, Nb, pc, tn, Ac, Mc1_bar, Mb2_bar, Gbc, N1, N1_wave, J1):
    """
    Assemble the 24x24 frictionless contact stiffness matrix.

    Parameters
    ----------
    Na, Nb : (4,) arrays
        Shape functions for slave and master surfaces.
    pc : float
        Contact penalty parameter.
    tn : float
        Normal contact traction.
    Ac, Mc1_bar, Mb2_bar : (3,12) arrays
        Auxiliary matrices related to geometry and projection.
    Gbc : (4,4) array
        Coupling matrix between slave and master surfaces.
    N1, N1_wave : (3,3) arrays
        Normal projection matrices.
    J1 : float
        Surface Jacobian.

    Returns
    -------
    FrictionlessK : (24,24) array
        Assembled frictionless contact stiffness matrix.
    """

    Frictionless_K11 = np.zeros((12, 12))
    Frictionless_K12 = np.zeros((12, 12))
    Frictionless_K21 = np.zeros((12, 12))
    Frictionless_K22 = np.zeros((12, 12))

    for aa in range(4):
        for bb in range(4):
            idxA = slice(3 * aa, 3 * aa + 3)
            idxB = slice(3 * bb, 3 * bb + 3)

            tempK = (-Na[aa] * Na[bb] * pc * N1_wave
                     - Na[aa] * tn * (Ac[:, idxB] + Mc1_bar[:, idxB] @ N1)) * J1
            Frictionless_K11[idxA, idxB] += tempK

            tempK = (Na[aa] * Nb[bb] * pc * N1_wave) * J1
            Frictionless_K12[idxA, idxB] += tempK

            tempK = (Nb[aa] * Na[bb] * pc * N1_wave
                     + Nb[aa] * tn * (Ac[:, idxB] + Mc1_bar[:, idxB] @ N1)
                     + Na[bb] * tn * Mb2_bar[:, idxA]
                     + Gbc[aa, bb] * tn * N1) * J1
            Frictionless_K21[idxA, idxB] += tempK

            tempK = (-Nb[aa] * Nb[bb] * pc * N1_wave
                     - Nb[bb] * tn * Mb2_bar[:, idxA]) * J1
            Frictionless_K22[idxA, idxB] += tempK

    # Combine into 24x24 block matrix
    FrictionlessK = np.block([
        [Frictionless_K11, Frictionless_K12],
        [Frictionless_K21, Frictionless_K22]
    ])

    return FrictionlessK


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
