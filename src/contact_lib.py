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

octave.addpath(octave.genpath("/home/felipe/sources/pyola_contact2/src/"))  # doctest: +SKIP

def DetermineContactState(FEMod, ContactPairs, Dt, PreDisp, GKF, Residual, Disp):
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
            
            GKF, Residual, ContactPairs =  CalculateContactKandF(
                FEMod, ContactPairs, Dt, PreDisp, i, GKF, Residual, Disp.reshape((-1,1)), IntegralPoint) 
            continue

        ContactPairs.CurContactState[i] = decide_stick_slip(FEMod, ContactPairs, Disp, PreDisp, i, IntegralPoint, FricFac, 
                          SlaveSurfXYZ[int(i/4)], SlavePoints[i], SlavePointsFrame[i])

        GKF, Residual, ContactPairs = CalculateContactKandF(
            FEMod, ContactPairs, Dt, PreDisp, i, GKF, Residual, Disp.reshape((-1,1)), IntegralPoint) 
        
        # Residual = Residual.flatten()
        # ContactPairs = flatteningContactPairs(ContactPairs)
        
    return ContactPairs, GKF, Residual


def DetermineContactState2(FEMod, ContactPairs, Dt, PreDisp, GKF, Residual, Disp):
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
            
            GKF, Residual, ContactPairs =  CalculateContactKandF(
                FEMod, ContactPairs, Dt, PreDisp, i, GKF, Residual, Disp.reshape((-1,1)), IntegralPoint) 
            continue

        ContactPairs.CurContactState[i] = decide_stick_slip(FEMod, ContactPairs, Disp, PreDisp, i, IntegralPoint, FricFac, 
                          SlaveSurfXYZ[int(i/4)], SlavePoints[i], SlavePointsFrame[i])

        GKF, Residual, ContactPairs = CalculateContactKandF(
            FEMod, ContactPairs, Dt, PreDisp, i, GKF, Residual, Disp.reshape((-1,1)), IntegralPoint) 
        
        # Residual = Residual.flatten()
        # ContactPairs = flatteningContactPairs(ContactPairs)
        
    return ContactPairs, GKF, Residual


def CalculateContactKandF(FEMod, ContactPairs, Dt, PreDisp, i, GKF, Residual, Disp, IntegralPoint):
    FricFac = FEMod.FricFac
    
    if ContactPairs.CurContactState[i] == 1:  # Stick contact
    
        KL, ResL, ContactPairs = CalculateContactKandF_stick(FEMod, ContactPairs, Dt, PreDisp, i, 
                                                               Disp, IntegralPoint)
        
        ResL = ResL.flatten()
        
    elif ContactPairs.CurContactState[i] == 2:  # Slip contact
        KL, ResL, ContactPairs = CalculateFrictionlessContactKandF(FEMod, ContactPairs, Dt, PreDisp, i, Disp, IntegralPoint)
        ResL = ResL.flatten()
        if FricFac != 0.0:
            KL2, ResL2, ContactPairs = CalculateContactKandF_onlyslip(FEMod, ContactPairs, Dt, PreDisp, i, 
                                                                        Disp, IntegralPoint)
        
            KL += KL2
            ResL += ResL2.flatten()        
    
    # ContactPairs = flatteningContactPairs(ContactPairs)
    
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
        
    return GKF, Residual, ContactPairs 

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


def CalculateFrictionlessContactKandF(FEMod, ContactPairs, Dt, PreDisp, i, Disp, IntegralPoint):
    """
    Python translation of the MATLAB CalculateFrictionlessContactKandF.
    - i is a zero-based index into ContactPairs (Python convention).
    - ContactPairs fields are numpy arrays (struct-like object from Oct2Py or dict-like).
    """

    # --- current slave geometry & previous slave geometry ---
    ip_idx = ContactPairs.SlaveIntegralPoint[i] - 1             # MATLAB->Python index
    CurIP = IntegralPoint[ip_idx, :].astype(np.float64)                                 # shape (2,)

    Na, dNa = GetSurfaceShapeFunction(CurIP)
    SlaveSurfNodes = GetSurfaceNode(FEMod.cells[ContactPairs.SlaveSurf[0,i]-1, :], 
                                                ContactPairs.SlaveSurf[1,i]-1)
    SlaveSurfDOF = get_dofs_given_nodes_ids(SlaveSurfNodes)
    CurSlaveSurfXYZ = get_deformed_position_given_dofs(SlaveSurfNodes, FEMod.X, Disp, SlaveSurfDOF)
    PreSlaveSurfXYZ = get_deformed_position_given_dofs(SlaveSurfNodes, FEMod.X, PreDisp, SlaveSurfDOF)
    
    # geometric quantities
    Cur_n, J1, Cur_N1Xa, Cur_N2Xa, Cur_x1 = get_surface_geometry(Na, dNa, CurSlaveSurfXYZ)
    _, _, Pre_N1Xa, Pre_N2Xa, Pre_x1 = get_surface_geometry(Na, dNa, PreSlaveSurfXYZ)

    # normal traction
    tn = ContactPairs.Cur_g[i] * ContactPairs.pc[i]

    dx1 = Cur_x1 - Pre_x1
    PN = np.eye(3) - np.outer(Cur_n, Cur_n)

    dg1_slave = Cur_N1Xa - Pre_N1Xa
    dg2_slave = Cur_N2Xa - Pre_N2Xa
    m1 = np.cross(dg1_slave, Cur_N2Xa) + np.cross(Cur_N1Xa, dg2_slave)
    c1 = PN.dot(m1) / J1

    # --- master geometry at current and previous steps ---
    Nb, dNb = GetSurfaceShapeFunction(np.array((ContactPairs.rc[i], ContactPairs.sc[i]), dtype = np.float64))
    MasterSurfNodes = GetSurfaceNode(FEMod.cells[ContactPairs.CurMasterSurf[0,i] - 1, :], 
                                                 ContactPairs.CurMasterSurf[1,i] - 1 )
    MasterSurfDOF = get_dofs_given_nodes_ids(MasterSurfNodes)
    CurMasterSurfXYZ = get_deformed_position_given_dofs(MasterSurfNodes, FEMod.X, Disp, MasterSurfDOF)
    PreMasterSurfXYZ = get_deformed_position_given_dofs(MasterSurfNodes, FEMod.X, PreDisp, MasterSurfDOF)

    _, _, Cur_N1Xb, Cur_N2Xb, Cur_x2 = get_surface_geometry(Nb, dNb, CurMasterSurfXYZ)
    _, _, _, _, Pre_x2 = get_surface_geometry(Nb, dNb, PreMasterSurfXYZ)

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
    mc1_bar = np.kron(dNa[0].reshape(1,4), g1_bar_slave.reshape(3,1)) + np.kron(dNa[1].reshape(1,4), g2_bar_slave.reshape(3,1))
    mb2_bar = np.kron(dNb[0].reshape(1,4), g1_bar_master.reshape(3,1)) + np.kron(dNb[1].reshape(1,4), g2_bar_master.reshape(3,1))

    Cur_g1_hat_slave = TransVect2SkewSym(Cur_N1Xa)
    Cur_g2_hat_slave = TransVect2SkewSym(Cur_N2Xa)
    Ac = (np.kron(dNa[0].reshape(1,4), Cur_g2_hat_slave) - np.kron(dNa[1].reshape(1,4), Cur_g1_hat_slave)) / J1

    # N1_wave is a 3x3 matrix: outer(Cur_n, N1_bar @ Cur_n)
    N1_wave = np.outer(Cur_n, N1_bar.dot(Cur_n))

    # Mc1_bar & Mb2_bar arranged as in original code (3 x 12)
    Mc1_bar = np.hstack([np.outer(Cur_n, mc1_bar[:, k]) for k in range(4)])
    Mb2_bar = np.hstack([-np.outer(Cur_n, mb2_bar[:, k]) for k in range(4)])

    # N12 arrays for Gbc: shape (2,4)
    N12a = np.vstack((dNa[0].reshape(1,4), dNa[1].reshape(1,4)))
    N12b = np.vstack((dNb[0].reshape(1,4), dNb[1].reshape(1,4)))
    Gbc = ContactPairs.Cur_g[i] * (N12b.T.dot(a_ab).dot(N12a))   # result 4x4

    # Assemble frictionless stiffness
    FrictionlessK = get_frictionless_K(Na, Nb, ContactPairs.pc[i], tn, Ac, Mc1_bar, Mb2_bar, Gbc, N1, N1_wave, J1)

    return -FrictionlessK, ContactNodeForce, ContactPairs




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


def CalculateContactKandF_stick(FEMod, ContactPairs, Dt, PreDisp, i, Disp, IntegralPoint):
    FricFac = FEMod.FricFac
    
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

    # --- relative sliding vector ---
    gs = Cur_x2_p - Cur_x1
    tv = ContactPairs.pc[i] * gs

    ContactPairs.Pressure[i] = abs(np.dot(tv, Cur_n))
    ContactPairs.Traction[i] = np.linalg.norm(tv)
    
    ContactNodeForce = assemble_contact_force(Na, Nb, J1, tv)   # 1D length 24

    # # --- stiffness ---
    Cur_g1_hat_slave = TransVect2SkewSym(Cur_NXa[0])
    Cur_g2_hat_slave = TransVect2SkewSym(Cur_NXa[1])
    Ac = (np.kron(dNa[0].T, Cur_g2_hat_slave) - np.kron(dNa[1].T, Cur_g1_hat_slave)) / J1
    
    
    Stick_K11 = np.zeros((12,12)); Stick_K12 = np.zeros((12,12))
    Stick_K21 = np.zeros((12,12)); Stick_K22 = np.zeros((12,12))

    # --- Precompute common constants --- 
    scale = ContactPairs.pc[i] * J1
    I3 = np.eye(3)
    
    # --- Loop over local shape functions ---
    for aa in range(4):
        NaA, NbA = Na[aa], Nb[aa]
        idxA = slice(3 * aa, 3 * aa + 3)
    
        for bb in range(4):
            NaB, NbB = Na[bb], Nb[bb]
            idxB = slice(3 * bb, 3 * bb + 3)
    
            # --- Common geometric term ---
            Ac_n = Ac[:, idxB] @ Cur_n
            outer_tv_Acn = np.outer(tv, Ac_n)
    
            Stick_K11[idxA, idxB] += (-NaA * NaB * scale * I3
                                      + NaA * J1 * outer_tv_Acn)
    
            Stick_K12[idxA, idxB] += (NaA * NbB * scale * I3)
    
            Stick_K21[idxA, idxB] += (NbA * NaB * scale * I3
                                      - NbA * J1 * outer_tv_Acn)
    
            Stick_K22[idxA, idxB] += (-NbA * NbB * scale * I3)


    Stick_K = np.block([[Stick_K11, Stick_K12],[Stick_K21, Stick_K22]])
    
    return -Stick_K, ContactNodeForce, ContactPairs 

# Avoid recomputing (done already for frictionless)
def CalculateContactKandF_onlyslip(FEMod, ContactPairs, Dt, PreDisp, i, Disp, IntegralPoint):
    Frictional_K = np.zeros((24,24))
    ContactNodeForce = np.zeros(24)
    FricFac = FEMod.FricFac
    # --- current slave geometry & previous slave geometry ---
    ip_idx = ContactPairs.SlaveIntegralPoint[i] - 1             # MATLAB->Python index
    CurIP = IntegralPoint[ip_idx, :].astype(np.float64)                                 # shape (2,)

    Na, dNa = GetSurfaceShapeFunction(CurIP)
    SlaveSurfNodes = GetSurfaceNode(FEMod.cells[ContactPairs.SlaveSurf[0,i]-1, :], 
                                                ContactPairs.SlaveSurf[1,i]-1)
    SlaveSurfDOF = get_dofs_given_nodes_ids(SlaveSurfNodes)
    CurSlaveSurfXYZ = get_deformed_position_given_dofs(SlaveSurfNodes, FEMod.X, Disp, SlaveSurfDOF)
    PreSlaveSurfXYZ = get_deformed_position_given_dofs(SlaveSurfNodes, FEMod.X, PreDisp, SlaveSurfDOF)
    
    # geometric quantities
    Cur_n, J1, Cur_N1Xa, Cur_N2Xa, Cur_x1 = get_surface_geometry(Na, dNa, CurSlaveSurfXYZ)
    _, _, Pre_N1Xa, Pre_N2Xa, Pre_x1 = get_surface_geometry(Na, dNa, PreSlaveSurfXYZ)

    # normal traction
    tn = ContactPairs.Cur_g[i] * ContactPairs.pc[i]

    dx1 = Cur_x1 - Pre_x1
    PN = np.eye(3) - np.outer(Cur_n, Cur_n)

    dg1_slave = Cur_N1Xa - Pre_N1Xa
    dg2_slave = Cur_N2Xa - Pre_N2Xa
    m1 = np.cross(dg1_slave, Cur_N2Xa) + np.cross(Cur_N1Xa, dg2_slave)
    c1 = PN.dot(m1) / J1

    # --- master geometry at current and previous steps ---
    Nb, dNb = GetSurfaceShapeFunction(np.array((ContactPairs.rc[i], ContactPairs.sc[i]), dtype = np.float64))
    MasterSurfNodes = GetSurfaceNode(FEMod.cells[ContactPairs.CurMasterSurf[0,i] - 1, :], 
                                                 ContactPairs.CurMasterSurf[1,i] - 1 )
    MasterSurfDOF = get_dofs_given_nodes_ids(MasterSurfNodes)
    CurMasterSurfXYZ = get_deformed_position_given_dofs(MasterSurfNodes, FEMod.X, Disp, MasterSurfDOF)
    PreMasterSurfXYZ = get_deformed_position_given_dofs(MasterSurfNodes, FEMod.X, PreDisp, MasterSurfDOF)

    _, _, Cur_N1Xb, Cur_N2Xb, Cur_x2 = get_surface_geometry(Nb, dNb, CurMasterSurfXYZ)
    _, _, _, _, Pre_x2 = get_surface_geometry(Nb, dNb, PreMasterSurfXYZ)

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
    tv = tn * FricFac * s1
    ContactNodeForce = assemble_contact_force(Na, Nb, J1, tv)   # 1D length 24

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
    mc1_bar = np.kron(dNa[0].reshape(1,4), g1_bar_slave.reshape(3,1)) + np.kron(dNa[1].reshape(1,4), g2_bar_slave.reshape(3,1))
    mb2_bar = np.kron(dNb[0].reshape(1,4), g1_bar_master.reshape(3,1)) + np.kron(dNb[1].reshape(1,4), g2_bar_master.reshape(3,1))

    Cur_g1_hat_slave = TransVect2SkewSym(Cur_N1Xa)
    Cur_g2_hat_slave = TransVect2SkewSym(Cur_N2Xa)
    Ac = (np.kron(dNa[0].reshape(1,4), Cur_g2_hat_slave) - np.kron(dNa[1].reshape(1,4), Cur_g1_hat_slave)) / J1

    # N1_wave is a 3x3 matrix: outer(Cur_n, N1_bar @ Cur_n)
    N1_wave = np.outer(Cur_n, N1_bar.dot(Cur_n))

    # Mc1_bar & Mb2_bar arranged as in original code (3 x 12)
    Mc1_bar = np.hstack([np.outer(Cur_n, mc1_bar[:, k]) for k in range(4)])
    Mb2_bar = np.hstack([-np.outer(Cur_n, mb2_bar[:, k]) for k in range(4)])

    # N12 arrays for Gbc: shape (2,4)
    N12a = np.vstack((dNa[0].reshape(1,4), dNa[1].reshape(1,4)))
    N12b = np.vstack((dNb[0].reshape(1,4), dNb[1].reshape(1,4)))
    Gbc = ContactPairs.Cur_g[i] * (N12b.T.dot(a_ab).dot(N12a))   # result 4x4
    
    if FricFac != 0 and np.linalg.norm(s1) > 1e-8:        
        Q1 = (np.dot(Cur_n, m1) * np.eye(3) + np.outer(Cur_n, m1)) / J1
        dg1_hat_slave = TransVect2SkewSym(dg1_slave)
        dg2_hat_slave = TransVect2SkewSym(dg2_slave)
        Ac1_bar = (np.kron(dNa[0].reshape((1,-1)), dg2_hat_slave) - np.kron(dNa[1].reshape((1,-1)), dg1_hat_slave)) / J1
        dh = np.linalg.norm(PN @ vr) * Dt
        Ps = (np.eye(3) - np.outer(s1, s1)) / dh
        R1 = (np.dot(Cur_n,r1) * np.eye(3) + np.outer(Cur_n, r1)) / ContactPairs.Cur_g[i]
        B1 = (Ps @ c1) @ (N1_bar @ Cur_n).T - Ps @ PN;
        L1 = ContactPairs.Cur_g[i] * Ps @ (PN @ Q1 + R1 - np.eye(3)) @ PN 
        Jc1 = L1 @ Ac - ContactPairs.Cur_g[i] * Ps @ PN @ Ac1_bar
        

        hc1_add = N1 @ mc1_bar + Ac @ np.kron(np.eye(4), Cur_n.reshape((-1,1)))
        hc1_sub = N1 @ mc1_bar - Ac @ np.kron(np.eye(4), Cur_n.reshape((-1,1)))
        S1 = np.outer(s1, Cur_n)
        S1_wave = np.outer(s1, N1_bar @ Cur_n)
        
        Frictional_K11 = np.zeros((12, 12))
        Frictional_K12 = np.zeros((12, 12))
        Frictional_K21 = np.zeros((12, 12))
        Frictional_K22 = np.zeros((12, 12))
        
        # --- Precompute common terms ---
        pc_term = ContactPairs.pc[i] * S1_wave + tn * B1   # Common part: pressure + normal term
        g_cur = ContactPairs.Cur_g[i]                      # Current gap scalar
        scale = FricFac * J1                               # Friction factor and Jacobian scaling
        
        # --- Loop over local nodes (4 nodes, 3 dofs each) ---
        for aa in range(4):
            for bb in range(4):
                # --- Local index ranges (Python uses 0-based indexing) ---
                idxA = slice(3 * aa, 3 * aa + 3)
                idxB = slice(3 * bb, 3 * bb + 3)
        
                # --- Frequently reused shape function values ---
                NaA, NaB = Na[aa], Na[bb]
                NbA, NbB = Nb[aa], Nb[bb]
        
                # --- Common sub-blocks for readability ---
                sub_hc = (np.outer(s1, hc1_sub[:, bb])
                          + g_cur * Ps @ np.outer(c1, hc1_add[:, bb])
                          - Jc1[:, idxB])
        
                sub_mb2 = -np.outer(s1, mb2_bar[:, aa])
        
                Frictional_K11[idxA, idxB] += scale * (-NaA * NaB * pc_term - NaA * tn * sub_hc)
                Frictional_K12[idxA, idxB] += scale * (NaA * NbB * pc_term)
                Frictional_K21[idxA, idxB] += scale * (NbA * NaB * pc_term
                                                       + NbA * tn * sub_hc
                                                       + NaB * tn * sub_mb2
                                                       + Gbc[aa, bb] * tn * S1)
                Frictional_K22[idxA, idxB] += scale * (-NbA * NbB * pc_term - NbB * tn * sub_mb2)
                
        
        Frictional_K = np.block([ [Frictional_K11, Frictional_K12],
                                  [Frictional_K21, Frictional_K22]])

    return -Frictional_K , ContactNodeForce, ContactPairs



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
