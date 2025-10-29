#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 20:20:32 2025

@author: frocha
"""


import numpy as np
from utils import (GetSurfaceNode, get_dofs_given_nodes_ids, get_deformed_position_given_dofs, 
                   GetSurfaceShapeFunction, TransVect2SkewSym, GetSurfaceXYZ, get_surface_geometry)
from ray_tracing import GetContactPointbyRayTracing
from scipy.spatial import cKDTree
from contact_lib import *
from oct2py import octave
import copy

octave.addpath(octave.genpath("/home/frocha/sources/pyola_contact2/src/matlab/"))  # doctest: +SKIP

def decide_stick_slip(FEMod, ContactPairs, Disp, PreDisp, i, IP, FricFac, SlaveSurfXYZ, SlavePoint, SlavePointsFrame):
        # --- Case 2: possible stick/slip contact ---
        # ensure integer index when indexing IntegralPoint (ContactPairs stores 1..4)
        ip_idx = ContactPairs.SlaveIntegralPoint[i].astype(np.int64) - 1
        CurIP = IP[ip_idx]        # shape (2,)
        Na, dNa = GetSurfaceShapeFunction(CurIP)

        # Slave surface coordinates (ensure integer surf spec [element, face])
        # slave_surf = ContactPairs.SlaveSurf[:, i] - 1
        
        # SlaveSurfNodes = GetSurfaceNode(slave_surf[0], slave_surf[1])
        # SlaveSurfDOF = get_dofs_given_nodes_ids(SlaveSurfNodes)
        # CurSlaveSurfXYZ = get_deformed_position_given_dofs(SlaveSurfNodes, FEMod.X, Disp, SlaveSurfDOF)
        # PreSlaveSurfXYZ = get_deformed_position_given_dofs(SlaveSurfNodes, FEMod.X, PreDisp, SlaveSurfDOF)
        
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
        ContactPairs.CurContactState[i] = 1 if fai < 0 else 2


# Chatgpt's second try
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
            

            # Residual = Residual.flatten()
            # CalculateFrictionlessContactKandF(
            #     # FEMod, ContactPairs, Dt, PreDisp, i, GKF, Residual, Disp.reshape((-1,1)), IntegralPoint) 
            continue

        decide_stick_slip(FEMod, ContactPairs, Disp, PreDisp, i, IntegralPoint, FricFac, 
                          SlaveSurfXYZ[int(i/4)], SlavePoints[i], SlavePointsFrame[i])

        # Update stiffness and force
        # i should start in 1 to be compatible with octave
        # GKF, Residual, ContactPairs = octave.CalculateContactKandF(
        #     FEMod, ContactPairs, Dt, PreDisp, i+1, GKF, Residual, Disp, IntegralPoint, nout = 3)
        # flattenising_struct(ContactPairs)
        
        GKF, Residual, ContactPairs = CalculateContactKandF(
            FEMod, ContactPairs, Dt, PreDisp, i, GKF, Residual, Disp.reshape((-1,1)), IntegralPoint) 
        
        # Residual = Residual.flatten()
        # ContactPairs = flatteningContactPairs(ContactPairs)
        
    return ContactPairs, GKF, Residual

def CalculateContactKandF_stick(FEMod, ContactPairs, Dt, PreDisp, i, GKF, Residual, Disp, IntegralPoint):
    FricFac = FEMod.FricFac
    
    # --- slave geometry at current IP ---
    ip_idx = int(ContactPairs.SlaveIntegralPoint[i]) - 1
    CurIP = IntegralPoint[ip_idx, :]
    Na, dNa = GetSurfaceShapeFunction(CurIP)
    CurSlaveSurfXYZ, SlaveSurfDOF = GetSurfaceXYZ(FEMod.cells, FEMod.X, Disp, ContactPairs.SlaveSurf[:, i])

    Cur_x1 = CurSlaveSurfXYZ.T @ Na
    Cur_NXa = dNa @ CurSlaveSurfXYZ

    Cur_n = np.cross(Cur_NXa[0], Cur_NXa[1])
    Cur_n /= np.linalg.norm(Cur_n)
    J1 = np.linalg.norm(np.cross(Cur_NXa[0], Cur_NXa[1]))

    # --- master geometry ---
    Nb, _, _ = GetSurfaceShapeFunction(ContactPairs.rp[i], ContactPairs.sc[i])
    CurMasterSurfXYZ_rpsp, MasterSurfDOF = GetSurfaceXYZ(FEMod.cells, FEMod.X, Disp, ContactPairs.PreMasterSurf[:, i])
    Cur_x2_p = Nb @ CurMasterSurfXYZ_rpsp

    # --- relative sliding vector ---
    gs = Cur_x2_p - Cur_x1
    tv = ContactPairs.pc[i] * gs

    ContactPairs.Pressure[i] = abs(np.dot(tv, Cur_n))
    ContactPairs.Traction[i] = np.linalg.norm(tv)

    # --- assemble nodal force ---
    ContactNodeForce = np.zeros(24)
    ContactDOF = np.concatenate([SlaveSurfDOF, MasterSurfDOF])
    for a in range(4):
        idxA = slice(3*a,3*a+3)
        ContactNodeForce[idxA] = Na[a] * J1 * tv
        ContactNodeForce[idxA.start+12:idxA.stop+12] = -Nb[a] * J1 * tv
        
    Residual[ContactDOF] += ContactNodeForce[:]

    # --- stiffness ---
    Cur_g1_hat_slave = TransVect2SkewSym(Cur_NXa[0])
    Cur_g2_hat_slave = TransVect2SkewSym(Cur_NXa[1])
    Ac = (np.kron(dNa[0].T, Cur_g2_hat_slave) - np.kron(dNa[1].T, Cur_g1_hat_slave)) / J1

    Stick_K11 = np.zeros((12,12)); Stick_K12 = np.zeros((12,12))
    Stick_K21 = np.zeros((12,12)); Stick_K22 = np.zeros((12,12))

    for aa in range(4):
        idxA = slice(3*aa,3*aa+3)
        for bb in range(4):
            idxB = slice(3*bb,3*bb+3)

            tempK11 = (-Na[aa]*Na[bb]*ContactPairs.pc[i]*np.eye(3)
                       - Na[aa]*(-tv @ (Ac[:, idxB] @ Cur_n).T)) * J1
            Stick_K11[idxA, idxB] += tempK11

            tempK12 = (Na[aa]*Nb[bb]*ContactPairs.pc[i]*np.eye(3)) * J1
            Stick_K12[idxA, idxB] += tempK12

            tempK21 = (Nb[aa]*Na[bb]*ContactPairs.pc[i]*np.eye(3)
                       + Nb[aa]*(-tv @ (Ac[:, idxB] @ Cur_n).T)) * J1
            Stick_K21[idxA, idxB] += tempK21

            tempK22 = (-Nb[aa]*Nb[bb]*ContactPairs.pc[i]*np.eye(3)) * J1
            Stick_K22[idxA, idxB] += tempK22

    Stick_K = np.block([[Stick_K11, Stick_K12],[Stick_K21, Stick_K22]])
    GKF[np.ix_(ContactDOF, ContactDOF)] -= Stick_K
    
    return GKF, Residual, ContactPairs 

def CalculateContactKandF_onlyslip2(FEMod, ContactPairs, Dt, PreDisp, i, Disp, IntegralPoint):
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
        # Q1 = ((Cur_n' * m1) * eye(3) + Cur_n * m1') / J1;
        # dg1_hat_slave = TransVect2SkewSym(dg1_slave);
        # dg2_hat_slave = TransVect2SkewSym(dg2_slave);
        # Ac1_bar = (kron(N1a', dg2_hat_slave) - kron(N2a', dg1_hat_slave)) / J1;
        
        # dh = sqrt((PN * vr)' * (PN * vr)) * Dt;
        # Ps = (eye(3) - s1 * s1') / dh;
        
        # R1 = ((Cur_n' * r1) * eye(3) + Cur_n * r1') / ContactPairs.Cur_g(i);
        # B1 = (Ps * c1) * (N1_bar * Cur_n)' - Ps * PN;
        # L1 = ContactPairs.Cur_g(i) * Ps * (PN * Q1 + R1 - eye(3)) * PN;
        
        # Jc1 = L1 * Ac - ContactPairs.Cur_g(i) * Ps * PN * Ac1_bar;
        
        # hc1_add = N1 * mc1_bar + Ac * kron(eye(4), Cur_n);
        # hc1_sub = N1 * mc1_bar - Ac * kron(eye(4), Cur_n);
        
        # S1 = s1 * Cur_n';
        # S1_wave = s1 * (N1_bar * Cur_n)';
        
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
        S1_wave = s1 @ (N1_bar @ Cur_n).T
        
        Frictional_K11 = np.zeros((12, 12))
        Frictional_K12 = np.zeros((12, 12))
        Frictional_K21 = np.zeros((12, 12))
        Frictional_K22 = np.zeros((12, 12))
        
        for aa in range(4):
            for bb in range(4):
                idxA = slice(3*aa, 3*aa + 3)
                idxB = slice(3*bb, 3*bb + 3)
        
                tempK = (
                    -Na[aa] * Na[bb] * FricFac * (ContactPairs.pc[i] * S1_wave + tn * B1)
                    - Na[aa] * FricFac * tn * (
                        s1 @ hc1_sub[:, bb].T
                        + ContactPairs.Cur_g[i] * Ps * c1 @ hc1_add[:, bb].T
                        - Jc1[:, idxB]
                    )
                ) * J1
                Frictional_K11[idxA, idxB] += tempK
        
                tempK = (Na[aa] * Nb[bb] * FricFac * (ContactPairs.pc[i] * S1_wave + tn * B1)) * J1
                Frictional_K12[idxA, idxB] += tempK
        
                tempK = (
                    Nb[aa] * Na[bb] * FricFac * (ContactPairs.pc[i] * S1_wave + tn * B1)
                    + Nb[aa] * FricFac * tn * (
                        s1 @ hc1_sub[:, bb].T
                        + ContactPairs.Cur_g[i] * Ps * c1 @ hc1_add[:, bb].T
                        - Jc1[:, idxB]
                    )
                    + Na[bb] * FricFac * tn * (-s1 @ mb2_bar[:, aa].T)
                    + Gbc[aa, bb] * FricFac * tn * S1
                ) * J1
                Frictional_K21[idxA, idxB] += tempK
        
                tempK = (
                    -Nb[aa] * Nb[bb] * FricFac * (ContactPairs.pc[i] * S1_wave + tn * B1)
                    - Nb[bb] * FricFac * tn * (-s1 @ mb2_bar[:, aa].T)
                ) * J1
                Frictional_K22[idxA, idxB] += tempK
        
    
    Frictional_K = np.block([ [Frictional_K11, Frictional_K12],
                              [Frictional_K21, Frictional_K22]])

    return -Frictional_K , ContactNodeForce, ContactPairs


    
    
def CalculateContactKandF_onlyslip(FEMod, ContactPairs, Dt, PreDisp, i, Disp, IntegralPoint):
    Frictional_K = np.zeros((24,24))
    ContactNodeForce = np.zeros(24)
    FricFac = FEMod.FricFac
    tn = ContactPairs.Cur_g[i] * ContactPairs.pc[i]

    # --- slave geometry current and previous ---
    ip_idx = int(ContactPairs.SlaveIntegralPoint[i]) - 1
    CurIP = IntegralPoint[ip_idx, :]
    Na, dNa = GetSurfaceShapeFunction(CurIP)
    CurSlaveSurfXYZ, SlaveSurfDOF = GetSurfaceXYZ(FEMod.cells, FEMod.X, Disp, ContactPairs.SlaveSurf[:, i] - 1)
    PreSlaveSurfNodeXYZ, _ = GetSurfaceXYZ(FEMod.cells, FEMod.X, PreDisp, ContactPairs.SlaveSurf[:, i] - 1)

    Cur_x1 = CurSlaveSurfXYZ.T @ Na
    Pre_x1 = PreSlaveSurfNodeXYZ.T @ Na
    dx1 = Cur_x1 - Pre_x1

    Pre_NXa = dNa @ PreSlaveSurfNodeXYZ
    Cur_NXa = dNa @ CurSlaveSurfXYZ

    Cur_n = np.cross(Cur_NXa[0], Cur_NXa[1])
    Cur_n /= np.linalg.norm(Cur_n)
    J1 = np.linalg.norm(np.cross(Cur_NXa[0], Cur_NXa[1]))
    PN = np.eye(3) - np.outer(Cur_n, Cur_n)

    dg1_slave = Cur_NXa[0] - Pre_NXa[0]
    dg2_slave = Cur_NXa[1] - Pre_NXa[1]
    m1 = np.cross(dg1_slave, Cur_NXa[1]) + np.cross(Cur_NXa[0], dg2_slave)
    c1 = PN @ m1 / J1

#     # --- master geometry ---
    Nb, dNb = GetSurfaceShapeFunction(np.array([ContactPairs.rc[i], ContactPairs.sc[i]]))
    CurMasterSurfNodeXYZ, MasterSurfDOF = GetSurfaceXYZ(FEMod.cells, FEMod.X, Disp, ContactPairs.CurMasterSurf[:, i] - 1)
    PreMasterSurfNodeXYZ, _ = GetSurfaceXYZ(FEMod.cells, FEMod.X, PreDisp, ContactPairs.CurMasterSurf[:, i] - 1)
    
    Cur_x2 = CurMasterSurfNodeXYZ.T@Nb  # shape (3,)
    Pre_x2 = PreMasterSurfNodeXYZ.T@Nb  # shape (3,)
    Cur_NXb = dNb@CurMasterSurfNodeXYZ  # shape (3,)
#        Cur_NXa = dNb@CurSlaveSurfXYZ # shape (3,)

    dx2 = Cur_x2 - Pre_x2

#     # --- tangential sliding ---
    r1 = ContactPairs.Cur_g[i] * c1 + dx1 - dx2
    vr = r1 / Dt
    s1_temp = PN @ vr
    s1 = s1_temp / np.linalg.norm(s1_temp) if np.linalg.norm(s1_temp) > 1e-8 else np.zeros(3)

    # --- contact nodal force ---

    tv = tn * FricFac * s1
    ContactPairs.Pressure[i] = abs(tn)
    ContactPairs.Traction[i] = np.linalg.norm(tv)

    for a in range(4):
        idxA = slice(3*a, 3*a+3)
        ContactNodeForce[idxA] = Na[a] * tv * J1
        ContactNodeForce[idxA.start+12:idxA.stop+12] = -Nb[a] * tv * J1

    # --- projection matrices ---
    A_ab = np.array([[Cur_NXa[0] @ Cur_NXa[0], Cur_NXa[0] @ Cur_NXa[1]],
                     [Cur_NXa[1] @ Cur_NXa[0], Cur_NXa[1] @ Cur_NXa[1]]])
    a_ab = np.linalg.inv(A_ab)

    g1_bar_slave  = a_ab[0,0]*Cur_NXa[0] + a_ab[1,0]*Cur_NXa[1]
    g2_bar_slave  = a_ab[0,1]*Cur_NXa[0] + a_ab[1,1]*Cur_NXa[1]
    # g1_bar_master = a_ab[0,0]*Cur_NXa[0] + a_ab[0,1]*Cur_NXa[1]
    # g2_bar_master = a_ab[1,0]*Cur_NXa[0] + a_ab[1,1]*Cur_NXa[1]

    N1_bar = np.eye(3) - np.outer(Cur_NXa[0], g1_bar_slave) - np.outer(Cur_NXa[1], g2_bar_slave)

    # --- Ac, mc1_bar, mb2_bar, Gbc ---
    Cur_g1_hat_slave = TransVect2SkewSym(Cur_NXa[0])
    Cur_g2_hat_slave = TransVect2SkewSym(Cur_NXa[1])
    Ac = (np.kron(dNa[0][:, None].T, Cur_g2_hat_slave) - np.kron(dNa[1][:, None].T, Cur_g1_hat_slave)) / J1
    
    # Felipe
    mc1_bar = a_ab[0,0] * Cur_NXb[0] + a_ab[1,0] * Cur_NXb[1]  # shape (3,)
    mb2_bar = a_ab[0,0] * Cur_NXa[0] + a_ab[0,1] * Cur_NXa[1]  # shape (3,) # why not [1,1]?
    
    # Mc1_bar = [ Cur_n * mc1_bar(:,1)' , Cur_n * mc1_bar(:,2)' , Cur_n * mc1_bar(:,3)' , Cur_n * mc1_bar(:,4)' ];
    # Mb2_bar = [ -Cur_n * mb2_bar(:,1)' , -Cur_n * mb2_bar(:,2)' , -Cur_n * mb2_bar(:,3)' , -Cur_n * mb2_bar(:,4)' ];
    # Mc1_bar = np.hstack([np.outer(Cur_n, mc1_bar[:, i]) for i in range(4)])
    # Mb2_bar = np.hstack([-np.outer(Cur_n, mb2_bar[:, i]) for i in range(4)])
                                   
    Gbc = ContactPairs.Cur_g[i] * (dNb.T @ a_ab @ dNa)  
    
    # # --- Frictional K additions ---
    
    if FricFac != 0 and np.linalg.norm(s1) > 1e-8:
        Q1 = ((Cur_n @ m1) * np.eye(3) + np.outer(Cur_n, m1)) / J1
        dh = np.linalg.norm(PN @ vr) * Dt
        Ps = (np.eye(3) - np.outer(s1, s1)) / dh
        R1 = ((Cur_n @ r1) * np.eye(3) + np.outer(Cur_n, r1)) / ContactPairs.Cur_g[i]
        B1 = (Ps @ c1) @ (N1_bar @ Cur_n).T - Ps @ PN;
        L1 = ContactPairs.Cur_g[i] * Ps @ (PN @ Q1 + R1 - np.eye(3)) @ PN
        Jc1 = L1 @ Ac - ContactPairs.Cur_g[i] * Ps @ PN @ (np.kron(dNa[0].T, TransVect2SkewSym(dg2_slave)) - np.kron(dNa[1].T, TransVect2SkewSym(dg1_slave))) / J1

        hc1_add = dNa @ mc1_bar + Ac @ np.kron(np.eye(4), Cur_n)
        hc1_sub = dNa @ mc1_bar - Ac @ np.kron(np.eye(4), Cur_n)
        S1_wave = s1 @ (N1_bar @ Cur_n).T
        S1 = np.outer(s1, Cur_n)
        
        Frictional_K11 = np.zeros((12, 12))
        Frictional_K12 = np.zeros((12, 12))
        Frictional_K21 = np.zeros((12, 12))
        Frictional_K22 = np.zeros((12, 12))

        
        for aa in range(4):
            for bb in range(4):
                idxA = slice(3*aa, 3*aa + 3)
                idxB = slice(3*bb, 3*bb + 3)
        
                tempK = (
                    -Na[aa] * Na[bb] * FricFac * (ContactPairs.pc[i] * S1_wave + tn * B1)
                    - Na[aa] * FricFac * tn * (
                        s1 @ hc1_sub[:, bb].T
                        + ContactPairs.Cur_g[i] * Ps * c1 @ hc1_add[:, bb].T
                        - Jc1[:, idxB]
                    )
                ) * J1
                Frictional_K11[idxA, idxB] += tempK
        
                tempK = (Na[aa] * Nb[bb] * FricFac * (ContactPairs.pc[i] * S1_wave + tn * B1)) * J1
                Frictional_K12[idxA, idxB] += tempK
        
                tempK = (
                    Nb[aa] * Na[bb] * FricFac * (ContactPairs.pc[i] * S1_wave + tn * B1)
                    + Nb[aa] * FricFac * tn * (
                        s1 @ hc1_sub[:, bb].T
                        + ContactPairs.Cur_g[i] * Ps * c1 @ hc1_add[:, bb].T
                        - Jc1[:, idxB]
                    )
                    + Na[bb] * FricFac * tn * (-s1 @ mb2_bar[:, aa].T)
                    + Gbc[aa, bb] * FricFac * tn * S1
                ) * J1
                Frictional_K21[idxA, idxB] += tempK
        
                tempK = (
                    -Nb[aa] * Nb[bb] * FricFac * (ContactPairs.pc[i] * S1_wave + tn * B1)
                    - Nb[bb] * FricFac * tn * (-s1 @ mb2_bar[:, aa].T)
                ) * J1
                Frictional_K22[idxA, idxB] += tempK
        
    
    Frictional_K = np.block([ [Frictional_K11, Frictional_K12],
                              [Frictional_K21, Frictional_K22]])

    return -Frictional_K , ContactNodeForce, ContactPairs

def CalculateContactKandF(FEMod, ContactPairs, Dt, PreDisp, i, GKF, Residual, Disp, IntegralPoint):
    FricFac = FEMod.FricFac
    
    if ContactPairs.CurContactState[i] == 1:  # Stick contact
    
        # print(Residual.shape)
        # Residual_ = copy.deepcopy(Residual.reshape((-1,1), order = 'F'))
        # print(Residual_.shape)

        # GKF, Residual, ContactPairs = CalculateContactKandF_stick(FEMod, ContactPairs, Dt, PreDisp, i, GKF, Residual, Disp, IntegralPoint)
        KL, ResL, ContactPairs = octave.CalculateContactKandF_stick2(FEMod, ContactPairs, Dt, PreDisp.reshape((-1,1)), i+1, 
                                                                     Disp.reshape((-1,1)), IntegralPoint, nout = 3)
        ResL = ResL.flatten()
    elif ContactPairs.CurContactState[i] == 2:  # Slip contact
        KL, ResL, ContactPairs = CalculateFrictionlessContactKandF(FEMod, ContactPairs, Dt, PreDisp, i, Disp, IntegralPoint)
        KL3, ResL3, ContactPairs = CalculateContactKandF_onlyslip2(FEMod, ContactPairs, Dt, PreDisp, i, 
                                                                    Disp, IntegralPoint)
        #if FricFac != 0:
        # GKF, Residual, ContactPairs = CalculateContactKandF_slip(FEMod, ContactPairs, Dt, PreDisp, i, GKF, Residual, Disp, IntegralPoint)
        KL2, ResL2, ContactPairs = octave.CalculateContactKandF_onlyslip(FEMod, ContactPairs, Dt, PreDisp.reshape((-1,1)), i+1, 
                                                                    Disp.reshape((-1,1)), IntegralPoint, nout = 3)
        
        KL += KL2
        ResL = ResL.flatten() + ResL2.flatten()
        # print(np.linalg.norm(ResL2))
        # print(np.linalg.norm(ResL3))
        print(KL2[0,0], np.linalg.norm(KL2))
        print(KL3[0,0], np.linalg.norm(KL3))
        # ResL = ResL.flatten()
        
    
    ContactPairs = flatteningContactPairs(ContactPairs)
    
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
