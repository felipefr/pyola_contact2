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
    
    idx_active = ContactSearch(FEMod, ContactPairs, Disp.reshape((-1,1)))

    FricFac = ContactPairs.FricFac
    
    # --- Loop over contact pairs
    for i in idx_active:
        sp = ContactPairs.slave_points[i]

        # Case 1: first contact or frictionless contact
        if (FricFac == 0) or (sp.master_surf_idx_old == -1): 
            sp.contact_state = 2  # Slip
        else:
            sp.contact_state = decide_stick_slip(sp, FricFac)
            
        if(sp.contact_state == 1): # stick
            KL, ResL, ContactPairs = CalculateContactKandF_stick(sp, FEMod, ContactPairs)
        
        elif(sp.contact_state == 2): # slip        
            sp.update_old(FEMod, PreDisp)
            KL, ResL, ContactPairs = CalculateContactKandF_slip2(sp, FEMod, ContactPairs, Dt)  

        dofs = sp.get_contact_dofs()
        Residual[dofs] += ResL
        GKF[np.ix_(dofs, dofs)] += KL
            
    return ContactPairs, GKF, Residual

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

# original version (Frictionless + Friction)
def CalculateContactKandF_slip(sp, FEMod, ContactPairs, Dt):
    """
    Python translation of the MATLAB CalculateFrictionlessContactKandF.
    - i is a zero-based index into ContactPairs (Python convention).
    - ContactPairs fields are numpy arrays (struct-like object from Oct2Py or dict-like).
    """
    
    FricFac = ContactPairs.FricFac
    I3 = np.eye(3)
    
    Na, dNa = FEMod.ShpfSurf[sp.idxIP]
    Cur_x1 = sp.point
    Cur_NXa = sp.frame[1:3]
    Cur_N1Xa, Cur_N2Xa = Cur_NXa
    Cur_n = sp.frame[0]
    J1 = sp.J 
    Pre_N1Xa, Pre_N2Xa, Pre_x1 = sp.frame_old[1], sp.frame_old[2], sp.point_old
    
    # --- master geometry at current geo, with current and previous steps ---
    Cur_x2 = sp.master_point
    Nb = sp.Nb
    dNb = sp.dNb
    Cur_NXb = sp.master_tangent
    Cur_x2 = sp.master_point
    Pre_x2 = sp.master_point_old


    # projection at slave
    dx1 = Cur_x1 - Pre_x1
    PN = I3 - np.outer(Cur_n, Cur_n)

    dg1_slave = Cur_N1Xa - Pre_N1Xa
    dg2_slave = Cur_N2Xa - Pre_N2Xa
    m1 = np.cross(dg1_slave, Cur_N2Xa) + np.cross(Cur_N1Xa, dg2_slave)
    c1 = (PN @ m1) / J1

    # --- master geometry at current and previous steps ---
    dx2 = Cur_x2 - Pre_x2

    # --- precompute projection matrices and related arrays ---    
    # Compute the 2x2 coupling matrix
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
    Gbc = sp.gap * (dNb.T @ a_ab @ dNa)   # result 4x4

    # --- relative velocity and tangential direction ---
    r1 = sp.gap * c1 + dx1 - dx2
    vr = r1 / Dt
    s1_temp = PN.dot(vr)

    if np.linalg.norm(s1_temp) > 1e-8:
        s1 = s1_temp / np.linalg.norm(s1_temp)
    else:
        s1 = np.zeros(3)
        dh = 0.0  # not used further here (kept for parity)

    # --- contact nodal force (frictionless baseline uses tv = tn * Cur_n) --
    # normal traction
    tn = sp.penc * sp.gap 
    tv = tn * Cur_n
    
    # Assemble frictionless stiffness
    KL = -get_frictionless_K(Na, Nb, sp.penc, tn, Ac, Mc1_bar, Mb2_bar, Gbc, N1, N1_wave, J1)
    
    if FricFac != 0 and np.linalg.norm(s1) > 1e-8:        
        tv += tn* FricFac * s1
        FrictionalK = get_frictional_K_slip(Na, Nb, sp.penc, tn, Ac, Mc1_bar, 
                                       Mb2_bar, Gbc, N1, N1_wave, J1, Cur_n, m1, 
                                       dg1_slave, dg2_slave, dNa, PN, vr, Dt, s1, r1, 
                                       sp.gap, FricFac, c1, N1_bar, mc1_bar, mb2_bar)
    
        
        KL -= FrictionalK   

    ContactNodeForce = assemble_contact_force(Na, Nb, J1, tv)   # 1D length 24
    sp.pressure  = abs(tn)
    sp.traction = np.linalg.norm(tv)
    
    return KL, ContactNodeForce, ContactPairs

def get_Ac_tilde(dNa, tau1, J1):
    I3 = np.eye(3)
    Cur_g1_hat_slave = TransVect2SkewSym(tau1[0])
    Cur_g2_hat_slave = TransVect2SkewSym(tau1[1])
    Ac = (Cur_g1_hat_slave @ np.kron(dNa[1], I3) - Cur_g2_hat_slave @ np.kron(dNa[0], I3))/J1 # different by a minus sign
    return np.block([Ac, np.zeros((3,12))])


def get_frictionless_aux_operators(Na, Nb, dNa, dNb, tau1, tau2, Cur_n, J1, gap):
    I3 = np.eye(3)
    Ngap = np.block([-np.kron(Na, I3), np.kron(Nb, I3)])
    
    # tau_i = Bi@(XL+ULi).ravel()
    B1 = np.kron(dNa, I3).reshape((2,3,12)) # derivative x spatial component x (node x spatial component)
    B2 = np.kron(dNb, I3).reshape((2,3,12)) # derivative x spatial component x (node x spatial component)
    
    B1tilde = np.concatenate((B1, np.zeros_like(B1)), axis=2)
    B2tilde = np.concatenate((np.zeros_like(B2), B2), axis=2)

    Ginv = np.linalg.inv(tau1 @ tau2.T)  # inverse
    Deta = - Ginv @ ( tau1@Ngap + gap*np.einsum('j, ijk -> ik', Cur_n, B1tilde))
    Ngap_star = Ngap + tau2.T@Deta
    
    # projection at slave
    PN = I3 - np.outer(Cur_n, Cur_n)
    Ac_tilde = get_Ac_tilde(dNa, tau1, J1)

    return Ngap, Ngap_star, Ac_tilde, Deta, B2tilde, PN   

# Cleanear implementation : Frictionless version (working)
def CalculateContactKandF_slip2(sp, FEMod, ContactPairs, Dt):
    gap = sp.gap
    eps = sp.penc
    
    Na, dNa = FEMod.ShpfSurf[sp.idxIP]
    tau1 = sp.frame[1:3]
    Cur_n = sp.frame[0]
    J1 = sp.J 
    
    # --- master geometry at current geo, with current and previous steps ---
    Nb = sp.Nb
    dNb = sp.dNb
    tau2 = sp.master_tangent
    
    tn = eps * gap 
    tv = tn * Cur_n
    
    Ngap, Ngap_star, Ac_tilde, Deta, B2tilde, PN = get_frictionless_aux_operators(Na, Nb, dNa, dNb, tau1, tau2, Cur_n, J1, gap)
    
    Dn = PN@Ac_tilde # [PN@Ac, 0]  
    Dtn_n = eps*np.outer(Cur_n,Cur_n) @ ( Ngap_star + gap*Dn) # Dtn[n] = eps*Dgn[n]
    Dtv = Dtn_n + tn*Dn # Dtn_n term is null if tn is indepent of u (e.g. Aug. Lagrangian formulations)
    DJ1 = J1*np.outer(tv,Cur_n)@Ac_tilde
    KL = Ngap.T@(J1*Dtv + DJ1)
    KL += np.einsum('i, jik -> kj', J1*tv, B2tilde)@Deta # virtual gap tangent term
        
    ContactNodeForce = -J1*Ngap.T@tv # the residual is perfect
    
    sp.pressure  = abs(tn)
    sp.traction = np.linalg.norm(tv)
    return KL, ContactNodeForce, ContactPairs

# Helper functions for slip3
def getQ(w,n):
    return -np.outer(n, w) - np.dot(n, w)*np.eye(3)

def getA(tau, dN, J):
    I3 = np.eye(3)
    tau1_hat = TransVect2SkewSym(tau[0])
    tau2_hat = TransVect2SkewSym(tau[1])
    A = (tau1_hat @ np.kron(dN[1], I3) - tau2_hat @ np.kron(dN[0], I3))/J
    return A

def get_slip(dn, gap, dx1, dx2, Dt, PN):
    I3 = np.eye(3)
    r1 = gap * dn + dx1 - dx2
    vr = r1 / Dt
    PNvr = PN.dot(vr)
    PNvr_norm = np.linalg.norm(PNvr)
    
    if PNvr_norm > 1e-8:
        s1 = PNvr / PNvr_norm
        Ps = (I3 - np.outer(s1,s1))/PNvr_norm
    else:
        Ps = I3
        s1 = np.zeros(3)
        dh = 0.0  # not used further here (kept for parity)

    return vr, s1, Ps

# cleaner version Frictionless + Friction (not working Friction)
def CalculateContactKandF_slip3(sp, FEMod, ContactPairs, Dt):
    
    FricFac = ContactPairs.FricFac
    I3 = np.eye(3)
    
    Na, dNa = FEMod.ShpfSurf[sp.idxIP]
    Cur_x1 = sp.point
    tau1 = sp.frame[1:3]
    Cur_n = sp.frame[0]
    J1 = sp.J 
    gap = sp.gap
    eps = sp.penc
    
    Pre_N1Xa, Pre_N2Xa, Pre_x1 = sp.frame_old[1], sp.frame_old[2], sp.point_old
    
    # --- master geometry at current geo, with current and previous steps ---
    Cur_x2 = sp.master_point
    Nb = sp.Nb
    dNb = sp.dNb
    tau2 = sp.master_tangent
    Cur_x2 = sp.master_point
    Pre_x2 = sp.master_point_old
    
    Ngap = np.block([-np.kron(Na, I3), np.kron(Nb, I3)])
    
    # i : spatial dimension derivative
    # j : nodes
    # k : spatial dimension component
    B1 = np.kron(dNa, I3).reshape((2,3,12))
    B2 = np.kron(dNb, I3).reshape((2,3,12))
    
    B1tilde = np.zeros((2,3,24))
    B1tilde[:,:,0:12] = B1
    
    B2tilde = np.zeros((2,3,24))
    B2tilde[:,:,12:] = B2
    
    Ginv = np.linalg.inv(tau1 @ tau2.T)  # inverse
    Deta = - Ginv @ ( tau1@Ngap + gap*np.einsum('j, ijk -> ik', Cur_n, B1tilde))
    Ngap_star = Ngap + tau2.T@Deta
    
    
    # projection at slave
    PN = I3 - np.outer(Cur_n, Cur_n)
    Ac = getA(tau1, dNa, J1)
    PNAc_tilde = np.block([PN@Ac, np.zeros((3,12))])  
    Ac_tilde = np.block([Ac, np.zeros((3,12))])
    Dgap = Ngap_star + gap*PNAc_tilde  # operator relatated to the variation of gap d(gap*w) = outer(w,n)@Dgap
    
    Dtn = eps*( gap*(np.outer(Cur_n,Cur_n) + I3)@PNAc_tilde + np.outer(Cur_n,Cur_n)@Ngap_star)
    
    # Frictionless stiffness
    KL = J1*Ngap.T@Dtn
    
    
    # increments
    dx1 = Cur_x1 - Pre_x1
    dx2 = Cur_x2 - Pre_x2
    
    # try with PN_old (staggered, so no need to linearise)
    dtau1 = tau1 - np.vstack((Pre_N1Xa, Pre_N2Xa))
    m1 = np.cross(dtau1[0], tau1[1]) + np.cross(tau1[0], dtau1[1])
    dn = (PN @ m1) / J1 
    
    # --- relative velocity and tangential direction ---
    vr, s, Ps = get_slip(dn, gap, dx1, dx2, Dt, PN)

    
    dAc = getA(dtau1, dNa, J1)
    dAc_tilde = np.block([dAc, np.zeros((3,12))])  
    
    AcDu = m1/J1 # A[Du]
    QAcDu = getQ(AcDu, Cur_n)
    # Ddn = QAcDu@PNAc_tilde - PN@np.outer(AcDu , Ac_tilde.T@Cur_n) + PN@dAc_tilde + PNAc_tilde 
    Ddn = QAcDu@PNAc_tilde - np.outer(dn , Ac_tilde.T@Cur_n) + PN@(dAc_tilde + Ac_tilde) 

    Dvr = np.outer(dn, Cur_n)@Dgap + gap*Ddn - Ngap_star # variation of the gap, dn and dx1-dx2, respectively
    Dvr = Dvr/Dt
    
    Qvr = getQ(vr,Cur_n)
    Ds = Ps @ (Qvr@PNAc_tilde + PN@Dvr)
    
    Dtmu = eps*FricFac*( np.outer(s,Cur_n)@Dgap + gap*Ds )
     
    tn = eps * gap 
    tv = tn * (Cur_n +  FricFac * s)
    
    
    KL += J1*Ngap.T@Dtmu
    KL += J1*Ngap.T@np.outer(tv,Cur_n)@Ac_tilde

    # simpler according to the ansatz of the FEbio paper
    vaux = J1*tv
    Kaux = np.einsum('i, jik -> kj', vaux, B2tilde)@Deta
    KL += Kaux
        
    ContactNodeForce = -J1*Ngap.T@tv # the residual is perfect
    
    sp.pressure  = abs(tn)
    sp.traction = np.linalg.norm(tv)
        
    
    return KL, ContactNodeForce, ContactPairs

# Original version
def CalculateContactKandF_stick2(sp, FEMod, ContactPairs):    
    Na, dNa = FEMod.ShpfSurf[sp.idxIP]
    Cur_x1 = sp.point
    Cur_NXa = sp.frame[1:3]
    Cur_n = sp.frame[0]
    J1 = sp.J 
    Cur_x2_p = sp.master_point_oldgeo
    Nb = sp.Nb_old
    
    Cur_g1_hat_slave = TransVect2SkewSym(Cur_NXa[0])
    Cur_g2_hat_slave = TransVect2SkewSym(Cur_NXa[1])
    Ac = (np.kron(dNa[0].T, Cur_g2_hat_slave) - np.kron(dNa[1].T, Cur_g1_hat_slave)) / J1
       
    # --- relative sliding vector ---
    gs = Cur_x2_p - Cur_x1
    tv = sp.penc * gs    
    
    ContactNodeForce = assemble_contact_force(Na, Nb, J1, tv)   # 1D length 24

    # # --- stiffness ---
    KL = -get_frictional_K_stick(Na, Nb, sp.penc, J1, tv, Ac, Cur_n)
    
    sp.pressure  = abs(np.dot(tv,Cur_n))
    sp.traction = np.linalg.norm(tv)
    
    return KL, ContactNodeForce, ContactPairs 



# My version (it converges the same but with + for KL, weirdly)
def CalculateContactKandF_stick(sp, FEMod, ContactPairs):  
    I3 = np.eye(3)
    Na, dNa = FEMod.ShpfSurf[sp.idxIP]
    Cur_x1 = sp.point
    Cur_NXa = sp.frame[1:3]
    Cur_n = sp.frame[0]
    J1 = sp.J 
    Cur_x2_p = sp.master_point_oldgeo
    Nb = sp.Nb_old
    
    Ngap = np.block([-np.kron(Na, I3), np.kron(Nb, I3)])
    
    Cur_g1_hat_slave = TransVect2SkewSym(Cur_NXa[0])
    Cur_g2_hat_slave = TransVect2SkewSym(Cur_NXa[1])
    Ac = (Cur_g1_hat_slave @ np.kron(dNa[1], I3) - Cur_g2_hat_slave @ np.kron(dNa[0], I3))/J1 # different by a minus sign
    Atilde = np.block([Ac, np.zeros((3,12))])
    
    # --- relative sliding vector ---
    gs = Cur_x2_p - Cur_x1
    tv = sp.penc * gs    
    ContactNodeForce = -J1*Ngap.T@tv

    # # --- stiffness ---
    KL = sp.penc*J1*(Ngap.T @ Ngap + Ngap.T @ np.outer(gs,Cur_n) @ Atilde) 
    
    sp.pressure  = abs(np.dot(tv,Cur_n))
    sp.traction = np.linalg.norm(tv)
    
    return KL, ContactNodeForce, ContactPairs 


def decide_stick_slip(sp, FricFac):
    # --- Case 2: possible stick/slip contact ---
    # ensure integer index when indexing IntegralPoint (ContactPairs stores 1..4)
    Cur_x1 = sp.point
    Cur_n = sp.frame[0]
    Cur_x2_p = sp.master_point_oldgeo # attention (it was computed in a strange way mixing old shape functions with current deformation)

    # Relative motion and projection
    gs = Cur_x2_p - Cur_x1
    tv = sp.penc * gs

    # Tangential/normal trial components
    tn_trial = abs(np.dot(tv, Cur_n))
    tt_trial = np.sqrt(np.linalg.norm(tv)**2 - tn_trial**2)

    # Slip/stick criterion
    fai = tt_trial - FricFac * tn_trial
    
    return 1 if fai < 0 else 2