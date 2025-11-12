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
            KL, ResL, ContactPairs = CalculateContactKandF_slip3(sp, FEMod, ContactPairs, Dt)   

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

# original version
def CalculateContactKandF_slip2(sp, FEMod, ContactPairs, Dt):
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


# original version
# Frictionless version
def CalculateContactKandF_slip3(sp, FEMod, ContactPairs, Dt):
    """
    Python translation of the MATLAB CalculateFrictionlessContactKandF.
    - i is a zero-based index into ContactPairs (Python convention).
    - ContactPairs fields are numpy arrays (struct-like object from Oct2Py or dict-like).
    """

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
    Cur_g1_hat_slave = TransVect2SkewSym(tau1[0])
    Cur_g2_hat_slave = TransVect2SkewSym(tau1[1])
    Ac = (Cur_g1_hat_slave @ np.kron(dNa[1], I3) - Cur_g2_hat_slave @ np.kron(dNa[0], I3))/J1 # different by a minus sign
    PNAc_tilde = np.block([PN@Ac, np.zeros((3,12))])  
    Ac_tilde = np.block([Ac, np.zeros((3,12))])     
    
    tn = eps * gap 
    tv = tn * Cur_n
    
    Dtv = eps*( gap*(np.outer(Cur_n,Cur_n) + I3)@PNAc_tilde + np.outer(Cur_n,Cur_n)@Ngap_star)
    KL = J1*Ngap.T@Dtv
    KL += J1*Ngap.T@np.outer(tv,Cur_n)@Ac_tilde
    
    # according to my reasoning
    # vaux = J1*tau1@tv
    # Kaux = Ngap_star.T@np.einsum('i, ijk -> jk', vaux, B1tilde)
    # KL += (Kaux + Kaux.T)
    
    # simpler according to the ansatz of the FEbio paper
    vaux = J1*tv
    Kaux = np.einsum('i, jik -> kj', vaux, B2tilde)@Deta
    KL += Kaux
        
    ContactNodeForce = -J1*Ngap.T@tv # the residual is perfect


    sp.pressure  = abs(tn)
    sp.traction = np.linalg.norm(tv)
    return KL, ContactNodeForce, ContactPairs


# My version: still does not converges
# Rtilde has terms with different dimensions
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
    
    tau1 = Cur_NXa.T
    tau2 = Cur_NXb.T

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
    
    Ginv = np.linalg.inv(Cur_NXa @ Cur_NXb.T)  # inverse
    Deta = - Ginv @ ( tau1.T@Ngap + sp.gap*np.einsum('j, ijk -> ik', Cur_n, B1tilde))
    Ngap_star = Ngap + tau2@Deta
    
    # increments
    dx1 = Cur_x1 - Pre_x1
    dx2 = Cur_x2 - Pre_x2
    gvec = Cur_x2 - Cur_x1
    
    # projection at slave
    PN = I3 - np.outer(Cur_n, Cur_n)
    Cur_g1_hat_slave = TransVect2SkewSym(Cur_NXa[0])
    Cur_g2_hat_slave = TransVect2SkewSym(Cur_NXa[1])
    Ac = (Cur_g1_hat_slave @ np.kron(dNa[1], I3) - Cur_g2_hat_slave @ np.kron(dNa[0], I3))/J1 # different by a minus sign
    # Ac = np.block([[-Cur_g2_hat_slave, Cur_g2_hat_slave]]) @ np.kron(dNa, I3)/J1 # different by a minus sign
    PNAc_tilde = np.block([PN@Ac, np.zeros((3,12))])  
    Ac_tilde = np.block([Ac, np.zeros((3,12))])     

    # dg1_slave = Cur_N1Xa - Pre_N1Xa
    # dg2_slave = Cur_N2Xa - Pre_N2Xa
    # m1 = np.cross(dg1_slave, Cur_N2Xa) + np.cross(Cur_N1Xa, dg2_slave)
    # c1 = (PN @ m1) / J1 # variation of the normal? Delta n?
    dn = Cur_n - sp.frame_old[0]
    
    # --- relative velocity and tangential direction ---
    r1 = sp.gap * dn + dx1 - dx2
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

    Qvr = -np.outer(Cur_n, vr) - np.dot(Cur_n, vr)*I3
    Rtilde = (np.outer(dn, Cur_n)@Ngap - I3 @ Ngap_star + (np.outer(dn, gvec) + sp.gap*I3)@PNAc_tilde)/Dt
    Ds = Ps @ (Qvr@PNAc_tilde + Rtilde)
    
    tn = sp.penc * sp.gap 
    tv = tn * (Cur_n +  FricFac * s1)
    
    Dtv = np.outer( sp.penc*tv/tn , Ngap_star.T@Cur_n + PNAc_tilde.T@gvec) + tn*PNAc_tilde + tn*FricFac*Ds 
     
    ContactNodeForce = -J1*Ngap.T@tv # the residual is perfect
    KL = -Ngap.T@ (J1*Dtv + np.outer(tv, Cur_n) @ Ac_tilde) - J1*np.einsum('j, ijk -> ki', tv, B2tilde) @ Deta

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
    KL = sp.penc*J1*(Ngap.T @ Ngap + Ngap.T @ np.outer(gs,Cur_n) @ Atilde) # it should be + to match the original
    
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