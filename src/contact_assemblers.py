#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 12:13:47 2025

@author: felipe
"""
import numpy as np
from numba import njit, float64, int64
from utils import *

@njit(cache=True)
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

@njit(cache=True)
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

    K11 = np.zeros((12, 12))
    K12 = np.zeros((12, 12))
    K21 = np.zeros((12, 12))
    K22 = np.zeros((12, 12))
            
    scale_pc = pc * J1                      # pressure contribution scaled by Jacobian
    scale_tn = tn * J1                      # normal contribution scaled by Jacobian

    for aa in range(4):
        NaA, NbA = Na[aa], Nb[aa]
        idxA = slice(3 * aa, 3 * aa + 3)
    
        for bb in range(4):
            NaB, NbB = Na[bb], Nb[bb]
            idxB = slice(3 * bb, 3 * bb + 3)
    
            AcB = Ac[:, idxB]
            Mc1B = Mc1_bar[:, idxB]
            Mb2A = Mb2_bar[:, idxA]
            term_hc = AcB + Mc1B @ N1
    
            K11[idxA, idxB] += (
                -NaA * NaB * scale_pc * N1_wave
                - NaA * scale_tn * term_hc
            )
    
            K12[idxA, idxB] += (
                NaA * NbB * scale_pc * N1_wave
            )
    
            K21[idxA, idxB] += (
                NbA * NaB * scale_pc * N1_wave
                + NbA * scale_tn * term_hc
                + NaB * scale_tn * Mb2A
                + Gbc[aa, bb] * scale_tn * N1
            )
    
            K22[idxA, idxB] += (
                -NbA * NbB * scale_pc * N1_wave
                - NbB * scale_tn * Mb2A
            )

    # K = np.block([ [K11, K12], [K21, K22]])
    return assemble_block(K11, K12, K21, K22)



@njit(cache=True)
def get_frictional_K_slip(Na, Nb, pc, tn, Ac, Mc1_bar, Mb2_bar, Gbc, N1, N1_wave, J1,
                     Cur_n, m1, dg1_slave, dg2_slave, dNa, PN, vr, Dt, s1, r1, 
                     g_cur, FricFac, c1, N1_bar, mc1_bar, mb2_bar):
    """
    Assemble the 24x24 frictional slip contact stiffness matrix.

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
    FrictionalK : (24,24) array
        Assembled frictionless contact stiffness matrix.
    """
    
    K11 = np.zeros((12, 12))
    K12 = np.zeros((12, 12))
    K21 = np.zeros((12, 12))
    K22 = np.zeros((12, 12))
    
    
    Q1 = (np.dot(Cur_n, m1) * np.eye(3) + np.outer(Cur_n, m1)) / J1
    dg1_hat_slave = TransVect2SkewSym(dg1_slave)
    dg2_hat_slave = TransVect2SkewSym(dg2_slave)
    Ac1_bar = (np.kron(dNa[0].reshape((1,-1)), dg2_hat_slave) - np.kron(dNa[1].reshape((1,-1)), dg1_hat_slave)) / J1
    dh = np.linalg.norm(PN @ vr) * Dt
    Ps = (np.eye(3) - np.outer(s1, s1)) / dh
    R1 = (np.dot(Cur_n,r1) * np.eye(3) + np.outer(Cur_n, r1)) / g_cur
    B1 = (Ps @ c1) @ (N1_bar @ Cur_n).T - Ps @ PN;
    L1 = g_cur * Ps @ (PN @ Q1 + R1 - np.eye(3)) @ PN 
    Jc1 = L1 @ Ac - g_cur * Ps @ PN @ Ac1_bar
    

    hc1_add = N1 @ mc1_bar + Ac @ np.kron(np.eye(4), Cur_n.reshape((-1,1)))
    hc1_sub = N1 @ mc1_bar - Ac @ np.kron(np.eye(4), Cur_n.reshape((-1,1)))
    S1 = np.outer(s1, Cur_n)
    S1_wave = np.outer(s1, N1_bar @ Cur_n)
    
    # --- Precompute common terms ---
    pc_term = pc * S1_wave + tn * B1   # Common part: pressure + normal term
    scale = FricFac * J1               # Friction factor and Jacobian scaling
    
    # --- Loop over local nodes (4 nodes, 3 dofs each) ---
    for aa in range(4):
        NaA, NbA = Na[aa], Nb[aa]
        idxA = slice(3 * aa, 3 * aa + 3)
        
        for bb in range(4):    
            NaB, NbB = Na[bb], Nb[bb]
            idxB = slice(3 * bb, 3 * bb + 3)
    
            # --- Common sub-blocks for readability ---
            sub_hc = (np.outer(s1, hc1_sub[:, bb])
                      + g_cur * Ps @ np.outer(c1, hc1_add[:, bb])
                      - Jc1[:, idxB])
    
            sub_mb2 = -np.outer(s1, mb2_bar[:, aa])
    
            K11[idxA, idxB] += scale * (-NaA * NaB * pc_term - NaA * tn * sub_hc)
            K12[idxA, idxB] += scale * (NaA * NbB * pc_term)
            K21[idxA, idxB] += scale * (NbA * NaB * pc_term
                                                   + NbA * tn * sub_hc
                                                   + NaB * tn * sub_mb2
                                                   + Gbc[aa, bb] * tn * S1)
            K22[idxA, idxB] += scale * (-NbA * NbB * pc_term - NbB * tn * sub_mb2)
            
    
    # K = np.block([ [K11, K12], [K21, K22]])
    return assemble_block(K11, K12, K21, K22)


@njit(cache=True)
def get_frictional_K_stick(Na, Nb, pc, J1, tv, Ac, Cur_n): 
    """
    Assemble the 24x24 frictional contact stiffness matrix.

    Parameters
    ----------
    Na, Nb : (4,) arrays
        Shape functions for slave and master surfaces.
    pc : float
        Contact penalty parameter.
    tv : contact traction.
    Ac : (3,12) arrays
        Auxiliary matrices related to geometry and projection.
    J1 : float
        Surface Jacobian.

    Returns
    -------
    FrictionalK : (24,24) array
        Assembled frictionless contact stiffness matrix.
    """
    
    K11 = np.zeros((12,12)); K12 = np.zeros((12,12))
    K21 = np.zeros((12,12)); K22 = np.zeros((12,12))

    # --- Precompute common constants --- 
    scale = pc * J1
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
    
            K11[idxA, idxB] += (-NaA * NaB * scale * I3
                                      + NaA * J1 * outer_tv_Acn)
    
            K12[idxA, idxB] += (NaA * NbB * scale * I3)
    
            K21[idxA, idxB] += (NbA * NaB * scale * I3
                                      - NbA * J1 * outer_tv_Acn)
    
            K22[idxA, idxB] += (-NbA * NbB * scale * I3)

    # K = np.block([[K11, K12],[K21, K22]])
    return assemble_block(K11, K12, K21, K22)

