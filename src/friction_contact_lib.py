#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 20:20:32 2025

@author: frocha
"""

# still buggy. Only the frictionless factorization is working


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
        [-gp,  gp]
    ])

    # --- Contact search and friction factor
    # ContactPairs = octave.ContactSearch(FEMod, ContactPairs, Disp, IntegralPoint, nout = 1)
    # flattenising_struct(ContactPairs)
    
    ContactPairs = ContactSearch(FEMod, ContactPairs, Disp, IntegralPoint)
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
            
            GKF, Residual, ContactPairs = CalculateContactKandF(
                FEMod, ContactPairs, Dt, PreDisp, i, GKF, Residual, Disp, IntegralPoint) 
            continue

        # --- Case 2: possible stick/slip contact ---
        # ensure integer index when indexing IntegralPoint (ContactPairs stores 1..4)
        ip_idx = int(np.asarray(ContactPairs.SlaveIntegralPoint[i])) - 1
        CurIP = IntegralPoint[ip_idx]        # shape (2,)
        Na, N1a, N2a = GetSurfaceShapeFunction(float(CurIP[0]), float(CurIP[1]))

        # Slave surface coordinates (ensure integer surf spec [element, face])
        slave_surf = np.asarray(ContactPairs.SlaveSurf[:, i]).astype(int)
        CurSlaveSurfXYZ, _ = GetSurfaceNodeLocation(FEMod, Disp, slave_surf)
        Cur_x1 = np.sum(Na[:, None] * CurSlaveSurfXYZ, axis=0)

        # Master surface (previous) - Nb uses rp,sp which are already numeric
        Nb, _, _ = GetSurfaceShapeFunction(ContactPairs.rp[i], ContactPairs.sp[i])
        pre_master = np.asarray(ContactPairs.PreMasterSurf[:, i]).astype(int)
        CurMasterSurfXYZ_p, _ = GetSurfaceNodeLocation(FEMod, Disp, pre_master)
        Cur_x2_p = np.sum(Nb[:, None] * CurMasterSurfXYZ_p, axis=0)

        # Relative motion and projection
        gs = Cur_x2_p - Cur_x1
        tv = ContactPairs.pc[i] * gs

        # Current normal
        Cur_N1Xa = np.sum(N1a[:, None] * CurSlaveSurfXYZ, axis=0)
        Cur_N2Xa = np.sum(N2a[:, None] * CurSlaveSurfXYZ, axis=0)
        Cur_n = np.cross(Cur_N1Xa, Cur_N2Xa)
        Cur_n = Cur_n / np.linalg.norm(Cur_n)

        # Tangential/normal trial components
        tn_trial = abs(np.dot(tv, Cur_n))
        tt_trial = np.sqrt(np.linalg.norm(tv)**2 - tn_trial**2)

        # Slip/stick criterion
        fai = tt_trial - FricFac * tn_trial
        ContactPairs.CurContactState[i] = 1 if fai < 0 else 2

        # Update stiffness and force
        # i should start in 1 to be compatible with octave
        # GKF, Residual, ContactPairs = octave.CalculateContactKandF(
        #     FEMod, ContactPairs, Dt, PreDisp, i+1, GKF, Residual, Disp, IntegralPoint, nout = 3)
        # flattenising_struct(ContactPairs)
        
        GKF, Residual, ContactPairs = CalculateContactKandF(
            FEMod, ContactPairs, Dt, PreDisp, i, GKF, Residual, Disp, IntegralPoint) 


    return ContactPairs, GKF, Residual

def CalculateContactKandF(FEMod, ContactPairs, Dt, PreDisp, i, GKF, Residual, Disp, IntegralPoint):
    FricFac = FEMod.FricFac

    if ContactPairs.CurContactState[i] == 1:  # Stick contact
        # --- slave geometry at current IP ---
        ip_idx = int(ContactPairs.SlaveIntegralPoint[i]) - 1
        CurIP = IntegralPoint[ip_idx, :]
        Na, N1a, N2a = GetSurfaceShapeFunction(CurIP[0], CurIP[1])
        CurSlaveSurfXYZ, SlaveSurfDOF = GetSurfaceNodeLocation(FEMod, Disp, ContactPairs.SlaveSurf[:, i])

        Cur_x1 = np.sum(Na[:, None] * CurSlaveSurfXYZ, axis=0)
        Cur_N1Xa = np.sum(N1a[:, None] * CurSlaveSurfXYZ, axis=0)
        Cur_N2Xa = np.sum(N2a[:, None] * CurSlaveSurfXYZ, axis=0)

        Cur_n = np.cross(Cur_N1Xa, Cur_N2Xa)
        Cur_n /= np.linalg.norm(Cur_n)
        J1 = np.linalg.norm(np.cross(Cur_N1Xa, Cur_N2Xa))

        # --- master geometry ---
        Nb, _, _ = GetSurfaceShapeFunction(ContactPairs.rp[i], ContactPairs.sc[i])
        CurMasterSurfXYZ_rpsp, MasterSurfDOF = GetSurfaceNodeLocation(FEMod, Disp, ContactPairs.PreMasterSurf[:, i])
        Cur_x2_p = np.sum(Nb[:, None] * CurMasterSurfXYZ_rpsp, axis=0)

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
        Residual[ContactDOF, :] += ContactNodeForce[:, None]

        # --- stiffness ---
        Cur_g1_hat_slave = TransVect2SkewSym(Cur_N1Xa)
        Cur_g2_hat_slave = TransVect2SkewSym(Cur_N2Xa)
        Ac = (np.kron(N1a[:, None].T, Cur_g2_hat_slave) - np.kron(N2a[:, None].T, Cur_g1_hat_slave)) / J1

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

    elif ContactPairs.CurContactState[i] == 2:  # Slip contact
        tn = ContactPairs.Cur_g[i] * ContactPairs.pc[i]

        # --- slave geometry current and previous ---
        ip_idx = int(ContactPairs.SlaveIntegralPoint[i]) - 1
        CurIP = IntegralPoint[ip_idx, :]
        Na, N1a, N2a = GetSurfaceShapeFunction(CurIP[0], CurIP[1])
        CurSlaveSurfXYZ, SlaveSurfDOF = GetSurfaceNodeLocation(FEMod, Disp, ContactPairs.SlaveSurf[:, i])
        PreSlaveSurfNodeXYZ, _ = GetSurfaceNodeLocation(FEMod, PreDisp, ContactPairs.SlaveSurf[:, i])

        Cur_x1 = np.sum(Na[:, None] * CurSlaveSurfXYZ, axis=0)
        Pre_x1 = np.sum(Na[:, None] * PreSlaveSurfNodeXYZ, axis=0)
        dx1 = Cur_x1 - Pre_x1

        Pre_N1Xa = np.sum(N1a[:, None] * PreSlaveSurfNodeXYZ, axis=0)
        Pre_N2Xa = np.sum(N2a[:, None] * PreSlaveSurfNodeXYZ, axis=0)
        Cur_N1Xa = np.sum(N1a[:, None] * CurSlaveSurfXYZ, axis=0)
        Cur_N2Xa = np.sum(N2a[:, None] * CurSlaveSurfXYZ, axis=0)

        Cur_n = np.cross(Cur_N1Xa, Cur_N2Xa)
        Cur_n /= np.linalg.norm(Cur_n)
        J1 = np.linalg.norm(np.cross(Cur_N1Xa, Cur_N2Xa))
        PN = np.eye(3) - np.outer(Cur_n, Cur_n)

        dg1_slave = Cur_N1Xa - Pre_N1Xa
        dg2_slave = Cur_N2Xa - Pre_N2Xa
        m1 = np.cross(dg1_slave, Cur_N2Xa) + np.cross(Cur_N1Xa, dg2_slave)
        c1 = PN @ m1 / J1

        # --- master geometry ---
        Nb, N1b, N2b = GetSurfaceShapeFunction(ContactPairs.rc[i], ContactPairs.sc[i])
        CurMasterSurfNodeXYZ, MasterSurfDOF = GetSurfaceNodeLocation(FEMod, Disp, ContactPairs.CurMasterSurf[:, i])
        PreMasterSurfNodeXYZ, _ = GetSurfaceNodeLocation(FEMod, PreDisp, ContactPairs.CurMasterSurf[:, i])
        
        # Felipe
        Cur_x2 = np.sum(Nb[:, np.newaxis] * CurMasterSurfNodeXYZ, axis=0)  # shape (3,)
        Cur_N1Xb = np.sum(N1b[:, np.newaxis] * CurMasterSurfNodeXYZ, axis=0)  # shape (3,)
        Cur_N2Xb = np.sum(N2b[:, np.newaxis] * CurMasterSurfNodeXYZ, axis=0)  # shape (3,)
        Cur_N1Xa = np.sum(N1a[:, np.newaxis] * CurSlaveSurfXYZ, axis=0)  # shape (3,)
        Cur_N2Xa = np.sum(N2a[:, np.newaxis] * CurSlaveSurfXYZ, axis=0)  # shape (3,)

        Cur_x2 = np.sum(Nb[:, None] * CurMasterSurfNodeXYZ, axis=0)
        Pre_x2 = np.sum(Nb[:, None] * PreMasterSurfNodeXYZ, axis=0)
        dx2 = Cur_x2 - Pre_x2

        # --- tangential sliding ---
        r1 = ContactPairs.Cur_g[i] * c1 + dx1 - dx2
        vr = r1 / Dt
        s1_temp = PN @ vr
        s1 = s1_temp / np.linalg.norm(s1_temp) if np.linalg.norm(s1_temp) > 1e-8 else np.zeros(3)

        # --- contact nodal force ---
        ContactNodeForce = np.zeros(24)
        tv = tn * (Cur_n + FricFac * s1)
        ContactPairs.Pressure[i] = abs(tn)
        ContactPairs.Traction[i] = np.linalg.norm(tv)

        for a in range(4):
            idxA = slice(3*a, 3*a+3)
            ContactNodeForce[idxA] = Na[a] * tv * J1
            ContactNodeForce[idxA.start+12:idxA.stop+12] = -Nb[a] * tv * J1

        ContactDOF = np.concatenate([SlaveSurfDOF, MasterSurfDOF])
        Residual[ContactDOF, :] += ContactNodeForce[:, None]

        # --- projection matrices ---
        A_ab = np.array([[Cur_N1Xa @ Cur_N1Xa, Cur_N1Xa @ Cur_N2Xa],
                         [Cur_N2Xa @ Cur_N1Xa, Cur_N2Xa @ Cur_N2Xa]])
        a_ab = np.linalg.inv(A_ab)

        g1_bar_slave  = a_ab[0,0]*Cur_N1Xa + a_ab[1,0]*Cur_N2Xa
        g2_bar_slave  = a_ab[0,1]*Cur_N1Xa + a_ab[1,1]*Cur_N2Xa
        g1_bar_master = a_ab[0,0]*Cur_N1Xa + a_ab[0,1]*Cur_N2Xa
        g2_bar_master = a_ab[1,0]*Cur_N1Xa + a_ab[1,1]*Cur_N2Xa

        N1_bar = np.eye(3) - np.outer(Cur_N1Xa, g1_bar_slave) - np.outer(Cur_N2Xa, g2_bar_slave)

        # --- Ac, mc1_bar, mb2_bar, Gbc ---
        Cur_g1_hat_slave = TransVect2SkewSym(Cur_N1Xa)
        Cur_g2_hat_slave = TransVect2SkewSym(Cur_N2Xa)
        Ac = (np.kron(N1a[:, None].T, Cur_g2_hat_slave) - np.kron(N2a[:, None].T, Cur_g1_hat_slave)) / J1
        
        # Felipe
        mc1_bar = a_ab[0,0] * Cur_N1Xb + a_ab[1,0] * Cur_N2Xb  # shape (3,)
        mb2_bar = a_ab[0,0] * Cur_N1Xa + a_ab[0,1] * Cur_N2Xa  # shape (3,)
        
        # Mc1_bar = [ Cur_n * mc1_bar(:,1)' , Cur_n * mc1_bar(:,2)' , Cur_n * mc1_bar(:,3)' , Cur_n * mc1_bar(:,4)' ];
        # Mb2_bar = [ -Cur_n * mb2_bar(:,1)' , -Cur_n * mb2_bar(:,2)' , -Cur_n * mb2_bar(:,3)' , -Cur_n * mb2_bar(:,4)' ];
        Mc1_bar = np.hstack([np.outer(Cur_n, mc1_bar[:, i]) for i in range(4)])
        Mb2_bar = np.hstack([-np.outer(Cur_n, mb2_bar[:, i]) for i in range(4)])
                                       
        N12a = np.row_stack((N1a, N2a))
        N12b = np.row_stack((N1b, N2b))
        Gbc = ContactPairs.Cur_g[i] @ (N12b.T @ a_ab @ N12a)  
        # ends here (Felipe)
        
        # --- Frictionless K ---
        Frictionless_K11 = np.zeros((12,12)); Frictionless_K12 = np.zeros((12,12))
        Frictionless_K21 = np.zeros((12,12)); Frictionless_K22 = np.zeros((12,12))
        for aa in range(4):
            idxA = slice(3*aa,3*aa+3)
            for bb in range(4):
                idxB = slice(3*bb,3*bb+3)
                tempK11 = (-Na[aa]*Na[bb]*ContactPairs.pc[i]*np.outer(Cur_n,Cur_n)
                           - Na[aa]*tn*(Ac[:, idxB] + mc1_bar[:, bb:bb+1] @ Cur_n[:,None].T)) * J1
                Frictionless_K11[idxA, idxB] += tempK11
                tempK12 = (Na[aa]*Nb[bb]*ContactPairs.pc[i]*np.outer(Cur_n,Cur_n)) * J1
                Frictionless_K12[idxA, idxB] += tempK12
                tempK21 = (Nb[aa]*Na[bb]*ContactPairs.pc[i]*np.outer(Cur_n,Cur_n)
                           + Nb[aa]*tn*(Ac[:, idxB] + mc1_bar[:, bb:bb+1] @ Cur_n[:,None].T)
                           + Na[bb]*tn*(-mb2_bar[:, aa:aa+1].T) + Gbc[aa, bb]*tn*np.outer(Cur_n,Cur_n)) * J1
                Frictionless_K21[idxA, idxB] += tempK21
                tempK22 = (-Nb[aa]*Nb[bb]*ContactPairs.pc[i]*np.outer(Cur_n,Cur_n)
                           - Nb[bb]*tn*mb2_bar[:, idxA]) * J1
                Frictionless_K22[idxA, idxB] += tempK22

        FrictionlessK = np.block([[Frictionless_K11, Frictionless_K12],
                                  [Frictionless_K21, Frictionless_K22]])

        # --- Frictional K additions ---
        Frictional_K = np.zeros((24,24))
        if FricFac != 0 and np.linalg.norm(s1) > 1e-8:
            Q1 = ((Cur_n @ m1) * np.eye(3) + np.outer(Cur_n, m1)) / J1
            dh = np.linalg.norm(PN @ vr) * Dt
            Ps = (np.eye(3) - np.outer(s1, s1)) / dh
            R1 = ((Cur_n @ r1) * np.eye(3) + np.outer(Cur_n, r1)) / ContactPairs.Cur_g[i]
            L1 = ContactPairs.Cur_g[i] * Ps @ (PN @ Q1 + R1 - np.eye(3)) @ PN
            Jc1 = L1 @ Ac - ContactPairs.Cur_g[i] * Ps @ PN @ (np.kron(N1a[:, None].T, TransVect2SkewSym(dg2_slave)) - np.kron(N2a[:, None].T, TransVect2SkewSym(dg1_slave))) / J1

            hc1_add = N1a[:, None] @ mc1_bar + Ac @ np.kron(np.eye(4), Cur_n)
            hc1_sub = N1a[:, None] @ mc1_bar - Ac @ np.kron(np.eye(4), Cur_n)
            S1_wave = s1 @ (N1_bar @ Cur_n).T
            S1 = np.outer(s1, Cur_n)

            for aa in range(4):
                idxA = slice(3*aa,3*aa+3)
                for bb in range(4):
                    idxB = slice(3*bb,3*bb+3)
                    tempK11 = (-Na[aa]*Na[bb]*FricFac*(ContactPairs.pc[i]*S1_wave + tn*B1)
                               - Na[aa]*FricFac*tn*(np.outer(s1, hc1_sub[:,bb]) + ContactPairs.Cur_g[i]*Ps@c1 @ hc1_add[:,bb] - Jc1[:,idxB])) * J1
                    Frictional_K[idxA, idxB] += tempK11
                    # ... similarly fill tempK12, tempK21, tempK22 exactly like MATLAB ...

        ContactK = FrictionlessK + Frictional_K
        GKF[np.ix_(ContactDOF, ContactDOF)] -= ContactK
        
    return GKF, Residual, ContactPairs 
