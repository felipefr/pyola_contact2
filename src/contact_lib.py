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

octave.addpath(octave.genpath("/home/felipe/sources/pyola_contact2/src/"))  # doctest: +SKIP

# === Obtain surface node numbers ===
def GetSurfaceNode(elementLE, SurfSign):
    """
    GetSurfaceNode - Return the node indices defining a surface of a hexahedral element.

    Parameters
    ----------
    elementLE : array_like
        1D array of 8 node indices for the element.
    SurfSign : int
        Surface identifier (1–6).

    Returns
    -------
    SurfNode : ndarray
        Array of 4 node indices for the specified face (1D, int).
    """
    if SurfSign == 1:
        SurfNode = elementLE[[3, 2, 1, 0]]   # face 1
    elif SurfSign == 2:
        SurfNode = elementLE[[5, 6, 7, 4]]   # face 2
    elif SurfSign == 3:
        SurfNode = elementLE[[1, 5, 4, 0]]   # face 3
    elif SurfSign == 4:
        SurfNode = elementLE[[1, 2, 6, 5]]   # face 4
    elif SurfSign == 5:
        SurfNode = elementLE[[2, 3, 7, 6]]   # face 5
    elif SurfSign == 6:
        SurfNode = elementLE[[4, 7, 3, 0]]   # face 6
    else:
        raise ValueError("SurfSign must be between 1 and 6.")
    return SurfNode - 1 # matlab -> python index


# === Surface shape function and derivatives ===
def GetSurfaceShapeFunction(r, s):
    """
    GetSurfaceShapeFunction - Compute shape functions and their derivatives
    for a quadrilateral surface element (4-node).

    Parameters
    ----------
    r, s : float
        Local coordinates (-1 ≤ r,s ≤ 1)

    Returns
    -------
    N : ndarray
        Shape function values (4,)
    N1 : ndarray
        Derivative with respect to r (4,)
    N2 : ndarray
        Derivative with respect to s (4,)
    """
    N  = np.array([
        0.25 * (r - 1) * (s - 1),
        -0.25 * (r + 1) * (s - 1),
        0.25 * (r + 1) * (s + 1),
        -0.25 * (r - 1) * (s + 1)
    ])
    N1 = np.array([
        0.25 * (s - 1),
        -0.25 * (s - 1),
        0.25 * (s + 1),
        -0.25 * (s + 1)
    ])
    N2 = np.array([
        0.25 * (r - 1),
        -0.25 * (r + 1),
        0.25 * (r + 1),
        -0.25 * (r - 1)
    ])
    return N, N1, N2


# === Obtain surface node coordinates and DOFs ===
def GetSurfaceNodeLocation(FEMod, Disp, Surf):
    """
    GetSurfaceNodeLocation - Return coordinates and DOF indices of nodes on a surface.

    Parameters
    ----------
    FEMod : Struct or dict
        Finite element model data; must contain:
            FEMod['Eles'] : element connectivity array (nElem × 8)
            FEMod['Nodes'] : node coordinates (nNode × 3)
    Disp : ndarray
        Global displacement vector (size = total DOFs)
    Surf : ndarray
        [element_index, surface_id] (2,)

    Returns
    -------
    SurfNodeXYZ : ndarray
        Coordinates (current) of surface nodes (4 × 3)
    SurfNodeDOF : ndarray
        DOF indices of surface nodes (12,)
    """
    # Extract element and surface
    element_index = int(Surf[0]) - 1  # MATLAB -> Python index
    SurfSign = int(Surf[1])

    # Get surface nodes (convert to 0-based indices)
    element_nodes = np.asarray(FEMod.Eles[element_index, :], dtype=int)
    SurfNode = GetSurfaceNode(element_nodes, SurfSign)

    # Build DOF indices (3 per node)
    SurfNodeDOF = np.zeros(len(SurfNode) * 3, dtype=int)
    for m, node in enumerate(SurfNode):
        SurfNodeDOF[3*m:3*m+3] = np.arange(3*node, 3*(node+1)) # node is already in python convention

    # Get nodal displacements and coordinates
    SurfNodeDis = Disp[SurfNodeDOF].reshape(3, len(SurfNode), order = 'F').T
    SurfNodeXYZ = FEMod.Nodes[SurfNode, :] + SurfNodeDis

    return SurfNodeXYZ, SurfNodeDOF


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
            contactPairs.PreMasterSurf[:, i] = np.array([0.0, 0.0])
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
        contactPairs.CurMasterSurf[:, i] = np.array([0.0, 0.0])
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
    SlaveSurf_arr = np.zeros((2, nPairs))
    SlaveIntegralPoint = np.zeros(nPairs, dtype=int)

    CurMasterSurf = np.zeros((2, nPairs))
    rc = np.zeros(nPairs)
    sc = np.zeros(nPairs)
    Cur_g = np.zeros(nPairs)
    Pre_g = np.zeros(nPairs)
    PreMasterSurf = np.zeros((2, nPairs))
    rp = np.zeros(nPairs)
    sp = np.zeros(nPairs)
    CurContactState = np.zeros(nPairs, dtype=int)
    PreContactState = np.zeros(nPairs, dtype=int)
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


def ContactSearch(FEMod, ContactPairs, Disp, IntegralPoint):
    """
    ContactSearch - conservative translation from MATLAB
    (GetContactPointbyRayTracing still in Octave, 1-based safe)
    """

    nPairs = ContactPairs.SlaveSurf.shape[1]

    for i in range(nPairs):

        # --- Get current slave surface geometry ---
        SlaveSurfNodeXYZ, _ = GetSurfaceNodeLocation(FEMod, Disp, ContactPairs.SlaveSurf[:, i])

        # Current integration point coordinates (MATLAB -> Python: subtract 1)
        ip_idx = int(ContactPairs.SlaveIntegralPoint[i]) - 1
        CurIP = IntegralPoint[ip_idx, :]
        N, N1, N2 = GetSurfaceShapeFunction(float(CurIP[0]), float(CurIP[1]))

        # Compute slave surface point and tangents
        SlavePoint = np.sum(N[:, None] * SlaveSurfNodeXYZ, axis=0).reshape((3,1)) # this is to match octave convention
        N1X = np.sum(N1[:, None] * SlaveSurfNodeXYZ, axis=0)
        N2X = np.sum(N2[:, None] * SlaveSurfNodeXYZ, axis=0)
        SlavePointTan = np.column_stack((N1X, N2X))

        # --- Find nearest master surface via ray tracing (still in Octave) ---
        rr2, ss2, MasterEle2, MasterSign2, gg2, Exist2 = octave.GetContactPointbyRayTracing(
            FEMod, Disp, SlavePoint, SlavePointTan, nout = 6)

        rr, ss, MasterEle, MasterSign, gg, Exist = GetContactPointbyRayTracing(
            FEMod, Disp, SlavePoint, SlavePointTan)

        # # --- Update contact pair information ---
        if(np.abs(gg2 - gg)>1e-8):
            print(i, gg, gg2, Exist, Exist2, MasterEle, MasterEle2)
            
            
        if Exist == 1:
            ContactPairs.CurMasterSurf[:, i] = np.array([MasterEle, MasterSign])
            ContactPairs.rc[i] = rr
            ContactPairs.sc[i] = ss
            ContactPairs.Cur_g[i] = gg
        else:
            # print("contact not found at ", i)
            ContactPairs.CurMasterSurf[:, i] = np.array([0, 0])
            ContactPairs.rc[i] = 0
            ContactPairs.sc[i] = 0
            ContactPairs.Cur_g[i] = 0
            ContactPairs.CurContactState[i] = 0

    return ContactPairs


# Todo1: node to python convention
# Todo2: eliminate repeated conde : "Build DOFs"
# Todo3: automate get deformed coordinates
# Todo4: Find the nearest node can be improved
def GetContactPointbyRayTracing(FEMod, Disp, SlavePoint, SlavePointTan):
    """
    Obtain master surface contact point by ray tracing.
    FEMod numbering follows MATLAB (1-based)
    """

    Tol = 1e-4
    Exist = -1
    MinDis = 1e8
    MinGrow = 0
    Ming = -1e3
    MinMasterPoint = None

    nMasterSurf = FEMod.MasterSurf.shape[1]
    AllMasterNode = np.zeros((nMasterSurf, 4), dtype=int)

    # --- Find node closest to integration point from slave surface ---
    for i in range(nMasterSurf):
        # MATLAB element index is 1-based
        MasterSurfNode = GetSurfaceNode(FEMod.Eles[int(FEMod.MasterSurf[0, i]) - 1, :].astype('int'),
                                        int(FEMod.MasterSurf[1, i]))
        AllMasterNode[i, :] = MasterSurfNode

        # Build DOF list
        MasterSurfDOF = np.zeros(len(MasterSurfNode) * 3, dtype=int)
        for m, node in enumerate(MasterSurfNode):
            MasterSurfDOF[3*m:3*m+3] = np.arange(3*node, 3*(node+1)) # python convention

        # Current deformed coordinates
        MasterSurfDis = Disp[MasterSurfDOF].reshape(3, len(MasterSurfNode), order = 'F').T
        MasterSurfXYZ = FEMod.Nodes[MasterSurfNode, :] + MasterSurfDis  

        # Find nearest node to slave point
        for j in range(4):
            ll = MasterSurfXYZ[j, :] - SlavePoint.flatten()
            Distance = np.linalg.norm(ll)
            if Distance < MinDis:
                MinDis = Distance
                MinMasterPoint = MasterSurfNode[j]
    
    
    # --- Determine candidate master surfaces ---
    AllMinMasterSurfNum = np.where(AllMasterNode == MinMasterPoint)[0]
    ContactCandidate = np.zeros((len(AllMinMasterSurfNum), 8))
    ContactCandidate[:, 4] = 1e7  # MATLAB column 5
    
    # --- Loop over candidate master surfaces ---
    for idx, surf_idx in enumerate(AllMinMasterSurfNum):
        MasterSurfNode = AllMasterNode[surf_idx, :]

        # Build DOFs
        MasterSurfDOF = np.zeros(len(MasterSurfNode) * 3, dtype=int)
        for m, node in enumerate(MasterSurfNode):
            MasterSurfDOF[3*m:3*m+3] = np.arange(3*node, 3*(node+1)) # python convention

        # Deformed coordinates
        MasterSurfDis = Disp[MasterSurfDOF].reshape(3, len(MasterSurfNode), order = 'F').T
        MasterSurfXYZ = FEMod.Nodes[MasterSurfNode, :] + MasterSurfDis

        # Ray-tracing Newton-Raphson iteration
        r = 0.0
        s = 0.0
        for j in range(int(1e8)):
            N, N1, N2 = GetSurfaceShapeFunction(r, s)

            NX = np.sum(N[:, None] * MasterSurfXYZ, axis=0)
            N1X = np.sum(N1[:, None] * MasterSurfXYZ, axis=0)
            N2X = np.sum(N2[:, None] * MasterSurfXYZ, axis=0)

            # felipe: flattening slavepoint
            fai = np.array([
                np.dot(SlavePoint.flatten() - NX, SlavePointTan[:, 0]),
                np.dot(SlavePoint.flatten() - NX, SlavePointTan[:, 1])
            ])

            if j == 500:
                r = 1e5
                Exist = -1
                break

            if np.max(np.abs(fai)) < Tol:
                break

            k11 = np.dot(N1X, SlavePointTan[:, 0])
            k12 = np.dot(N2X, SlavePointTan[:, 0])
            k21 = np.dot(N1X, SlavePointTan[:, 1])
            k22 = np.dot(N2X, SlavePointTan[:, 1])

            KT = np.array([[k11, k12], [k21, k22]])
            drs = np.linalg.solve(KT, fai)

            r += drs[0]
            s += drs[1]

        # --- Save nearest RayTracing point ---
        if abs(r) <= 1.01 and abs(s) <= 1.01:
            v = np.cross(SlavePointTan[:, 0], SlavePointTan[:, 1])
            v /= np.linalg.norm(v)
            
            # felipe: flattening slavepoint
            g = np.dot(NX - SlavePoint.flatten(), v)

            ContactCandidate[idx, 0] = FEMod.MasterSurf[0, surf_idx]
            ContactCandidate[idx, 1] = FEMod.MasterSurf[1, surf_idx]
            ContactCandidate[idx, 2:5] = np.array([r, s, g])
            ContactCandidate[idx, 5:8] = v

            if Exist <= 0:
                if g >= 0 and abs(Ming) > abs(g):
                    Exist = 0; MinGrow = idx; Ming = g
                elif g < 0:
                    Exist = 1; MinGrow = idx; Ming = g
            elif Exist == 1:
                if g < 0 and abs(Ming) > abs(g):
                    Exist = 1; MinGrow = idx; Ming = g

    # --- Final contact outputs ---
    if Exist == 0 or Exist == 1:
        MasterEle = ContactCandidate[MinGrow, 0]
        MasterSign = ContactCandidate[MinGrow, 1]
        rr = ContactCandidate[MinGrow, 2]
        ss = ContactCandidate[MinGrow, 3]
        gg = ContactCandidate[MinGrow, 4]
    else:
        MasterEle = 1e10
        MasterSign = 1e10
        rr = 1e10
        ss = 1e10
        gg = 1e10

    return rr, ss, MasterEle, MasterSign, gg, Exist


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
