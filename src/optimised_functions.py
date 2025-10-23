#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 17:17:18 2025

@author: frocha
"""

import numba
import numpy as np
# from utils import *

# === Surface shape function and derivatives ===
@numba.jit(nopython=True)
def GetSurfaceShapeFunction(rs):
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
    dN : ndarray
        Derivative with respect to rs (2,4)
    """
    r, s = rs
    
    N  = 0.25*np.array([(r - 1) * (s - 1), -(r + 1) * (s - 1), (r + 1) * (s + 1), -(r - 1) * (s + 1)])
    dN = 0.25*np.array([[(s - 1),-(s - 1), (s + 1), -(s + 1)], 
                        [(r - 1), -(r + 1), (r + 1), -(r - 1)]])
    
    return N, dN

FACE_INDEX = np.array([[3, 2, 1, 0], 
                       [5, 6, 7, 4],
                       [1, 5, 4, 0],
                       [1, 2, 6, 5],
                       [2, 3, 7, 6],
                       [4, 7, 3, 0]], dtype=np.int64)

@numba.jit(nopython=True)
def GetSurfaceNode(elementLE, SurfSign):
    """
    GetSurfaceNode - Return the node indices defining a surface of a hexahedral element.

    Parameters
    ----------
    elementLE : array_like (expected to be a 1D NumPy array)
        1D array of 8 node indices for the element.
    SurfSign : int
        Surface identifier (0–5).

    Returns
    -------
    SurfNode : ndarray
        Array of 4 node indices for the specified face (1D, int).
    """

    return elementLE[FACE_INDEX[SurfSign]].astype(np.int64)



@numba.jit(nopython=True)
def solve_2x2_system_nb(A, b):
    """
    Solves a 2x2 linear system Ax = b for x, where:
    A = [[a11, a12], [a21, a22]]
    b = [b1, b2]

    Parameters
    ----------
    A : np.ndarray (float, 2x2)
        The coefficient matrix.
    b : np.ndarray (float, 1D, size 2)
        The right-hand side vector.

    Returns
    -------
    x : np.ndarray (float, 1D, size 2)
        The solution vector [x1, x2].
    """
    # Unpack matrix A elements for clarity
    a11 = A[0, 0]
    a12 = A[0, 1]
    a21 = A[1, 0]
    a22 = A[1, 1]

    # Calculate the determinant of A
    det_A = a11 * a22 - a12 * a21

    # Check for singularity (det_A close to zero)
    if abs(det_A) < 1e-15:
        # Return zeros or raise an error for a singular matrix
        return np.zeros(2, dtype=A.dtype)

    # Calculate the inverse of A: inv(A) = (1/det_A) * [[a22, -a12], [-a21, a11]]
    inv_det = 1.0 / det_A

    # Calculate x = inv(A) @ b
    x1 = inv_det * (a22 * b[0] - a12 * b[1])
    x2 = inv_det * (-a21 * b[0] + a11 * b[1])

    # Create the solution vector
    x = np.empty(2, dtype=A.dtype)
    x[0] = x1
    x[1] = x2

    return x

@numba.jit(nopython=True)
def newton_raphson_raytracing(SlavePoint, SlavePointTan, MasterSurfXYZ, Exist, Tol):
    rs = np.zeros(2)
    for j in range(int(1e8)):
        N, dN = GetSurfaceShapeFunction(rs)

        NX = MasterSurfXYZ.T@N
        NTX = dN @ MasterSurfXYZ # (2,4)x(4,3) --> (2,3)
                
        # SlavePointTan is (3,2) , fai is (2,), SlavePoint and Nx are (3,)
        fai = SlavePointTan.T @ (SlavePoint - NX)

        if j == 500:
            rs[0] = 1e5
            Exist = -1
            break

        if np.max(np.abs(fai)) < Tol:
            break

        KT = (NTX@SlavePointTan).T
        drs = solve_2x2_system_nb(KT, fai)

        rs += drs

    return rs[0], rs[1], Exist

@numba.jit(nopython=True)
def get_dofs_given_nodes_ids(nodes_ids):
    DOFs = np.empty(nodes_ids.shape[0] * 3, dtype=np.int64)
    for m, node in enumerate(nodes_ids):
        DOFs[3*m:3*m+3] = np.arange(3*node, 3*(node+1)) # python convention

    return DOFs


def get_deformed_position(nodes_ids, nodes, disp):
    # Build DOF list
    DOFs = get_dofs_given_nodes_ids(nodes_ids)
    n_nodes = nodes_ids.shape[0]
    # Current deformed coordinates
    u = disp[DOFs].reshape((3, n_nodes), order = 'F').T # numba does not like 'F'
    x = nodes[nodes_ids, :] + u  
    return x 


# Todo1: node to python convention
# Todo2: eliminate repeated conde : "Build DOFs"
# Todo3: automate get deformed coordinates
# Todo4: Find the nearest node can be improved
# @numba.jit
def GetContactPointbyRayTracing(Eles, Nodes, MasterSurf, Disp, SlavePoint, SlavePointTan):
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

    nMasterSurf = MasterSurf.shape[1]
    AllMasterNode = np.zeros((nMasterSurf, 4), dtype = np.int64)
    
    SlavePoint_ = SlavePoint.flatten().astype(np.float64)
    
    # --- Find node closest to integration point from slave surface ---
    for i in range(nMasterSurf):
        # MATLAB element index is 1-based
        MasterSurfNode = GetSurfaceNode(Eles[MasterSurf[0, i], :],
                                        MasterSurf[1, i])
        AllMasterNode[i, :] = MasterSurfNode

        MasterSurfXYZ = get_deformed_position(MasterSurfNode, Nodes, Disp)
                
        # Find nearest node to slave point
        ll = MasterSurfXYZ - SlavePoint_  # Result is a (4, 3) array
        Distances = np.linalg.norm(ll, axis=1) # Result is a (4,) array
        min_idx = np.argmin(Distances)
        current_min_distance = Distances[min_idx]
        if current_min_distance < MinDis:
            MinDis = current_min_distance
            MinMasterPoint = MasterSurfNode[min_idx]
    
    
    # --- Determine candidate master surfaces ---
    AllMinMasterSurfNum = np.where(AllMasterNode == MinMasterPoint)[0]
    ContactCandidate = np.zeros((len(AllMinMasterSurfNum), 8))
    ContactCandidate[:, 4] = 1e7  # MATLAB column 5
    
    # --- Loop over candidate master surfaces ---
    for idx, surf_idx in enumerate(AllMinMasterSurfNum):
        MasterSurfNode = AllMasterNode[surf_idx, :]        
        MasterSurfXYZ = get_deformed_position(MasterSurfNode, Nodes, Disp)

        # Ray-tracing Newton-Raphson iteration
        r, s, Exist = newton_raphson_raytracing(SlavePoint_, SlavePointTan, MasterSurfXYZ, Exist, Tol)

        # --- Save nearest RayTracing point ---
        if abs(r) <= 1.01 and abs(s) <= 1.01:
            v = np.cross(SlavePointTan[:, 0], SlavePointTan[:, 1])
            v /= np.linalg.norm(v)
            
            N, _ = GetSurfaceShapeFunction(np.array((r,s)))
            NX = MasterSurfXYZ.T@N
            
            g = np.dot(NX - SlavePoint_, v)

            ContactCandidate[idx, 0] = MasterSurf[0, surf_idx]
            ContactCandidate[idx, 1] = MasterSurf[1, surf_idx]
            ContactCandidate[idx, 2:5] = np.array((r, s, g))
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
        MasterEle = ContactCandidate[MinGrow, 0] + 1
        MasterSign = ContactCandidate[MinGrow, 1] + 1
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