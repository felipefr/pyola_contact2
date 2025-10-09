#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 10:35:06 2025

@author: frocha
"""
import numpy as np

# Convert a 3D vector to its skew-symmetric matrix (cross-product matrix).
def TransVect2SkewSym(Vect):
    Vect = np.asarray(Vect).flatten()
    SkewSym = np.array([[0, -Vect[2], Vect[1]],
                        [Vect[2], 0, -Vect[0]],
                        [-Vect[1], Vect[0], 0]])
    return SkewSym

def flattenising_struct(s):
    for k in s.keys(): 
        if(s[k].shape[0] == 1 and s[k].shape[1]!=1):
            s[k] = s[k].flatten()
            

def get_surface_geometry(N, N1, N2, SurfXYZ):
    """
    Compute surface geometry quantities from shape functions and nodal coordinates.

    Parameters
    ----------
    N, N1, N2 : (n_nodes,) arrays
        Shape function and its derivatives with respect to local coordinates.
    SurfXYZ : (n_nodes, 3) array
        Nodal coordinates of the surface.

    Returns
    -------
    n : (3,) array
        Unit normal vector.
    J : float
        Surface Jacobian (norm of cross product).
    N1X, N2X : (3,) arrays
        Tangent vectors in local directions.
    x : (3,) array
        Current surface point coordinates.
    """
    x   = np.sum(N[:, None]  * SurfXYZ, axis=0)
    N1X = np.sum(N1[:, None] * SurfXYZ, axis=0)
    N2X = np.sum(N2[:, None] * SurfXYZ, axis=0)

    n = np.cross(N1X, N2X)
    J = np.linalg.norm(n)
    n = n / J  # normalize

    return n, J, N1X, N2X, x


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