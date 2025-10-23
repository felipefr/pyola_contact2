#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 10:35:06 2025

@author: frocha
"""
import numpy as np
import numba
# from utils import *
from fem_lib import get_dofs_given_nodes_ids


FACE_INDEX = np.array([[3, 2, 1, 0], 
                       [5, 6, 7, 4],
                       [1, 5, 4, 0],
                       [1, 2, 6, 5],
                       [2, 3, 7, 6],
                       [4, 7, 3, 0]], dtype=np.int64)

# Convert a 3D vector to its skew-symmetric matrix (cross-product matrix).
@numba.jit(nopython=True)
def TransVect2SkewSym(Vect):
    Vect = np.asarray(Vect).flatten()
    SkewSym = np.array([[0, -Vect[2], Vect[1]],
                        [Vect[2], 0, -Vect[0]],
                        [-Vect[1], Vect[0], 0]])
    return SkewSym


# get N1 and N2 as dN
@numba.jit(nopython=True)
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
    x   = SurfXYZ.T@N
    NTx = np.vstack((N1,N2))@SurfXYZ # (2,4)x(4,3) --> (2,3)

    n = np.cross(NTx[0,:], NTx[1,:])
    J = np.linalg.norm(n)
    n = n / J  # normalize

    return n, J, NTx[0,:], NTx[1,:], x



@numba.jit(nopython=True)
def GetSurfaceNode(elementLE, SurfSign):
    """
    GetSurfaceNode - Return the node indices defining a surface of a hexahedral element.

    Parameters
    ----------
    elementLE : array_like (expected to be a 1D NumPy array)
        1D array of 8 node indices for the element.
    SurfSign : int
        Surface identifier (1–6).

    Returns
    -------
    SurfNode : ndarray
        Array of 4 node indices for the specified face (1D, int).
    """

    return elementLE[FACE_INDEX[SurfSign-1]].astype(np.int64) - 1


@numba.jit(nopython=True)
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
    dN : ndarray
        Derivative with respect to rs (2,4)
    """
    
    N  = 0.25*np.array([(r - 1) * (s - 1), -(r + 1) * (s - 1), (r + 1) * (s + 1), -(r - 1) * (s + 1)])
    dN = 0.25*np.array([[(s - 1),-(s - 1), (s + 1), -(s + 1)], 
                        [(r - 1), -(r + 1), (r + 1), -(r - 1)]])
    
    return N, dN[0,:], dN[1,:]

@numba.jit(nopython=True)
def get_deformed_position(nodes_ids, nodes, disp):
    # Build DOF list
    DOFs = get_dofs_given_nodes_ids(nodes_ids)
    # return deformed coordinates
    return nodes[nodes_ids, :] + disp[DOFs].reshape((nodes_ids.shape[0],3)) 

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
    SurfNodeDOF = get_dofs_given_nodes_ids(SurfNode)

    # Get nodal displacements and coordinates
    SurfNodeDis = Disp[SurfNodeDOF].reshape((len(SurfNode),3))
    SurfNodeXYZ = FEMod.Nodes[SurfNode, :] + SurfNodeDis

    return SurfNodeXYZ, SurfNodeDOF


def get_isotropic_celas(E, nu):
    """
    Return 6x6 isotropic elasticity (tangent) matrix for 3D elasticity.
    
    Parameters
    ----------
    E : float
        Young's modulus
    nu : float
        Poisson's ratio
    
    Returns
    -------
    Dtan : (6,6) ndarray
        Constitutive matrix in Voigt notation
    """
    fac = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
    
    Dtan = np.array([
        [1,           nu / (1 - nu), nu / (1 - nu), 0, 0, 0],
        [nu / (1 - nu), 1,           nu / (1 - nu), 0, 0, 0],
        [nu / (1 - nu), nu / (1 - nu), 1,           0, 0, 0],
        [0, 0, 0, (1 - 2*nu) / (2 * (1 - nu)), 0, 0],
        [0, 0, 0, 0, (1 - 2*nu) / (2 * (1 - nu)), 0],
        [0, 0, 0, 0, 0, (1 - 2*nu) / (2 * (1 - nu))]
    ], dtype=float)
    
    Dtan *= fac
    return Dtan

# def flattenising_struct(s):
#     for k in s.keys(): 
#         if(s[k].shape[0] == 1 and s[k].shape[1]!=1):
#             s[k] = s[k].flatten()
            


