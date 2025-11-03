#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 10:35:06 2025

@author: frocha
"""
import numpy as np
import numba
from numba import float64


@numba.jit(nopython=True, cache=True)
def assemble_block(K11, K12, K21, K22):
    n = K11.shape[0]  # assuming square sub-blocks
    K = np.empty((2*n, 2*n), dtype= K11.dtype)

    # Fill block by block
    K[0:n,     0:n    ] = K11
    K[0:n,     n:2*n  ] = K12
    K[n:2*n,   0:n    ] = K21
    K[n:2*n,   n:2*n  ] = K22

    return K

@numba.jit(nopython=True, cache=True)
def ten2voigt(T, fac):
    return np.array([T[0,0], T[1,1], T[2,2],
                     fac*T[0,1], fac*T[1,2], fac*T[0,2]], dtype = np.float64)

@numba.jit(nopython=True, cache=True)
def voigt2ten(v, fac):
    T = np.array([[v[0], v[3]/fac, v[5]/fac],
                  [v[3]/fac, v[1], v[4]/fac],
                  [v[5]/fac, v[4]/fac, v[2]]], dtype = np.float64)
    return T

@numba.jit(nopython=True, cache=True)
def get_dofs_given_nodes_ids(nodes_ids):
    DOFs = np.empty(nodes_ids.shape[0] * 3, dtype=np.int64)
    for m, node in enumerate(nodes_ids):
        DOFs[3*m:3*m+3] = np.arange(3*node, 3*(node+1)) # python convention

    return DOFs


FACE_INDEX = np.array([[3, 2, 1, 0], 
                       [5, 6, 7, 4],
                       [1, 5, 4, 0],
                       [1, 2, 6, 5],
                       [2, 3, 7, 6],
                       [4, 7, 3, 0]], dtype=np.int64)

# Convert a 3D vector to its skew-symmetric matrix (cross-product matrix).
@numba.jit(nopython=True, cache=True)
def TransVect2SkewSym(Vect):
    Vect = np.asarray(Vect).flatten()
    SkewSym = np.array([[0, -Vect[2], Vect[1]],
                        [Vect[2], 0, -Vect[0]],
                        [-Vect[1], Vect[0], 0]])
    return SkewSym


# get N1 and N2 as dN
@numba.jit(nopython=True, cache=True)
def get_surface_geometry(N, dN, SurfXYZ):
    """
    Compute surface geometry quantities from shape functions and nodal coordinates.

    Parameters
    ----------
    N, dN : (n_nodes,)  and (2,n_nodes) arrays
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
    NTx = dN@SurfXYZ # (2,4)x(4,3) --> (2,3)

    n = np.cross(NTx[0,:], NTx[1,:])
    J = np.linalg.norm(n)
    n = n / J  # normalize

    return n, J, NTx[0,:], NTx[1,:], x


@numba.jit(nopython=True, cache=True)
def get_surface_frame(N, dN, SurfXYZ):
    """
    Compute surface geometry quantities from shape functions and nodal coordinates.

    Parameters
    ----------
    N, dN : (n_nodes,)  and (2,n_nodes) arrays
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
    
    frame = np.empty((3, 3), dtype=np.float64) # (:, [normal, t1, t2], ndim) not the best convention
    x = SurfXYZ.T@N
    frame[1:3] = dN @ SurfXYZ #  tangents
    frame[0] = np.cross(frame[1], frame[2]) # normal
    J = np.linalg.norm(frame[0])
    frame[0] /= J
    return x, frame, J

@numba.jit(nopython=True, cache=True)
def GetSurfaceNode(elementLE, SurfSign, matlab_shift = 0):
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

    return elementLE[FACE_INDEX[SurfSign-matlab_shift]].astype(np.int64) - matlab_shift

#@numba.jit(nopython=True, cache=True)
def GetSurfaceXYZ(cells, X, Disp, surf):
    SurfNodes = GetSurfaceNode(cells[surf[0],:], surf[1])
    SurfDOF = get_dofs_given_nodes_ids(SurfNodes)
    XYZ = get_deformed_position_given_dofs(SurfNodes, X, Disp, SurfDOF)
    return XYZ, SurfDOF


@numba.jit(nopython=True, cache=True)
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

# === Surface shape function and derivatives ===
@numba.jit(nopython=True, cache=True)
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

# return deformed coordinates
@numba.jit(nopython=True, cache=True)
def get_deformed_position(nodes_ids, nodes, disp):
    return nodes[nodes_ids, :] + disp.reshape((-1,3))[nodes_ids, :] 

# return deformed coordinates
@numba.jit(nopython=True, cache=True)
def get_deformed_position_given_dofs(DOFs, nodes, disp):
    return (nodes.reshape((-1,))[DOFs] + disp[DOFs]).reshape((-1,3)) 


sig = 'float64[:,:](float64, float64)'
@numba.jit([sig], nopython=True, cache=True)
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
    ], dtype=np.float64)
    
    Dtan *= fac
    return Dtan

# def flattenising_struct(s):
#     for k in s.keys(): 
#         if(s[k].shape[0] == 1 and s[k].shape[1]!=1):
#             s[k] = s[k].flatten()
            


