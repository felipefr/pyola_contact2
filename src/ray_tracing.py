#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 18:59:53 2025

@author: frocha
"""


import numba
import numpy as np
from utils import *
from scipy.spatial import cKDTree
# from sklearn.neighbors import KDTree
import copy


@numba.jit(nopython=True, cache=True)
def newton_raphson_raytracing(SlavePoint, SlavePointFrame, MasterSurfXYZ):
    Tol = 1e-8
    rs = np.zeros(2, dtype = np.float64)
    SlavePointTan = SlavePointFrame[1:3]
    
    for j in range(int(1e8)):
        N, dN = GetSurfaceShapeFunction(rs)

        NX = MasterSurfXYZ.T@N
        NTX = dN @ MasterSurfXYZ # (2,4)x(4,3) --> (2,3)
                
        # SlavePointTan is (3,2) , fai is (2,), SlavePoint and Nx are (3,)
        fai = SlavePointTan @ (SlavePoint - NX)

        if j == 100: # failed to converge
            rs[0] = 1e5
            return rs, -1, 1e7

        if np.max(np.abs(fai)) < Tol:
            break

        KT = SlavePointTan@NTX.T
        drs = solve_2x2_system_nb(KT, fai)

        rs += drs
        
    if np.max(np.abs(rs)) > 1+Tol:
        return rs, -1, 1e7
    
    N, _ = GetSurfaceShapeFunction(rs)
    NX = MasterSurfXYZ.T@N
    
    g = np.dot(NX - SlavePoint, SlavePointFrame[0,:])
    Exist = 0 if g>0.0 else 1

    return rs, Exist, g


def GetContactPointbyRayTracing(FEMod, Disp, SlavePoint, SlavePointFrame, MasterSurfXYZ, tree, method = "newton"):
    """
    Obtain master surface contact point by ray tracing.
    FEMod numbering follows MATLAB (1-based)
    """

    SlavePoint_ = SlavePoint.flatten().astype(np.float64)
    Eles = FEMod.cells
    Nodes = FEMod.X
    MasterSurf = FEMod.MasterSurf - 1
    master_surf_cells = FEMod.master_surf_cells 
    master_surf_nodes = FEMod.master_surf_nodes
    
    MinMasterPoint = master_surf_nodes[ tree.query(SlavePoint_.reshape((1,3)))[1][0] ]
    
    # --- Determine candidate master surfaces ---
    AllMinMasterSurfNum = np.where(FEMod.master_surf_cells  == MinMasterPoint)[0]
    ContactCandidate = np.zeros((AllMinMasterSurfNum.shape[0], 6))
    ContactCandidate[:,4] = 1e7 # reserved for the gap
    
    if(method == "newton"):
        raytracing = newton_raphson_raytracing
    else:
        raytracing = raytracing_moller_trumbore_quad
    
    # --- Loop over candidate master surfaces ---
    for idx, surf_idx in enumerate(AllMinMasterSurfNum):
        rs, Exist, g = raytracing(SlavePoint_, SlavePointFrame, MasterSurfXYZ[surf_idx, :, :])
            
        ContactCandidate[idx, 0] = MasterSurf[0, surf_idx]
        ContactCandidate[idx, 1] = MasterSurf[1, surf_idx]
        ContactCandidate[idx, 2:4] = rs[:]
        ContactCandidate[idx, 4] = g
        ContactCandidate[idx, 5] = Exist
        
        
    # --- Final contact outputs ---
    Exist = int(np.max(ContactCandidate[:, -1]))
    if Exist == 0 or Exist == 1:
        idxmin = np.argmin(ContactCandidate[:, 4])
        MasterEle = ContactCandidate[idxmin, 0] + 1
        MasterSign = ContactCandidate[idxmin, 1] + 1
        rr = ContactCandidate[idxmin, 2]
        ss = ContactCandidate[idxmin, 3]
        gg = ContactCandidate[idxmin, 4]
    else:
        MasterEle = 1e10
        MasterSign = 1e10
        rr = 1e10
        ss = 1e10
        gg = 1e10

    return rr, ss, MasterEle, MasterSign, gg, Exist

@numba.jit(nopython=True, cache=True)
def raytracing_moller_trumbore_tri(o, d, v0, v1, v2, eps=1e-9):
    """
    Compute ray-triangle intersection using Möller–Trumbore algorithm.
    
    Parameters
    ----------
    o : array-like, shape (3,)
        Ray origin
    d : array-like, shape (3,)
        Ray direction (need not be normalized)
    v0, v1, v2 : array-like, shape (3,)
        Triangle vertices
    eps : float
        Small threshold to handle floating-point parallelism
    
    Returns
    -------
    hit : bool
        Whether intersection occurs
    t : float
        Distance along the ray to intersection
    u, v : float
        Barycentric coordinates (v0 + u*(v1-v0) + v*(v2-v0))
    p : np.ndarray, shape (3,)
        Intersection point (if hit)
    """    
    e1 = v1 - v0
    e2 = v2 - v0
    pvec = np.cross(d, e2)
    det = np.dot(e1, pvec)
    
    # Ray parallel to triangle
    if abs(det) < eps:
        return -1, 1e7, 1e7, 1e7 # Exists, u, v, g

    inv_det = 1.0 / det
    tvec = o - v0
    u = np.dot(tvec, pvec) * inv_det
    if u < -eps or u > 1.+eps:
        return -1, 1e7, 1e7, 1e7
    
    qvec = np.cross(tvec, e1)
    v = np.dot(d, qvec) * inv_det
    if v < -eps or u + v > 1.+eps :
        return -1, 1e7, 1e7, 1e7
    
    t = np.dot(e2, qvec) * inv_det
    if t > 0: # positive gap
        return 0, u, v, t
    else:
        return 1, u, v, t


@numba.jit(nopython=True, cache=True)
def raytracing_moller_trumbore_quad(SlavePoint, SlavePointFrame, MasterSurfXYZ):

    hit1 = raytracing_moller_trumbore_tri(SlavePoint, SlavePointFrame[0], MasterSurfXYZ[0], MasterSurfXYZ[1], MasterSurfXYZ[2])
    hit2 = raytracing_moller_trumbore_tri(SlavePoint, SlavePointFrame[0], MasterSurfXYZ[0], MasterSurfXYZ[2], MasterSurfXYZ[3])

    triidx = -1
    if (hit1[0] == 1 and hit2[0] == 1):
        if hit1[3] < hit2[3]:
            triidx = 0
        else:
            triidx = 1
    elif (hit1[0] == 1):
        triidx = 0
    elif (hit2[0] == 1):
        triidx = 1
    else:
        return np.zeros(2), -1, 1e7

    if triidx == 0:
        u, v, g = hit1[1], hit1[2], hit1[3]
        rs = np.array([-1 + 2*(u + v), -1 + 2*v], dtype = np.float64)
    else:
        u, v, g = hit2[1], hit2[2], hit2[3]
        rs = np.array([-1 + 2*u, -1 + 2*(u + v)], dtype = np.float64)

    return rs, 1, g




# Todo1: node to python convention
# Todo2: eliminate repeated conde : "Build DOFs"
# Todo3: automate get deformed coordinates
# Todo4: Find the nearest node can be improved

# def GetContactPointbyRayTracing(Eles, Nodes, MasterSurf, Disp, SlavePoint, SlavePointTan):
#     """
#     Obtain master surface contact point by ray tracing.
#     FEMod numbering follows MATLAB (1-based)
#     """

#     Tol = 1e-4
#     Exist = -1
#     MinDis = 1e8
#     MinGrow = 0
#     Ming = -1e3
#     MinMasterPoint = None

#     nMasterSurf = MasterSurf.shape[1]
#     AllMasterNode = np.zeros((nMasterSurf, 4), dtype = np.int64)
    
#     SlavePoint_ = SlavePoint.flatten().astype(np.float64)
    
#     # --- Find node closest to integration point from slave surface ---
#     for i in range(nMasterSurf):
#         # MATLAB element index is 1-based
#         MasterSurfNode = GetSurfaceNode(Eles[MasterSurf[0, i], :],
#                                         MasterSurf[1, i])
#         AllMasterNode[i, :] = MasterSurfNode

#         MasterSurfXYZ = get_deformed_position(MasterSurfNode, Nodes, Disp)
                
#         # Find nearest node to slave point
#         ll = MasterSurfXYZ - SlavePoint_  # Result is a (4, 3) array
#         Distances = np.linalg.norm(ll, axis=1) # Result is a (4,) array
#         min_idx = np.argmin(Distances)
#         current_min_distance = Distances[min_idx]
#         if current_min_distance < MinDis:
#             MinDis = current_min_distance
#             MinMasterPoint = MasterSurfNode[min_idx]
    
    
#     # --- Determine candidate master surfaces ---
#     AllMinMasterSurfNum = np.where(AllMasterNode == MinMasterPoint)[0]
#     ContactCandidate = np.zeros((AllMinMasterSurfNum.shape[0], 8))
#     ContactCandidate[:, 4] = 1e7  # MATLAB column 5
    
#     # --- Loop over candidate master surfaces ---
#     for idx, surf_idx in enumerate(AllMinMasterSurfNum):
#         MasterSurfNode = AllMasterNode[surf_idx, :]        
#         MasterSurfXYZ = get_deformed_position(MasterSurfNode, Nodes, Disp)

#         # Ray-tracing Newton-Raphson iteration
#         rs, Exist = newton_raphson_raytracing(SlavePoint_, SlavePointTan, MasterSurfXYZ, Exist, Tol)

#         # --- Save nearest RayTracing point ---
#         if np.max(np.abs(rs)) <= 1.01:
#             v = np.cross(SlavePointTan[:, 0], SlavePointTan[:, 1])
#             v /= np.linalg.norm(v)
            
#             N, _ = GetSurfaceShapeFunction(rs)
#             NX = MasterSurfXYZ.T@N
            
#             g = np.dot(NX - SlavePoint_, v)

#             ContactCandidate[idx, 0] = MasterSurf[0, surf_idx]
#             ContactCandidate[idx, 1] = MasterSurf[1, surf_idx]
#             ContactCandidate[idx, 2:5] = np.array((rs[0], rs[1], g))
#             ContactCandidate[idx, 5:8] = v

#             if Exist <= 0:
#                 if g >= 0 and abs(Ming) > abs(g):
#                     Exist = 0; MinGrow = idx; Ming = g
#                 elif g < 0:
#                     Exist = 1; MinGrow = idx; Ming = g
#             elif Exist == 1:
#                 if g < 0 and abs(Ming) > abs(g):
#                     Exist = 1; MinGrow = idx; Ming = g

#     # --- Final contact outputs ---
#     if Exist == 0 or Exist == 1:
#         MasterEle = ContactCandidate[MinGrow, 0] + 1
#         MasterSign = ContactCandidate[MinGrow, 1] + 1
#         rr = ContactCandidate[MinGrow, 2]
#         ss = ContactCandidate[MinGrow, 3]
#         gg = ContactCandidate[MinGrow, 4]
#     else:
#         MasterEle = 1e10
#         MasterSign = 1e10
#         rr = 1e10
#         ss = 1e10
#         gg = 1e10

#     return rr, ss, MasterEle, MasterSign, gg, Exist
