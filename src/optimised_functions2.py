#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 28 13:48:41 2025

@author: frocha
"""

import numba
import numpy as np
from utils import *
from scipy.spatial import cKDTree
# from sklearn.neighbors import KDTree
import copy


def ContactSearch(FEMod, ContactPairs, Disp, IntegralPoint):
    """
    ContactSearch - conservative translation from MATLAB
    (GetContactPointbyRayTracing still in Octave, 1-based safe)
    """
    
    method = "newton"
    nPairs = ContactPairs.SlaveSurf.shape[1]
    SlaveSurf = ContactPairs.SlaveSurf.astype(np.int64) - 1
    
    MasterSurfXYZ = np.array([ get_deformed_position(msc, FEMod.X, Disp) for msc in FEMod.master_surf_cells]).reshape((-1,4,3)) # redudant computations
    MasterSurfNodeXYZ = get_deformed_position(FEMod.master_surf_nodes, FEMod.X, Disp) # redudant computations
    tree = cKDTree(MasterSurfNodeXYZ)
    
    SlaveSurfNodeXYZ =  np.array([ get_deformed_position(ssc, FEMod.X, Disp) for ssc in FEMod.slave_surf_cells]) # 120x4x3
    SlavePoint = np.empty((nPairs, 3))
    SlavePointFrame = np.empty((nPairs, 3, 3)) # (:, [normal, t1, t2], ndim) not the best convention
    
    for i in range(FEMod.slave_surf_cells.shape[0]):
        for j in range(4):
            ipair = 4*i + j
            SlavePoint[ipair, :] = SlaveSurfNodeXYZ[i].T@FEMod.ShpfSurf[j][0]
            SlavePointFrame[ipair, 1:3, :] = FEMod.ShpfSurf[j][1] @ SlaveSurfNodeXYZ[i]
            SlavePointFrame[ipair, 0, :] = np.cross(SlavePointFrame[ipair, 1, :], SlavePointFrame[ipair, 2, :])
            SlavePointFrame[ipair, 0, :] /= np.linalg.norm(SlavePointFrame[ipair, 0, :])


    for i in range(FEMod.slave_surf_cells.shape[0]):        
        for j in range(4): # gauss point number
            ipair = 4*i + j
            rr, ss, MasterEle, MasterSign, gg, Exist = GetContactPointbyRayTracing(
                FEMod, Disp, SlavePoint[ipair,:], SlavePointFrame[ipair,:,:], MasterSurfXYZ, tree, method)
            
            
            if Exist == 1:
                ContactPairs.CurMasterSurf[:, ipair] = np.array([MasterEle, MasterSign])
                ContactPairs.rc[ipair] = rr
                ContactPairs.sc[ipair] = ss
                ContactPairs.Cur_g[ipair] = gg
            else:
                # print("contact not found at ", i)
                ContactPairs.CurMasterSurf[:, ipair] = 0 
                ContactPairs.rc[ipair] = 0
                ContactPairs.sc[ipair] = 0
                ContactPairs.Cur_g[ipair] = 0
                ContactPairs.CurContactState[ipair] = 0

    return ContactPairs


@numba.jit(nopython=True)
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

@numba.jit(nopython=True)
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


@numba.jit(nopython=True)
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