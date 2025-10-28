#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 17:17:18 2025

@author: frocha
"""

import numba
import numpy as np
from utils import *
from scipy.spatial import cKDTree
from sklearn.neighbors import KDTree


@numba.jit(nopython=True)
def ray_triangle_moller_trumbore(o, d, vT, eps=1e-9):
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
    v0, v1, v2 = vT[0], vT[1], vT[2] 
    
    e1 = v1 - v0
    e2 = v2 - v0
    pvec = np.cross(d, e2)
    det = np.dot(e1, pvec)
    
    if abs(det) < eps:
        return False, None, None, None, None  # Ray parallel to triangle
    
    inv_det = 1.0 / det
    tvec = o - v0
    u = np.dot(tvec, pvec) * inv_det
    # if u < 0.0 or u > 1.0:
        # return False, None, None, None, None
    
    qvec = np.cross(tvec, e1)
    v = np.dot(d, qvec) * inv_det
    # if v < 0.0 or u + v > 1.0:
        # return False, None, None, None, None
    
    t = np.dot(e2, qvec) * inv_det
    # if t <= 0:
        # return False, None, None, None, None  # Intersection behind ray origin
    
    p = o + t * d
    return True, t, u, v, p


def raytracing_quad(SlavePoint, SlavePointTan, MasterSurfXYZ, Exist, Tol):
    rs = []
    Exist_list = []
    v = np.cross(SlavePointTan[:, 0], SlavePointTan[:, 1])
    v /= np.linalg.norm(v)
    
    hit1 = ray_triangle_moller_trumbore(SlavePoint, v, MasterSurfXYZ[[0,1,2]], Tol)
    hit2 = ray_triangle_moller_trumbore(SlavePoint, v, MasterSurfXYZ[[0,2,3]], Tol)
    
    if(hit1[0]):
        u = hit1[2]
        v = hit1[3]
        rs_trial = np.array([-1 + 2*(u + v), -1 + 2*v], dtype = np.float64)
        if(np.max(np.abs(rs))<=1.01): 
            rs.append(rs_trial)    
            Exist_list.append(Exist)
            
    if(hit2[0]):
        u = hit2[2]
        v = hit2[3]
        rs_trial = np.array([-1 + 2*u, -1 + 2*(u + v)], dtype = np.float64)
        if(np.max(np.abs(rs))<=1.01): 
            rs.append(rs_trial)    
            Exist_list.append(Exist)
        
    # if((not hit1[0]) and (not hit1[0])):
    #     rs.append(np.array([1e5,0]))
    #     Exist_list.append(-1)
        
    return rs, Exist_list





def ContactSearch(FEMod, ContactPairs, Disp, IntegralPoint):
    """
    ContactSearch - conservative translation from MATLAB
    (GetContactPointbyRayTracing still in Octave, 1-based safe)
    """
    
    nPairs = ContactPairs.SlaveSurf.shape[1]
    MasterSurf_ = FEMod.MasterSurf - 1
    SlaveSurf = ContactPairs.SlaveSurf.astype(np.int64) - 1

    # MasterSurfXYZ = np.array([ get_deformed_position(msc, FEMod.X, Disp) for msc in FEMod.master_surf_cells]).reshape((-1,3))
    MasterSurfNodeXYZ = get_deformed_position(FEMod.master_surf_nodes, FEMod.X, Disp)
    # tree = KDTree(MasterSurfXYZ)
    # MasterSurfXYZ = None 
    tree = None
    
    SlaveSurfNodeXYZ =  np.array([ get_deformed_position(ssc, FEMod.X, Disp) for ssc in FEMod.slave_surf_cells]) # 120x4x3
    SlavePoint = np.empty((nPairs, 3))
    SlavePointTan = np.empty((nPairs, 3, 2)) # not the best convention
    
    for i in range(FEMod.slave_surf_cells.shape[0]):
        for j in range(4):
            ipair = 4*i + j
            SlavePoint[ipair, :] = SlaveSurfNodeXYZ[i].T@FEMod.ShpfSurf[j][0]
            SlavePointTan[ipair, :, :] = (FEMod.ShpfSurf[j][1] @ SlaveSurfNodeXYZ[i]).T

    for i in range(FEMod.slave_surf_cells.shape[0]):        
        for j in range(4):
            ipair = 4*i + j
            rr, ss, MasterEle, MasterSign, gg, Exist = GetContactPointbyRayTracing(
                FEMod.cells, FEMod.X, FEMod.master_surf_cells, FEMod.master_surf_nodes, 
                MasterSurfNodeXYZ, MasterSurf_, tree, Disp, SlavePoint[ipair,:], SlavePointTan[ipair,:,:])
            
            # Eles, Nodes, master_surf_cells, master_surf_nodes, master_surf_nodesXYZ,
            #                                 MasterSurf, tree, Disp, SlavePoint, SlavePointTan
            
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

    return rs, Exist

# Todo1: node to python convention
# Todo2: eliminate repeated conde : "Build DOFs"
# Todo3: automate get deformed coordinates
# Todo4: Find the nearest node can be improved

    

# --- Find node closest to integration point from slave surface ---
@numba.jit(nopython=True)
def nearest_neighbour(Nodes, Disp, master_surf_cells, SlavePoint):
    MinDis = 1e8
    MinMasterPoint = None
    nMasterSurf = master_surf_cells.shape[0]
    
    for i in range(nMasterSurf):
        MasterSurfXYZ = get_deformed_position(master_surf_cells[i], Nodes, Disp)
                
        # Find nearest node to slave point
        ll = MasterSurfXYZ - SlavePoint  # Result is a (4, 3) array
        Distances = np.linalg.norm(ll, axis=1) # Result is a (4,) array
        min_idx = np.argmin(Distances)
        current_min_distance = Distances[min_idx]
        if current_min_distance < MinDis:
            MinDis = current_min_distance
            MinMasterPoint = master_surf_cells[i,min_idx]
    
    return MinMasterPoint
    
# @numba.jit(nopython=True)
def nearest_neighbour2(master_surf_nodes, MasterSurfNodeXYZ, SlavePoint):

    ll = MasterSurfNodeXYZ - SlavePoint  # Result is a (4, 3) array
    Distances = np.linalg.norm(ll, axis=1) # Result is a (4,) array
    min_idx = np.argmin(Distances)
    return master_surf_nodes[min_idx]

def GetContactPointbyRayTracing(Eles, Nodes, master_surf_cells, master_surf_nodes, master_surf_nodesXYZ,
                                MasterSurf, tree, Disp, SlavePoint, SlavePointTan):
    """
    Obtain master surface contact point by ray tracing.
    FEMod numbering follows MATLAB (1-based)
    """

    Tol = 1e-4
    Exist = -1
    
    MinGrow = 0
    Ming = -1e3

    
    SlavePoint_ = SlavePoint.flatten().astype(np.float64)

    # --- Find node closest to integration point from slave surface ---
    # MinMasterPoint = nearest_neighbour(Nodes, Disp, master_surf_cells, SlavePoint)
    MinMasterPoint = nearest_neighbour2(master_surf_nodes, master_surf_nodesXYZ, SlavePoint_)
    # MinMasterPoint = master_surf_nodes[ tree.query(SlavePoint_.reshape((1,3)))[1][0] ]
    # print(MinMasterPoint)
    
    # --- Determine candidate master surfaces ---
    AllMinMasterSurfNum = np.where(master_surf_cells == MinMasterPoint)[0]
    ContactCandidate = np.zeros((AllMinMasterSurfNum.shape[0], 8))
    ContactCandidate[:, 4] = 1e7  # MATLAB column 5
    
    # --- Loop over candidate master surfaces ---
    for idx, surf_idx in enumerate(AllMinMasterSurfNum):
        MasterSurfNode = master_surf_cells[surf_idx, :]        
        MasterSurfXYZ = get_deformed_position(MasterSurfNode, Nodes, Disp)

        # Ray-tracing Newton-Raphson iteration
        # rs, Exist = newton_raphson_raytracing(SlavePoint_, SlavePointTan, MasterSurfXYZ, Exist, Tol)
        
        
        rs_list, Exist_list = raytracing_quad(SlavePoint_, SlavePointTan, MasterSurfXYZ, Exist, Tol)
        # print(Exist2, Exist)
        # print(rs2, rs)
        # input()
        # --- Save nearest RayTracing point ---
        for i, rs in enumerate(rs_list):
            Exist = Exist_list[i]
            if np.max(np.abs(rs)) <= 1.01:
                v = np.cross(SlavePointTan[:, 0], SlavePointTan[:, 1])
                v /= np.linalg.norm(v)
                
                N, _ = GetSurfaceShapeFunction(rs)
                NX = MasterSurfXYZ.T@N
                
                g = np.dot(NX - SlavePoint_, v)
    
                ContactCandidate[idx, 0] = MasterSurf[0, surf_idx]
                ContactCandidate[idx, 1] = MasterSurf[1, surf_idx]
                ContactCandidate[idx, 2:5] = np.array((rs[0], rs[1], g))
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