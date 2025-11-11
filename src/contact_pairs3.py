#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  2 09:46:55 2025

@author: felipe
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 14:01:20 2025

@author: frocha
"""
import numpy as np
from utils import *

from numba import int64, float64, njit
from numba.experimental import jitclass

# spec = [
#     ('penc', float64),          # pc
#     ('surf', int64),            # Slave surface ID
#     ('surf_facet', int64),      # (if you plan to remove it, you can drop it here too)
#     ('idxIP', int64),           # Gauss integration point index

#     ('master_surf', int64),     # CurMasterSurf[0]
#     ('master_surf_facet', int64),  # can also be removed if unnecessary
#     ('master_surf_old', int64),    # PreMasterSurf[0]
#     ('master_surf_facet_old', int64),

#     ('Xi', float64[:]),         # [rc, sc]
#     ('Xi_old', float64[:]),     # [rp, sp]

#     ('gap', float64),           # Cur_g
#     ('gap_old', float64),       # Pre_g

#     ('contact_state', int64),   # CurContactState
#     ('contact_state_old', int64), # PreContactState

#     ('pressure', float64),      # Pressure
#     ('traction', float64),      # Traction
# ]

# @jitclass(spec)
class SlavePointData:
    def __init__(self, idx):
        self.idx = idx
        self.penc = 1e6 # pc
        self.surf = 0 # SlaveSurf[0]
        self.surf_facet = 0 # SlaveSurf[1] should be removed
        self.idxIP = 0 # SlaveIntegrationPoint
        
        self.master_surf = -1 # CurMasterSurf[0]
        self.master_surf_facet = -1 # CurMasterSurf[1] should be removed
        self.master_surf_old = -1 # PreMasterSurf[0]
        self.master_surf_facet_old = -1 # PreMasterSurf[1] should be removed
        
        self.Xi = np.zeros(2, dtype = np.float64) # rc, sc
        self.Xi_old = np.zeros(2, dtype = np.float64) # rp, sp
        
        self.gap = 0.0 # Cur_g
        self.gap_old = 0.0 # Pre_g
        
        self.contact_state = 0 # CurContactState
        self.contact_state_old = 0 # PreContactState
        
        self.pressure = 0.0 # Pressure
        self.traction = 0.0 # Traction
        
    def update(self, FEMod, Disp): # integral points already chosen
        self.surf_nodes = GetSurfaceNode(FEMod.cells[self.surf, :], self.surf_facet)  # todo: do once for all    
        self.surf_dofs = get_dofs_given_nodes_ids(self.surf_nodes)
        self.surfXYZ =  get_deformed_position_given_dofs(self.surf_nodes, FEMod.X, Disp, self.surf_dofs)
        self.frame = np.empty((3, 3), dtype=np.float64) # (:, [normal, t1, t2], ndim) not the best convention
        
        self.point = self.surfXYZ.T@FEMod.ShpfSurf[self.idxIP][0]
        self.frame[1:3] = FEMod.ShpfSurf[self.idxIP][1] @ self.surfXYZ #  tangents
        self.frame[0] = np.cross(self.frame[1], self.frame[2]) # normal
        self.J = np.linalg.norm(self.frame[0])
        self.frame[0] /= self.J
        
        # Master surface (previous) - Nb uses rp,sp which are already numeric
        self.Nb_old, _ = GetSurfaceShapeFunction(self.Xi_old)
        self.Nb, self.dNb = GetSurfaceShapeFunction(self.Xi)
        
        self.master_surf_nodes_old = GetSurfaceNode(FEMod.cells[self.master_surf_old,:], self.master_surf_facet_old)
        self.master_surf_dofs_old = get_dofs_given_nodes_ids(self.master_surf_nodes_old)
        
        self.master_surf_nodes = GetSurfaceNode(FEMod.cells[self.master_surf,:], self.master_surf_facet)
        self.master_surf_dofs = get_dofs_given_nodes_ids(self.master_surf_nodes)
        
        self.master_surf_XYZ_oldgeo = get_deformed_position_given_dofs(self.master_surf_nodes_old, FEMod.X, Disp, self.master_surf_dofs_old)
        self.master_surf_XYZ = get_deformed_position_given_dofs(self.master_surf_nodes, FEMod.X, Disp, self.master_surf_dofs)
        
        # attention !! I think there is a bug in the assembling for the matlab version (copying as it is). note that oldgeo is different then old
        self.master_point_oldgeo = self.Nb_old @ self.master_surf_XYZ_oldgeo 
        self.master_point = self.Nb @ self.master_surf_XYZ
        
        self.master_tangent = self.dNb @ self.master_surf_XYZ # master tangents
        
        
    def update_old(self, FEMod, PreDisp):
        self.surfXYZ_old =  get_deformed_position_given_dofs(self.surf_nodes, FEMod.X, PreDisp, self.surf_dofs)
        self.frame_old = np.empty((3, 3), dtype=np.float64) # (:, [normal, t1, t2], ndim) not the best convention
        
        self.point_old = self.surfXYZ_old.T@FEMod.ShpfSurf[self.idxIP][0]
        self.frame_old[1:3] = FEMod.ShpfSurf[self.idxIP][1] @ self.surfXYZ_old #  tangents
        self.frame_old[0] = np.cross(self.frame_old[1], self.frame_old[2]) # normal
        self.J_old = np.linalg.norm(self.frame_old[0]) # unused
        self.frame_old[0] /= self.J_old # unused
        
        # Master surface (previous) - Nb uses rp,sp which are already numeric        
        self.master_surf_XYZ_old = get_deformed_position_given_dofs(self.master_surf_nodes, FEMod.X, PreDisp, self.master_surf_dofs)
        self.master_point_old = self.Nb @ self.master_surf_XYZ_old
        
    
class ContactPairs:
    """
    ContactPairs - Class to hold and initialize contact pair data for a FEM model.
    """

    def __init__(self, FEMod, nGauss=4, FricFac = 0.0, master_surf_id = 0, slave_surf_id = 1):
        """
        Initialize a ContactPairs object from a FEM model.

        Parameters
        ----------
        FEMod : dict or object
            FEM data structure; must include attribute or key 'SlaveSurf' (2×n array)
        nGauss : int, optional
            Number of Gauss points per slave surface element (default=4)
        """
        
        self.FricFac = FricFac 
        
        
        self.SlaveSurf_mesh = FEMod.facets[slave_surf_id]
        self.MasterSurf_mesh = FEMod.facets[master_surf_id] 
        
        self.nMasterSurf = self.MasterSurf_mesh.shape[1]
        self.master_surf_cells = np.array([ GetSurfaceNode(FEMod.cells[self.MasterSurf_mesh[0, i], :], 
                                                            self.MasterSurf_mesh[1, i]) 
                                            for i in range(self.nMasterSurf)], dtype = np.int64) 
        
        self.nSlaveSurf = self.SlaveSurf_mesh.shape[1]
        self.slave_surf_cells = np.array([ GetSurfaceNode(FEMod.cells[self.SlaveSurf_mesh[0, i], :], 
                                                            self.SlaveSurf_mesh[1, i]) 
                                            for i in range(self.nSlaveSurf)], dtype = np.int64)   
        
        self.master_surf_nodes = np.setdiff1d(self.master_surf_cells.flatten(), [])

        nSlave = self.SlaveSurf_mesh.shape[1]
        nPairs = nSlave * nGauss

        # --- Initialize arrays ---
        self.pc  = 1e6 * np.ones(nPairs) # penalisation
        self.SlaveSurf = np.zeros((2, nPairs), dtype=np.int64)
        self.SlaveIntegralPoint = np.zeros(nPairs, dtype=np.int64)

        self.CurMasterSurf = np.zeros((2, nPairs), dtype=np.int64)
        self.rc = np.zeros(nPairs)
        self.sc = np.zeros(nPairs)
        self.Cur_g = np.zeros(nPairs)
        self.Pre_g = np.zeros(nPairs)
        self.PreMasterSurf = np.zeros((2, nPairs), dtype=np.int64)
        self.rp = np.zeros(nPairs)
        self.sp = np.zeros(nPairs)
        self.CurContactState = np.zeros(nPairs, dtype=np.int64)
        self.PreContactState = np.zeros(nPairs, dtype=np.int64)
        self.Pressure = np.zeros(nPairs)
        self.Traction = np.zeros(nPairs)

        # --- Populate fields based on Gauss integration ---
        for i in range(nSlave):
            for j in range(nGauss):
                k = i * nGauss + j
                self.SlaveSurf[:, k] = self.SlaveSurf_mesh[:, i] + 1 # keep MATLAB-style 1-based Gauss index
                self.SlaveIntegralPoint[k] = j 
                
        self.slave_points = []
        for i in range(nPairs):
            self.slave_points.append(SlavePointData(i))
    
    def update_master_slave_XYZ(self, FEMod, Disp):
        self.master_surf_XYZ = np.array([ get_deformed_position(msc, FEMod.X, Disp) for msc in self.master_surf_cells]).reshape((-1,4,3)) 
        self.slave_surf_XYZ =  np.array([ get_deformed_position(ssc, FEMod.X, Disp) for ssc in self.slave_surf_cells]).reshape((-1,4,3))  # 120x4x3

    def get_contact_dofs(self, FEMod, i):
        # MasterSurfNodes = GetSurfaceNode(FEMod.cells[self.CurMasterSurf[0,i] - 1, :], 
        #                                              self.CurMasterSurf[1,i] - 1 )
        # MasterSurfDOF = get_dofs_given_nodes_ids(MasterSurfNodes)

        # SlaveSurfNodes = GetSurfaceNode(FEMod.cells[self.SlaveSurf[0,i]-1, :], 
        #                                             self.SlaveSurf[1,i]-1) 
        
        # SlaveSurfDOF = get_dofs_given_nodes_ids(SlaveSurfNodes)

        # contact_dofs = np.concatenate([np.asarray(SlaveSurfDOF), 
        #                              np.asarray(MasterSurfDOF)])
            
        contact_dofs = np.concatenate([self.slave_points[i].surf_dofs, 
                                       self.slave_points[i].master_surf_dofs])
        return contact_dofs            

    def update_contact(self):
        """
        Update contact history between time steps.
        """

        nPairs = self.SlaveSurf.shape[1]

        for i in range(nPairs):
            if self.CurContactState[i] == 0:
                # --- No contact ---
                self.PreMasterSurf[:, i] = 0
                self.rp[i] = 0.0
                self.sp[i] = 0.0
                self.PreContactState[i] = 0
                self.Pre_g[i] = 0.0
                self.Pressure[i] = 0.0
                self.Traction[i] = 0.0
            else:
                # --- Slip or stick contact ---
                self.PreMasterSurf[:, i] = self.CurMasterSurf[:, i]
                self.rp[i] = self.rc[i]
                self.sp[i] = self.sc[i]
                self.PreContactState[i] = self.CurContactState[i]
                self.Pre_g[i] = self.Cur_g[i]

            # --- Reset current step quantities ---
            self.rc[i] = 0.0
            self.sc[i] = 0.0
            self.Cur_g[i] = 0.0
            self.CurMasterSurf[:, i] = 0
            self.CurContactState[i] = 0
            
    def update_history_slave_points(self):
        """
        Update contact history between time steps.
        """
        for sp in self.slave_points:
            if sp.contact_state == 0:
                # --- No contact ---
                sp.master_surf_old = -1
                sp.master_surf_facet_old = -1
                sp.Xi_old.fill(0.)
                sp.contact_state_old = 0
                sp.gap_old = 0.0
                sp.pressure = 0.0
                sp.traction = 0.0
            else:
                # --- Slip or stick contact ---
                sp.master_surf_old  = sp.master_surf
                sp.master_surf_facet_old  = sp.master_surf_facet
                sp.Xi_old[:] = sp.Xi[:]
                sp.contact_state_old = sp.contact_state
                sp.gap_old = sp.gap

            # --- Reset current step quantities ---
            sp.Xi.fill(0.)
            sp.gap = 0.0
            sp.master_surf = -1
            sp.master_surf_facet = -1
            sp.contact_state = 0
    
    def copy_state_arrays2slavepoints(self):
        copy_arrays_to_slave_points(self.slave_points, 
                                     self.pc, self.SlaveSurf, self.SlaveIntegralPoint,
                                     self.CurMasterSurf, self.rc, self.sc, self.Cur_g, self.Pre_g,
                                     self.PreMasterSurf, self.rp, self.sp,
                                     self.CurContactState, self.PreContactState,
                                     self.Pressure, self.Traction)    
        

    def copy_state_slavepoints2arrays(self):
        copy_slave_points_to_arrays(self.slave_points, 
                                     self.pc, self.SlaveSurf, self.SlaveIntegralPoint,
                                     self.CurMasterSurf, self.rc, self.sc, self.Cur_g, self.Pre_g,
                                     self.PreMasterSurf, self.rp, self.sp,
                                     self.CurContactState, self.PreContactState,
                                     self.Pressure, self.Traction)    
        
# @njit
def copy_arrays_to_slave_points(slave_points, 
                             pc, SlaveSurf, SlaveIntegralPoint,
                             CurMasterSurf, rc, sc, Cur_g, Pre_g,
                             PreMasterSurf, rp, sp,
                             CurContactState, PreContactState,
                             Pressure, Traction):
    """
    Copy all data from the old array-based structure into a List of SlavePointData.
    """

    nPairs = len(pc)
    for k in range(nPairs):
        p = slave_points[k]

        # Scalar copies
        p.penc = pc[k]
        p.surf = SlaveSurf[0, k] - 1
        p.surf_facet = SlaveSurf[1, k] - 1  # optional
        p.idxIP = SlaveIntegralPoint[k]

        p.master_surf = CurMasterSurf[0, k] - 1
        p.master_surf_facet = CurMasterSurf[1, k] - 1
        p.master_surf_old = PreMasterSurf[0, k] - 1
        p.master_surf_facet_old = PreMasterSurf[1, k] - 1

        # 2D coordinates (Xi, Xi_old)
        p.Xi[0] = rc[k]
        p.Xi[1] = sc[k]
        p.Xi_old[0] = rp[k]
        p.Xi_old[1] = sp[k]

        # Gap quantities
        p.gap = Cur_g[k]
        p.gap_old = Pre_g[k]

        # Contact states
        p.contact_state = CurContactState[k]
        p.contact_state_old = PreContactState[k]

        # Contact response
        p.pressure = Pressure[k]
        p.traction = Traction[k]
        
# @njit
def copy_slave_points_to_arrays(slave_points,
                                pc, SlaveSurf, SlaveIntegralPoint,
                                CurMasterSurf, rc, sc, Cur_g, Pre_g,
                                PreMasterSurf, rp, sp,
                                CurContactState, PreContactState,
                                Pressure, Traction):
    """
    Copy all data from a List of SlavePointData objects back into array form.
    """

    nPairs = len(slave_points)
    for k in range(nPairs):
        p = slave_points[k]

        pc[k] = p.penc
        SlaveSurf[0, k] = p.surf + 1 
        SlaveSurf[1, k] = p.surf_facet + 1  # optional
        SlaveIntegralPoint[k] = p.idxIP

        CurMasterSurf[0, k] = p.master_surf + 1
        CurMasterSurf[1, k] = p.master_surf_facet + 1 

        PreMasterSurf[0, k] = p.master_surf_old + 1
        PreMasterSurf[1, k] = p.master_surf_facet_old + 1

        rc[k] = p.Xi[0]
        sc[k] = p.Xi[1]
        rp[k] = p.Xi_old[0]
        sp[k] = p.Xi_old[1]

        Cur_g[k] = p.gap
        Pre_g[k] = p.gap_old

        CurContactState[k] = p.contact_state
        PreContactState[k] = p.contact_state_old

        Pressure[k] = p.pressure
        Traction[k] = p.traction



