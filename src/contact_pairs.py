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
    master_surf_cells = None
    
    def __init__(self, FEMod, idx, surf, surf_facet, idxIP, penc = 1e6):
        self.is_active = False
        self.idx = idx
        self.surf = surf # SlaveSurf[0]
        self.surf_facet = surf_facet # SlaveSurf[1] should be removed
        self.idxIP = idxIP # SlaveIntegrationPoint
        self.penc = penc
        
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

        self.surf_nodes = GetSurfaceNode(FEMod.cells[self.surf, :], self.surf_facet)
        self.surf_dofs = get_dofs_given_nodes_ids(self.surf_nodes)       
        
    def update_slave(self, FEMod, Disp): # integral points already chosen
        surfXYZ =  get_deformed_position(self.surf_nodes, FEMod.X, Disp)
        self.point, self.frame, self.J = get_surface_frame(FEMod.ShpfSurf[self.idxIP][0], FEMod.ShpfSurf[self.idxIP][1], surfXYZ)
    
    def update_master(self, FEMod, Disp):
        # Master surface (previous) - Nb uses rp,sp which are already numeric
        self.Nb_old, _ = GetSurfaceShapeFunction(self.Xi_old)
        self.Nb, self.dNb = GetSurfaceShapeFunction(self.Xi)
        
        self.master_surf_nodes_old = GetSurfaceNode(FEMod.cells[self.master_surf_old,:], self.master_surf_facet_old)
        
        self.master_surf_nodes = GetSurfaceNode(FEMod.cells[self.master_surf,:], self.master_surf_facet)
        self.master_surf_dofs = get_dofs_given_nodes_ids(self.master_surf_nodes)
        
        master_surf_XYZ_oldgeo = get_deformed_position(self.master_surf_nodes_old, FEMod.X, Disp)
        master_surf_XYZ = get_deformed_position(self.master_surf_nodes, FEMod.X, Disp)
        
        # attention !! I think there is a bug in the assembling for the matlab version (copying as it is). note that oldgeo is different then old
        self.master_point_oldgeo = self.Nb_old @ master_surf_XYZ_oldgeo 
        self.master_point = self.Nb @ master_surf_XYZ
        
        self.master_tangent = self.dNb @ master_surf_XYZ # master tangents
        
        
    def update_old(self, FEMod, PreDisp):
        self.surfXYZ_old =  get_deformed_position(self.surf_nodes, FEMod.X, PreDisp)
        self.point_old, self.frame_old, self.J_old = get_surface_frame(FEMod.ShpfSurf[self.idxIP][0], 
                                                                       FEMod.ShpfSurf[self.idxIP][1], 
                                                                       self.surfXYZ_old)
        
        # Master surface (previous) - Nb uses rp,sp which are already numeric        
        self.master_surf_XYZ_old = get_deformed_position(self.master_surf_nodes, FEMod.X, PreDisp)
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

        self.slave_points = []
        for i in range(nSlave):
            for j in range(nGauss):
                k = i*nGauss + j
                self.slave_points.append(SlavePointData(FEMod, k, self.SlaveSurf_mesh[0,i], self.SlaveSurf_mesh[1,i], j))

        SlavePointData.master_surf_cells = self.master_surf_cells
        
    def update_master_slave_XYZ(self, FEMod, Disp):
        self.master_surf_XYZ = get_deformed_position(self.master_surf_cells.flatten(), FEMod.X, Disp).reshape((-1,4,3)) 
        self.slave_surf_XYZ =  get_deformed_position(self.slave_surf_cells.flatten(), FEMod.X, Disp).reshape((-1,4,3))  # 120x4x3

    def get_contact_dofs(self, FEMod, i):
        contact_dofs = np.concatenate([self.slave_points[i].surf_dofs, 
                                       self.slave_points[i].master_surf_dofs])
        return contact_dofs            

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
        


    def assert_state(self):
        assert_slave_points_equal_arrays(self.slave_points, 
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


def assert_slave_points_equal_arrays(
    slave_points, 
    pc, SlaveSurf, SlaveIntegralPoint,
    CurMasterSurf, rc, sc, Cur_g, Pre_g,
    PreMasterSurf, rp, sp,
    CurContactState, PreContactState,
    Pressure, Traction,
):
    """
    Assert that the data stored in each SlavePointData object
    matches the corresponding entries in the original arrays.
    """
    nPairs = len(pc)
    assert len(slave_points) == nPairs, "Mismatch in number of slave points"

    for k in range(nPairs):
        p = slave_points[k]

        # Scalars
        assert p.penc == pc[k], f"penc mismatch at {k}"
        assert p.surf == SlaveSurf[0, k] - 1, f"surf mismatch at {k}"
        assert p.surf_facet == SlaveSurf[1, k] - 1, f"surf_facet mismatch at {k}"
        assert p.idxIP == SlaveIntegralPoint[k], f"idxIP mismatch at {k}"

        assert p.master_surf == CurMasterSurf[0, k] - 1, f"master_surf mismatch at {k}"
        assert p.master_surf_facet == CurMasterSurf[1, k] - 1, f"master_surf_facet mismatch at {k}"
        assert p.master_surf_old == PreMasterSurf[0, k] - 1, f"master_surf_old mismatch at {k}"
        assert p.master_surf_facet_old == PreMasterSurf[1, k] - 1, f"master_surf_facet_old mismatch at {k}"

        # Coordinates
        assert p.Xi[0] == rc[k], f"Xi[0] mismatch at {k}"
        assert p.Xi[1] == sc[k], f"Xi[1] mismatch at {k}"
        assert p.Xi_old[0] == rp[k], f"Xi_old[0] mismatch at {k}"
        assert p.Xi_old[1] == sp[k], f"Xi_old[1] mismatch at {k}"

        # Gaps
        assert p.gap == Cur_g[k], f"gap mismatch at {k}"
        assert p.gap_old == Pre_g[k], f"gap_old mismatch at {k}"

        # States
        assert p.contact_state == CurContactState[k], f"contact_state mismatch at {k}"
        assert p.contact_state_old == PreContactState[k], f"contact_state_old mismatch at {k}"

        # Response
        assert p.pressure == Pressure[k], f"pressure mismatch at {k}"
        assert p.traction == Traction[k], f"traction mismatch at {k}"


