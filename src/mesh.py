#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 12:54:49 2025

@author: frocha
"""

import numpy as np
from utils import *

import numpy as np
import meshio

class Mesh:
    def __init__(self, meshfile, param=None):
        self.meshfile = meshfile
        m = meshio.read(meshfile)
        self.X = m.points.astype(np.float64)
        self.cells = m.cells_dict['hexahedron'].astype(np.int64) # python convention
        self.param = param
        self.n_cells = len(self.cells)
        self.n_nodes = len(self.X)
        self.ndim = self.X.shape[1]
        
        # Force Bcs
        ForceNode = m.point_sets['SET-4'].astype(np.int64) + 1 # matlab convention
        self.ExtF = np.zeros((len(ForceNode), 3)).astype(np.float64)
        for i, node in enumerate(ForceNode):
            self.ExtF[i, :] = [node, 2, -4e4]
        
        # Displacement boundary conditions
        ConNode = m.point_sets['CONNODE'].astype(np.int64) + 1 # matlab convention
        self.Cons = np.zeros((len(ConNode) * 3, 3)).astype(np.float64)
        for i, node in enumerate(ConNode):
            self.Cons[3 * i + 0, :] = [node, 1, 0]
            self.Cons[3 * i + 1, :] = [node, 2, 0]
            self.Cons[3 * i + 2, :] = [node, 3, 0]
        
        # Contact boundaries
        self.MasterSurf = m.cell_sets_dict['_MASTERSURF_S4']['hexahedron'].astype(np.int64) + 1 # matlab convention
        self.SlaveSurf = m.cell_sets_dict['_SLAVESURF_S6']['hexahedron'].astype(np.int64) + 1 # matlab convention
        self.MasterSurf = np.vstack((self.MasterSurf, 4 * np.ones_like(self.MasterSurf)))
        self.SlaveSurf = np.vstack((self.SlaveSurf, 6 * np.ones_like(self.SlaveSurf)))
        

        self.nMasterSurf = self.MasterSurf.shape[1]
        self.master_surf_cells = np.array([ GetSurfaceNode(self.cells[self.MasterSurf[0, i] - 1, :], 
                                                            self.MasterSurf[1, i] - 1) 
                                            for i in range(self.nMasterSurf)], dtype = np.int64) 

        self.nSlaveSurf = self.SlaveSurf.shape[1]
        self.slave_surf_cells = np.array([ GetSurfaceNode(self.cells[self.SlaveSurf[0, i] - 1, :], 
                                                            self.SlaveSurf[1, i] - 1) 
                                            for i in range(self.nSlaveSurf)], dtype = np.int64)   
        
        self.master_surf_nodes = np.setdiff1d(self.master_surf_cells.flatten(), [])
        
        gp = 1.0 / np.sqrt(3.0)
        self.SurfIPs = np.array([ [-gp, -gp], [ gp, -gp], [ gp,  gp], [-gp,  gp]], dtype = np.float64) 
        
        self.ShpfSurf = [ GetSurfaceShapeFunction(ip) for ip in self.SurfIPs ]
        
        
        self.FricFac = 0.0
        self.Prop = np.array([[210000, 0.3]], dtype = np.float64)
        