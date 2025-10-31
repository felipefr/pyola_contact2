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
    def __init__(self, meshfile, cell_type = 'hexahedron', facets_id = [], 
                  force_bnd_id = None, dirichlet_bnd_id = None, param=None):
        self.meshfile = meshfile
        m = meshio.read(meshfile)
        self.X = m.points.astype(np.float64)
        self.cells = m.cells_dict[cell_type].astype(np.int64) # python convention
        self.param = param
        self.n_cells = len(self.cells)
        self.n_nodes = len(self.X)
        self.ndim = self.X.shape[1]
        
        # Force Bcs
        ForceNode = m.point_sets[force_bnd_id].astype(np.int64) + 1 # matlab convention
        self.ExtF = np.zeros((len(ForceNode), 3)).astype(np.float64)
        for i, node in enumerate(ForceNode):
            self.ExtF[i, :] = [node, 2, -4e4]
        
        # Displacement boundary conditions
        ConNode = m.point_sets[dirichlet_bnd_id].astype(np.int64) + 1 # matlab convention
        self.Cons = np.zeros((len(ConNode) * 3, 3)).astype(np.float64)
        for i, node in enumerate(ConNode):
            for j in range(self.ndim):
                self.Cons[3 * i + j, :] = [node, j+1, 0]
                
        # Contact boundaries
        self.facets = []
        for name, surf_id in facets_id:
            facets =  m.cell_sets_dict[name][cell_type].astype(np.int64)
            facets = np.vstack((facets, surf_id * np.ones_like(facets)))
            self.facets.append(facets)

        
        gp = 1.0 / np.sqrt(3.0)
        self.SurfIPs = np.array([ [-gp, -gp], [ gp, -gp], [ gp,  gp], [-gp,  gp]], dtype = np.float64) 
        
        self.ShpfSurf = [ GetSurfaceShapeFunction(ip) for ip in self.SurfIPs ]
        
        self.Prop = np.array([[210000, 0.3]], dtype = np.float64)