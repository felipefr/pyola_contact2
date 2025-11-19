#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 12:54:49 2025

@author: frocha
"""

import numpy as np
from utils import *
from fem_lib import get_local_shape_derivative

import numpy as np
import meshio

class MeshINP:
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
        ForceNode = m.point_sets[force_bnd_id].astype(np.int64) # python convention
        self.ExtF = np.zeros((len(ForceNode), 3)).astype(np.float64)
        for i, node in enumerate(ForceNode):
            self.ExtF[i, :] = [node, 1, -4e4]
        
        # Displacement boundary conditions
        ConNode = m.point_sets[dirichlet_bnd_id].astype(np.int64) # python convention
        self.Cons = np.zeros((len(ConNode) * 3, 3)).astype(np.float64)
        for i, node in enumerate(ConNode):
            for j in range(self.ndim):
                self.Cons[3 * i + j, :] = [node, j, 0]
                
        # Facets
        self.facets = []
        self.facets_cells = []
        for f in facets_id:
            
            if(len(facets_id)==2):
                name, surf_id = f
                facets_cells =  m.cell_sets_dict[name][cell_type].astype(np.int64)
                facets = np.array([ GetSurfaceNode(self.cells[fc, :], surf_id) 
                                                    for fc in facets_cells], dtype = np.int64)
                
                facets_cells_ = np.vstack((facets_cells, surf_id * np.ones_like(facets_cells)))
            
                self.facets_cells.append(facets_cells_)
                self.facets.append(facets)
                    
            elif(len(f)==1):
                # not implemented (only self.facets should be appended)
                pass
                

        gp = 1.0 / np.sqrt(3.0)
        self.SurfIPs = np.array([ [-gp, -gp], [ gp, -gp], [ gp,  gp], [-gp,  gp]], dtype = np.float64) 
        
        self.ShpfSurf = [ GetSurfaceShapeFunction(ip) for ip in self.SurfIPs ]
        

        XG = np.array([-0.57735026918963, 0.57735026918963])
        WGT = np.array([1.0, 1.0])
        
        self.IPs = np.array([[x, y, z] for x in XG for y in XG for z in XG])
        self.WIP = np.array([wx*wy*wz for wx in WGT for wy in WGT for wz in WGT], dtype = np.float64)
        self.DSF = np.array([ get_local_shape_derivative(xi) for xi in self.IPs ], dtype = np.float64)
        

class MeshMSH:
    def __init__(self, meshfile, cell_type = 'tetra', facets_id = [], 
                  mark_force_bnd = None, mark_dirichlet_bnd = None, param=None):
        
        
        self.meshfile = meshfile
        m = meshio.read(meshfile)
        self.X = m.points.astype(np.float64)
        self.cells = m.cells_dict[cell_type].astype(np.int64) # python convention
        self.param = param
        self.n_cells = len(self.cells)
        self.n_nodes = len(self.X)
        self.ndim = self.X.shape[1]
        
        ForceNode = np.where(mark_force_bnd(self.X))[0]
        self.ExtF=np.zeros((len(ForceNode),3), dtype=np.float64)
        for i, node in enumerate(ForceNode):
            self.ExtF[i, :] = [node, 1, -4e4]
            
        ConNode = np.where(mark_dirichlet_bnd(self.X))[0] # noeuds à fixer
        self.Cons= np.zeros((len(ConNode)*3, 3), dtype=np.float64) # condition de dirichlet
        for i, node in enumerate(ConNode):
            for j in range(self.ndim):
                self.Cons[3 * i + j, :] = [node, j, 0]
            
            
        self.facets = []
        if(cell_type == 'tetra'):
            facet_type = 'triangle'
            
        triangles = m.cells_dict[facet_type]
        tag = m.cell_data_dict['gmsh:geometrical'][facet_type]
        for f in facets_id:
            self.facets.append(triangles[tag==f])
                
        gp = 1.0 / 3.0
        self.SurfIPs = np.array([ [gp, gp] ], dtype = np.float64) 
        
        self.ShpfSurf = [ GetSurfaceShapeFunction(ip) for ip in self.SurfIPs ]