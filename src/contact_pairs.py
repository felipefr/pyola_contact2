#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 14:01:20 2025

@author: frocha
"""

import numpy as np
from utils import *

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
                self.SlaveSurf[:, k] = self.SlaveSurf_mesh[:, i] + 1
                self.SlaveIntegralPoint[k] = j # keep MATLAB-style 1-based Gauss index
                
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
    


