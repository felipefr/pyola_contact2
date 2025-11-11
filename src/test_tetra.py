#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 15:56:46 2025

@author: felipe
"""

from mesh import *

# --- Mesh and model ---
FEMod = MeshMSH('mesh1.msh', 
             facets_id = [5, 11], # slave, master
             mark_force_bnd = lambda x: (x[:,1] == 50.) & (x[:,2]==19.75),
             mark_dirichlet_bnd = lambda x: x[:,1]==0)
