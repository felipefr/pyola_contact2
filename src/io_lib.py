#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 31 15:23:54 2025

@author: felipe
"""

import numpy as np
import pyvista as pv

def plot_structural_contours(FEMod, point_field, U = None):
    field_name, field_value = list(point_field.items())[0]
    n_nodes = FEMod.X.shape[0]
    n_elems = FEMod.cells.shape[0]
    
    n_verts_per_elem = 8
    cells = np.hstack(
        np.c_[np.full(n_elems, n_verts_per_elem), FEMod.cells]
    ).ravel()
    
    celltypes = np.full(n_elems, pv.CellType.HEXAHEDRON, dtype=np.uint8)
    
    if type(U) == type(None):
        x = FEMod.X
    else:
        x = FEMod.X + U
        
    grid = pv.UnstructuredGrid(cells, celltypes, x)
    
    grid.point_data[field_name] = field_value

    plotter = pv.Plotter()
    plotter.add_mesh(
        grid,
        #style="wireframe", 
        # color="black", 
        # line_width=1,
        scalars=field_name,
        cmap="jet",
        show_edges=True,
        lighting=True,
        smooth_shading=True,
    )
    
    plotter.add_scalar_bar(
        title=field_name,
        vertical=True,
        title_font_size=10,
        label_font_size=8,
        position_x=0.85,
        position_y=0.05,
        height=0.9,
        width=0.05,
    )
    
    plotter.show()