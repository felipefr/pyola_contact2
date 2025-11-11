#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 21:07:07 2025

@author: felipe
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from fem_lib import *
from contact_lib import *
from utils import *
from timeit import default_timer as timer
from mesh import *
from contact_pairs import ContactPairs

I3 = np.eye(3)

gp = 1.0 / np.sqrt(3.0)
SurfIPs = np.array([ [-gp, -gp], [ gp, -gp], [ gp,  gp], [-gp,  gp]], dtype = np.float64) 

ShpfSurf = [ GetSurfaceShapeFunction(ip) for ip in SurfIPs ]

for i in range(4):
    N, dN = ShpfSurf[0]
    x = np.random.rand(12)
    tau = dN@x.reshape((4,3))
    B = np.kron(dN, I3).reshape((2,3,12))
    tau2 = B@x
    assert np.allclose(tau, tau2)