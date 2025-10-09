#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  9 10:35:06 2025

@author: frocha
"""
import numpy as np

# Convert a 3D vector to its skew-symmetric matrix (cross-product matrix).
def TransVect2SkewSym(Vect):
    Vect = np.asarray(Vect).flatten()
    SkewSym = np.array([[0, -Vect[2], Vect[1]],
                        [Vect[2], 0, -Vect[0]],
                        [-Vect[1], Vect[0], 0]])
    return SkewSym

def flattenising_struct(s):
    for k in s.keys(): 
        if(s[k].shape[0] == 1 and s[k].shape[1]!=1):
            s[k] = s[k].flatten()