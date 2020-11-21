#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 20:41:15 2020

@author: franco
"""

class Material():
    
    def __init__(self, name, density, n_int, young, poisson):
        self.name = name
        self.density = density
        self.n_int = n_int
        self.poisson = poisson
        self.young = young
        
        
