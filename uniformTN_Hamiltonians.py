#!/usr/bin/env python# -*- coding: utf-8 -*-

from __future__ import division
import math
import numpy as np
import scipy as sp

class localH:
    def __init__(self,H):
        self.matrix = H

class oneBodyH(localH):
    def __init__(self,H):
        super().__init__(H)
        self.tensor = self.matrix.view()
        self.tensor.shape = np.array([2,2])

class twoBodyH(localH):
    def __init__(self,H):
        super().__init__(H)
        self.tensor = self.matrix.view()
        self.tensor.shape = np.array([2,2,2,2])

class plaquetteH(localH):
    def __init__(self,H):
        super().__init__(H)
        self.tensor = self.matrix.view()
        self.tensor.shape = np.array([2,2,2,2,2,2,2,2])



