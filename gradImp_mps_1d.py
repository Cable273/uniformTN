#!/usr/bin/env python# -*- coding: utf-8 -*-
from __future__ import division
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from scipy.sparse import linalg as sparse_linalg
from ncon import ncon
from einsumt import einsumt as einsum
from abc import ABC, abstractmethod
from uniformTN_transfers import *
import copy

from uniformTN_Hamiltonians import *

#Abstract implementations
class gradImplementation_uniform_1d(ABC):
    def __init__(self,psi,H):
        self.psi = psi
        self.H = H

    @abstractmethod
    def buildLeftEnv(self):
        pass
    @abstractmethod
    def buildRightEnv(self):
        pass
    @abstractmethod
    def getCentralTerms(self):
        pass

# -------------------------------------------------------------------------------------------------------------------------------------
#Concrete implementations
# -------------------------------------------------------------------------------------------------------------------------------------
#gradImplementation_uniform_1d 
class gradImplementation_uniform_1d_oneSite_oneBodyH(gradImplementation_uniform_1d):
    def getCentralTerms(self):
        return ncon([self.psi.mps,self.H,self.psi.L.tensor,self.psi.R.tensor],((1,3,4),(-2,1),(-5,3),(-6,4)),forder=(-2,-5,-6))
    def buildLeftEnv(self):
        return ncon([self.psi.mps,self.H,self.psi.mps.conj(),self.psi.L.tensor],((1,3,-5),(2,1),(2,4,-6),(4,3)),forder=(-6,-5),order=(4,3,1,2))
    def buildRightEnv(self):
        return ncon([self.psi.mps,self.H,self.psi.mps.conj(),self.psi.R.tensor],((1,-3,4),(2,1),(2,-6,5),(5,4)),forder=(-6,-3),order=(5,4,1,2))

class gradImplementation_uniform_1d_oneSite_twoBodyH(gradImplementation_uniform_1d):
    def getCentralTerms(self):
        grad = ncon([self.psi.mps,self.psi.mps,self.H,self.psi.mps.conj(),self.psi.L.tensor,self.psi.R.tensor],((1,5,6),(3,6,7),(-2,4,1,3),(4,-9,8),(-10,5),(8,7)),forder=(-2,-10,-9),order=(7,8,3,4,6,1,5))
        grad += ncon([self.psi.mps,self.psi.mps,self.H,self.psi.mps.conj(),self.psi.L.tensor,self.psi.R.tensor],((1,5,6),(3,6,7),(2,-4,1,3),(2,10,-9),(10,5),(-8,7)),forder=(-4,-9,-8),order=(10,5,1,2,6,3,7))
        return grad
    def buildLeftEnv(self):
        return ncon([self.psi.mps,self.psi.mps,self.H,self.psi.mps.conj(),self.psi.mps.conj(),self.psi.L.tensor],((1,5,6),(3,6,-7),(2,4,1,3),(2,10,9),(4,9,-8),(10,5)),forder=(-8,-7),order=(10,5,1,2,6,9,3,4))
    def buildRightEnv(self):
        return ncon([self.psi.mps,self.psi.mps,self.H,self.psi.mps.conj(),self.psi.mps.conj(),self.psi.R.tensor],((1,-5,6),(3,6,7),(2,4,1,3),(2,-10,9),(4,9,8),(8,7)),forder=(-10,-5))

# -----------------------------
class gradImplementation_uniform_1d_oneSiteLeft_oneBodyH(gradImplementation_uniform_1d):
    def getCentralTerms(self):
        return ncon([self.psi.mps,self.H,self.psi.R.tensor],((1,-3,4),(-2,1),(-5,4)),forder=(-2,-3,-5))
    def buildLeftEnv(self):
        return ncon([self.psi.mps,self.H,self.psi.mps.conj()],((1,3,-4),(2,1),(2,3,-5)),forder=(-5,-4),order=(3,1,2))
    def buildRightEnv(self):
        return ncon([self.psi.mps,self.H,self.psi.mps.conj(),self.psi.R.tensor],((1,-3,4),(2,1),(2,-6,5),(5,4)),forder=(-6,-3),order=(5,4,1,2))

class gradImplementation_uniform_1d_oneSiteLeft_twoBodyH(gradImplementation_uniform_1d):
    def getCentralTerms(self):
        grad = ncon([self.psi.mps,self.psi.mps,self.H,self.psi.mps.conj(),self.psi.R.tensor],((1,-5,6),(3,6,7),(-2,4,1,3),(4,-9,8),(8,7)),forder=(-2,-5,-9),order=(8,7,3,4,6))
        grad += ncon([self.psi.mps,self.psi.mps,self.H,self.psi.mps.conj(),self.psi.R.tensor],((1,5,6),(3,6,7),(2,-4,1,3),(2,5,-9),(-8,7)),forder=(-4,-9,-8),order=(5,1,2,6,7))
        return grad
    def buildLeftEnv(self):
        return ncon([self.psi.mps,self.psi.mps,self.H,self.psi.mps.conj(),self.psi.mps.conj()],((1,5,6),(3,6,-7),(2,4,1,3),(2,5,9),(4,9,-8)),forder=(-8,-7),order=(5,1,2,6,9,3,4))
    def buildRightEnv(self):
        return ncon([self.psi.mps,self.psi.mps,self.H,self.psi.mps.conj(),self.psi.mps.conj(),self.psi.R.tensor],((1,-5,6),(3,6,7),(2,4,1,3),(2,-10,9),(4,9,8),(8,7)),forder=(-10,-5))

# -----------------------------
class gradImplementation_uniform_1d_twoSiteLeft(gradImplementation_uniform_1d):
    def __init__(self,psi,imp_site1,imp_site2):
        self.psi = psi
        self.imp_site1 = imp_site1
        self.imp_site2 = imp_site2
    def getCentralTerms(self):
        return 1/2*(self.imp_site1.getCentralTerms() + self.imp_site2.getCentralTerms())
    def buildLeftEnv(self):
        return 1/2*(self.imp_site1.buildLeftEnv() + self.imp_site2.buildLeftEnv())
    def buildRightEnv(self):
        return 1/2*(self.imp_site1.buildRightEnv() + self.imp_site2.buildRightEnv())

class gradImplementation_uniform_1d_twoSiteLeft_oneBodyH_site1(gradImplementation_uniform_1d):
    def getCentralTerms(self):
        return ncon([self.psi.mps,self.H,self.psi.R.tensor],((1,-2,-4,5),(-3,1),(-6,5)),forder=(-3,-2,-4,-6))
    def buildLeftEnv(self):
        return ncon([self.psi.mps,self.H,self.psi.mps.conj()],((1,2,4,-5),(3,1),(3,2,4,-6)),forder=(-6,-5))
    def buildRightEnv(self):
        return ncon([self.psi.mps,self.H,self.psi.mps.conj(),self.psi.R.tensor],((1,2,-4,5),(3,1),(3,2,-7,6),(6,5)),forder=(-7,-4))
class gradImplementation_uniform_1d_twoSiteLeft_oneBodyH_site2(gradImplementation_uniform_1d):
    def getCentralTerms(self):
        return ncon([self.psi.mps,self.H,self.psi.R.tensor],((-1,2,-4,5),(-3,2),(-6,5)),forder=(-1,-3,-4,-6))
    def buildLeftEnv(self):
        return ncon([self.psi.mps,self.H,self.psi.mps.conj()],((1,2,4,-5),(3,2),(1,3,4,-6)),forder=(-6,-5))
    def buildRightEnv(self):
        return ncon([self.psi.mps,self.H,self.psi.mps.conj(),self.psi.R.tensor],((1,2,-4,5),(3,2),(1,3,-7,6),(6,5)),forder=(-7,-4))

class gradImplementation_uniform_1d_twoSiteLeft_twoBodyH_site1(gradImplementation_uniform_1d):
    def getCentralTerms(self):
        return ncon([self.psi.mps,self.H,self.psi.R.tensor],((1,2,-5,6),(-3,-4,1,2),(-7,6)),forder=(-3,-4,-5,-7),order=(6,2,1))
    def buildLeftEnv(self):
        return ncon([self.psi.mps,self.H,self.psi.mps.conj()],((1,2,5,-6),(3,4,1,2),(3,4,5,-7)),forder=(-7,-6),order=(5,1,3,2,4))
    def buildRightEnv(self):
        return ncon([self.psi.mps,self.H,self.psi.mps.conj(),self.psi.R.tensor],((1,2,-5,6),(3,4,1,2),(3,4,-8,7),(7,6)),forder=(-8,-5),order=(7,6,2,4,1,3))
class gradImplementation_uniform_1d_twoSiteLeft_twoBodyH_site2(gradImplementation_uniform_1d):
    def getCentralTerms(self):
        grad = ncon([self.psi.mps,self.psi.mps,self.H,self.psi.mps.conj(),self.psi.R.tensor],((-1,2,-7,8),(3,4,8,9),(-5,6,2,3),(6,4,-11,10),(10,9)),forder=(-1,-5,-7,-11),order=(10,9,4,3,6,8,2))
        grad += ncon([self.psi.mps,self.psi.mps,self.H,self.psi.mps.conj(),self.psi.R.tensor],((1,2,7,8),(3,-4,8,9),(5,-6,2,3),(1,5,7,-11),(-10,9)),forder=(-6,-4,-11,-10),order=(7,1,2,5,8,3,9))
        return grad
    def buildLeftEnv(self):
        return ncon([self.psi.mps,self.psi.mps,self.H,self.psi.mps.conj(),self.psi.mps.conj()],((1,2,7,8),(3,4,8,-9),(5,6,2,3),(1,5,7,11),(6,4,11,-10)),forder=(-10,-9),order=(7,1,2,5,8,11,3,6,4))
    def buildRightEnv(self):
        return ncon([self.psi.mps,self.psi.mps,self.H,self.psi.mps.conj(),self.psi.mps.conj(),self.psi.R.tensor],((1,2,-7,8),(3,4,8,9),(5,6,2,3),(1,5,-12,11),(6,4,11,10),(10,9)),forder=(-12,-7),order=(10,9,4,3,6,8,11,2,5,1))
# -----------------------------

class gradImplementation_bipartite_1d(gradImplementation_uniform_1d):
    def __init__(self,psi,H,H_index,grad_index):
        super().__init__(psi,H)
        self.H_index = H_index #the index above the first site of H
        self.grad_index = grad_index #the mps tensor to take gradient of
        if self.grad_index == 1:
            self.index1 = 1
            self.index2 = 2
        elif self.grad_index == 2:
            self.index1 = 2
            self.index2 = 1

class gradImplementation_bipartite_1d_left_oneBodyH(gradImplementation_bipartite_1d):
    def getCentralTerms(self):
        if self.H_index == self.grad_index:
            return ncon([self.psi.mps[self.index1],self.H,self.psi.R[self.index2].tensor],((1,-3,4),(-2,1),(-5,4)),forder=(-2,-3,-5))
        else:
            return np.zeros(self.psi.mps[self.index1].shape).astype(complex)
    def buildLeftEnv(self):
        if self.H_index == self.grad_index:
            return ncon([self.psi.mps[self.index1],self.H,self.psi.mps[self.index1].conj(),self.psi.mps[self.index2],self.psi.mps[self.index2].conj()],((1,3,4),(2,1),(2,3,5),(6,4,-7),(6,5,-8)),forder=(-8,-7),order=(3,1,2,4,5,6))
        else:
            return ncon([self.psi.mps[self.index2],self.H,self.psi.mps[self.index2].conj()],((1,3,-4),(2,1),(2,3,-5)),forder=(-5,-4),order=(3,1,2))
    def buildRightEnv(self):
        if self.H_index == self.grad_index:
            return ncon([self.psi.mps[self.index1],self.H,self.psi.mps[self.index1].conj(),self.psi.R[self.index2].tensor,self.psi.mps[self.index2],self.psi.mps[self.index2].conj()],((1,3,4),(2,1),(2,6,5),(5,4),(7,-9,3),(7,-8,6)),forder=(-8,-9),order=(5,4,1,2,3,6,7))
        else:
            return ncon([self.psi.mps[self.index2],self.H,self.psi.mps[self.index2].conj(),self.psi.R[self.index1].tensor],((1,-3,4),(2,1),(2,-6,5),(5,4)),forder=(-6,-3),order=(5,4,1,2))

class gradImplementation_bipartite_1d_left_twoBodyH(gradImplementation_bipartite_1d):
    def getCentralTerms(self):
        if self.H_index == self.grad_index:
            return ncon([self.psi.mps[self.index1],self.psi.mps[self.index2],self.H,self.psi.mps[self.index2].conj(),self.psi.R[self.index1].tensor],((1,-5,6),(3,6,7),(-2,4,1,3),(4,-9,8),(8,7)),forder=(-2,-5,-9),order=(8,7,3,4,6))
        else:
            return ncon([self.psi.mps[self.index2],self.psi.mps[self.index1],self.H,self.psi.mps[self.index2].conj(),self.psi.R[self.index2].tensor],((1,5,6),(3,6,7),(2,-4,1,3),(2,5,-9),(-8,7)),forder=(-4,-9,-8),order=(5,1,2,6,7))
    def buildLeftEnv(self):
        if self.H_index == self.grad_index:
            return ncon([self.psi.mps[self.index1],self.psi.mps[self.index2],self.H,self.psi.mps[self.index1].conj(),self.psi.mps[self.index2].conj()],((1,5,6),(3,6,-7),(2,4,1,3),(2,5,9),(4,9,-8)),forder=(-8,-7),order=(5,1,2,6,9,3,4))
        else:
            return ncon([self.psi.mps[self.index2],self.psi.mps[self.index1],self.H,self.psi.mps[self.index2].conj(),self.psi.mps[self.index1].conj(),self.psi.mps[self.index2],self.psi.mps[self.index2].conj()],((1,5,6),(3,6,7),(2,4,1,3),(2,5,9),(4,9,8),(10,7,-12),(10,8,-11)),forder=(-11,-12),order=(5,1,2,6,9,3,4,7,8,10))
    def buildRightEnv(self):
        if self.H_index == self.grad_index:
            return ncon([self.psi.mps[self.index1],self.psi.mps[self.index2],self.H,self.psi.mps[self.index1].conj(),self.psi.mps[self.index2].conj(),self.psi.R[self.index1].tensor,self.psi.mps[self.index2],self.psi.mps[self.index2].conj()],((1,5,6),(3,6,7),(2,4,1,3),(2,10,9),(4,9,8),(8,7),(11,-13,5),(11,-12,10)),forder=(-12,-13))
        else:
            return ncon([self.psi.mps[self.index2],self.psi.mps[self.index1],self.H,self.psi.mps[self.index2].conj(),self.psi.mps[self.index1].conj(),self.psi.R[self.index2].tensor],((1,-5,6),(3,6,7),(2,4,1,3),(2,-10,9),(4,9,8),(8,7)),forder=(-10,-5))

