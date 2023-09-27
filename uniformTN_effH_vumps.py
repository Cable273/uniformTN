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
from uniformTN_Hamiltonians import *

#all vumps code is for 1d only
class vumpsEffH:
    def __init__(self,psi,H):
        self.psi = psi
        self.H = H

        self.H_imp = dict()
        for n in range(0,len(self.H.terms)):
            self.H_imp[n] = self.fetch_implementation(self.H.terms[n])

    def fetch_implementation(self,localH_term):
        if type(localH_term) == oneBodyH:
            return vumpsEffH_1d_implementation_oneBodyH(self.psi,localH_term.tensor)
        elif type(localH_term) == twoBodyH:
            return vumpsEffH_1d_implementation_twoBodyH(self.psi,localH_term.tensor)

    def get_effH(self):
        physDim = self.psi.mps.shape[0]

        leftEnv = self.H_imp[0].buildLeftEnv()
        rightEnv = self.H_imp[0].buildRightEnv()
        H_ac = self.H_imp[0].getCentreTerms_Ac()
        H_c = self.H_imp[0].getCentreTerms_C()
        for n in range(1,len(self.H_imp)):
            leftEnv += self.H_imp[n].buildLeftEnv()
            rightEnv += self.H_imp[n].buildRightEnv()
            H_ac += self.H_imp[n].getCentreTerms_Ac()
            H_c += self.H_imp[n].getCentreTerms_C()

        #apply pseudo inverses
        leftEnv = self.psi.Ta_inv.applyRight(leftEnv.reshape(self.psi.D**2)).reshape(self.psi.D,self.psi.D)
        rightEnv = self.psi.Ta_inv.applyLeft(rightEnv.reshape(self.psi.D**2)).reshape(self.psi.D,self.psi.D)

        H_ac += ncon([leftEnv,np.eye(physDim),np.eye(self.psi.D)],((-5,-3),(-2,-1),(-6,-4)),forder=(-2,-5,-6,-1,-3,-4))
        H_ac += ncon([np.eye(self.psi.D),np.eye(physDim),rightEnv],((-5,-3),(-2,-1),(-6,-4)),forder=(-2,-5,-6,-1,-3,-4))
        H_c += ncon([leftEnv,np.eye(self.psi.D)],((-3,-1),(-4,-2)),forder=(-3,-4,-1,-2))
        H_c += ncon([np.eye(self.psi.D),rightEnv],((-3,-1),(-4,-2)),forder=(-3,-4,-1,-2))

        H_ac = H_ac.reshape(physDim*self.psi.D**2,physDim*self.psi.D**2)
        H_c = H_c.reshape(self.psi.D**2,self.psi.D**2)
        return H_ac,H_c

#-------------------------------------------------------------------------------------------------------------------------------------------------
class vumpsEffH_implementation(ABC):
    def __init__(self,psi,H_tensor):
        self.psi = psi
        self.H_tensor = H_tensor

    @abstractmethod
    def buildLeftEnv(self):
        pass
    @abstractmethod
    def buildRightEnv(self):
        pass
    @abstractmethod
    def getCentreTerms_Ac(self):
        pass
    @abstractmethod
    def getCentreTerms_C(self):
        pass

class vumpsEffH_1d_implementation_oneBodyH(vumpsEffH_implementation):
    def buildLeftEnv(self):
        return ncon([self.psi.Al,self.H_tensor,self.psi.Al.conj()],((1,3,-4),(2,1),(2,3,-5)),forder=(-5,-4))
    def buildRightEnv(self):
        return ncon([self.psi.Ar,self.H_tensor,self.psi.Ar.conj()],((1,-3,4),(2,1),(2,-5,4)),forder=(-5,-3))
    def getCentreTerms_Ac(self):
        return ncon([np.eye(self.psi.D),self.H_tensor,np.eye(self.psi.D)],((-5,-3),(-2,-1),(-6,-4)),forder=(-2,-5,-6,-1,-3,-4)).astype(complex)
    def getCentreTerms_C(self):
        return np.zeros((self.psi.D,self.psi.D,self.psi.D,self.psi.D)).astype(complex)

class vumpsEffH_1d_implementation_twoBodyH(vumpsEffH_implementation):
    def buildLeftEnv(self):
        return ncon([self.psi.Al,self.psi.Al,self.H_tensor,self.psi.Al.conj(),self.psi.Al.conj()],((1,5,6),(3,6,-7),(2,4,1,3),(2,5,9),(4,9,-8)),forder=(-8,-7))
    def buildRightEnv(self):
        return ncon([self.psi.Ar,self.psi.Ar,self.H_tensor,self.psi.Ar.conj(),self.psi.Ar.conj()],((1,-5,6),(3,6,7),(2,4,1,3),(2,-9,8),(4,8,7)),forder=(-9,-5))
    def getCentreTerms_Ac(self):
        H = ncon([self.psi.Al,self.H_tensor,self.psi.Al.conj(),np.eye(self.psi.D)],((1,6,-5),(3,-4,1,-2),(3,6,-7),(-8,-9)),forder=(-4,-7,-8,-2,-5,-9)).astype(complex)
        H += ncon([self.psi.Ar,self.H_tensor,self.psi.Ar.conj(),np.eye(self.psi.D)],((2,-5,6),(-3,4,-1,2),(4,-7,6),(-8,-9)),forder=(-3,-8,-7,-1,-9,-5))
        return H
    def getCentreTerms_C(self):
        return ncon([self.psi.Al,self.psi.Ar,self.H_tensor,self.psi.Al.conj(),self.psi.Ar.conj()],((1,10,-5),(2,-6,7),(3,4,1,2),(3,10,-9),(4,-8,7)),forder=(-9,-8,-5,-6)).astype(complex)
