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

# -------------------------------------------------------------------------------------------------------------------------------------
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

class gradImplementation_mpso_2d_mps_uniform(ABC):
    def __init__(self,psi,H):
        self.psi = psi
        self.H = H

    #gets effective 1d Hamiltonians to compute mps gradient of 2d term
    #consists of two terms, effH_centre + effH_shifted
    @abstractmethod
    def getEffectiveH(self):
        pass

class gradImplementation_mpso_2d_mpo_uniform(ABC):
    def __init__(self,psi,H):
        self.psi = psi
        self.H = H

        @abstractmethod
        def getFixedPoints(self):
            pass
        @abstractmethod
        def getOuterContracts(self):
            pass
        @abstractmethod
        def getCentralTerms(self):
            pass
        @abstractmethod
        def buildLeftEnv(self):
            pass
        @abstractmethod
        def buildRightEnv(self):
            pass
        @abstractmethod
        def buildTopEnvGeo(self):
            pass
        @abstractmethod
        def buildBotEnvGeo(self):
            pass
        @abstractmethod
        def buildTopEnvGeo_quadrants(self):
            pass
        @abstractmethod
        def buildBotEnvGeo_quadrants(self):
            pass
        @abstractmethod
        def attachRightMax(self):
            pass
        @abstractmethod
        def attachLeftMax(self):
            pass

    def attachRightSingle(self,env):
        return ncon([self.psi.mpo,env,self.psi.R.tensor,self.outerContract],((-2,1,4,5),(-6,4),(-7,5),(-3,1)),forder=(-3,-2,-6,-7),order=(4,1,5))
    def attachLeftSingle(self,env):
        return ncon([self.psi.mpo,env,self.outerContract],((-2,1,-4,5),(-6,5),(-3,1)),forder=(-3,-2,-4,-6),order=(5,1))

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
class gradImplementation_uniform_1d_twoSiteLeft_oneBodyH(gradImplementation_uniform_1d):
    def getCentralTerms(self):
        grad = ncon([self.psi.mps,self.H,self.psi.R.tensor],((1,-2,-4,5),(-3,1),(-6,5)),forder=(-3,-2,-4,-6))
        grad += ncon([self.psi.mps,self.H,self.psi.R.tensor],((-1,2,-4,5),(-3,2),(-6,5)),forder=(-1,-3,-4,-6))
        return grad/2
    def buildLeftEnv(self):
        env = ncon([self.psi.mps,self.H,self.psi.mps.conj()],((1,2,4,-5),(3,1),(3,2,4,-6)),forder=(-6,-5))
        env += ncon([self.psi.mps,self.H,self.psi.mps.conj()],((1,2,4,-5),(3,2),(1,3,4,-6)),forder=(-6,-5))
        return env/2
    def buildRightEnv(self):
        env = ncon([self.psi.mps,self.H,self.psi.mps.conj(),self.psi.R.tensor],((1,2,-4,5),(3,1),(3,2,-7,6),(6,5)),forder=(-7,-4))
        env += ncon([self.psi.mps,self.H,self.psi.mps.conj(),self.psi.R.tensor],((1,2,-4,5),(3,2),(1,3,-7,6),(6,5)),forder=(-7,-4))
        return env/2

class gradImplementation_uniform_1d_twoSiteLeft_twoBodyH(gradImplementation_uniform_1d):
    def getCentralTerms(self):
        grad = ncon([self.psi.mps,self.H,self.psi.R.tensor],((1,2,-5,6),(-3,-4,1,2),(-7,6)),forder=(-3,-4,-5,-7),order=(6,2,1))
        grad += ncon([self.psi.mps,self.psi.mps,self.H,self.psi.mps.conj(),self.psi.R.tensor],((-1,2,-7,8),(3,4,8,9),(-5,6,2,3),(6,4,-11,10),(10,9)),forder=(-1,-5,-7,-11),order=(10,9,4,3,6,8,2))
        grad += ncon([self.psi.mps,self.psi.mps,self.H,self.psi.mps.conj(),self.psi.R.tensor],((1,2,7,8),(3,-4,8,9),(5,-6,2,3),(1,5,7,-11),(-10,9)),forder=(-6,-4,-11,-10),order=(7,1,2,5,8,3,9))
        return grad/2
    def buildLeftEnv(self):
        env = ncon([self.psi.mps,self.H,self.psi.mps.conj()],((1,2,5,-6),(3,4,1,2),(3,4,5,-7)),forder=(-7,-6),order=(5,1,3,2,4))
        env += ncon([self.psi.mps,self.psi.mps,self.H,self.psi.mps.conj(),self.psi.mps.conj()],((1,2,7,8),(3,4,8,-9),(5,6,2,3),(1,5,7,11),(6,4,11,-10)),forder=(-10,-9),order=(7,1,2,5,8,11,3,6,4))
        return env/2
    def buildRightEnv(self):
        env = ncon([self.psi.mps,self.H,self.psi.mps.conj(),self.psi.R.tensor],((1,2,-5,6),(3,4,1,2),(3,4,-8,7),(7,6)),forder=(-8,-5),order=(7,6,2,4,1,3))
        env += ncon([self.psi.mps,self.psi.mps,self.H,self.psi.mps.conj(),self.psi.mps.conj(),self.psi.R.tensor],((1,2,-7,8),(3,4,8,9),(5,6,2,3),(1,5,-12,11),(6,4,11,10),(10,9)),forder=(-12,-7),order=(10,9,4,3,6,8,11,2,5,1))
        return env/2

# -----------------------------
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

# -------------------------------------------------------------------------------------------------------------------------------------
#2d mps implementations
class gradImplementation_mpso_2d_mps_uniform_oneBodyH(gradImplementation_mpso_2d_mps_uniform):
    def getEffectiveH(self):
        #under current Hamiltonian
        outerContract =  ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
        H_eff_centre = ncon([self.H,self.psi.mpo,self.psi.mpo.conj(),self.psi.R.tensor],((3,2),(2,-1,5,6),(3,-4,5,7),(7,6)),forder=(-4,-1),order=(5,2,3,6,7)) 
        #right half of plane (shifted)
        env =  ncon([self.H,self.psi.mpo,self.psi.mpo.conj(),outerContract],((3,2),(2,1,5,-6),(3,4,5,-7),(4,1)),forder=(-7,-6),order=(5,1,2,3,4))
        env = self.psi.Tb_inv.applyRight(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        H_eff_shifted = ncon([env,self.psi.mpo,self.psi.mpo.conj(),self.psi.R.tensor],((6,4),(2,-1,4,5),(2,-3,6,7),(7,5)),forder=(-3,-1),order=(4,6,2,5,7))
        return oneBodyH(H_eff_centre + H_eff_shifted)

class gradImplementation_mpso_2d_mps_uniform_twoBodyH_hori(gradImplementation_mpso_2d_mps_uniform):
    def getEffectiveH(self):
        #under current Hamiltonian
        outerContract =  ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
        centreEnv =  ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.R.tensor],((1,-10,5,6),(2,-11,6,7),(3,4,1,2),(3,-12,5,9),(4,-13,9,8),(8,7)),forder=(-12,-13,-10,-11),order=(5,6,9,7,8,1,2,3,4))
        H_eff_centre = ncon([centreEnv,outerContract],((1,-2,3,-4),(1,3)),forder=(-2,-4)) + ncon([centreEnv,outerContract],((-1,2,-3,4),(2,4)),forder=(-1,-3))
        #right half of plane (shifted)
        env =  ncon([self.H,self.psi.mpo,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo.conj(),outerContract,outerContract],((3,4,1,2),(1,10,5,6),(2,11,6,-7),(3,12,5,9),(4,13,9,-8),(12,10),(13,11)),forder=(-8,-7),order=(5,10,1,3,12,6,9,11,2,4,13))
        env = self.psi.Tb_inv.applyRight(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        H_eff_shifted = ncon([env,self.psi.mpo,self.psi.mpo.conj(),self.psi.R.tensor],((6,4),(2,-1,4,5),(2,-3,6,7),(7,5)),forder=(-3,-1),order=(4,6,2,5,7))
        return oneBodyH(H_eff_centre + H_eff_shifted)

class gradImplementation_mpso_2d_mps_uniform_twoBodyH_vert(gradImplementation_mpso_2d_mps_uniform):
    def getEffectiveH(self):
        H_eff_centre =  ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.RR.tensor],((1,-11,5,6),(2,-12,8,9),(3,4,1,2),(3,-13,5,7),(4,-14,8,10),(7,6,10,9)),forder=(-13,-14,-11,-12),order=(6,7,10,9,5,8,1,2,3,4))

        env = ncon([self.H,self.psi.mps,self.psi.mps,self.psi.mpo,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mps.conj(),self.psi.mps.conj(),self.psi.T.tensor],((3,7,2,6),(1,9,10),(5,10,11),(2,1,14,-15),(6,5,17,-18),(3,4,14,-16),(7,8,17,-19),(4,9,13),(8,13,12),(12,11)),forder=(-16,-15,-19,-18),order=(9,10,13,11,12,1,14,2,3,4,5,17,6,7,8))
        env = self.psi.Tb2_inv.applyRight(env.reshape(self.psi.D_mpo**4)).reshape(self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo)
        H_eff_shifted = ncon([self.psi.mpo,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo.conj(),env,self.psi.RR.tensor],((2,-1,7,9),(5,-4,11,13),(2,-3,8,10),(5,-6,12,14),(8,7,12,11),(10,9,14,13)),forder=(-3,-6,-1,-4),order=(13,14,5,9,10,2,8,7,12,11))
        return twoBodyH(H_eff_centre + H_eff_shifted)

# -----------------------------

class gradImplementation_mpso_2d_mps_bipartite_oneBodyH(gradImplementation_mpso_2d_mps_uniform):
    def getEffectiveH(self,index):
        if index == 1:
            index1 = 1
            index2 = 2
        else:
            index1 = 2
            index2 = 1
        outerContract_1 =  ncon([self.psi.mps[index1],self.psi.mps[index1].conj(),self.psi.T[index2].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
        outerContract_2 =  ncon([self.psi.mps[index2],self.psi.mps[index2].conj(),self.psi.T[index1].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))

        #under current Hamiltonian
        H_eff_centre = ncon([self.psi.mpo[index1],self.H,self.psi.mpo[index1].conj(),self.psi.R[index2].tensor],((2,-1,5,6),(3,2),(3,-4,5,8),(8,6)),forder=(-4,-1),order=(5,2,3,6,8))

        #right half of plane (shifted)
        env_1 = ncon([self.psi.mpo[index1],self.H,self.psi.mpo[index1].conj(),outerContract_1],((2,1,5,-6),(3,2),(3,4,5,-7),(4,1)),forder=(-7,-6),order=(5,1,2,3,4))
        transfer = mpsu1Transfer_left_oneLayer(self.psi.mps[index2],self.psi.mpo[index2],self.psi.T[index1])
        env_1 = transfer.applyRight(env_1.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)

        env_2 = ncon([self.psi.mpo[index2],self.H,self.psi.mpo[index2].conj(),outerContract_2],((2,1,5,-6),(3,2),(3,4,5,-7),(4,1)),forder=(-7,-6),order=(5,1,2,3,4))

        env = env_1 + env_2
        env = self.psi.Tb_inv[index1].applyRight(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        H_eff_shifted = ncon([env,self.psi.mpo[index1],self.psi.mpo[index1].conj(),self.psi.R[index2].tensor],((6,4),(2,-1,4,5),(2,-3,6,7),(7,5)),forder=(-3,-1),order=(4,6,2,5,7))
        return oneBodyH(H_eff_centre + H_eff_shifted)

class gradImplementation_mpso_2d_mps_bipartite_twoBodyH_hori(gradImplementation_mpso_2d_mps_uniform):
    def getEffectiveH(self,index):
        if index == 1:
            index1 = 1
            index2 = 2
        else:
            index1 = 2
            index2 = 1
        #under current Hamiltonian
        outerContract_1 =  ncon([self.psi.mps[index1],self.psi.mps[index1].conj(),self.psi.T[index2].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
        outerContract_2 =  ncon([self.psi.mps[index2],self.psi.mps[index2].conj(),self.psi.T[index1].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
        centreEnv_1 =  ncon([self.psi.mpo[index1],self.psi.mpo[index2],self.H,self.psi.mpo[index1].conj(),self.psi.mpo[index2].conj(),self.psi.R[index1].tensor],((1,-10,5,6),(2,-11,6,7),(3,4,1,2),(3,-12,5,9),(4,-13,9,8),(8,7)),forder=(-12,-13,-10,-11),order=(5,6,9,7,8,1,2,3,4))
        centreEnv_2 =  ncon([self.psi.mpo[index2],self.psi.mpo[index1],self.H,self.psi.mpo[index2].conj(),self.psi.mpo[index1].conj(),self.psi.R[index2].tensor],((1,-10,5,6),(2,-11,6,7),(3,4,1,2),(3,-12,5,9),(4,-13,9,8),(8,7)),forder=(-12,-13,-10,-11),order=(5,6,9,7,8,1,2,3,4))
        H_eff_centre = ncon([centreEnv_2,outerContract_2],((1,-2,3,-4),(1,3)),forder=(-2,-4)) + ncon([centreEnv_1,outerContract_2],((-1,2,-3,4),(2,4)),forder=(-1,-3))

        #right half of plane (shifted)
        env_1 =  ncon([self.H,self.psi.mpo[index1],self.psi.mpo[index2],self.psi.mpo[index1].conj(),self.psi.mpo[index2].conj(),outerContract_1,outerContract_2],((3,4,1,2),(1,10,5,6),(2,11,6,-7),(3,12,5,9),(4,13,9,-8),(12,10),(13,11)),forder=(-8,-7),order=(5,10,1,3,12,6,9,11,2,4,13))
        env_2 =  ncon([self.H,self.psi.mpo[index2],self.psi.mpo[index1],self.psi.mpo[index2].conj(),self.psi.mpo[index1].conj(),outerContract_2,outerContract_1],((3,4,1,2),(1,10,5,6),(2,11,6,-7),(3,12,5,9),(4,13,9,-8),(12,10),(13,11)),forder=(-8,-7),order=(5,10,1,3,12,6,9,11,2,4,13))
        env_2 = ncon([env_2,self.psi.mpo[index2],self.psi.mpo[index2].conj(),outerContract_2],((6,4),(2,1,4,-5),(2,3,6,-7),(3,1)),forder=(-7,-5))
        env = env_1 + env_2

        env = self.psi.Tb_inv[index1].applyRight(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        H_eff_shifted = ncon([env,self.psi.mpo[index1],self.psi.mpo[index1].conj(),self.psi.R[index2].tensor],((6,4),(2,-1,4,5),(2,-3,6,7),(7,5)),forder=(-3,-1),order=(4,6,2,5,7))
        return oneBodyH(H_eff_centre + H_eff_shifted)

class gradImplementation_mpso_2d_mps_bipartite_twoBodyH_vert(gradImplementation_mpso_2d_mps_uniform):
    def getEffectiveH(self,index):
        if index == 1:
            index1 = 1
            index2 = 2
        else:
            index1 = 2
            index2 = 1
        #under current Hamiltonian
        H_eff_centre = ncon([self.psi.mpo[index1],self.psi.mpo[index2],self.H,self.psi.mpo[index1].conj(),self.psi.mpo[index2].conj(),self.psi.RR[index2].tensor],((1,-11,5,6),(2,-12,8,9),(3,4,1,2),(3,-13,5,7),(4,-14,8,10),(7,6,10,9)),forder=(-13,-14,-11,-12),order=(6,7,10,9,5,8,1,2,3,4))

        #right half of plane (shifted)
        env_1 = ncon([self.H,self.psi.mps[index1],self.psi.mps[index2],self.psi.mpo[index1],self.psi.mpo[index2],self.psi.mpo[index1].conj(),self.psi.mpo[index2].conj(),self.psi.mps[index1].conj(),self.psi.mps[index2].conj(),self.psi.T[index1].tensor],((3,7,2,6),(1,9,10),(5,10,11),(2,1,14,-15),(6,5,17,-18),(3,4,14,-16),(7,8,17,-19),(4,9,13),(8,13,12),(12,11)),forder=(-16,-15,-19,-18),order=(9,10,13,11,12,1,14,2,3,4,5,17,6,7,8))
        transfer = ncon([self.psi.mps[index2],self.psi.mps[index1],self.psi.mpo[index2],self.psi.mpo[index1],self.psi.mpo[index2].conj(),self.psi.mpo[index1].conj(),self.psi.mps[index2].conj(),self.psi.mps[index1].conj(),self.psi.T[index2].tensor],((1,7,8),(4,8,9),(2,1,-12,-14),(5,4,-16,-18),(2,3,-13,-15),(5,6,-17,-19),(3,7,11),(6,11,10),(10,9)),forder=(-13,-12,-17,-16,-15,-14,-19,-18),order=(7,8,9,10,11,1,2,3,4,5,6)).reshape(self.psi.D_mpo**4,self.psi.D_mpo**4)
        env_1 = np.dot(env_1.reshape(self.psi.D_mpo**4),transfer).reshape(self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo)
        env_2 = ncon([self.H,self.psi.mps[index2],self.psi.mps[index1],self.psi.mpo[index2],self.psi.mpo[index1],self.psi.mpo[index2].conj(),self.psi.mpo[index1].conj(),self.psi.mps[index2].conj(),self.psi.mps[index1].conj(),self.psi.T[index2].tensor],((3,7,2,6),(1,9,10),(5,10,11),(2,1,14,-15),(6,5,17,-18),(3,4,14,-16),(7,8,17,-19),(4,9,13),(8,13,12),(12,11)),forder=(-16,-15,-19,-18),order=(9,10,13,11,12,1,14,2,3,4,5,17,6,7,8))

        env = env_1 + env_2
        env = self.psi.Tb2_inv[index1].applyRight(env.reshape(self.psi.D_mpo**4)).reshape(self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo)
        H_eff_shifted = ncon([env,self.psi.mpo[index2],self.psi.mpo[index2].conj(),self.psi.mpo[index1],self.psi.mpo[index1].conj(),self.psi.RR[index2].tensor],((13,11,9,7),(2,-1,7,8),(2,-3,9,10),(5,-4,11,12),(5,-6,13,14),(14,12,10,8)),forder=(-6,-3,-4,-1),order=(7,9,2,11,13,5,8,10,12,14))
        return twoBodyH(H_eff_centre + H_eff_shifted)

# -------------------------------------------------------------------------------------------------------------------------------------
#2d mpo implementations
class gradImplementation_mpso_2d_mpo_uniform_twoBodyH_hori(gradImplementation_mpso_2d_mpo_uniform):
    def __init__(self,psi,H):
        super().__init__(psi,H)
        #objects needed to construct terms
        innerContract = ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.mpo.conj()],((2,-1,9,10),(6,-5,10,-11),(3,7,2,6),(3,-4,9,13),(7,-8,13,-12)),forder=(-4,-8,-1,-5,-12,-11),order=(9,2,3,10,13,6,7))
        self.outerContract =  ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
        exp = ncon([innerContract,self.outerContract,self.outerContract,self.psi.R.tensor],((1,2,3,4,5,6),(1,3),(2,4),(5,6)),order=(1,3,2,4,5,6))

        self.h_tilde = (self.H.reshape(4,4)-exp*np.eye(4)).reshape(2,2,2,2)
        self.innerContract= ncon([self.psi.mpo,self.psi.mpo,self.h_tilde,self.psi.mpo.conj(),self.psi.mpo.conj()],((2,-1,9,10),(6,-5,10,-11),(3,7,2,6),(3,-4,9,13),(7,-8,13,-12)),forder=(-4,-8,-1,-5,-12,-11),order=(9,2,3,10,13,6,7))
    def getFixedPoints(self,d,Td):
        if d == 0:
            RR_d = self.psi.RR
        else:
            TT_d = mpsu1Transfer_left_twoLayerWithMpsInsert(self.psi.mps,self.psi.mpo,self.psi.T,Td)
            RR_d = TT_d.findRightEig()
            RR_d.norm_pairedCanon()
        fp = dict()
        fp['RR_d'] = RR_d.tensor
        fp['RR_d'] = RR_d.tensor
        return fp
    def getOuterContracts(self,Td):
        outerContractDouble = ncon([self.psi.mps,self.psi.mps,self.psi.mps.conj(),self.psi.mps.conj(),Td,self.psi.T.tensor],((-1,5,6),(-2,7,8),(-3,5,11),(-4,10,9),(11,6,10,7),(9,8)),forder=(-3,-4,-1,-2),order=(5,6,11,7,10,8,9))
        outerContract_open_top = ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor,Td],((-1,4,5),(-2,7,6),(6,5),(-8,-3,7,4)),forder=(-2,-1,-8,-3),order=(5,6,4,7))
        outerContract_open_bot = ncon([self.psi.mps,self.psi.mps.conj(),Td],((-1,3,4),(-2,3,5),(5,4,-7,-6)),forder=(-2,-1,-7,-6),order=(3,4,5))
        fp = dict()
        fp['outerContractDouble'] = outerContractDouble
        fp['outerContract_open_top'] = outerContract_open_top
        fp['outerContract_open_bot'] = outerContract_open_bot
        return fp

    def getCentralTerms(self):
        grad = ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.R.tensor,self.outerContract,self.outerContract],((2,1,-9,10),(6,5,10,11),(-3,7,2,6),(7,8,-13,12),(12,11),(-4,1),(8,5)),forder=(-4,-3,-9,-13),order=(12,11,5,8,6,7,10,1))
        grad += ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.R.tensor,self.outerContract,self.outerContract],((2,1,9,10),(6,5,10,11),(3,-7,2,6),(3,4,9,-13),(-12,11),(4,1),(-8,5)),forder=(-8,-7,-13,-12),order=(9,1,2,3,4,10,5,6,11))
        return grad

    def buildLeftEnv(self,H=None):
        if H is None:
            H = self.H
        env = ncon([self.psi.mpo,self.psi.mpo,H,self.psi.mpo.conj(),self.psi.mpo.conj(),self.outerContract,self.outerContract],((2,1,9,10),(6,5,10,-11),(3,7,2,6),(3,4,9,13),(7,8,13,-12),(4,1),(8,5)),forder=(-12,-11),order=(9,1,2,3,4,10,13,6,7,5,8))
        env = self.psi.Tb_inv.applyRight(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        return env
    def buildRightEnv(self,H=None):
        if H is None:
            H = self.H
        env = ncon([self.psi.mpo,self.psi.mpo,H,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.R.tensor,self.outerContract,self.outerContract],((2,1,-9,10),(6,5,10,11),(3,7,2,6),(3,4,-14,13),(7,8,13,12),(12,11),(4,1),(8,5)),forder=(-14,-9),order=(11,12,5,6,7,8,10,13,2,3,1,4))
        env = self.psi.Tb_inv.applyLeft(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        return env

    def attachRightMax(self,env):
        return super().attachRightSingle(env)
    def attachLeftMax(self,env):
        return super().attachLeftSingle(env)

    def buildTopEnvGeo(self,fixedPoints,outers):
        env = ncon([self.innerContract,fixedPoints['RR_d'],self.psi.mpo,self.psi.mpo.conj(),outers['outerContractDouble'],outers['outerContract_open_top']],((1,2,3,4,5,6),(10,8,5,6),(12,11,-7,8),(12,13,-9,10),(13,2,11,4),(1,3,-14,-15)),forder=(-14,-9,-7,-15),order=(5,6,8,10,4,2,11,12,13,1,3))
        env += ncon([self.innerContract,fixedPoints['RR_d'],self.outerContract,outers['outerContract_open_top']],((1,2,3,4,5,6),(-8,-7,5,6),(1,3),(2,4,-9,-10)),forder=(-9,-8,-7,-10),order=(5,6,1,3,2,4))
        return env
    def buildBotEnvGeo(self,fixedPoints,outers):
        env = ncon([self.innerContract,fixedPoints['RR_d'],self.psi.mpo,self.psi.mpo.conj(),outers['outerContractDouble'],outers['outerContract_open_bot']],((1,2,3,4,5,6),(5,6,7,8),(10,9,-12,8),(10,11,-13,7),(2,11,4,9),(1,3,-14,-15)),forder=(-14,-13,-12,-15),order=(5,6,7,8,9,10,11,4,2,1,3))
        env += ncon([self.innerContract,fixedPoints['RR_d'],self.outerContract,outers['outerContract_open_bot']],((1,2,3,4,5,6),(5,6,-7,-8),(1,3),(2,4,-9,-10)),forder=(-9,-7,-8,-10),order=(5,6,1,3,2,4))
        return env
    def buildTopEnvGeo_quadrants(self,fixedPoints,outers,leftEnv,):
        env = ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj(),fixedPoints['RR_d'],outers['outerContract_open_top']],((6,4),(2,1,4,5),(2,3,6,7),(-9,-8,7,5),(3,1,-10,-11)),forder=(-10,-9,-8,-11),order=(6,4,2,5,7,1,3))
        return env
    def buildBotEnvGeo_quadrants(self,fixedPoints,outers,leftEnv,):
        env = ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj(),fixedPoints['RR_d'],outers['outerContract_open_bot']],((6,4),(2,1,4,5),(2,3,6,7),(7,5,-8,-9),(3,1,-10,-11)),forder=(-10,-8,-9,-11),order=(4,6,2,5,7,1,3))
        return env
    def buildRightEnvGeo_quadrants(self,fixedPoints,outers):
        env = ncon([self.innerContract,fixedPoints['RR_d'],self.psi.mpo,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo.conj(),outers['outerContractDouble'],outers['outerContractDouble']],((1,2,3,4,5,6),(18,15,5,6),(11,10,14,15),(8,7,-13,14),(11,12,17,18),(8,9,-16,17),(9,1,7,3),(12,2,10,4)),forder=(-16,-13),order=(5,6,15,18,11,4,2,10,12,14,17,8,1,3,7,9))
        env += ncon([self.innerContract,fixedPoints['RR_d'],self.psi.mpo,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo.conj(),outers['outerContractDouble'],outers['outerContractDouble']],((1,2,3,4,5,6),(5,6,18,15),(11,10,14,15),(8,7,-13,14),(11,12,17,18),(8,9,-16,17),(1,9,3,7),(2,12,4,10)),forder=(-16,-13),order=(5,6,18,15,11,10,12,4,2,14,17,8,7,9,3,1))
        env = self.psi.Tb_inv.applyLeft(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        return env

# -----------------------------

class gradImplementation_mpso_2d_mpo_uniform_twoBodyH_vert(gradImplementation_mpso_2d_mpo_uniform):
    def __init__(self,psi,H):
        super().__init__(psi,H)
        #objects needed to construct terms
        self.outerContract =  ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
        self.outerContractDouble = ncon([self.psi.mps,self.psi.mps,self.psi.mps.conj(),self.psi.mps.conj(),self.psi.T.tensor],((-1,5,6),(-2,6,7),(-3,5,9),(-4,9,8),(8,7)),forder=(-3,-4,-1,-2))

        leftEnv = ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.mpo.conj(),self.outerContractDouble],((2,1,9,-10),(6,5,11,-12),(3,7,2,6),(3,4,9,-13),(7,8,11,-14),(4,8,1,5)),forder=(-13,-10,-14,-12),order=(9,1,2,3,4,11,5,6,7,8))
        exp = np.einsum('abcd,abcd',leftEnv,self.psi.RR.tensor)
        self.h_tilde = (self.H.reshape(4,4)-exp*np.eye(4)).reshape(2,2,2,2)

        self.innerContract= ncon([self.psi.mpo,self.psi.mpo,self.h_tilde,self.psi.mpo.conj(),self.psi.mpo.conj()],((2,-1,9,-10),(6,-5,11,-12),(3,7,2,6),(3,-4,9,-13),(7,-8,11,-14)),forder=(-4,-8,-1,-5,-13,-10,-14,-12),order=(9,2,3,11,6,7))
    def getFixedPoints(self,d,Td):
        Twd_lower = mpsu1Transfer_left_threeLayerWithMpsInsert_lower(self.psi.mps,self.psi.mpo,self.psi.T,Td)
        R_Twd_lower = Twd_lower.findRightEig()
        R_Twd_lower.norm_pairedCanon()
        Twd_upper = mpsu1Transfer_left_threeLayerWithMpsInsert_upper(self.psi.mps,self.psi.mpo,self.psi.T,Td)
        R_Twd_upper = Twd_upper.findRightEig()
        R_Twd_upper.norm_pairedCanon()
        fp = dict()
        fp['RRR_d_lower'] = R_Twd_lower.tensor
        fp['RRR_d_upper'] = R_Twd_upper.tensor
        return fp
    def getOuterContracts(self,Td):
        outercontractTriple_upper = ncon([self.psi.mps,self.psi.mps,self.psi.mps,self.psi.mps.conj(),self.psi.mps.conj(),self.psi.mps.conj(),Td,self.psi.T.tensor],((-1,7,8),(-3,8,9),(-5,10,11),(-2,7,15),(-4,15,14),(-6,13,12),(14,9,13,10),(12,11)),forder=(-2,-4,-6,-1,-3,-5),order=(7,8,15,9,14,10,13,11,12))
        outercontractTriple_lower = ncon([self.psi.mps,self.psi.mps,self.psi.mps,self.psi.mps.conj(),self.psi.mps.conj(),self.psi.mps.conj(),Td,self.psi.T.tensor],((-1,7,8),(-3,9,10),(-5,10,11),(-2,7,15),(-4,14,13),(-6,13,12),(15,8,14,9),(12,11)),forder=(-2,-4,-6,-1,-3,-5),order=(7,8,15,9,14,10,13,11,12))

        outerContract_open_top = ncon([self.psi.mps,self.psi.mps,self.psi.mps.conj(),self.psi.mps.conj(),self.psi.T.tensor,Td],((-1,6,7),(-3,7,8),(-2,11,10),(-4,10,9),(9,8),(-12,-5,11,6)),forder=(-2,-4,-1,-3,-12,-5),order=(9,8,7,10,6,11))
        outerContract_open_bot = ncon([self.psi.mps,self.psi.mps,self.psi.mps.conj(),self.psi.mps.conj(),Td],((-1,5,6),(-3,6,7),(-2,5,11),(-4,11,10),(10,7,-9,-8)),forder=(-2,-4,-1,-3,-9,-8),order=(5,6,11,10,7))
        fp = dict()
        fp['outerContractTriple_upper'] = outercontractTriple_upper
        fp['outerContractTriple_lower'] = outercontractTriple_lower
        fp['outerContract_open_top'] = outerContract_open_top
        fp['outerContract_open_bot'] = outerContract_open_bot
        return fp
    def getCentralTerms(self):
        grad = ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.RR.tensor,self.outerContractDouble],((2,1,-9,10),(4,3,11,12),(-5,7,2,4),(7,8,11,14),(-13,10,14,12),(-6,8,1,3)),forder=(-6,-5,-9,-13),order=(12,14,11,4,7,3,8,10,2,1))
        grad += ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.RR.tensor,self.outerContractDouble],((2,1,9,10),(6,5,-11,12),(3,-7,2,6),(3,4,9,13),(13,10,-14,12),(4,-8,1,5)),forder=(-8,-7,-11,-14),order=(10,13,9,2,3,1,4,12,5))
        return grad
    def buildLeftEnv(self,H=None):
        if H is None:
            H = self.H
        env = ncon([self.psi.mpo,self.psi.mpo,H,self.psi.mpo.conj(),self.psi.mpo.conj(),self.outerContractDouble],((2,1,9,-10),(6,5,11,-12),(3,7,2,6),(3,4,9,-13),(7,8,11,-14),(4,8,1,5)),forder=(-13,-10,-14,-12),order=(9,1,2,3,4,11,5,6,7,8))
        env = self.psi.Tb2_inv.applyRight(env.reshape(self.psi.D_mpo**4)).reshape(self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo)
        return env
    def buildRightEnv(self,H=None):
        if H is None:
            H = self.H
        env = ncon([self.psi.mpo,self.psi.mpo,H,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.RR.tensor,self.outerContractDouble],((2,1,-9,10),(6,5,-11,12),(3,7,2,6),(3,4,-13,14),(7,8,-15,16),(14,10,16,12),(4,8,1,5)),forder=(-13,-9,-15,-11),order=(12,16,5,6,7,8,10,14,1,2,3,4))
        env = self.psi.Tb2_inv.applyLeft(env.reshape(self.psi.D_mpo**4)).reshape(self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo)
        return env
    def attachRightMax(self,leftEnv):
        grad = ncon([leftEnv,self.psi.mpo,self.psi.mpo,self.psi.mpo.conj(),self.psi.RR.tensor,self.outerContractDouble],((13,11,-9,7),(-2,1,7,8),(5,4,11,12),(5,6,13,14),(14,12,-10,8),(6,-3,4,1)),forder=(-3,-2,-9,-10),order=(14,12,5,4,6,8,1,13,11,7))
        grad += ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,self.psi.RR.tensor,self.outerContractDouble],((-10,9,8,7),(2,1,7,11),(2,3,8,12),(-5,4,9,13),(-14,13,12,11),(-6,3,4,1)),forder=(-6,-5,-10,-14),order=(11,12,2,1,3,7,8,9,13,4))
        return grad
    def attachLeftMax(self,rightEnv):
        rightEnv1 = np.einsum('abcc->ab',rightEnv)
        rightEnv2 = np.einsum('aabc->bc',rightEnv)
        grad = ncon([self.psi.mpo,rightEnv1,self.outerContract],((-2,1,-4,5),(-6,5),(-3,1)),forder=(-3,-2,-4,-6),order=(5,1))
        grad += ncon([self.psi.mpo,rightEnv2,self.outerContract],((-2,1,-4,5),(-6,5),(-3,1)),forder=(-3,-2,-4,-6),order=(5,1))
        return grad

    def buildTopEnvGeo(self,fixedPoints,outers):
        return ncon([self.innerContract,fixedPoints['RRR_d_lower'],outers['outerContract_open_top']],((1,2,3,4,5,6,7,8),(-9,-10,5,6,7,8),(1,2,3,4,-11,-12)),forder=(-11,-9,-10,-12),order=(5,6,7,8,1,2,3,4))
    def buildBotEnvGeo(self,fixedPoints,outers):
        return ncon([self.innerContract,fixedPoints['RRR_d_upper'],outers['outerContract_open_bot']],((1,2,3,4,5,6,7,8),(5,6,7,8,-9,-10),(1,2,3,4,-11,-12)),forder=(-11,-9,-10,-12),order=(5,6,7,8,1,2,3,4))
    def buildTopEnvGeo_quadrants(self,fixedPoints,outers,leftEnv,):
        return ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo.conj(),fixedPoints['RRR_d_lower'],outers['outerContract_open_top']],((13,11,9,7),(2,1,7,8),(2,3,9,10),(5,4,11,12),(5,6,13,14),(-16,-15,14,12,10,8),(6,3,4,1,-17,-18)),forder=(-17,-16,-15,-18),order=(7,9,2,11,13,5,8,10,12,14,1,3,4,6))
    def buildBotEnvGeo_quadrants(self,fixedPoints,outers,leftEnv,):
        return ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo.conj(),fixedPoints['RRR_d_upper'],outers['outerContract_open_bot']],((13,11,9,7),(2,1,7,8),(2,3,9,10),(5,4,11,12),(5,6,13,14),(14,12,10,8,-15,-16),(6,3,4,1,-17,-18)),forder=(-17,-15,-16,-18),order=(7,9,2,11,13,5,8,10,12,14,1,3,4,6))
    def buildRightEnvGeo_quadrants(self,fixedPoints,outers):
        env1 = self.buildTopEnvGeo(fixedPoints,outers)
        env1 = ncon([env1,self.psi.mps,self.psi.mpo,self.psi.mpo.conj(),self.psi.mps.conj()],((10,9,7,5),(1,4,5),(2,1,-6,7),(2,3,-8,9),(3,4,10)),forder=(-8,-6),order=(5,7,9,10,1,2,3,4))
        env2 = self.buildBotEnvGeo(fixedPoints,outers)
        env2 = ncon([env2,self.psi.mps,self.psi.mpo,self.psi.mpo.conj(),self.psi.mps.conj(),self.psi.T.tensor],((10,9,7,4),(1,4,5),(2,1,-6,7),(2,3,-8,9),(3,10,11),(11,5)),forder=(-8,-6),order=(4,7,9,10,1,2,3,5,11))
        env = env1 + env2
        env = self.psi.Tb_inv.applyLeft(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        return env
