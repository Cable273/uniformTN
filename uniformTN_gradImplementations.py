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

    #d dep bipartite transfers for geosum
    def init_mps_transfers(self):
        Td_matrix = np.eye(self.psi.D_mps**2)
        Td = Td_matrix.reshape(self.psi.D_mps,self.psi.D_mps,self.psi.D_mps,self.psi.D_mps)
        return Td_matrix,Td
    def apply_mps_transfers(self,Td_matrix):
        Td_matrix = np.dot(Td_matrix,self.psi.Ta.matrix)
        Td = Td_matrix.reshape(self.psi.D_mps,self.psi.D_mps,self.psi.D_mps,self.psi.D_mps)
        return Td_matrix,Td

class gradImplementation_mpso_2d_mpo_bipartite(gradImplementation_mpso_2d_mpo_uniform):
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
class gradImplementation_mpso_2d_mps_twoSite_oneBodyH(gradImplementation_mpso_2d_mps_uniform):
    def getEffectiveH(self,site):
        if site == 1:
            outerContract= ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((-1,3,4,5),(-2,3,4,6),(6,5)),forder=(-2,-1),order=(4,3,5,6))
            tensorLabel = 'bot'
        elif site == 2:
            outerContract= ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((1,-2,4,5),(1,-3,4,6),(6,5)),forder=(-3,-2),order=(4,1,5,6))
            tensorLabel = 'top'
        
        H_eff = ncon([self.psi.mpo,self.psi.mpo.conj(),self.H,outerContract,self.psi.R[tensorLabel].tensor],((2,6,-1,5,8,9),(3,6,-4,7,8,10),(3,2),(7,5),(10,9)),forder=(-4,-1),order=(10,9,5,6,7,2,3,8))
        H_eff += ncon([self.psi.mpo,self.psi.mpo.conj(),self.H,outerContract,self.psi.R[tensorLabel].tensor],((2,6,1,-5,8,9),(3,6,4,-7,8,10),(3,2),(4,1),(10,9)),forder=(-7,-5),order=(8,1,2,3,4,6,9,10))
        H_eff += ncon([self.psi.mpo,self.psi.mpo.conj(),self.H,outerContract,self.psi.R[tensorLabel].tensor],((2,5,-1,4,8,9),(2,6,-3,7,8,10),(6,5),(7,4),(10,9)),forder=(-3,-1),order=(10,9,4,5,6,7,2,8))
        H_eff += ncon([self.psi.mpo,self.psi.mpo.conj(),self.H,outerContract,self.psi.R[tensorLabel].tensor],((2,5,1,-4,8,9),(2,6,3,-7,8,10),(6,5),(3,1),(10,9)),forder=(-7,-4),order=(8,1,2,3,5,6,9,10))

        leftEnv = ncon([self.psi.mpo,self.psi.mpo.conj(),self.H,outerContract,outerContract],((2,6,1,5,8,-9),(3,6,4,7,8,-10),(3,2),(4,1),(7,5)),forder=(-10,-9),order=(8,2,3,6,1,4,5,7))
        leftEnv += ncon([self.psi.mpo,self.psi.mpo.conj(),self.H,outerContract,outerContract],((2,5,1,4,8,-9),(2,6,3,7,8,-10),(6,5),(3,1),(7,4)),forder=(-10,-9),order=(8,2,5,6,1,3,4,7))
        leftEnv = self.psi.Tb_inv[tensorLabel].applyRight(leftEnv.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)

        H_eff += ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj(),outerContract,self.psi.R[tensorLabel].tensor],((10,7),(2,5,-1,4,7,8),(2,5,-3,6,10,9),(6,4),(9,8)),forder=(-3,-1),order=(8,9,4,5,6,2,7,10))
        H_eff += ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj(),outerContract,self.psi.R[tensorLabel].tensor],((10,7),(2,5,1,-4,7,8),(2,5,3,-6,10,9),(3,1),(9,8)),forder=(-6,-4),order=(7,10,1,2,3,5,8,9))
        return oneBodyH(1/2*H_eff)

class gradImplementation_mpso_2d_mps_twoSite_twoBodyH_hori(gradImplementation_mpso_2d_mps_uniform):
    def getEffectiveH(self,site):
        if site == 1:
            outerContract= ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((-1,3,4,5),(-2,3,4,6),(6,5)),forder=(-2,-1),order=(4,3,5,6))
            tensorLabel = 'bot'
        elif site == 2:
            outerContract= ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((1,-2,4,5),(1,-3,4,6),(6,5)),forder=(-3,-2),order=(4,1,5,6))
            tensorLabel = 'top'

        H_eff = ncon([self.psi.mpo,self.H,self.psi.mpo.conj(),outerContract,self.psi.R[tensorLabel].tensor],((2,6,-1,5,9,10),(3,7,2,6),(3,7,-4,8,9,11),(8,5),(11,10)),forder=(-4,-1),order=(11,10,5,6,7,8,2,3,9))
        H_eff += ncon([self.psi.mpo,self.H,self.psi.mpo.conj(),outerContract,self.psi.R[tensorLabel].tensor],((2,6,1,-5,9,10),(3,7,2,6),(3,7,4,-8,9,11),(4,1),(11,10)),forder=(-8,-5),order=(9,1,2,3,4,6,7,10,11))
        H_eff += ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.mpo.conj(),outerContract,outerContract,outerContract,self.psi.R[tensorLabel].tensor],((2,5,1,-4,15,16),(9,13,8,12,16,17),(6,10,5,9),(2,6,3,-7,15,19),(10,13,11,14,19,18),(3,1),(11,8),(14,12),(18,17)),forder=(-7,-4),order=(18,17,12,13,14,8,9,10,11,16,19,5,6,1,2,3,15))
        H_eff += ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.mpo.conj(),outerContract,outerContract,outerContract,self.psi.R[tensorLabel].tensor],((2,5,1,4,15,16),(9,13,-8,12,16,17),(6,10,5,9),(2,6,3,7,15,19),(10,13,-11,14,19,18),(3,1),(7,4),(14,12),(18,17)),forder=(-11,-8),order=(15,1,2,3,4,5,6,7,16,19,9,10,12,13,14,17,18))

        leftEnv = ncon([self.psi.mpo,self.H,self.psi.mpo.conj(),outerContract,outerContract],((2,6,1,5,9,-10),(3,7,2,6),(3,7,4,8,9,-11),(4,1),(8,5)),forder=(-11,-10),order=(9,1,2,3,4,5,6,7,8))
        leftEnv += ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.mpo.conj(),outerContract,outerContract,outerContract,outerContract],((2,5,1,4,15,16),(9,13,8,12,16,-17),(6,10,5,9),(2,6,3,7,15,19),(10,13,11,14,19,-18),(3,1),(7,4),(11,8),(14,12)),forder=(-18,-17),order=(15,1,2,3,4,5,6,7,16,19,8,9,10,11,12,13,14))
        leftEnv = self.psi.Tb_inv[tensorLabel].applyRight(leftEnv.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)

        H_eff += ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj(),outerContract,self.psi.R[tensorLabel].tensor],((10,7),(2,5,-1,4,7,8),(2,5,-3,6,10,9),(6,4),(9,8)),forder=(-3,-1),order=(8,9,4,5,6,2,7,10))
        H_eff += ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj(),outerContract,self.psi.R[tensorLabel].tensor],((10,7),(2,5,1,-4,7,8),(2,5,3,-6,10,9),(3,1),(9,8)),forder=(-6,-4),order=(7,10,1,2,3,5,8,9))
        return oneBodyH(1/2*H_eff)

class gradImplementation_mpso_2d_mps_twoSite_twoBodyH_vert(gradImplementation_mpso_2d_mps_uniform):
    def getEffectiveH(self,site):
        if site == 1:
            outerContract= ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((-1,-3,5,6),(-2,-4,5,7),(6,7)),forder=(-2,-4,-1,-3),order=(5,6,7))
            tensorLabel = 'square'
        elif site == 2:
            outerContract= ncon([self.psi.mps,self.psi.mps,self.psi.mps.conj(),self.psi.mps.conj(),self.psi.T.tensor],((1,-2,7,8),(-4,6,8,9),(1,-3,7,11),(-5,6,11,10),(10,9)),forder=(-3,-5,-2,-4),order=(7,1,8,11,6,9,10))
            tensorLabel = 'prong'

        H_eff = ncon([self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo.conj(),self.H,outerContract,self.psi.RR[tensorLabel].tensor],((2,6,-1,5,15,16),(3,6,-4,7,15,17),(9,13,-8,12,18,19),(10,13,-11,14,18,20),(10,3,9,2),(14,7,12,5),(20,19,17,16)),forder=(-11,-4,-8,-1),order=(18,9,10,12,13,14,19,20,17,16,5,6,7,2,3,15))
        H_eff += ncon([self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo.conj(),self.H,outerContract,self.psi.RR[tensorLabel].tensor],((2,6,1,-5,15,16),(3,6,4,-7,15,17),(9,13,8,-12,18,19),(10,13,11,-14,18,20),(10,3,9,2),(11,4,8,1),(20,19,17,16)),forder=(-14,-7,-12,-5),order=(18,8,9,10,11,13,19,20,17,16,6,1,2,3,4,15))
        H_eff += ncon([self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo.conj(),self.H,outerContract,self.psi.RR[tensorLabel].tensor],((2,5,-1,4,15,16),(2,6,-3,7,15,17),(9,12,-8,11,18,19),(9,13,-10,14,18,20),(13,6,12,5),(14,7,11,4),(20,19,17,16)),forder=(-10,-3,-8,-1),order=(18,9,11,12,13,14,20,19,17,16,4,5,6,7,2,15))
        H_eff += ncon([self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo.conj(),self.H,outerContract,self.psi.RR[tensorLabel].tensor],((2,5,1,-4,15,16),(2,6,3,-7,15,17),(9,12,8,-11,18,19),(9,13,10,-14,18,20),(13,6,12,5),(10,3,8,1),(20,19,17,16)),forder=(-14,-7,-11,-4),order=(18,8,9,10,12,13,20,19,17,16,5,6,1,2,3,15))

        leftEnv = ncon([self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo.conj(),self.H,outerContract,outerContract],((2,6,1,5,15,-16),(3,6,4,7,15,-17),(9,13,8,12,18,-19),(10,13,11,14,18,-20),(10,3,9,2),(11,4,8,1),(14,7,12,5)),forder=(-20,-19,-17,-16),order=(18,9,10,13,8,11,12,14,15,2,3,6,1,4,5,7)) 
        leftEnv += ncon([self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo.conj(),self.H,outerContract,outerContract],((2,5,1,4,15,-16),(2,6,3,7,15,-17),(9,12,8,11,18,-19),(9,13,10,14,18,-20),(13,6,12,5),(10,3,8,1),(14,7,11,4)),forder=(-20,-19,-17,-16),order=(18,9,12,13,8,10,11,14,15,2,5,6,1,3,4,7)) 
        leftEnv = self.psi.Tb2_inv[tensorLabel].applyRight(leftEnv.reshape(self.psi.D_mpo**4)).reshape(self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo)

        H_eff += ncon([self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo.conj(),leftEnv,outerContract,self.psi.RR[tensorLabel].tensor],((2,5,-1,4,13,14),(2,5,-3,6,15,16),(8,11,-7,10,17,18),(8,11,-9,12,19,20),(19,17,15,13),(12,6,10,4),(20,18,16,14)),forder=(-9,-3,-7,-1),order=(19,17,15,13,8,11,2,5,12,10,4,6,20,18,16,14))
        H_eff += ncon([self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo.conj(),leftEnv,outerContract,self.psi.RR[tensorLabel].tensor],((2,5,1,-4,13,14),(2,5,3,-6,15,16),(8,11,7,-10,17,18),(8,11,9,-12,19,20),(19,17,15,13),(9,3,7,1),(20,18,16,14)),forder=(-12,-6,-10,-4),order=(19,17,15,13,2,5,8,11,1,3,7,9,20,18,16,14))
        return twoBodyH(1/2*H_eff.reshape(4,4))
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
        fp = dict()
        fp['outerContractDouble'] = outerContractDouble
        return fp

    def getCentralTerms(self):
        grad = ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.R.tensor,self.outerContract,self.outerContract],((2,1,-9,10),(6,5,10,11),(-3,7,2,6),(7,8,-13,12),(12,11),(-4,1),(8,5)),forder=(-4,-3,-9,-13),order=(12,11,5,8,6,7,10,1))
        grad += ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.R.tensor,self.outerContract,self.outerContract],((2,1,9,10),(6,5,10,11),(3,-7,2,6),(3,4,9,-13),(-12,11),(4,1),(-8,5)),forder=(-8,-7,-13,-12),order=(9,1,2,3,4,10,5,6,11))
        return grad

    def buildLeftEnv(self,H=None,wrapAround=False):
        if H is None:
            H = self.H
        env = ncon([self.psi.mpo,self.psi.mpo,H,self.psi.mpo.conj(),self.psi.mpo.conj(),self.outerContract,self.outerContract],((2,1,9,10),(6,5,10,-11),(3,7,2,6),(3,4,9,13),(7,8,13,-12),(4,1),(8,5)),forder=(-12,-11),order=(9,1,2,3,4,10,13,6,7,5,8))
        env = self.psi.Tb_inv.applyRight(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        if wrapAround == True:
            env = ncon([env,self.psi.R.tensor,self.outerContract],((-1,-2),(-3,-4),(-5,-6)),forder=(-5,-6,-1,-2,-3,-4))
        return env
    def buildRightEnv(self,H=None):
        if H is None:
            H = self.H
        env = ncon([self.psi.mpo,self.psi.mpo,H,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.R.tensor,self.outerContract,self.outerContract],((2,1,-9,10),(6,5,10,11),(3,7,2,6),(3,4,-14,13),(7,8,13,12),(12,11),(4,1),(8,5)),forder=(-14,-9),order=(11,12,5,6,7,8,10,13,2,3,1,4))
        env = self.psi.Tb_inv.applyLeft(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        env = ncon([self.outerContract,env],((-1,-2),(-3,-4)),forder=(-1,-2,-3,-4))
        return env

    def buildTopEnvGeo(self,fixedPoints,outers):
        env = ncon([self.innerContract,fixedPoints['RR_d'],self.psi.mpo,self.psi.mpo.conj(),outers['outerContractDouble'],outers['outerContractDouble']],((1,2,3,4,5,6),(10,8,5,6),(12,11,-7,8),(12,13,-9,10),(13,2,11,4),(-14,1,-15,3)),forder=(-14,-15,-9,-7),order=(5,6,8,10,4,2,11,12,13,1,3))
        env += ncon([self.innerContract,fixedPoints['RR_d'],self.outerContract,outers['outerContractDouble']],((1,2,3,4,5,6),(-8,-7,5,6),(1,3),(-9,2,-10,4)),forder=(-9,-10,-8,-7),order=(5,6,1,3,2,4))
        return env
    def buildBotEnvGeo(self,fixedPoints,outers):
        env = ncon([self.innerContract,fixedPoints['RR_d'],self.psi.mpo,self.psi.mpo.conj(),outers['outerContractDouble'],outers['outerContractDouble']],((1,2,3,4,5,6),(5,6,7,8),(10,9,-12,8),(10,11,-13,7),(2,11,4,9),(1,-14,3,-15)),forder=(-14,-15,-13,-12),order=(5,6,7,8,9,10,11,4,2,1,3))
        env += ncon([self.innerContract,fixedPoints['RR_d'],self.outerContract,outers['outerContractDouble']],((1,2,3,4,5,6),(5,6,-7,-8),(1,3),(2,-9,4,-10)),forder=(-9,-10,-7,-8),order=(5,6,1,3,2,4))
        return env
    def buildTopEnvGeo_quadrants(self,fixedPoints,outers,leftEnv,):
        env = ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj(),fixedPoints['RR_d'],outers['outerContractDouble']],((6,4),(2,1,4,5),(2,3,6,7),(-9,-8,7,5),(-10,3,-11,1)),forder=(-10,-11,-9,-8),order=(6,4,2,5,7,1,3))
        return env
    def buildBotEnvGeo_quadrants(self,fixedPoints,outers,leftEnv,):
        env = ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj(),fixedPoints['RR_d'],outers['outerContractDouble']],((6,4),(2,1,4,5),(2,3,6,7),(7,5,-8,-9),(3,-10,1,-11)),forder=(-10,-11,-8,-9),order=(4,6,2,5,7,1,3))
        return env
    def buildRightEnvGeo_quadrants(self,fixedPoints,outers):
        env = ncon([self.innerContract,fixedPoints['RR_d'],self.psi.mpo,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo.conj(),outers['outerContractDouble'],outers['outerContractDouble']],((1,2,3,4,5,6),(18,15,5,6),(11,10,14,15),(8,7,-13,14),(11,12,17,18),(8,9,-16,17),(9,1,7,3),(12,2,10,4)),forder=(-16,-13),order=(5,6,15,18,11,4,2,10,12,14,17,8,1,3,7,9))
        env += ncon([self.innerContract,fixedPoints['RR_d'],self.psi.mpo,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo.conj(),outers['outerContractDouble'],outers['outerContractDouble']],((1,2,3,4,5,6),(5,6,18,15),(11,10,14,15),(8,7,-13,14),(11,12,17,18),(8,9,-16,17),(1,9,3,7),(2,12,4,10)),forder=(-16,-13),order=(5,6,18,15,11,10,12,4,2,14,17,8,7,9,3,1))
        env = self.psi.Tb_inv.applyLeft(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        env = ncon([self.outerContract,env],((-1,-2),(-3,-4)),forder=(-1,-2,-3,-4))
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
    def buildLeftEnv(self,H=None,wrapAround = False):
        if H is None:
            H = self.H
        env = ncon([self.psi.mpo,self.psi.mpo,H,self.psi.mpo.conj(),self.psi.mpo.conj(),self.outerContractDouble],((2,1,9,-10),(6,5,11,-12),(3,7,2,6),(3,4,9,-13),(7,8,11,-14),(4,8,1,5)),forder=(-13,-10,-14,-12),order=(9,1,2,3,4,11,5,6,7,8))
        env = self.psi.Tb2_inv.applyRight(env.reshape(self.psi.D_mpo**4)).reshape(self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo)
        if wrapAround == True:
            env1 = ncon([env,self.psi.mpo,self.psi.mpo.conj(),self.psi.RR.tensor,self.outerContractDouble],((-10,-8,6,4),(2,1,4,5),(2,3,6,7),(-11,-9,7,5),(-12,3,-13,1)),forder=(-12,-13,-10,-8,-11,-9),order=(4,6,2,5,7,1,3))
            env2 = ncon([env,self.psi.mpo,self.psi.mpo.conj(),self.psi.RR.tensor,self.outerContractDouble],((6,4,-10,-8),(2,1,4,5),(2,3,6,7),(7,5,-11,-9),(3,-12,1,-13)),forder=(-12,-13,-10,-8,-11,-9),order=(4,6,2,5,7,1,3))
            env = env1 + env2
        return env
    def buildRightEnv(self,H=None):
        if H is None:
            H = self.H
        env = ncon([self.psi.mpo,self.psi.mpo,H,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.RR.tensor,self.outerContractDouble],((2,1,-9,10),(6,5,-11,12),(3,7,2,6),(3,4,-13,14),(7,8,-15,16),(14,10,16,12),(4,8,1,5)),forder=(-13,-9,-15,-11),order=(12,16,5,6,7,8,10,14,1,2,3,4))
        env = self.psi.Tb2_inv.applyLeft(env.reshape(self.psi.D_mpo**4)).reshape(self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo)
        env  = ncon([self.outerContract,env],((-1,-2),(-3,-4,5,5)),forder=(-1,-2,-3,-4)) + ncon([self.outerContract,env],((-1,-2),(3,3,-4,-5)),forder=(-1,-2,-4,-5))
        return env

    def buildTopEnvGeo(self,fixedPoints,outers):
        return ncon([self.innerContract,fixedPoints['RRR_d_lower'],outers['outerContractTriple_lower']],((1,2,3,4,5,6,7,8),(-9,-10,5,6,7,8),(-11,1,2,-12,3,4)),forder=(-11,-12,-9,-10),order=(5,6,7,8,1,2,3,4))
    def buildBotEnvGeo(self,fixedPoints,outers):
        return ncon([self.innerContract,fixedPoints['RRR_d_upper'],outers['outerContractTriple_upper']],((1,2,3,4,5,6,7,8),(5,6,7,8,-9,-10),(1,2,-11,3,4,-12)),forder=(-11,-12,-9,-10),order=(5,6,7,8,1,2,3,4))
    def buildTopEnvGeo_quadrants(self,fixedPoints,outers,leftEnv,):
        return ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo.conj(),fixedPoints['RRR_d_lower'],outers['outerContractTriple_lower']],((13,11,9,7),(2,1,7,8),(2,3,9,10),(5,4,11,12),(5,6,13,14),(-16,-15,14,12,10,8),(-17,6,3,-18,4,1)),forder=(-17,-18,-16,-15),order=(7,9,2,11,13,5,8,10,12,14,1,3,4,6))
    def buildBotEnvGeo_quadrants(self,fixedPoints,outers,leftEnv,):
        return ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo.conj(),fixedPoints['RRR_d_upper'],outers['outerContractTriple_upper']],((13,11,9,7),(2,1,7,8),(2,3,9,10),(5,4,11,12),(5,6,13,14),(14,12,10,8,-15,-16),(6,3,-17,4,1,-18)),forder=(-17,-18,-15,-16),order=(7,9,2,11,13,5,8,10,12,14,1,3,4,6))
    def buildRightEnvGeo_quadrants(self,fixedPoints,outers):
        env1 = self.buildTopEnvGeo(fixedPoints,outers)
        env1 = ncon([env1,self.psi.mpo,self.psi.mpo.conj()],((3,1,7,5),(2,1,-4,5),(2,3,-6,7)),forder=(-6,-4),order=(5,7,1,2,3))
        env2 = self.buildBotEnvGeo(fixedPoints,outers)
        env2 = ncon([env2,self.psi.mpo,self.psi.mpo.conj()],((3,1,7,5),(2,1,-4,5),(2,3,-6,7)),forder=(-6,-4),order=(5,7,1,2,3))
        env = env1 + env2
        env = self.psi.Tb_inv.applyLeft(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        env = ncon([self.outerContract,env],((-1,-2),(-3,-4)),forder=(-1,-2,-3,-4))
        return env
# -----------------------------
class gradImplementation_mpso_2d_mpo_bipartite_twoBodyH_hori(gradImplementation_mpso_2d_mpo_bipartite):
    def __init__(self,psi,H,H_index,grad_index):
        super().__init__(psi,H,H_index,grad_index)
        # objects needed to construct terms
        self.outerContract = dict()

        self.outerContract[1] =  ncon([self.psi.mps[1],self.psi.mps[1].conj(),self.psi.T[2].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
        self.outerContract[2] =  ncon([self.psi.mps[2],self.psi.mps[2].conj(),self.psi.T[1].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))

        if self.H_index == self.grad_index:
            innerContract = ncon([self.psi.mpo[self.index1],self.psi.mpo[self.index2],self.H,self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2].conj()],((2,-1,9,10),(6,-5,10,-11),(3,7,2,6),(3,-4,9,13),(7,-8,13,-12)),forder=(-4,-8,-1,-5,-12,-11),order=(9,2,3,10,13,6,7))
            exp = np.real(ncon([innerContract,self.outerContract[self.index1],self.outerContract[self.index2],self.psi.R[self.index1].tensor],((1,2,3,4,5,6),(1,3),(2,4),(5,6)),order=(1,3,2,4,5,6)))
        else:
            innerContract = ncon([self.psi.mpo[self.index2],self.psi.mpo[self.index1],self.H,self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1].conj()],((2,-1,9,10),(6,-5,10,-11),(3,7,2,6),(3,-4,9,13),(7,-8,13,-12)),forder=(-4,-8,-1,-5,-12,-11),order=(9,2,3,10,13,6,7))
            exp = np.real(ncon([innerContract,self.outerContract[self.index2],self.outerContract[self.index1],self.psi.R[self.index2].tensor],((1,2,3,4,5,6),(1,3),(2,4),(5,6)),order=(1,3,2,4,5,6)))
        self.h_tilde = (self.H.reshape(4,4)-exp*np.eye(4)).reshape(2,2,2,2)

        if self.H_index == self.grad_index:
            self.innerContract = ncon([self.psi.mpo[self.index1],self.psi.mpo[self.index2],self.h_tilde,self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2].conj()],((2,-1,9,10),(6,-5,10,-11),(3,7,2,6),(3,-4,9,13),(7,-8,13,-12)),forder=(-4,-8,-1,-5,-12,-11),order=(9,2,3,10,13,6,7))
        else:
            self.innerContract = ncon([self.psi.mpo[self.index2],self.psi.mpo[self.index1],self.h_tilde,self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1].conj()],((2,-1,9,10),(6,-5,10,-11),(3,7,2,6),(3,-4,9,13),(7,-8,13,-12)),forder=(-4,-8,-1,-5,-12,-11),order=(9,2,3,10,13,6,7))


    #d dep bipartite transfers for geosum
    def init_mps_transfers(self):
        Td_matrix = dict()
        Td = dict()
        Td_matrix[1] = np.eye(self.psi.D_mps**2)
        Td[1] = Td_matrix[1].reshape(self.psi.D_mps,self.psi.D_mps,self.psi.D_mps,self.psi.D_mps)
        Td_matrix[2] = Td_matrix[1]
        Td[2] = Td[1]
        return Td_matrix,Td
    def apply_mps_transfers(self,Td_matrix):
        Td_matrix[1] = np.dot(Td_matrix[1],self.psi.Ta[1].matrix)
        Td_matrix[2] = np.dot(Td_matrix[2],self.psi.Ta[2].matrix)
        Td = dict()
        Td[1] = Td_matrix[1].reshape(self.psi.D_mps,self.psi.D_mps,self.psi.D_mps,self.psi.D_mps)
        Td[2] = Td_matrix[2].reshape(self.psi.D_mps,self.psi.D_mps,self.psi.D_mps,self.psi.D_mps)
        return Td_matrix,Td

    def getFixedPoints(self,d,Td):
        RR_d = dict()
        RR_d_p1 = dict()

        TT_d_12 = mpsu1Transfer_left_twoLayerWithMpsInsertBip(self.psi.mps[1],self.psi.mps[2],self.psi.mpo[1],self.psi.mpo[2],self.psi.T[1],self.psi.T[2],Td[1],Td[2])
        RR_d[1] = TT_d_12.findRightEig()
        RR_d[1].norm_pairedCanon()
        RR_d[1] = RR_d[1].tensor
        del TT_d_12

        TT_d_21 = mpsu1Transfer_left_twoLayerWithMpsInsertBip(self.psi.mps[2],self.psi.mps[1],self.psi.mpo[2],self.psi.mpo[1],self.psi.T[2],self.psi.T[1],Td[2],Td[1])
        RR_d[2] = TT_d_21.findRightEig()
        RR_d[2].norm_pairedCanon()
        RR_d[2] = RR_d[2].tensor
        del TT_d_21

        TT_d_12_p1 = mpsu1Transfer_left_twoLayerWithMpsInsertBip_plusOne(self.psi.mps[1],self.psi.mps[2],self.psi.mpo[1],self.psi.mpo[2],self.psi.T[1],self.psi.T[2],Td[1],Td[2])
        RR_d_p1[1] = TT_d_12_p1.findRightEig()
        RR_d_p1[1].norm_pairedCanon()
        RR_d_p1[1] = RR_d_p1[1].tensor
        del TT_d_12_p1

        TT_d_21_p1 = mpsu1Transfer_left_twoLayerWithMpsInsertBip_plusOne(self.psi.mps[2],self.psi.mps[1],self.psi.mpo[2],self.psi.mpo[1],self.psi.T[2],self.psi.T[1],Td[2],Td[1])
        RR_d_p1[2] = TT_d_21_p1.findRightEig()
        RR_d_p1[2].norm_pairedCanon()
        RR_d_p1[2] = RR_d_p1[2].tensor
        del TT_d_21_p1

        fp = dict()
        fp['RR_d'] = RR_d
        fp['RR_d_p1'] = RR_d_p1
        return fp

    def getOuterContracts(self,Td):
        outerContractDouble = dict()
        outerContractDouble_p1 = dict()

        outerContractDouble[1] = ncon([self.psi.mps[1],self.psi.mps[2],self.psi.mps[1].conj(),self.psi.mps[2].conj(),Td[2],self.psi.T[1].tensor],((-1,5,6),(-3,7,8),(-2,5,11),(-4,10,9),(11,6,10,7),(9,8)),forder=(-2,-4,-1,-3),order=(5,6,11,7,10,8,9))
        outerContractDouble[2] = ncon([self.psi.mps[2],self.psi.mps[1],self.psi.mps[2].conj(),self.psi.mps[1].conj(),Td[1],self.psi.T[2].tensor],((-1,5,6),(-3,7,8),(-2,5,11),(-4,10,9),(11,6,10,7),(9,8)),forder=(-2,-4,-1,-3),order=(5,6,11,7,10,8,9))

        outerContractDouble_p1[1] = ncon([self.psi.mps[1],self.psi.mps[2],self.psi.mps[1],self.psi.mps[1].conj(),self.psi.mps[2].conj(),self.psi.mps[1].conj(),Td[2],self.psi.T[2].tensor],((-1,6,7),(3,8,9),(-4,9,10),(-2,6,14),(3,13,12),(-5,12,11),(14,7,13,8),(11,10)),forder=(-2,-5,-1,-4),order=(6,7,14,8,13,3,9,12,10,11))
        outerContractDouble_p1[2] = ncon([self.psi.mps[2],self.psi.mps[1],self.psi.mps[2],self.psi.mps[2].conj(),self.psi.mps[1].conj(),self.psi.mps[2].conj(),Td[1],self.psi.T[1].tensor],((-1,6,7),(3,8,9),(-4,9,10),(-2,6,14),(3,13,12),(-5,12,11),(14,7,13,8),(11,10)),forder=(-2,-5,-1,-4),order=(6,7,14,8,13,3,9,12,10,11))

        fp = dict()
        fp['outerContractDouble'] = outerContractDouble
        fp['outerContractDouble_p1'] = outerContractDouble_p1
        return fp

    def getCentralTerms(self):
        if self.H_index == self.grad_index:
            return ncon([self.psi.mpo[self.index1],self.psi.mpo[self.index2],self.H,self.psi.mpo[self.index2].conj(),self.psi.R[self.index1].tensor,self.outerContract[self.index1],self.outerContract[self.index2]],((2,1,-9,10),(6,5,10,11),(-3,7,2,6),(7,8,-13,12),(12,11),(-4,1),(8,5)),forder=(-4,-3,-9,-13),order=(12,11,5,8,6,7,10,1))
        else:
            return ncon([self.psi.mpo[self.index2],self.psi.mpo[self.index1],self.H,self.psi.mpo[self.index2].conj(),self.psi.R[self.index2].tensor,self.outerContract[self.index2],self.outerContract[self.index1]],((2,1,9,10),(6,5,10,11),(3,-7,2,6),(3,4,9,-13),(-12,11),(4,1),(-8,5)),forder=(-8,-7,-13,-12),order=(9,1,2,3,4,1,5,6,11))

    def buildLeftEnv(self,H=None,wrapAround=False):
        if H is None:
            H = self.H
            appendExtraSite = True #init left env for gradient in line
        else:
            appendExtraSite = False #left env for quadrants, dont append extra site, is done later

        if self.H_index == self.grad_index:
            env = ncon([self.psi.mpo[self.index1],self.psi.mpo[self.index2],H,self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2].conj(),self.outerContract[self.index1],self.outerContract[self.index2]],((2,1,9,10),(6,5,10,-11),(3,7,2,6),(3,4,9,13),(7,8,13,-12),(4,1),(8,5)),forder=(-12,-11),order=(9,1,2,3,4,10,13,6,7,5,8))
            env = self.psi.Tb_inv[self.index1].applyRight(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        else:
            env = ncon([self.psi.mpo[self.index2],self.psi.mpo[self.index1],H,self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1].conj(),self.outerContract[self.index2],self.outerContract[self.index1]],((2,1,9,10),(6,5,10,-11),(3,7,2,6),(3,4,9,13),(7,8,13,-12),(4,1),(8,5)),forder=(-12,-11),order=(9,1,2,3,4,10,13,6,7,5,8))
            if appendExtraSite is True:
                env = ncon([env,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.outerContract[self.index2]],((6,4),(2,1,4,-5),(2,3,6,-7),(3,1)),forder=(-7,-5),order=(4,6,1,2,3))
                env = self.psi.Tb_inv[self.index1].applyRight(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
            else:
                env = self.psi.Tb_inv[self.index2].applyRight(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)

        if wrapAround == True:
            env = ncon([env,self.psi.R[self.index2].tensor,self.outerContract[self.index1]],((-1,-2),(-3,-4),(-5,-6)),forder=(-5,-6,-1,-2,-3,-4))
        return env

    def buildRightEnv(self,H=None):
        if H is None:
            H = self.H
        if self.H_index == self.grad_index:
            env = ncon([self.psi.mpo[self.index1],self.psi.mpo[self.index2],H,self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2].conj(),self.psi.R[self.index1].tensor,self.outerContract[self.index1],self.outerContract[self.index2]],((2,1,-9,10),(6,5,10,11),(3,7,2,6),(3,4,-14,13),(7,8,13,12),(12,11),(4,1),(8,5)),forder=(-14,-9),order=(11,12,5,6,7,8,10,13,2,3,1,4))
            env = ncon([env,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.outerContract[self.index2]],((7,5),(2,1,-4,5),(2,3,-6,7),(3,1)),forder=(-6,-4),order=(5,7,1,2,3))
        else:
            env = ncon([self.psi.mpo[self.index2],self.psi.mpo[self.index1],H,self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1].conj(),self.psi.R[self.index2].tensor,self.outerContract[self.index2],self.outerContract[self.index1]],((2,1,-9,10),(6,5,10,11),(3,7,2,6),(3,4,-14,13),(7,8,13,12),(12,11),(4,1),(8,5)),forder=(-14,-9),order=(11,12,5,6,7,8,10,13,2,3,1,4))

        env = self.psi.Tb_inv[self.index2].applyLeft(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        env = ncon([self.outerContract[self.index1],env],((-1,-2),(-3,-4)),forder=(-1,-2,-3,-4))
        return env

    def buildTopEnvGeo(self,fixedPoints,outers):
        if self.H_index == self.grad_index:
            env = ncon([self.innerContract,fixedPoints['RR_d_p1'][self.index1],self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),outers['outerContractDouble_p1'][self.index2],outers['outerContractDouble_p1'][self.index1]],((1,2,3,4,5,6),(10,8,5,6),(12,11,-7,8),(12,13,-9,10),(13,2,11,4),(-14,1,-15,3)),forder=(-14,-15,-9,-7),order=(5,6,8,10,4,2,11,12,13,1,3))
            env += ncon([self.innerContract,fixedPoints['RR_d'][self.index2],self.outerContract[self.index1],outers['outerContractDouble'][self.index1]],((1,2,3,4,5,6),(-8,-7,5,6),(1,3),(-9,2,-10,4)),forder=(-9,-10,-8,-7),order=(5,6,1,3,2,4))
            return env
        else:
            env = ncon([self.innerContract,fixedPoints['RR_d'][self.index1],self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),outers['outerContractDouble'][self.index2],outers['outerContractDouble'][self.index1]],((1,2,3,4,5,6),(10,8,5,6),(12,11,-7,8),(12,13,-9,10),(13,2,11,4),(-14,1,-15,3)),forder=(-14,-15,-9,-7),order=(5,6,8,10,4,2,11,12,13,1,3))
            env += ncon([self.innerContract,fixedPoints['RR_d_p1'][self.index2],self.outerContract[self.index2],outers['outerContractDouble_p1'][self.index1]],((1,2,3,4,5,6),(-8,-7,5,6),(1,3),(-9,2,-10,4)),forder=(-9,-10,-8,-7),order=(5,6,1,3,2,4))
            return env

    def buildBotEnvGeo(self,fixedPoints,outers):
        if self.H_index == self.grad_index:
            env = ncon([self.innerContract,fixedPoints['RR_d_p1'][self.index1],self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),outers['outerContractDouble_p1'][self.index2],outers['outerContractDouble_p1'][self.index1]],((1,2,3,4,5,6),(5,6,7,8),(10,9,-12,8),(10,11,-13,7),(2,11,4,9),(1,-14,3,-15)),forder=(-14,-15,-13,-12),order=(5,6,7,8,9,10,11,4,2,1,3))
            env += ncon([self.innerContract,fixedPoints['RR_d'][self.index1],self.outerContract[self.index1],outers['outerContractDouble'][self.index2]],((1,2,3,4,5,6),(5,6,-7,-8),(1,3),(2,-9,4,-10)),forder=(-9,-10,-7,-8),order=(5,6,1,3,2,4))
            return env
        else:
            env = ncon([self.innerContract,fixedPoints['RR_d'][self.index2],self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),outers['outerContractDouble'][self.index1],outers['outerContractDouble'][self.index2]],((1,2,3,4,5,6),(5,6,7,8),(10,9,-12,8),(10,11,-13,7),(2,11,4,9),(1,-14,3,-15)),forder=(-14,-15,-13,-12),order=(5,6,7,8,9,10,11,4,2,1,3))
            env += ncon([self.innerContract,fixedPoints['RR_d_p1'][self.index2],self.outerContract[self.index2],outers['outerContractDouble_p1'][self.index1]],((1,2,3,4,5,6),(5,6,-7,-8),(1,3),(2,-9,4,-10)),forder=(-9,-10,-7,-8),order=(5,6,1,3,2,4))
            return env

    def buildTopEnvGeo_quadrants(self,fixedPoints,outers,leftEnv):
        if self.H_index == self.grad_index:
            env = ncon([leftEnv,self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj(),fixedPoints['RR_d_p1'][self.index2],outers['outerContractDouble_p1'][self.index1]],((6,4),(2,1,4,5),(2,3,6,7),(-9,-8,7,5),(-10,3,-11,1)),forder=(-10,-11,-9,-8),order=(6,4,2,5,7,1,3))
            newLeftEnv = ncon([leftEnv,self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj(),self.outerContract[self.index1]],((6,4),(2,1,4,-5),(2,3,6,-7),(3,1)),forder=(-7,-5),order=(4,6,1,2,3))
            env += ncon([newLeftEnv,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),fixedPoints['RR_d'][self.index2],outers['outerContractDouble'][self.index1]],((6,4),(2,1,4,5),(2,3,6,7),(-9,-8,7,5),(-10,3,-11,1)),forder=(-10,-11,-9,-8),order=(6,4,2,5,7,1,3))
            return env
        else:
            env = ncon([leftEnv,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),fixedPoints['RR_d'][self.index2],outers['outerContractDouble'][self.index1]],((6,4),(2,1,4,5),(2,3,6,7),(-9,-8,7,5),(-10,3,-11,1)),forder=(-10,-11,-9,-8),order=(6,4,2,5,7,1,3))
            newLeftEnv = ncon([leftEnv,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.outerContract[self.index2]],((6,4),(2,1,4,-5),(2,3,6,-7),(3,1)),forder=(-7,-5),order=(4,6,1,2,3))
            env += ncon([newLeftEnv,self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj(),fixedPoints['RR_d_p1'][self.index2],outers['outerContractDouble_p1'][self.index1]],((6,4),(2,1,4,5),(2,3,6,7),(-9,-8,7,5),(-10,3,-11,1)),forder=(-10,-11,-9,-8),order=(6,4,2,5,7,1,3))
            return env

    def buildBotEnvGeo_quadrants(self,fixedPoints,outers,leftEnv):
        if self.H_index == self.grad_index:
            env = ncon([leftEnv,self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj(),fixedPoints['RR_d_p1'][self.index2],outers['outerContractDouble_p1'][self.index1]],((6,4),(2,1,4,5),(2,3,6,7),(7,5,-8,-9),(3,-10,1,-11)),forder=(-10,-11,-8,-9),order=(4,6,2,5,7,1,3))
            newLeftEnv = ncon([leftEnv,self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj(),self.outerContract[self.index1]],((6,4),(2,1,4,-5),(2,3,6,-7),(3,1)),forder=(-7,-5),order=(4,6,1,2,3))
            env += ncon([newLeftEnv,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),fixedPoints['RR_d'][self.index1],outers['outerContractDouble'][self.index2]],((6,4),(2,1,4,5),(2,3,6,7),(7,5,-8,-9),(3,-10,1,-11)),forder=(-10,-11,-8,-9),order=(4,6,2,5,7,1,3))
            return env
        else:
            env = ncon([leftEnv,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),fixedPoints['RR_d'][self.index1],outers['outerContractDouble'][self.index2]],((6,4),(2,1,4,5),(2,3,6,7),(7,5,-8,-9),(3,-10,1,-11)),forder=(-10,-11,-8,-9),order=(4,6,2,5,7,1,3))
            newLeftEnv = ncon([leftEnv,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.outerContract[self.index2]],((6,4),(2,1,4,-5),(2,3,6,-7),(3,1)),forder=(-7,-5),order=(4,6,1,2,3))
            env += ncon([newLeftEnv,self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj(),fixedPoints['RR_d_p1'][self.index2],outers['outerContractDouble_p1'][self.index1]],((6,4),(2,1,4,5),(2,3,6,7),(7,5,-8,-9),(3,-10,1,-11)),forder=(-10,-11,-8,-9),order=(4,6,2,5,7,1,3))
            return env

    def buildRightEnvGeo_quadrants(self,fixedPoints,outers):
        if self.H_index == self.grad_index:
            env1 = ncon([self.innerContract,fixedPoints['RR_d'][self.index2],self.psi.mpo[self.index1],self.psi.mpo[self.index2],self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2].conj(),outers['outerContractDouble'][self.index2],outers['outerContractDouble'][self.index1]],((1,2,3,4,5,6),(18,15,5,6),(11,10,14,15),(8,7,-13,14),(11,12,17,18),(8,9,-16,17),(9,1,7,3),(12,2,10,4)),forder=(-16,-13),order=(5,6,15,18,11,4,2,10,12,14,17,8,1,3,7,9))
            env1 += ncon([self.innerContract,fixedPoints['RR_d'][self.index1],self.psi.mpo[self.index1],self.psi.mpo[self.index2],self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2].conj(),outers['outerContractDouble'][self.index1],outers['outerContractDouble'][self.index2]],((1,2,3,4,5,6),(5,6,18,15),(11,10,14,15),(8,7,-13,14),(11,12,17,18),(8,9,-16,17),(1,9,3,7),(2,12,4,10)),forder=(-16,-13),order=(5,6,18,15,11,10,12,4,2,14,17,8,7,9,3,1))
            env2 = ncon([self.innerContract,fixedPoints['RR_d_p1'][self.index1],self.psi.mpo[self.index2],self.psi.mpo[self.index1],self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1].conj(),outers['outerContractDouble_p1'][self.index1],outers['outerContractDouble_p1'][self.index2]],((1,2,3,4,5,6),(18,15,5,6),(11,10,14,15),(8,7,-13,14),(11,12,17,18),(8,9,-16,17),(9,1,7,3),(12,2,10,4)),forder=(-16,-13),order=(5,6,15,18,11,4,2,10,12,14,17,8,1,3,7,9))
            env2 += ncon([self.innerContract,fixedPoints['RR_d_p1'][self.index1],self.psi.mpo[self.index2],self.psi.mpo[self.index1],self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1].conj(),outers['outerContractDouble_p1'][self.index1],outers['outerContractDouble_p1'][self.index2]],((1,2,3,4,5,6),(5,6,18,15),(11,10,14,15),(8,7,-13,14),(11,12,17,18),(8,9,-16,17),(1,9,3,7),(2,12,4,10)),forder=(-16,-13),order=(5,6,18,15,11,10,12,4,2,14,17,8,7,9,3,1))
        else:
            env1 = ncon([self.innerContract,fixedPoints['RR_d_p1'][self.index2],self.psi.mpo[self.index1],self.psi.mpo[self.index2],self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2].conj(),outers['outerContractDouble_p1'][self.index2],outers['outerContractDouble_p1'][self.index1]],((1,2,3,4,5,6),(18,15,5,6),(11,10,14,15),(8,7,-13,14),(11,12,17,18),(8,9,-16,17),(9,1,7,3),(12,2,10,4)),forder=(-16,-13),order=(5,6,15,18,11,4,2,10,12,14,17,8,1,3,7,9))
            env1 += ncon([self.innerContract,fixedPoints['RR_d_p1'][self.index2],self.psi.mpo[self.index1],self.psi.mpo[self.index2],self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2].conj(),outers['outerContractDouble_p1'][self.index2],outers['outerContractDouble_p1'][self.index1]],((1,2,3,4,5,6),(5,6,18,15),(11,10,14,15),(8,7,-13,14),(11,12,17,18),(8,9,-16,17),(1,9,3,7),(2,12,4,10)),forder=(-16,-13),order=(5,6,18,15,11,10,12,4,2,14,17,8,7,9,3,1))
            env2 = ncon([self.innerContract,fixedPoints['RR_d'][self.index1],self.psi.mpo[self.index2],self.psi.mpo[self.index1],self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1].conj(),outers['outerContractDouble'][self.index1],outers['outerContractDouble'][self.index2]],((1,2,3,4,5,6),(18,15,5,6),(11,10,14,15),(8,7,-13,14),(11,12,17,18),(8,9,-16,17),(9,1,7,3),(12,2,10,4)),forder=(-16,-13),order=(5,6,15,18,11,4,2,10,12,14,17,8,1,3,7,9))
            env2 += ncon([self.innerContract,fixedPoints['RR_d'][self.index2],self.psi.mpo[self.index2],self.psi.mpo[self.index1],self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1].conj(),outers['outerContractDouble'][self.index2],outers['outerContractDouble'][self.index1]],((1,2,3,4,5,6),(5,6,18,15),(11,10,14,15),(8,7,-13,14),(11,12,17,18),(8,9,-16,17),(1,9,3,7),(2,12,4,10)),forder=(-16,-13),order=(5,6,18,15,11,10,12,4,2,14,17,8,7,9,3,1))

        env2 = ncon([env2,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.outerContract[self.index2]],((7,5),(2,1,-4,5),(2,3,-6,7),(3,1)),forder=(-6,-4),order=(5,7,1,2,3))
        env = env1 + env2
        env = self.psi.Tb_inv[self.index2].applyLeft(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)

        env = ncon([self.outerContract[self.index1],env],((-1,-2),(-3,-4)),forder=(-1,-2,-3,-4))
        return env

# -----------------------------
class gradImplementation_mpso_2d_mpo_bipartite_twoBodyH_vert(gradImplementation_mpso_2d_mpo_bipartite):
    def __init__(self,psi,H,H_index,grad_index):
        super().__init__(psi,H,H_index,grad_index)
        #objects needed to construct terms
        self.outerContract = dict()
        self.outerContractDouble = dict()
        self.outerContract[1] =  ncon([self.psi.mps[1],self.psi.mps[1].conj(),self.psi.T[2].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
        self.outerContract[2] =  ncon([self.psi.mps[2],self.psi.mps[2].conj(),self.psi.T[1].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
        self.outerContractDouble[1] = ncon([self.psi.mps[1],self.psi.mps[2],self.psi.mps[1].conj(),self.psi.mps[2].conj(),self.psi.T[1].tensor],((-1,5,6),(-2,6,7),(-3,5,9),(-4,9,8),(8,7)),forder=(-3,-4,-1,-2))
        self.outerContractDouble[2] = ncon([self.psi.mps[2],self.psi.mps[1],self.psi.mps[2].conj(),self.psi.mps[1].conj(),self.psi.T[2].tensor],((-1,5,6),(-2,6,7),(-3,5,9),(-4,9,8),(8,7)),forder=(-3,-4,-1,-2))

        if self.H_index == self.grad_index:
            leftEnv = ncon([self.psi.mpo[self.index1],self.psi.mpo[self.index2],self.H,self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2].conj(),self.outerContractDouble[self.index1]],((2,1,9,-10),(6,5,11,-12),(3,7,2,6),(3,4,9,-13),(7,8,11,-14),(4,8,1,5)),forder=(-13,-10,-14,-12),order=(9,1,2,3,4,11,5,6,7,8))
            exp = np.einsum('abcd,abcd',leftEnv,self.psi.RR[self.index2].tensor)
        else:
            leftEnv = ncon([self.psi.mpo[self.index2],self.psi.mpo[self.index1],self.H,self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1].conj(),self.outerContractDouble[self.index2]],((2,1,9,-10),(6,5,11,-12),(3,7,2,6),(3,4,9,-13),(7,8,11,-14),(4,8,1,5)),forder=(-13,-10,-14,-12),order=(9,1,2,3,4,11,5,6,7,8))
            exp = np.einsum('abcd,abcd',leftEnv,self.psi.RR[self.index1].tensor)
        self.h_tilde = (self.H.reshape(4,4)-exp*np.eye(4)).reshape(2,2,2,2)

        if self.H_index == self.grad_index:
            self.innerContract= ncon([self.psi.mpo[self.index1],self.psi.mpo[self.index2],self.h_tilde,self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2].conj()],((2,-1,9,-10),(6,-5,11,-12),(3,7,2,6),(3,-4,9,-13),(7,-8,11,-14)),forder=(-4,-8,-1,-5,-13,-10,-14,-12),order=(9,2,3,11,6,7))
        else:
            self.innerContract= ncon([self.psi.mpo[self.index2],self.psi.mpo[self.index1],self.h_tilde,self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1].conj()],((2,-1,9,-10),(6,-5,11,-12),(3,7,2,6),(3,-4,9,-13),(7,-8,11,-14)),forder=(-4,-8,-1,-5,-13,-10,-14,-12),order=(9,2,3,11,6,7))

    #d dep bipartite transfers for geosum
    def init_mps_transfers(self):
        Td_matrix = dict()
        Td = dict()
        Td_matrix[1] = np.eye(self.psi.D_mps**2)
        Td[1] = Td_matrix[1].reshape(self.psi.D_mps,self.psi.D_mps,self.psi.D_mps,self.psi.D_mps)
        Td_matrix[2] = Td_matrix[1]
        Td[2] = Td[1]
        return Td_matrix,Td
    def apply_mps_transfers(self,Td_matrix):
        Td_matrix[1] = np.dot(Td_matrix[1],self.psi.Ta[1].matrix)
        Td_matrix[2] = np.dot(Td_matrix[2],self.psi.Ta[2].matrix)
        Td = dict()
        Td[1] = Td_matrix[1].reshape(self.psi.D_mps,self.psi.D_mps,self.psi.D_mps,self.psi.D_mps)
        Td[2] = Td_matrix[2].reshape(self.psi.D_mps,self.psi.D_mps,self.psi.D_mps,self.psi.D_mps)
        return Td_matrix,Td

    def getFixedPoints(self,d,Td):
        RRR_d_lower = dict()
        RRR_d_upper = dict()
        RRR_d_lower_p1 = dict()
        RRR_d_upper_p1 = dict()

        #get new fixed points 
        TTT_d_u_12 = mpsu1Transfer_left_threeLayerWithMpsInsert_upperBip(self.psi.mps[1],self.psi.mps[2],self.psi.mpo[1],self.psi.mpo[2],self.psi.T[1],self.psi.T[2],Td[1],Td[2])
        RRR_d_u_1 = TTT_d_u_12.findRightEig()
        RRR_d_u_1.norm_pairedCanon()
        RRR_d_upper[1] = RRR_d_u_1.tensor
        del TTT_d_u_12

        TTT_d_u_21 = mpsu1Transfer_left_threeLayerWithMpsInsert_upperBip(self.psi.mps[2],self.psi.mps[1],self.psi.mpo[2],self.psi.mpo[1],self.psi.T[2],self.psi.T[1],Td[2],Td[1])
        RRR_d_u_2 = TTT_d_u_21.findRightEig()
        RRR_d_u_2.norm_pairedCanon()
        RRR_d_upper[2] = RRR_d_u_2.tensor
        del TTT_d_u_21

        TTT_d_u_12_p1 = mpsu1Transfer_left_threeLayerWithMpsInsert_upperBip_plusOne(self.psi.mps[1],self.psi.mps[2],self.psi.mpo[1],self.psi.mpo[2],self.psi.T[1],self.psi.T[2],Td[1],Td[2])
        RRR_d_u_1_p1 = TTT_d_u_12_p1.findRightEig()
        RRR_d_u_1_p1.norm_pairedCanon()
        RRR_d_upper_p1[1] = RRR_d_u_1_p1.tensor
        del TTT_d_u_12_p1

        TTT_d_u_21_p1 = mpsu1Transfer_left_threeLayerWithMpsInsert_upperBip_plusOne(self.psi.mps[2],self.psi.mps[1],self.psi.mpo[2],self.psi.mpo[1],self.psi.T[2],self.psi.T[1],Td[2],Td[1])
        RRR_d_u_2_p1 = TTT_d_u_21_p1.findRightEig()
        RRR_d_u_2_p1.norm_pairedCanon()
        RRR_d_upper_p1[2] = RRR_d_u_2_p1.tensor
        del TTT_d_u_21_p1

        TTT_d_l_12 = mpsu1Transfer_left_threeLayerWithMpsInsert_lowerBip(self.psi.mps[1],self.psi.mps[2],self.psi.mpo[1],self.psi.mpo[2],self.psi.T[1],self.psi.T[2],Td[1],Td[2])
        RRR_d_l_1 = TTT_d_l_12.findRightEig()
        RRR_d_l_1.norm_pairedCanon()
        RRR_d_lower[1] = RRR_d_l_1.tensor
        del TTT_d_l_12

        TTT_d_l_21 = mpsu1Transfer_left_threeLayerWithMpsInsert_lowerBip(self.psi.mps[2],self.psi.mps[1],self.psi.mpo[2],self.psi.mpo[1],self.psi.T[2],self.psi.T[1],Td[2],Td[1])
        RRR_d_l_2 = TTT_d_l_21.findRightEig()
        RRR_d_l_2.norm_pairedCanon()
        RRR_d_lower[2] = RRR_d_l_2.tensor
        del TTT_d_l_21

        TTT_d_l_12_p1 = mpsu1Transfer_left_threeLayerWithMpsInsert_lowerBip_plusOne(self.psi.mps[1],self.psi.mps[2],self.psi.mpo[1],self.psi.mpo[2],self.psi.T[1],self.psi.T[2],Td[1],Td[2])
        RRR_d_l_1_p1 = TTT_d_l_12_p1.findRightEig()
        RRR_d_l_1_p1.norm_pairedCanon()
        RRR_d_lower_p1[1] = RRR_d_l_1_p1.tensor
        del TTT_d_l_12_p1

        TTT_d_l_21_p1 = mpsu1Transfer_left_threeLayerWithMpsInsert_lowerBip_plusOne(self.psi.mps[2],self.psi.mps[1],self.psi.mpo[2],self.psi.mpo[1],self.psi.T[2],self.psi.T[1],Td[2],Td[1])
        RRR_d_l_2_p1 = TTT_d_l_21_p1.findRightEig()
        RRR_d_l_2_p1.norm_pairedCanon()
        RRR_d_lower_p1[2] = RRR_d_l_2_p1.tensor
        del TTT_d_l_21_p1

        fp = dict()
        fp['RRR_d_lower'] = RRR_d_lower
        fp['RRR_d_upper'] = RRR_d_upper
        fp['RRR_d_lower_p1'] = RRR_d_lower_p1
        fp['RRR_d_upper_p1'] = RRR_d_upper_p1
        return fp

    def getOuterContracts(self,Td):
        outerContractTriple_upper = dict()
        outerContractTriple_lower = dict()
        outerContractTriple_upper_p1 = dict()
        outerContractTriple_lower_p1 = dict()

        outerContractTriple_upper[1] = ncon([self.psi.mps[1],self.psi.mps[2],self.psi.mps[1],self.psi.mps[1].conj(),self.psi.mps[2].conj(),self.psi.mps[1].conj(),Td[1],self.psi.T[2].tensor],((-1,7,8),(-3,8,9),(-5,10,11),(-2,7,15),(-4,15,14),(-6,13,12),(14,9,13,10),(12,11)),forder=(-2,-4,-6,-1,-3,-5),order=(7,8,15,9,14,10,13,11,12))
        outerContractTriple_upper[2] = ncon([self.psi.mps[2],self.psi.mps[1],self.psi.mps[2],self.psi.mps[2].conj(),self.psi.mps[1].conj(),self.psi.mps[2].conj(),Td[2],self.psi.T[1].tensor],((-1,7,8),(-3,8,9),(-5,10,11),(-2,7,15),(-4,15,14),(-6,13,12),(14,9,13,10),(12,11)),forder=(-2,-4,-6,-1,-3,-5),order=(7,8,15,9,14,10,13,11,12))

        outerContractTriple_lower[1] = ncon([self.psi.mps[1],self.psi.mps[2],self.psi.mps[1],self.psi.mps[1].conj(),self.psi.mps[2].conj(),self.psi.mps[1].conj(),Td[2],self.psi.T[2].tensor],((-1,7,8),(-3,9,10),(-5,10,11),(-2,7,15),(-4,14,13),(-6,13,12),(15,8,14,9),(12,11)),forder=(-2,-4,-6,-1,-3,-5),order=(7,8,15,9,14,10,13,11,12))
        outerContractTriple_lower[2] = ncon([self.psi.mps[2],self.psi.mps[1],self.psi.mps[2],self.psi.mps[2].conj(),self.psi.mps[1].conj(),self.psi.mps[2].conj(),Td[1],self.psi.T[1].tensor],((-1,7,8),(-3,9,10),(-5,10,11),(-2,7,15),(-4,14,13),(-6,13,12),(15,8,14,9),(12,11)),forder=(-2,-4,-6,-1,-3,-5),order=(7,8,15,9,14,10,13,11,12))

        outerContractTriple_upper_p1[1] = ncon([self.psi.mps[1],self.psi.mps[2],self.psi.mps[1],self.psi.mps[2],self.psi.mps[1].conj(),self.psi.mps[2].conj(),self.psi.mps[1].conj(),self.psi.mps[2].conj(),Td[1],self.psi.T[1].tensor],((-1,8,9),(-3,9,10),(5,11,12),(-6,12,13),(-2,8,18),(-4,18,17),(5,16,15),(-7,15,14),(17,10,16,11),(14,13)),forder=(-2,-4,-7,-1,-3,-6),order=(8,9,18,10,17,11,16,5,12,15,13,14))
        outerContractTriple_upper_p1[2] = ncon([self.psi.mps[2],self.psi.mps[1],self.psi.mps[2],self.psi.mps[1],self.psi.mps[2].conj(),self.psi.mps[1].conj(),self.psi.mps[2].conj(),self.psi.mps[1].conj(),Td[2],self.psi.T[2].tensor],((-1,8,9),(-3,9,10),(5,11,12),(-6,12,13),(-2,8,18),(-4,18,17),(5,16,15),(-7,15,14),(17,10,16,11),(14,13)),forder=(-2,-4,-7,-1,-3,-6),order=(8,9,18,10,17,11,16,5,12,15,13,14))

        outerContractTriple_lower_p1[1] = ncon([self.psi.mps[1],self.psi.mps[2],self.psi.mps[1],self.psi.mps[2],self.psi.mps[1].conj(),self.psi.mps[2].conj(),self.psi.mps[1].conj(),self.psi.mps[2].conj(),Td[1],self.psi.T[1].tensor],((-1,8,9),(3,9,10),(-4,11,12),(-6,12,13),(-2,8,18),(3,18,17),(-5,16,15),(-7,15,14),(17,10,16,11),(14,13)),forder=(-2,-5,-7,-1,-4,-6),order=(8,9,18,3,10,17,11,16,12,15,13,14))
        outerContractTriple_lower_p1[2] = ncon([self.psi.mps[2],self.psi.mps[1],self.psi.mps[2],self.psi.mps[1],self.psi.mps[2].conj(),self.psi.mps[1].conj(),self.psi.mps[2].conj(),self.psi.mps[1].conj(),Td[2],self.psi.T[2].tensor],((-1,8,9),(3,9,10),(-4,11,12),(-6,12,13),(-2,8,18),(3,18,17),(-5,16,15),(-7,15,14),(17,10,16,11),(14,13)),forder=(-2,-5,-7,-1,-4,-6),order=(8,9,18,3,10,17,11,16,12,15,13,14))

        fp = dict()
        fp['outerContractTriple_upper'] = outerContractTriple_upper
        fp['outerContractTriple_lower'] = outerContractTriple_lower
        fp['outerContractTriple_upper_p1'] = outerContractTriple_upper_p1
        fp['outerContractTriple_lower_p1'] = outerContractTriple_lower_p1
        return fp

    def getCentralTerms(self):
        if self.H_index == self.grad_index:
            return ncon([self.psi.mpo[self.index1],self.psi.mpo[self.index2],self.H,self.psi.mpo[self.index2].conj(),self.psi.RR[self.index2].tensor,self.outerContractDouble[self.index1]],((2,1,-9,10),(4,3,11,12),(-5,7,2,4),(7,8,11,14),(-13,10,14,12),(-6,8,1,3)),forder=(-6,-5,-9,-13),order=(12,14,11,4,7,3,8,10,2,1))
        else:
            return ncon([self.psi.mpo[self.index2],self.psi.mpo[self.index1],self.H,self.psi.mpo[self.index2].conj(),self.psi.RR[self.index1].tensor,self.outerContractDouble[self.index2]],((2,1,9,10),(6,5,-11,12),(3,-7,2,6),(3,4,9,13),(13,10,-14,12),(4,-8,1,5)),forder=(-8,-7,-11,-14),order=(10,13,9,2,3,1,4,12,5))

    def buildLeftEnv(self,H=None,wrapAround=False):
        if H is None:
            H = self.H
        if self.H_index == self.grad_index:
            env = ncon([self.psi.mpo[self.index1],self.psi.mpo[self.index2],H,self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2].conj(),self.outerContractDouble[self.index1]],((2,1,9,-10),(6,5,11,-12),(3,7,2,6),(3,4,9,-13),(7,8,11,-14),(4,8,1,5)),forder=(-13,-10,-14,-12),order=(9,1,2,3,4,11,5,6,7,8))
            env = self.psi.Tb2_inv[self.index2].applyRight(env.reshape(self.psi.D_mpo**4)).reshape(self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo)

            if wrapAround == True:
                env1 = ncon([env,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.psi.RR[self.index1].tensor,self.outerContractDouble[self.index2]],((6,4,-10,-8),(2,1,4,5),(2,3,6,7),(7,5,-11,-9),(3,-12,1,-13)),forder=(-12,-13,-10,-8,-11,-9),order=(4,6,2,5,7,1,3))
                env = ncon([env,self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.outerContractDouble[self.index2]],((13,11,9,7),(2,1,7,-8),(2,3,9,-10),(5,4,11,-12),(5,6,13,-14),(6,3,4,1)),forder=(-14,-12,-10,-8),order=(7,9,2,11,13,5,1,3,4,6))
                env2 = ncon([env,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.psi.RR[self.index2].tensor,self.outerContractDouble[self.index1]],((-10,-8,6,4),(2,1,4,5),(2,3,6,7),(-11,-9,7,5),(-12,3,-13,1)),forder=(-12,-13,-10,-8,-11,-9),order=(4,6,2,5,7,1,3))
                env = env1 + env2

        else:
            env = ncon([self.psi.mpo[self.index2],self.psi.mpo[self.index1],H,self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1].conj(),self.outerContractDouble[self.index2]],((2,1,9,-10),(6,5,11,-12),(3,7,2,6),(3,4,9,-13),(7,8,11,-14),(4,8,1,5)),forder=(-13,-10,-14,-12),order=(9,1,2,3,4,11,5,6,7,8))
            env = self.psi.Tb2_inv[self.index1].applyRight(env.reshape(self.psi.D_mpo**4)).reshape(self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo)

            if wrapAround == True:
                env1 = ncon([env,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.psi.RR[self.index2].tensor,self.outerContractDouble[self.index1]],((-10,-8,6,4),(2,1,4,5),(2,3,6,7),(-11,-9,7,5),(-12,3,-13,1)),forder=(-12,-13,-10,-8,-11,-9),order=(4,6,2,5,7,1,3))
                env = ncon([env,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj(),self.outerContractDouble[self.index1]],((13,11,9,7),(2,1,7,-8),(2,3,9,-10),(5,4,11,-12),(5,6,13,-14),(6,3,4,1)),forder=(-14,-12,-10,-8),order=(7,9,2,11,13,5,1,3,4,6))
                env2 = ncon([env,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.psi.RR[self.index1].tensor,self.outerContractDouble[self.index2]],((6,4,-10,-8),(2,1,4,5),(2,3,6,7),(7,5,-11,-9),(3,-12,1,-13)),forder=(-12,-13,-10,-8,-11,-9),order=(4,6,2,5,7,1,3))
                env = env1 + env2

        return env

    def buildRightEnv(self,H=None):
        if H is None:
            H = self.H
        if self.H_index == self.grad_index:
            env = ncon([self.psi.mpo[self.index1],self.psi.mpo[self.index2],H,self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2].conj(),self.psi.RR[self.index2].tensor,self.outerContractDouble[self.index1]],((2,1,-9,10),(6,5,-11,12),(3,7,2,6),(3,4,-13,14),(7,8,-15,16),(14,10,16,12),(4,8,1,5)),forder=(-13,-9,-15,-11),order=(12,16,5,6,7,8,10,14,1,2,3,4))
            env = self.psi.Tb2_inv[self.index1].applyLeft(env.reshape(self.psi.D_mpo**4)).reshape(self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo)

            env1  = ncon([self.outerContract[self.index1],env],((-1,-2),(3,3,-4,-5)),forder=(-1,-2,-4,-5)) 
            env2 =  ncon([self.outerContract[self.index1],env,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.outerContract[self.index2]],((-8,-9),(7,5,10,10),(2,1,-4,5),(2,3,-6,7),(3,1)),forder=(-8,-9,-6,-4),order=[10,5,7,1,2,3])
            env = env1 + env2
        else:
            env = ncon([self.psi.mpo[self.index2],self.psi.mpo[self.index1],H,self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1].conj(),self.psi.RR[self.index1].tensor,self.outerContractDouble[self.index2]],((2,1,-9,10),(6,5,-11,12),(3,7,2,6),(3,4,-13,14),(7,8,-15,16),(14,10,16,12),(4,8,1,5)),forder=(-13,-9,-15,-11),order=(12,16,5,6,7,8,10,14,1,2,3,4))
            env = self.psi.Tb2_inv[self.index2].applyLeft(env.reshape(self.psi.D_mpo**4)).reshape(self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo)

            env1  = ncon([self.outerContract[self.index1],env],((-1,-2),(-3,-4,5,5)),forder=(-1,-2,-3,-4)) 
            env2 =  ncon([self.outerContract[self.index1],env,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.outerContract[self.index2]],((-8,-9),(10,10,7,5),(2,1,-4,5),(2,3,-6,7),(3,1)),forder=(-8,-9,-6,-4),order=[10,5,7,1,2,3])
            env = env1 + env2
        return env

    def buildTopEnvGeo(self,fixedPoints,outers):
        if self.H_index == self.grad_index:
            return ncon([self.innerContract,fixedPoints['RRR_d_lower_p1'][self.index2],outers['outerContractTriple_lower_p1'][self.index1]],((1,2,3,4,5,6,7,8),(-9,-10,5,6,7,8),(-11,1,2,-12,3,4)),forder=(-11,-12,-9,-10),order=(5,6,7,8,1,2,3,4))
        else:
            return ncon([self.innerContract,fixedPoints['RRR_d_lower'][self.index2],outers['outerContractTriple_lower'][self.index1]],((1,2,3,4,5,6,7,8),(-9,-10,5,6,7,8),(-11,1,2,-12,3,4)),forder=(-11,-12,-9,-10),order=(5,6,7,8,1,2,3,4))

    def buildBotEnvGeo(self,fixedPoints,outers):
        if self.H_index == self.grad_index:
            return ncon([self.innerContract,fixedPoints['RRR_d_upper'][self.index2],outers['outerContractTriple_upper'][self.index1]],((1,2,3,4,5,6,7,8),(5,6,7,8,-9,-10),(1,2,-11,3,4,-12)),forder=(-11,-12,-9,-10),order=(5,6,7,8,1,2,3,4))
        else:
            return ncon([self.innerContract,fixedPoints['RRR_d_upper_p1'][self.index1],outers['outerContractTriple_upper_p1'][self.index2]],((1,2,3,4,5,6,7,8),(5,6,7,8,-9,-10),(1,2,-11,3,4,-12)),forder=(-11,-12,-9,-10),order=(5,6,7,8,1,2,3,4))

    def buildTopEnvGeo_quadrants(self,fixedPoints,outers,leftEnv):
        if self.H_index == self.grad_index:
            env =  ncon([leftEnv,self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),fixedPoints['RRR_d_lower'][self.index2],outers['outerContractTriple_lower'][self.index1]],((13,11,9,7),(2,1,7,8),(2,3,9,10),(5,4,11,12),(5,6,13,14),(-16,-15,14,12,10,8),(-17,6,3,-18,4,1)),forder=(-17,-18,-16,-15),order=(7,9,2,11,13,5,8,10,12,14,1,3,4,6))
            newEnv = ncon([leftEnv,self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.outerContractDouble[self.index2]],((13,11,9,7),(2,1,7,-8),(2,3,9,-10),(5,4,11,-12),(5,6,13,-14),(6,3,4,1)),forder=(-14,-12,-10,-8),order=(7,9,11,13,1,2,3,4,5,6))
            env +=  ncon([newEnv,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj(),fixedPoints['RRR_d_lower_p1'][self.index2],outers['outerContractTriple_lower_p1'][self.index1]],((13,11,9,7),(2,1,7,8),(2,3,9,10),(5,4,11,12),(5,6,13,14),(-16,-15,14,12,10,8),(-17,6,3,-18,4,1)),forder=(-17,-18,-16,-15),order=(7,9,2,11,13,5,8,10,12,14,1,3,4,6))
        else:
            env =  ncon([leftEnv,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj(),fixedPoints['RRR_d_lower_p1'][self.index2],outers['outerContractTriple_lower_p1'][self.index1]],((13,11,9,7),(2,1,7,8),(2,3,9,10),(5,4,11,12),(5,6,13,14),(-16,-15,14,12,10,8),(-17,6,3,-18,4,1)),forder=(-17,-18,-16,-15),order=(7,9,2,11,13,5,8,10,12,14,1,3,4,6))
            newEnv = ncon([leftEnv,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj(),self.outerContractDouble[self.index1]],((13,11,9,7),(2,1,7,-8),(2,3,9,-10),(5,4,11,-12),(5,6,13,-14),(6,3,4,1)),forder=(-14,-12,-10,-8),order=(7,9,11,13,1,2,3,4,5,6))
            env +=  ncon([newEnv,self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),fixedPoints['RRR_d_lower'][self.index2],outers['outerContractTriple_lower'][self.index1]],((13,11,9,7),(2,1,7,8),(2,3,9,10),(5,4,11,12),(5,6,13,14),(-16,-15,14,12,10,8),(-17,6,3,-18,4,1)),forder=(-17,-18,-16,-15),order=(7,9,2,11,13,5,8,10,12,14,1,3,4,6))
        return env

    def buildBotEnvGeo_quadrants(self,fixedPoints,outers,leftEnv,):
        if self.H_index == self.grad_index:
            env = ncon([leftEnv,self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),fixedPoints['RRR_d_upper_p1'][self.index1],outers['outerContractTriple_upper_p1'][self.index2]],((13,11,9,7),(2,1,7,8),(2,3,9,10),(5,4,11,12),(5,6,13,14),(14,12,10,8,-15,-16),(6,3,-17,4,1,-18)),forder=(-17,-18,-15,-16),order=(7,9,2,11,13,5,8,10,12,14,1,3,4,6))
            newEnv = ncon([leftEnv,self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.outerContractDouble[self.index2]],((13,11,9,7),(2,1,7,-8),(2,3,9,-10),(5,4,11,-12),(5,6,13,-14),(6,3,4,1)),forder=(-14,-12,-10,-8),order=(7,9,11,13,1,2,3,4,5,6))
            env += ncon([newEnv,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj(),fixedPoints['RRR_d_upper'][self.index2],outers['outerContractTriple_upper'][self.index1]],((13,11,9,7),(2,1,7,8),(2,3,9,10),(5,4,11,12),(5,6,13,14),(14,12,10,8,-15,-16),(6,3,-17,4,1,-18)),forder=(-17,-18,-15,-16),order=(7,9,2,11,13,5,8,10,12,14,1,3,4,6))
        else:
            env = ncon([leftEnv,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj(),fixedPoints['RRR_d_upper'][self.index2],outers['outerContractTriple_upper'][self.index1]],((13,11,9,7),(2,1,7,8),(2,3,9,10),(5,4,11,12),(5,6,13,14),(14,12,10,8,-15,-16),(6,3,-17,4,1,-18)),forder=(-17,-18,-15,-16),order=(7,9,2,11,13,5,8,10,12,14,1,3,4,6))
            newEnv = ncon([leftEnv,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj(),self.outerContractDouble[self.index1]],((13,11,9,7),(2,1,7,-8),(2,3,9,-10),(5,4,11,-12),(5,6,13,-14),(6,3,4,1)),forder=(-14,-12,-10,-8),order=(7,9,11,13,1,2,3,4,5,6))
            env += ncon([newEnv,self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),fixedPoints['RRR_d_upper_p1'][self.index1],outers['outerContractTriple_upper_p1'][self.index2]],((13,11,9,7),(2,1,7,8),(2,3,9,10),(5,4,11,12),(5,6,13,14),(14,12,10,8,-15,-16),(6,3,-17,4,1,-18)),forder=(-17,-18,-15,-16),order=(7,9,2,11,13,5,8,10,12,14,1,3,4,6))
        return env


    def buildRightEnvGeo_quadrants(self,fixedPoints,outers):
        if self.H_index == self.grad_index:
            #bot env
            env1 = ncon([self.innerContract,fixedPoints['RRR_d_upper_p1'][self.index2],outers['outerContractTriple_upper_p1'][self.index1]],((1,2,3,4,5,6,7,8),(5,6,7,8,-9,-10),(1,2,-11,3,4,-12)),forder=(-11,-12,-9,-10),order=(5,6,7,8,1,2,3,4))
            env1 = ncon([env1,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj()],((3,1,7,5),(2,1,-4,5),(2,3,-6,7)),forder=(-6,-4),order=(5,7,1,2,3))
            env2 = ncon([self.innerContract,fixedPoints['RRR_d_upper'][self.index2],outers['outerContractTriple_upper'][self.index1]],((1,2,3,4,5,6,7,8),(5,6,7,8,-9,-10),(1,2,-11,3,4,-12)),forder=(-11,-12,-9,-10),order=(5,6,7,8,1,2,3,4))
            env2 = ncon([env2,self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj()],((3,1,7,5),(2,1,-4,5),(2,3,-6,7)),forder=(-6,-4),order=(5,7,1,2,3))
            #append extra site to left
            env2 = ncon([env2,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.outerContract[self.index2]],((7,5),(2,1,-4,5),(2,3,-6,7),(3,1)),forder=(-6,-4),order=(5,7,1,2,3))
            botEnv = env1 + env2

            #top env
            env1 = ncon([self.innerContract,fixedPoints['RRR_d_lower'][self.index1],outers['outerContractTriple_lower'][self.index2]],((1,2,3,4,5,6,7,8),(-9,-10,5,6,7,8),(-11,1,2,-12,3,4)),forder=(-11,-12,-9,-10),order=(5,6,7,8,1,2,3,4))
            env1 = ncon([env1,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj()],((3,1,7,5),(2,1,-4,5),(2,3,-6,7)),forder=(-6,-4),order=(5,7,1,2,3))
            env2 = ncon([self.innerContract,fixedPoints['RRR_d_lower_p1'][self.index2],outers['outerContractTriple_lower_p1'][self.index1]],((1,2,3,4,5,6,7,8),(-9,-10,5,6,7,8),(-11,1,2,-12,3,4)),forder=(-11,-12,-9,-10),order=(5,6,7,8,1,2,3,4))
            env2 = ncon([env2,self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj()],((3,1,7,5),(2,1,-4,5),(2,3,-6,7)),forder=(-6,-4),order=(5,7,1,2,3))
            #append extra site to left
            env2 = ncon([env2,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.outerContract[self.index2]],((7,5),(2,1,-4,5),(2,3,-6,7),(3,1)),forder=(-6,-4),order=(5,7,1,2,3))
            topEnv = env1 + env2

        else:
            #bot env
            env1 = ncon([self.innerContract,fixedPoints['RRR_d_upper'][self.index1],outers['outerContractTriple_upper'][self.index2]],((1,2,3,4,5,6,7,8),(5,6,7,8,-9,-10),(1,2,-11,3,4,-12)),forder=(-11,-12,-9,-10),order=(5,6,7,8,1,2,3,4))
            env1 = ncon([env1,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj()],((3,1,7,5),(2,1,-4,5),(2,3,-6,7)),forder=(-6,-4),order=(5,7,1,2,3))
            env2 = ncon([self.innerContract,fixedPoints['RRR_d_upper_p1'][self.index1],outers['outerContractTriple_upper_p1'][self.index2]],((1,2,3,4,5,6,7,8),(5,6,7,8,-9,-10),(1,2,-11,3,4,-12)),forder=(-11,-12,-9,-10),order=(5,6,7,8,1,2,3,4))
            env2 = ncon([env2,self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj()],((3,1,7,5),(2,1,-4,5),(2,3,-6,7)),forder=(-6,-4),order=(5,7,1,2,3))
            #append extra site to left
            env2 = ncon([env2,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.outerContract[self.index2]],((7,5),(2,1,-4,5),(2,3,-6,7),(3,1)),forder=(-6,-4),order=(5,7,1,2,3))
            botEnv = env1 + env2

            #top env
            env1 = ncon([self.innerContract,fixedPoints['RRR_d_lower_p1'][self.index1],outers['outerContractTriple_lower_p1'][self.index2]],((1,2,3,4,5,6,7,8),(-9,-10,5,6,7,8),(-11,1,2,-12,3,4)),forder=(-11,-12,-9,-10),order=(5,6,7,8,1,2,3,4))
            env1 = ncon([env1,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj()],((3,1,7,5),(2,1,-4,5),(2,3,-6,7)),forder=(-6,-4),order=(5,7,1,2,3))
            env2 = ncon([self.innerContract,fixedPoints['RRR_d_lower'][self.index2],outers['outerContractTriple_lower'][self.index1]],((1,2,3,4,5,6,7,8),(-9,-10,5,6,7,8),(-11,1,2,-12,3,4)),forder=(-11,-12,-9,-10),order=(5,6,7,8,1,2,3,4))
            env2 = ncon([env2,self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj()],((3,1,7,5),(2,1,-4,5),(2,3,-6,7)),forder=(-6,-4),order=(5,7,1,2,3))
            #append extra site to left
            env2 = ncon([env2,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.outerContract[self.index2]],((7,5),(2,1,-4,5),(2,3,-6,7),(3,1)),forder=(-6,-4),order=(5,7,1,2,3))
            topEnv = env1 + env2

        env = botEnv + topEnv
        env = self.psi.Tb_inv[self.index2].applyLeft(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        env = ncon([self.outerContract[self.index1],env],((-1,-2),(-3,-4)),forder=(-1,-2,-3,-4))
        return env
