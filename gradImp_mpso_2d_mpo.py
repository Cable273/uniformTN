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
        def wrapLeftEnv(self):
            pass
        @abstractmethod
        def wrapRightEnv(self):
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

# -------------------------------------------------------------------------------------------------------------------------------------
#Concrete implementations
# -------------------------------------------------------------------------------------------------------------------------------------
#2d mpo implementations
class gradImplementation_mpso_2d_mpo_uniform_H_verticalLength_1(gradImplementation_mpso_2d_mpo_uniform):
    def __init__(self,psi,H):
        super().__init__(psi,H)
        self.outerContract =  ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))

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

    def buildTopEnvGeo_H_horiLength_1(self,fixedPoints,outers,innerContract):
        return ncon([innerContract,fixedPoints['RR_d'],outers['outerContractDouble']],((1,2,3,4),(-5,-6,3,4),(-7,1,-8,2)),forder=(-7,-8,-5,-6),order=(3,4,1,2))
    def buildTopEnvGeo_H_horiLength_2(self,fixedPoints,outers,innerContract):
        return ncon([innerContract,fixedPoints['RR_d'],self.psi.mpo,self.psi.mpo.conj(),outers['outerContractDouble'],outers['outerContractDouble']],((1,2,3,4,5,6),(10,8,5,6),(12,11,-7,8),(12,13,-9,10),(13,2,11,4),(-14,1,-15,3)),forder=(-14,-15,-9,-7),order=(5,6,8,10,4,2,11,12,13,1,3))
    def buildBotEnvGeo_H_horiLength_1(self,fixedPoints,outers,innerContract):
        return ncon([innerContract,fixedPoints['RR_d'],outers['outerContractDouble']],((1,2,3,4),(3,4,-5,-6),(1,-7,2,-8)),forder=(-7,-8,-5,-6),order=(3,4,1,2))
    def buildBotEnvGeo_H_horiLength_2(self,fixedPoints,outers,innerContract):
        return ncon([innerContract,fixedPoints['RR_d'],self.psi.mpo,self.psi.mpo.conj(),outers['outerContractDouble'],outers['outerContractDouble']],((1,2,3,4,5,6),(10,8,5,6),(12,11,-7,8),(12,13,-9,10),(13,2,11,4),(-14,1,-15,3)),forder=(-14,-15,-9,-7),order=(5,6,8,10,4,2,11,12,13,1,3))

    def buildTopEnvGeo_quadrants(self,fixedPoints,outers):
        return self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight)
    def buildBotEnvGeo_quadrants(self,fixedPoints,outers):
        return self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight)

    def wrapRightEnv(self,env):
        return ncon([self.outerContract,env],((-1,-2),(-3,-4)),forder=(-1,-2,-3,-4))
    def wrapLeftEnv(self,env):
        return ncon([env,self.psi.R.tensor,self.outerContract],((-1,-2),(-3,-4),(-5,-6)),forder=(-5,-6,-1,-2,-3,-4))

class gradImplementation_mpso_2d_mpo_uniform_oneBodyH(gradImplementation_mpso_2d_mpo_uniform_H_verticalLength_1):
    def __init__(self,psi,H):
        super().__init__(psi,H)
        innerContract = ncon([self.psi.mpo,self.psi.mpo.conj(),self.H],((2,-1,5,-6),(3,-4,5,-7),(3,2)),forder=(-4,-1,-7,-6))
        exp = ncon([innerContract,self.outerContract,self.psi.R.tensor],((1,2,3,4),(1,2),(3,4)),order=(3,4,1,2))
        self.h_tilde = self.H - exp*np.eye(2)

        self.innerContract = ncon([self.psi.mpo,self.psi.mpo.conj(),self.h_tilde],((2,-1,5,-6),(3,-4,5,-7),(3,2)),forder=(-4,-1,-7,-6))
        leftEnv = self.buildLeftEnv(H = self.h_tilde)
        self.innerContract_shiftedRight = ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj()],((5,3),(2,-1,3,-4),(2,-7,5,-6)),forder=(-7,-1,-6,-4),order=(3,5,2))

    def getCentralTerms(self):
        return ncon([self.psi.mpo,self.H,self.outerContract,self.psi.R.tensor],((2,1,-5,6),(-3,2),(-4,1),(-7,6)),forder=(-3,-4,-5,-7),order=(6,1,2))
    def buildLeftEnv(self,H=None):
        if H is None:
            H = self.H
        env = ncon([self.psi.mpo,H,self.psi.mpo.conj(),self.outerContract],((2,1,5,-6),(3,2),(3,4,5,-7),(4,1)),forder=(-7,-6),order=(5,1,2,3,4))
        env = self.psi.Tb_inv.applyRight(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        return env

    def buildRightEnv(self,H=None):
        if H is None:
            H = self.H
        env = ncon([self.psi.mpo,H,self.psi.mpo.conj(),self.outerContract,self.psi.R.tensor],((2,1,-5,6),(3,2),(3,4,-7,8),(4,1),(8,6)),forder=(-7,-5),order=(8,6,1,2,3,4))
        env = self.psi.Tb_inv.applyLeft(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        return env

    def buildTopEnvGeo(self,fixedPoints,outers):
        return self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract)
    def buildBotEnvGeo(self,fixedPoints,outers):
        return self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract)

    def buildRightEnvGeo_quadrants(self,fixedPoints,outers):
        env = self.buildTopEnvGeo(fixedPoints,outers)
        env += self.buildBotEnvGeo(fixedPoints,outers)
        env = ncon([env,self.psi.mpo,self.psi.mpo.conj()],((3,1,7,5),(2,1,-4,5),(2,3,-6,7)),forder=(-6,-4),order=(5,7,1,2,3))
        env = self.psi.Tb_inv.applyLeft(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        env = ncon([self.outerContract,env],((-1,-2),(-3,-4)),forder=(-1,-2,-3,-4))
        return env

class gradImplementation_mpso_2d_mpo_uniform_twoBodyH_hori(gradImplementation_mpso_2d_mpo_uniform_H_verticalLength_1):
    def __init__(self,psi,H):
        super().__init__(psi,H)
        innerContract = ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.mpo.conj()],((2,-1,9,10),(6,-5,10,-11),(3,7,2,6),(3,-4,9,13),(7,-8,13,-12)),forder=(-4,-8,-1,-5,-12,-11),order=(9,2,3,10,13,6,7))
        exp = ncon([innerContract,self.outerContract,self.outerContract,self.psi.R.tensor],((1,2,3,4,5,6),(1,3),(2,4),(5,6)),order=(1,3,2,4,5,6))
        self.h_tilde = (self.H.reshape(4,4)-exp*np.eye(4)).reshape(2,2,2,2)

        self.innerContract= ncon([self.psi.mpo,self.psi.mpo,self.h_tilde,self.psi.mpo.conj(),self.psi.mpo.conj()],((2,-1,9,10),(6,-5,10,-11),(3,7,2,6),(3,-4,9,13),(7,-8,13,-12)),forder=(-4,-8,-1,-5,-12,-11),order=(9,2,3,10,13,6,7))
        leftEnv = self.buildLeftEnv(H = self.h_tilde)
        self.innerContract_shiftedRight = ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj()],((5,3),(2,-1,3,-4),(2,-7,5,-6)),forder=(-7,-1,-6,-4),order=(3,5,2))

    def getCentralTerms(self):
        grad = ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.R.tensor,self.outerContract,self.outerContract],((2,1,-9,10),(6,5,10,11),(-3,7,2,6),(7,8,-13,12),(12,11),(-4,1),(8,5)),forder=(-3,-4,-9,-13),order=(12,11,5,8,6,7,10,1))
        grad += ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.R.tensor,self.outerContract,self.outerContract],((2,1,9,10),(6,5,10,11),(3,-7,2,6),(3,4,9,-13),(-12,11),(4,1),(-8,5)),forder=(-7,-8,-13,-12),order=(9,1,2,3,4,10,5,6,11))
        return grad

    def buildLeftEnv(self,H=None,wrapAround=False):
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

    def buildTopEnvGeo(self,fixedPoints,outers):
        innerContract1 = ncon([self.innerContract,self.outerContract],((1,-2,3,-4,-5,-6),(1,3)),forder=(-2,-4,-5,-6))
        env = self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,innerContract1)
        env += self.buildTopEnvGeo_H_horiLength_2(fixedPoints,outers,self.innerContract)
        return env
    def buildBotEnvGeo(self,fixedPoints,outers):
        innerContract1 = ncon([self.innerContract,self.outerContract],((1,-2,3,-4,-5,-6),(1,3)),forder=(-2,-4,-5,-6))
        env = self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,innerContract1)
        env += self.buildBotEnvGeo_H_horiLength_2(fixedPoints,outers,self.innerContract)
        return env

    def buildRightEnvGeo_quadrants(self,fixedPoints,outers):
        env = self.buildTopEnvGeo_H_horiLength_2(fixedPoints,outers,self.innerContract)
        env += self.buildBotEnvGeo_H_horiLength_2(fixedPoints,outers,self.innerContract)
        env = ncon([env,self.psi.mpo,self.psi.mpo.conj()],((3,1,7,5),(2,1,-4,5),(2,3,-6,7)),forder=(-6,-4),order=(5,7,1,2,3))
        env = self.psi.Tb_inv.applyLeft(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        env = ncon([self.outerContract,env],((-1,-2),(-3,-4)),forder=(-1,-2,-3,-4))
        return env

# -----------------------------
class gradImplementation_mpso_2d_mpo_uniform_H_verticalLength_2(gradImplementation_mpso_2d_mpo_uniform):
    def __init__(self,psi,H):
        super().__init__(psi,H)
        self.outerContract =  ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
        self.outerContractDouble = ncon([self.psi.mps,self.psi.mps,self.psi.mps.conj(),self.psi.mps.conj(),self.psi.T.tensor],((-1,5,6),(-2,6,7),(-3,5,9),(-4,9,8),(8,7)),forder=(-3,-4,-1,-2))

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

    def buildTopEnvGeo_H_horiLength_1(self,fixedPoints,outers,innerContract):
        return ncon([innerContract,fixedPoints['RRR_d_lower'],outers['outerContractTriple_lower']],((1,2,3,4,5,6,7,8),(-9,-10,5,6,7,8),(-11,1,2,-12,3,4)),forder=(-11,-12,-9,-10),order=(5,6,7,8,1,2,3,4))
    def buildBotEnvGeo_H_horiLength_1(self,fixedPoints,outers,innerContract):
        return ncon([innerContract,fixedPoints['RRR_d_upper'],outers['outerContractTriple_upper']],((1,2,3,4,5,6,7,8),(5,6,7,8,-9,-10),(1,2,-11,3,4,-12)),forder=(-11,-12,-9,-10),order=(5,6,7,8,1,2,3,4))

    def buildTopEnvGeo_quadrants(self,fixedPoints,outers):
        return self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight)
    def buildBotEnvGeo_quadrants(self,fixedPoints,outers):
        return self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight)

    def wrapRightEnv(self,env):
        return ncon([self.outerContract,env],((-1,-2),(-3,-4,5,5)),forder=(-1,-2,-3,-4)) + ncon([self.outerContract,env],((-1,-2),(3,3,-4,-5)),forder=(-1,-2,-4,-5))
    def wrapLeftEnv(self,env):
        env1 = ncon([env,self.psi.mpo,self.psi.mpo.conj(),self.psi.RR.tensor,self.outerContractDouble],((-10,-8,6,4),(2,1,4,5),(2,3,6,7),(-11,-9,7,5),(-12,3,-13,1)),forder=(-12,-13,-10,-8,-11,-9),order=(4,6,2,5,7,1,3))
        env2 = ncon([env,self.psi.mpo,self.psi.mpo.conj(),self.psi.RR.tensor,self.outerContractDouble],((6,4,-10,-8),(2,1,4,5),(2,3,6,7),(7,5,-11,-9),(3,-12,1,-13)),forder=(-12,-13,-10,-8,-11,-9),order=(4,6,2,5,7,1,3))
        env = env1 + env2
        return env

class gradImplementation_mpso_2d_mpo_uniform_twoBodyH_vert(gradImplementation_mpso_2d_mpo_uniform_H_verticalLength_2):
    def __init__(self,psi,H):
        super().__init__(psi,H)

        leftEnv = ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.mpo.conj(),self.outerContractDouble],((2,1,9,-10),(6,5,11,-12),(3,7,2,6),(3,4,9,-13),(7,8,11,-14),(4,8,1,5)),forder=(-13,-10,-14,-12),order=(9,1,2,3,4,11,5,6,7,8))
        exp = np.einsum('abcd,abcd',leftEnv,self.psi.RR.tensor)
        self.h_tilde = (self.H.reshape(4,4)-exp*np.eye(4)).reshape(2,2,2,2)

        self.innerContract= ncon([self.psi.mpo,self.psi.mpo,self.h_tilde,self.psi.mpo.conj(),self.psi.mpo.conj()],((2,-1,9,-10),(6,-5,11,-12),(3,7,2,6),(3,-4,9,-13),(7,-8,11,-14)),forder=(-4,-8,-1,-5,-13,-10,-14,-12),order=(9,2,3,11,6,7))
        leftEnv = self.buildLeftEnv(H = self.h_tilde)
        self.innerContract_shiftedRight  = ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo.conj()],((13,11,9,7),(2,-1,7,-8),(2,-3,9,-10),(5,-4,11,-12),(5,-6,13,-14)),forder=(-6,-3,-4,-1,-14,-12,-10,-8),order=(7,9,2,11,13,5))

    def getCentralTerms(self):
        grad = ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.RR.tensor,self.outerContractDouble],((2,1,-9,10),(4,3,11,12),(-5,7,2,4),(7,8,11,14),(-13,10,14,12),(-6,8,1,3)),forder=(-5,-6,-9,-13),order=(12,14,11,4,7,3,8,10,2,1))
        grad += ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.RR.tensor,self.outerContractDouble],((2,1,9,10),(6,5,-11,12),(3,-7,2,6),(3,4,9,13),(13,10,-14,12),(4,-8,1,5)),forder=(-7,-8,-11,-14),order=(10,13,9,2,3,1,4,12,5))
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

    def buildTopEnvGeo(self,fixedPoints,outers):
        return self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract)
    def buildBotEnvGeo(self,fixedPoints,outers):
        return self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract)
    def buildRightEnvGeo_quadrants(self,fixedPoints,outers):
        env = self.buildTopEnvGeo(fixedPoints,outers)
        env += self.buildBotEnvGeo(fixedPoints,outers)
        env = ncon([env,self.psi.mpo,self.psi.mpo.conj()],((3,1,7,5),(2,1,-4,5),(2,3,-6,7)),forder=(-6,-4),order=(5,7,1,2,3))
        env = self.psi.Tb_inv.applyLeft(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        env = ncon([self.outerContract,env],((-1,-2),(-3,-4)),forder=(-1,-2,-3,-4))
        return env
# -----------------------------
class gradImplementation_mpso_2d_mpo_bipartite_H_verticalLength_1(gradImplementation_mpso_2d_mpo_bipartite):
    def __init__(self,psi,H,H_index,grad_index):
        super().__init__(psi,H,H_index,grad_index)
        self.outerContract = dict()
        self.outerContract[1] =  ncon([self.psi.mps[1],self.psi.mps[1].conj(),self.psi.T[2].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
        self.outerContract[2] =  ncon([self.psi.mps[2],self.psi.mps[2].conj(),self.psi.T[1].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))

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

    def buildTopEnvGeo_H_horiLength_1(self,fixedPoints,outers,innerContract,p1,firstIndex):
        if p1 == False:
            RR_d_label,outerContractDouble_label = 'RR_d', 'outerContractDouble'
            if firstIndex == 1:
                index1,index2 = 1,2
            elif firstIndex == 2:
                index1,index2 = 2,1
        elif p1 == True:
            RR_d_label,outerContractDouble_label = 'RR_d_p1', 'outerContractDouble_p1'
            if firstIndex == 1:
                index1,index2 = 2,1
            elif firstIndex == 2:
                index1,index2 = 1,2
        return ncon([innerContract,fixedPoints[RR_d_label][index1],outers[outerContractDouble_label][index2]],((1,2,3,4),(-5,-6,3,4),(-7,1,-8,2)),forder=(-7,-8,-5,-6),order=(3,4,1,2))

    def buildTopEnvGeo_H_horiLength_2(self,fixedPoints,outers,innerContract,p1,firstIndex):
        if p1 == False:
            RR_d_label, outerContractDouble_label = 'RR_d', 'outerContractDouble'
            if firstIndex == 1:
                index1,index2 = 1,2
            elif firstIndex == 2:
                index1,index2 = 2,1
        elif p1 == True:
            RR_d_label, outerContractDouble_label = 'RR_d_p1', 'outerContractDouble_p1'
            if firstIndex == 1:
                index1,index2= 2,1
            elif firstIndex == 2:
                index1,index2 = 1,2
        return ncon([innerContract,fixedPoints[RR_d_label][index2],self.psi.mpo[index1],self.psi.mpo[index1].conj(),outers[outerContractDouble_label][index1],outers[outerContractDouble_label][index2]],((1,2,3,4,5,6),(10,8,5,6),(12,11,-7,8),(12,13,-9,10),(13,2,11,4),(-14,1,-15,3)),forder=(-14,-15,-9,-7),order=(5,6,8,10,4,2,11,12,13,1,3))

    def buildBotEnvGeo_H_horiLength_1(self,fixedPoints,outers,innerContract,p1,firstIndex):
        if p1 == False:
            RR_d_label,outerContractDouble_label = 'RR_d', 'outerContractDouble'
        elif p1 == True:
            RR_d_label,outerContractDouble_label = 'RR_d_p1', 'outerContractDouble_p1'

        if firstIndex == 1:
            index1,index2 = 1,2
        elif firstIndex == 2:
            index1,index2 = 2,1

        return ncon([innerContract,fixedPoints[RR_d_label][index2],outers[outerContractDouble_label][index1]],((1,2,3,4),(3,4,-5,-6),(1,-7,2,-8)),forder=(-7,-8,-5,-6),order=(3,4,1,2))

    def buildBotEnvGeo_H_horiLength_2(self,fixedPoints,outers,innerContract,p1,firstIndex):
        if p1 == False:
            RR_d_label,outerContractDouble_label = 'RR_d', 'outerContractDouble'
            if firstIndex == 1:
                index1,index2,mpoIndex = 1,2,1
            elif firstIndex == 2:
                index1,index2,mpoIndex = 2,1,2
        elif p1 == True:
            RR_d_label,outerContractDouble_label = 'RR_d_p1', 'outerContractDouble_p1'
            if firstIndex == 1:
                index1,index2,mpoIndex = 1,2,2
            elif firstIndex == 2:
                index1,index2,mpoIndex = 2,1,1

        return ncon([self.innerContract,fixedPoints[RR_d_label][index1],self.psi.mpo[mpoIndex],self.psi.mpo[mpoIndex].conj(),outers[outerContractDouble_label][index2],outers[outerContractDouble_label][index1]],((1,2,3,4,5,6),(5,6,7,8),(10,9,-12,8),(10,11,-13,7),(2,11,4,9),(1,-14,3,-15)),forder=(-14,-15,-13,-12),order=(5,6,7,8,9,10,11,4,2,1,3))

    def wrapRightEnv(self,env):
        return ncon([self.outerContract[self.index1],env],((-1,-2),(-3,-4)),forder=(-1,-2,-3,-4))
    def wrapLeftEnv(self,env):
        return ncon([env,self.psi.R[self.index2].tensor,self.outerContract[self.index1]],((-1,-2),(-3,-4),(-5,-6)),forder=(-5,-6,-1,-2,-3,-4))

class gradImplementation_mpso_2d_mpo_bipartite_oneBodyH(gradImplementation_mpso_2d_mpo_bipartite_H_verticalLength_1):
    def __init__(self,psi,H,H_index,grad_index):
        super().__init__(psi,H,H_index,grad_index)
        if self.H_index == self.grad_index:
            innerContract = ncon([self.psi.mpo[self.index1],self.H,self.psi.mpo[self.index1].conj()],((2,-1,5,-6),(3,2),(3,-4,5,-7)),forder=(-4,-1,-7,-6),order=(5,2,3))
            exp = np.real(ncon([innerContract,self.outerContract[self.index1],self.psi.R[self.index2].tensor],((1,2,3,4),(1,2),(3,4)),order=(1,2,3,4)))
        else:
            innerContract = ncon([self.psi.mpo[self.index2],self.H,self.psi.mpo[self.index2].conj()],((2,-1,5,-6),(3,2),(3,-4,5,-7)),forder=(-4,-1,-7,-6),order=(5,2,3))
            exp = np.real(ncon([innerContract,self.outerContract[self.index2],self.psi.R[self.index1].tensor],((1,2,3,4),(1,2),(3,4)),order=(1,2,3,4)))
        self.h_tilde = self.H-exp*np.eye(2)

        leftEnv = self.buildLeftEnv(H = self.h_tilde)
        if self.H_index == self.grad_index:
            self.innerContract = ncon([self.psi.mpo[self.index1],self.h_tilde,self.psi.mpo[self.index1].conj()],((2,-1,5,-6),(3,2),(3,-4,5,-7)),forder=(-4,-1,-7,-6),order=(5,2,3))
            self.innerContract_shiftedRight = ncon([leftEnv,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj()],((6,4),(2,-1,4,-5),(2,-3,6,-7)),forder=(-3,-1,-7,-5),order=(4,6,2))
            self.innerContract_shiftedRight_p1 = ncon([self.innerContract_shiftedRight,self.outerContract[self.index2],self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj()],((1,2,3,4),(1,2),(6,-5,4,-8),(6,-7,3,-9)),forder=(-7,-5,-9,-8),order=(1,2,4,3,6))
        else:
            self.innerContract = ncon([self.psi.mpo[self.index2],self.h_tilde,self.psi.mpo[self.index2].conj()],((2,-1,5,-6),(3,2),(3,-4,5,-7)),forder=(-4,-1,-7,-6),order=(5,2,3))
            self.innerContract_shiftedRight = ncon([leftEnv,self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj()],((6,4),(2,-1,4,-5),(2,-3,6,-7)),forder=(-3,-1,-7,-5),order=(4,6,2))
            self.innerContract_shiftedRight_p1 = ncon([self.innerContract_shiftedRight,self.outerContract[self.index1],self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj()],((1,2,3,4),(1,2),(6,-5,4,-8),(6,-7,3,-9)),forder=(-7,-5,-9,-8),order=(1,2,4,3,6))

    def getCentralTerms(self):
        if self.H_index == self.grad_index:
            return ncon([self.psi.mpo[self.index1],self.H,self.outerContract[self.index1],self.psi.R[self.index2].tensor],((2,1,-5,6),(-3,2),(-4,1),(-7,6)),forder=(-3,-4,-5,-7),order=(6,1,2))
        else:
            return np.zeros(self.psi.mpo[1].shape).astype(complex)

    def buildLeftEnv(self,H=None):
        if H is None:
            H = self.H
        if self.H_index == self.grad_index:
            return ncon([self.psi.mpo[self.index1],H,self.psi.mpo[self.index1].conj(),self.outerContract[self.index1]],((2,1,5,-6),(3,2),(3,4,5,-7),(4,1)),forder=(-7,-6),order=(5,2,3,1,4))
        else:
            return ncon([self.psi.mpo[self.index2],H,self.psi.mpo[self.index2].conj(),self.outerContract[self.index2]],((2,1,5,-6),(3,2),(3,4,5,-7),(4,1)),forder=(-7,-6),order=(5,2,3,1,4))

    def buildRightEnv(self,H=None):
        if H is None:
            H = self.H
        if self.H_index == self.grad_index:
            return ncon([self.psi.mpo[self.index1],H,self.psi.mpo[self.index1].conj(),self.outerContract[self.index1],self.psi.R[self.index2].tensor],((2,1,-5,6),(3,2),(3,4,-7,8),(4,1),(8,6)),forder=(-7,-5),order=(8,6,2,3,1,4))
        else:
            return ncon([self.psi.mpo[self.index2],H,self.psi.mpo[self.index2].conj(),self.outerContract[self.index2],self.psi.R[self.index1].tensor],((2,1,-5,6),(3,2),(3,4,-7,8),(4,1),(8,6)),forder=(-7,-5),order=(8,6,2,3,1,4))

    def buildTopEnvGeo(self,fixedPoints,outers):
        if self.H_index == self.grad_index:
            return self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract,p1=True,firstIndex=self.index1)
        else:
            return self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract,p1=False,firstIndex=self.index2)
    def buildBotEnvGeo(self,fixedPoints,outers):
        if self.H_index == self.grad_index:
            return self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract,p1=True,firstIndex=self.index1)
        else:
            return self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract,p1=False,firstIndex=self.index2)

    def buildTopEnvGeo_quadrants(self,fixedPoints,outers):
        if self.H_index == self.grad_index:
            env = self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight,p1=False,firstIndex=self.index2)
            env += self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight_p1,p1=True,firstIndex=self.index1)
        else:
            env = self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight,p1=True,firstIndex=self.index1)
            env += self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight_p1,p1=False,firstIndex=self.index2)
        return env

    def buildBotEnvGeo_quadrants(self,fixedPoints,outers):
        if self.H_index == self.grad_index:
            env = self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight,p1=False,firstIndex=self.index2)
            env += self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight_p1,p1=True,firstIndex=self.index1)
        else:
            env = self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight,p1=True,firstIndex=self.index1)
            env += self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight_p1,p1=False,firstIndex=self.index2)
        return env

    def buildRightEnvGeo_quadrants(self,fixedPoints,outers):
        if self.H_index == self.grad_index:
            env1 = self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract,p1=False,firstIndex=self.index1)
            env1 += self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract,p1=False,firstIndex=self.index1)
            env2 = self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract,p1=True,firstIndex=self.index1)
            env2 += self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract,p1=True,firstIndex=self.index1)
        else:
            env1 = self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract,p1=True,firstIndex=self.index2)
            env1 += self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract,p1=True,firstIndex=self.index2)
            env2 = self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract,p1=False,firstIndex=self.index2)
            env2 += self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract,p1=False,firstIndex=self.index2)
        env1 = ncon([env1,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj()],((3,1,7,5),(2,1,-4,5),(2,3,-6,7)),forder=(-6,-4),order=(5,7,1,2,3))
        env2 = ncon([env2,self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj()],((3,1,7,5),(2,1,-4,5),(2,3,-6,7)),forder=(-6,-4),order=(5,7,1,2,3))
        env2 = ncon([env2,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.outerContract[self.index2]],((7,5),(2,1,-4,5),(2,3,-6,7),(3,1)),forder=(-6,-4),order=(5,7,1,2,3))
        env = env1 + env2
        env = self.psi.Tb_inv[self.index2].applyLeft(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        return self.wrapRightEnv(env)


class gradImplementation_mpso_2d_mpo_bipartite_twoBodyH_hori(gradImplementation_mpso_2d_mpo_bipartite_H_verticalLength_1):
    def __init__(self,psi,H,H_index,grad_index):
        super().__init__(psi,H,H_index,grad_index)

        if self.H_index == self.grad_index:
            innerContract = ncon([self.psi.mpo[self.index1],self.psi.mpo[self.index2],self.H,self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2].conj()],((2,-1,9,10),(6,-5,10,-11),(3,7,2,6),(3,-4,9,13),(7,-8,13,-12)),forder=(-4,-8,-1,-5,-12,-11),order=(9,2,3,10,13,6,7))
            exp = np.real(ncon([innerContract,self.outerContract[self.index1],self.outerContract[self.index2],self.psi.R[self.index1].tensor],((1,2,3,4,5,6),(1,3),(2,4),(5,6)),order=(1,3,2,4,5,6)))
        else:
            innerContract = ncon([self.psi.mpo[self.index2],self.psi.mpo[self.index1],self.H,self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1].conj()],((2,-1,9,10),(6,-5,10,-11),(3,7,2,6),(3,-4,9,13),(7,-8,13,-12)),forder=(-4,-8,-1,-5,-12,-11),order=(9,2,3,10,13,6,7))
            exp = np.real(ncon([innerContract,self.outerContract[self.index2],self.outerContract[self.index1],self.psi.R[self.index2].tensor],((1,2,3,4,5,6),(1,3),(2,4),(5,6)),order=(1,3,2,4,5,6)))
        self.h_tilde = (self.H.reshape(4,4)-exp*np.eye(4)).reshape(2,2,2,2)

        leftEnv = self.buildLeftEnv(H = self.h_tilde)
        if self.H_index == self.grad_index:
            self.innerContract = ncon([self.psi.mpo[self.index1],self.psi.mpo[self.index2],self.h_tilde,self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2].conj()],((2,-1,9,10),(6,-5,10,-11),(3,7,2,6),(3,-4,9,13),(7,-8,13,-12)),forder=(-4,-8,-1,-5,-12,-11),order=(9,2,3,10,13,6,7))
            self.innerContract_shiftedRight = ncon([leftEnv,self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj()],((6,4),(2,-1,4,-5),(2,-3,6,-7)),forder=(-3,-1,-7,-5),order=(4,6,2))
            self.innerContract_shiftedRight_p1 = ncon([self.innerContract_shiftedRight,self.outerContract[self.index1],self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj()],((1,2,3,4),(1,2),(6,-5,4,-8),(6,-7,3,-9)),forder=(-7,-5,-9,-8),order=(1,2,4,3,6))
        else:
            self.innerContract = ncon([self.psi.mpo[self.index2],self.psi.mpo[self.index1],self.h_tilde,self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1].conj()],((2,-1,9,10),(6,-5,10,-11),(3,7,2,6),(3,-4,9,13),(7,-8,13,-12)),forder=(-4,-8,-1,-5,-12,-11),order=(9,2,3,10,13,6,7))
            self.innerContract_shiftedRight = ncon([leftEnv,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj()],((6,4),(2,-1,4,-5),(2,-3,6,-7)),forder=(-3,-1,-7,-5),order=(4,6,2))
            self.innerContract_shiftedRight_p1 = ncon([self.innerContract_shiftedRight,self.outerContract[self.index2],self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj()],((1,2,3,4),(1,2),(6,-5,4,-8),(6,-7,3,-9)),forder=(-7,-5,-9,-8),order=(1,2,4,3,6))


    def getCentralTerms(self):
        if self.H_index == self.grad_index:
            return ncon([self.psi.mpo[self.index1],self.psi.mpo[self.index2],self.H,self.psi.mpo[self.index2].conj(),self.psi.R[self.index1].tensor,self.outerContract[self.index1],self.outerContract[self.index2]],((2,1,-9,10),(6,5,10,11),(-3,7,2,6),(7,8,-13,12),(12,11),(-4,1),(8,5)),forder=(-3,-4,-9,-13),order=(12,11,5,8,6,7,10,1))
        else:
            return ncon([self.psi.mpo[self.index2],self.psi.mpo[self.index1],self.H,self.psi.mpo[self.index2].conj(),self.psi.R[self.index2].tensor,self.outerContract[self.index2],self.outerContract[self.index1]],((2,1,9,10),(6,5,10,11),(3,-7,2,6),(3,4,9,-13),(-12,11),(4,1),(-8,5)),forder=(-7,-8,-13,-12),order=(9,1,2,3,4,1,5,6,11))

    def buildLeftEnv(self,H=None):
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
        return env

    def buildTopEnvGeo(self,fixedPoints,outers):
        if self.H_index == self.grad_index:
            env = self.buildTopEnvGeo_H_horiLength_2(fixedPoints,outers,self.innerContract,p1=True,firstIndex = self.index1)
            innerContract = ncon([self.innerContract,self.outerContract[self.index1]],((1,-2,3,-4,-5,-6),(1,3)),forder=(-2,-4,-5,-6))
            env += self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,innerContract,p1=False,firstIndex = self.index2)
            return env
        else:
            env = self.buildTopEnvGeo_H_horiLength_2(fixedPoints,outers,self.innerContract,p1=False,firstIndex = self.index2)
            innerContract = ncon([self.innerContract,self.outerContract[self.index2]],((1,-2,3,-4,-5,-6),(1,3)),forder=(-2,-4,-5,-6))
            env += self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,innerContract,p1=True,firstIndex = self.index1)
            return env

    def buildBotEnvGeo(self,fixedPoints,outers):
        if self.H_index == self.grad_index:
            env = self.buildBotEnvGeo_H_horiLength_2(fixedPoints,outers,self.innerContract,p1=True,firstIndex = self.index1)
            innerContract = ncon([self.innerContract,self.outerContract[self.index1]],((1,-2,3,-4,-5,-6),(1,3)),forder=(-2,-4,-5,-6))
            env += self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,innerContract,p1=False,firstIndex = self.index2)
            return env
        else:
            env = self.buildBotEnvGeo_H_horiLength_2(fixedPoints,outers,self.innerContract,p1=False,firstIndex = self.index2)
            innerContract = ncon([self.innerContract,self.outerContract[self.index2]],((1,-2,3,-4,-5,-6),(1,3)),forder=(-2,-4,-5,-6))
            env += self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,innerContract,p1=True,firstIndex = self.index1)
            return env

    def buildTopEnvGeo_quadrants(self,fixedPoints,outers):
        if self.H_index == self.grad_index:
            env = self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight,p1=True,firstIndex = self.index1)
            env += self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight_p1,p1=False,firstIndex = self.index2)
        else:
            env = self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight,p1=False,firstIndex = self.index2)
            env += self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight_p1,p1=True,firstIndex = self.index1)
        return env

    def buildBotEnvGeo_quadrants(self,fixedPoints,outers):
        leftEnv = self.buildLeftEnv(H = self.h_tilde)
        if self.H_index == self.grad_index:
            env = self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight,p1=True,firstIndex = self.index1)
            env += self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight_p1,p1=False,firstIndex = self.index2)
        else:
            env = self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight,p1=False,firstIndex = self.index2)
            env += self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight_p1,p1=True,firstIndex = self.index1)
        return env

    def buildRightEnvGeo_quadrants(self,fixedPoints,outers):
        if self.H_index == self.grad_index:
            env1 = self.buildTopEnvGeo_H_horiLength_2(fixedPoints,outers,self.innerContract,p1=False,firstIndex = self.index1)
            env1 += self.buildBotEnvGeo_H_horiLength_2(fixedPoints,outers,self.innerContract,p1=False,firstIndex = self.index1)
            env2 = self.buildTopEnvGeo_H_horiLength_2(fixedPoints,outers,self.innerContract,p1=True,firstIndex = self.index1)
            env2 += self.buildBotEnvGeo_H_horiLength_2(fixedPoints,outers,self.innerContract,p1=True,firstIndex = self.index1)
        else:
            env1 = self.buildTopEnvGeo_H_horiLength_2(fixedPoints,outers,self.innerContract,p1=True,firstIndex = self.index2)
            env1 += self.buildBotEnvGeo_H_horiLength_2(fixedPoints,outers,self.innerContract,p1=True,firstIndex = self.index2)
            env2 = self.buildTopEnvGeo_H_horiLength_2(fixedPoints,outers,self.innerContract,p1=False,firstIndex = self.index2)
            env2 += self.buildBotEnvGeo_H_horiLength_2(fixedPoints,outers,self.innerContract,p1=False,firstIndex = self.index2)
        env1 = ncon([env1,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj()],((3,1,7,5),(2,1,-4,5),(2,3,-6,7)),forder=(-6,-4),order=(5,7,1,2,3))
        env2 = ncon([env2,self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj()],((3,1,7,5),(2,1,-4,5),(2,3,-6,7)),forder=(-6,-4),order=(5,7,1,2,3))
        env2 = ncon([env2,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.outerContract[self.index2]],((7,5),(2,1,-4,5),(2,3,-6,7),(3,1)),forder=(-6,-4),order=(5,7,1,2,3))
        env = env1 + env2
        env = self.psi.Tb_inv[self.index2].applyLeft(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        return self.wrapRightEnv(env)

# -----------------------------
class gradImplementation_mpso_2d_mpo_bipartite_H_verticalLength_2(gradImplementation_mpso_2d_mpo_bipartite):
    def __init__(self,psi,H,H_index,grad_index):
        super().__init__(psi,H,H_index,grad_index)
        self.outerContract = dict()
        self.outerContractDouble = dict()
        self.outerContract[1] =  ncon([self.psi.mps[1],self.psi.mps[1].conj(),self.psi.T[2].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
        self.outerContract[2] =  ncon([self.psi.mps[2],self.psi.mps[2].conj(),self.psi.T[1].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
        self.outerContractDouble[1] = ncon([self.psi.mps[1],self.psi.mps[2],self.psi.mps[1].conj(),self.psi.mps[2].conj(),self.psi.T[1].tensor],((-1,5,6),(-2,6,7),(-3,5,9),(-4,9,8),(8,7)),forder=(-3,-4,-1,-2))
        self.outerContractDouble[2] = ncon([self.psi.mps[2],self.psi.mps[1],self.psi.mps[2].conj(),self.psi.mps[1].conj(),self.psi.T[2].tensor],((-1,5,6),(-2,6,7),(-3,5,9),(-4,9,8),(8,7)),forder=(-3,-4,-1,-2))

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

    def wrapLeftEnv(self,env):
        if self.H_index == self.grad_index:
            env1 = ncon([env,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.psi.RR[self.index1].tensor,self.outerContractDouble[self.index2]],((6,4,-10,-8),(2,1,4,5),(2,3,6,7),(7,5,-11,-9),(3,-12,1,-13)),forder=(-12,-13,-10,-8,-11,-9),order=(4,6,2,5,7,1,3))
            env = ncon([env,self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.outerContractDouble[self.index2]],((13,11,9,7),(2,1,7,-8),(2,3,9,-10),(5,4,11,-12),(5,6,13,-14),(6,3,4,1)),forder=(-14,-12,-10,-8),order=(7,9,2,11,13,5,1,3,4,6))
            env2 = ncon([env,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.psi.RR[self.index2].tensor,self.outerContractDouble[self.index1]],((-10,-8,6,4),(2,1,4,5),(2,3,6,7),(-11,-9,7,5),(-12,3,-13,1)),forder=(-12,-13,-10,-8,-11,-9),order=(4,6,2,5,7,1,3))
            env = env1 + env2
        else:
            env1 = ncon([env,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.psi.RR[self.index2].tensor,self.outerContractDouble[self.index1]],((-10,-8,6,4),(2,1,4,5),(2,3,6,7),(-11,-9,7,5),(-12,3,-13,1)),forder=(-12,-13,-10,-8,-11,-9),order=(4,6,2,5,7,1,3))
            env = ncon([env,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj(),self.outerContractDouble[self.index1]],((13,11,9,7),(2,1,7,-8),(2,3,9,-10),(5,4,11,-12),(5,6,13,-14),(6,3,4,1)),forder=(-14,-12,-10,-8),order=(7,9,2,11,13,5,1,3,4,6))
            env2 = ncon([env,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.psi.RR[self.index1].tensor,self.outerContractDouble[self.index2]],((6,4,-10,-8),(2,1,4,5),(2,3,6,7),(7,5,-11,-9),(3,-12,1,-13)),forder=(-12,-13,-10,-8,-11,-9),order=(4,6,2,5,7,1,3))
            env = env1 + env2
        return env

    def wrapRightEnv(self,env):
        if self.H_index == self.grad_index:
            env1  = ncon([self.outerContract[self.index1],env],((-1,-2),(3,3,-4,-5)),forder=(-1,-2,-4,-5)) 
            env2 =  ncon([self.outerContract[self.index1],env,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.outerContract[self.index2]],((-8,-9),(7,5,10,10),(2,1,-4,5),(2,3,-6,7),(3,1)),forder=(-8,-9,-6,-4),order=[10,5,7,1,2,3])
            env = env1 + env2
        else:
            env1  = ncon([self.outerContract[self.index1],env],((-1,-2),(-3,-4,5,5)),forder=(-1,-2,-3,-4)) 
            env2 =  ncon([self.outerContract[self.index1],env,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.outerContract[self.index2]],((-8,-9),(10,10,7,5),(2,1,-4,5),(2,3,-6,7),(3,1)),forder=(-8,-9,-6,-4),order=[10,5,7,1,2,3])
            env = env1 + env2
        return env

    def buildTopEnvGeo_H_horiLength_1(self,fixedPoints,outers,innerContract,p1,firstIndex):
        if p1 == False:
            RRR_d_label, outerContractTriple_label = 'RRR_d_lower', 'outerContractTriple_lower'
            if firstIndex == 1:
                index1,index2 = 1,2
            elif firstIndex == 2:
                index1,index2 = 2,1
        elif p1 == True:
            RRR_d_label, outerContractTriple_label = 'RRR_d_lower_p1', 'outerContractTriple_lower_p1'
            if firstIndex == 1:
                index1,index2 = 2,1
            elif firstIndex == 2:
                index1,index2 = 1,2
        return ncon([innerContract,fixedPoints[RRR_d_label][index1],outers[outerContractTriple_label][index2]],((1,2,3,4,5,6,7,8),(-9,-10,5,6,7,8),(-11,1,2,-12,3,4)),forder=(-11,-12,-9,-10),order=(5,6,7,8,1,2,3,4))

    def buildBotEnvGeo_H_horiLength_1(self,fixedPoints,outers,innerContract,p1,firstIndex):
        if p1 == False:
            RRR_d_label,outerContractTriple_label = 'RRR_d_upper', 'outerContractTriple_upper'
        elif p1 == True:
            RRR_d_label,outerContractTriple_label = 'RRR_d_upper_p1', 'outerContractTriple_upper_p1'
        if firstIndex == 1:
            index1,index2 = 1,2
        elif firstIndex == 2:
            index1,index2 = 2,1
        return ncon([innerContract,fixedPoints[RRR_d_label][index2],outers[outerContractTriple_label][index1]],((1,2,3,4,5,6,7,8),(5,6,7,8,-9,-10),(1,2,-11,3,4,-12)),forder=(-11,-12,-9,-10),order=(5,6,7,8,1,2,3,4))


class gradImplementation_mpso_2d_mpo_bipartite_twoBodyH_vert(gradImplementation_mpso_2d_mpo_bipartite_H_verticalLength_2):
    def __init__(self,psi,H,H_index,grad_index):
        super().__init__(psi,H,H_index,grad_index)

        if self.H_index == self.grad_index:
            leftEnv = ncon([self.psi.mpo[self.index1],self.psi.mpo[self.index2],self.H,self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2].conj(),self.outerContractDouble[self.index1]],((2,1,9,-10),(6,5,11,-12),(3,7,2,6),(3,4,9,-13),(7,8,11,-14),(4,8,1,5)),forder=(-13,-10,-14,-12),order=(9,1,2,3,4,11,5,6,7,8))
            exp = np.einsum('abcd,abcd',leftEnv,self.psi.RR[self.index2].tensor)
        else:
            leftEnv = ncon([self.psi.mpo[self.index2],self.psi.mpo[self.index1],self.H,self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1].conj(),self.outerContractDouble[self.index2]],((2,1,9,-10),(6,5,11,-12),(3,7,2,6),(3,4,9,-13),(7,8,11,-14),(4,8,1,5)),forder=(-13,-10,-14,-12),order=(9,1,2,3,4,11,5,6,7,8))
            exp = np.einsum('abcd,abcd',leftEnv,self.psi.RR[self.index1].tensor)
        self.h_tilde = (self.H.reshape(4,4)-exp*np.eye(4)).reshape(2,2,2,2)

        leftEnv = self.buildLeftEnv(H=self.h_tilde)
        if self.H_index == self.grad_index:
            self.innerContract= ncon([self.psi.mpo[self.index1],self.psi.mpo[self.index2],self.h_tilde,self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2].conj()],((2,-1,9,-10),(6,-5,11,-12),(3,7,2,6),(3,-4,9,-13),(7,-8,11,-14)),forder=(-4,-8,-1,-5,-13,-10,-14,-12),order=(9,2,3,11,6,7))
            self.innerContract_shiftedRight  = ncon([leftEnv,self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj()],((13,11,9,7),(2,-1,7,-8),(2,-3,9,-10),(5,-4,11,-12),(5,-6,13,-14)),forder=(-6,-3,-4,-1,-14,-12,-10,-8),order=(7,9,2,11,13,5))
            self.innerContract_shiftedRight_p1  = ncon([self.innerContract_shiftedRight,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj(),self.outerContractDouble[self.index2]],((1,2,3,4,5,6,7,8),(10,-9,8,-15),(10,-11,7,-16),(13,-12,6,-17),(13,-14,5,-18),(1,2,3,4)),forder=(-14,-11,-12,-9,-18,-17,-16,-15),order=(1,2,3,4,8,7,10,6,5,13))
        else:
            self.innerContract= ncon([self.psi.mpo[self.index2],self.psi.mpo[self.index1],self.h_tilde,self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1].conj()],((2,-1,9,-10),(6,-5,11,-12),(3,7,2,6),(3,-4,9,-13),(7,-8,11,-14)),forder=(-4,-8,-1,-5,-13,-10,-14,-12),order=(9,2,3,11,6,7))
            self.innerContract_shiftedRight  = ncon([leftEnv,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj()],((13,11,9,7),(2,-1,7,-8),(2,-3,9,-10),(5,-4,11,-12),(5,-6,13,-14)),forder=(-6,-3,-4,-1,-14,-12,-10,-8),order=(7,9,2,11,13,5))
            self.innerContract_shiftedRight_p1  = ncon([self.innerContract_shiftedRight,self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.outerContractDouble[self.index1]],((1,2,3,4,5,6,7,8),(10,-9,8,-15),(10,-11,7,-16),(13,-12,6,-17),(13,-14,5,-18),(1,2,3,4)),forder=(-14,-11,-12,-9,-18,-17,-16,-15),order=(1,2,3,4,8,7,10,6,5,13))


    def getCentralTerms(self):
        if self.H_index == self.grad_index:
            return ncon([self.psi.mpo[self.index1],self.psi.mpo[self.index2],self.H,self.psi.mpo[self.index2].conj(),self.psi.RR[self.index2].tensor,self.outerContractDouble[self.index1]],((2,1,-9,10),(4,3,11,12),(-5,7,2,4),(7,8,11,14),(-13,10,14,12),(-6,8,1,3)),forder=(-5,-6,-9,-13),order=(12,14,11,4,7,3,8,10,2,1))
        else:
            return ncon([self.psi.mpo[self.index2],self.psi.mpo[self.index1],self.H,self.psi.mpo[self.index2].conj(),self.psi.RR[self.index1].tensor,self.outerContractDouble[self.index2]],((2,1,9,10),(6,5,-11,12),(3,-7,2,6),(3,4,9,13),(13,10,-14,12),(4,-8,1,5)),forder=(-7,-8,-11,-14),order=(10,13,9,2,3,1,4,12,5))

    def buildLeftEnv(self,H=None):
        if H is None:
            H = self.H
        if self.H_index == self.grad_index:
            env = ncon([self.psi.mpo[self.index1],self.psi.mpo[self.index2],H,self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2].conj(),self.outerContractDouble[self.index1]],((2,1,9,-10),(6,5,11,-12),(3,7,2,6),(3,4,9,-13),(7,8,11,-14),(4,8,1,5)),forder=(-13,-10,-14,-12),order=(9,1,2,3,4,11,5,6,7,8))
            env = self.psi.Tb2_inv[self.index2].applyRight(env.reshape(self.psi.D_mpo**4)).reshape(self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo)
        else:
            env = ncon([self.psi.mpo[self.index2],self.psi.mpo[self.index1],H,self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1].conj(),self.outerContractDouble[self.index2]],((2,1,9,-10),(6,5,11,-12),(3,7,2,6),(3,4,9,-13),(7,8,11,-14),(4,8,1,5)),forder=(-13,-10,-14,-12),order=(9,1,2,3,4,11,5,6,7,8))
            env = self.psi.Tb2_inv[self.index1].applyRight(env.reshape(self.psi.D_mpo**4)).reshape(self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo)
        return env

    def buildRightEnv(self,H=None):
        if H is None:
            H = self.H
        if self.H_index == self.grad_index:
            env = ncon([self.psi.mpo[self.index1],self.psi.mpo[self.index2],H,self.psi.mpo[self.index1].conj(),self.psi.mpo[self.index2].conj(),self.psi.RR[self.index2].tensor,self.outerContractDouble[self.index1]],((2,1,-9,10),(6,5,-11,12),(3,7,2,6),(3,4,-13,14),(7,8,-15,16),(14,10,16,12),(4,8,1,5)),forder=(-13,-9,-15,-11),order=(12,16,5,6,7,8,10,14,1,2,3,4))
            env = self.psi.Tb2_inv[self.index1].applyLeft(env.reshape(self.psi.D_mpo**4)).reshape(self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo)
        else:
            env = ncon([self.psi.mpo[self.index2],self.psi.mpo[self.index1],H,self.psi.mpo[self.index2].conj(),self.psi.mpo[self.index1].conj(),self.psi.RR[self.index1].tensor,self.outerContractDouble[self.index2]],((2,1,-9,10),(6,5,-11,12),(3,7,2,6),(3,4,-13,14),(7,8,-15,16),(14,10,16,12),(4,8,1,5)),forder=(-13,-9,-15,-11),order=(12,16,5,6,7,8,10,14,1,2,3,4))
            env = self.psi.Tb2_inv[self.index2].applyLeft(env.reshape(self.psi.D_mpo**4)).reshape(self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo)
        return env

    def buildTopEnvGeo(self,fixedPoints,outers):
        if self.H_index == self.grad_index:
            return self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract,p1=True,firstIndex=self.index1)
        else:
            return self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract,p1=False,firstIndex=self.index2)

    def buildBotEnvGeo(self,fixedPoints,outers):
        if self.H_index == self.grad_index:
            return self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract,p1=False,firstIndex=self.index1)
        else:
            return self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract,p1=True,firstIndex=self.index2)

    def buildTopEnvGeo_quadrants(self,fixedPoints,outers):
        if self.H_index == self.grad_index:
            env = self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight,p1=False,firstIndex=self.index2)
            env += self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight_p1,p1=True,firstIndex=self.index1)
        else:
            env = self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight,p1=True,firstIndex=self.index1)
            env += self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight_p1,p1=False,firstIndex=self.index2)
        return env

    def buildBotEnvGeo_quadrants(self,fixedPoints,outers):
        leftEnv = self.buildLeftEnv(H=self.h_tilde)
        if self.H_index == self.grad_index:
            env = self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight,p1=True,firstIndex=self.index2)
            env += self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight_p1,p1=False,firstIndex=self.index1)
        else:
            env = self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight,p1=False,firstIndex=self.index1)
            env += self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract_shiftedRight_p1,p1=True,firstIndex=self.index2)
        return env

    def buildRightEnvGeo_quadrants(self,fixedPoints,outers):
        if self.H_index == self.grad_index:
            #bot env
            env1 = self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract,p1=True,firstIndex=self.index1)
            env1 += self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract,p1=False,firstIndex=self.index1)
            env2 = self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract,p1=False,firstIndex = self.index1)
            env2 += self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract,p1=True,firstIndex=self.index1)
        else:
            #bot env
            env1 = self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract,p1=False,firstIndex=self.index2)
            env1 += self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract,p1=True,firstIndex=self.index2)
            env2 = self.buildBotEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract,p1=True,firstIndex = self.index2)
            env2 += self.buildTopEnvGeo_H_horiLength_1(fixedPoints,outers,self.innerContract,p1=False,firstIndex=self.index2)
        env1 = ncon([env1,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj()],((3,1,7,5),(2,1,-4,5),(2,3,-6,7)),forder=(-6,-4),order=(5,7,1,2,3))
        env2 = ncon([env2,self.psi.mpo[self.index1],self.psi.mpo[self.index1].conj()],((3,1,7,5),(2,1,-4,5),(2,3,-6,7)),forder=(-6,-4),order=(5,7,1,2,3))
        #append extra site to left
        env2 = ncon([env2,self.psi.mpo[self.index2],self.psi.mpo[self.index2].conj(),self.outerContract[self.index2]],((7,5),(2,1,-4,5),(2,3,-6,7),(3,1)),forder=(-6,-4),order=(5,7,1,2,3))
        env = env1 + env2
        env = self.psi.Tb_inv[self.index2].applyLeft(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        return ncon([self.outerContract[self.index1],env],((-1,-2),(-3,-4)),forder=(-1,-2,-3,-4))
# -----------------------------
class gradImplementation_mpso_2d_mpo_twoSite_twoBodyH_hori(gradImplementation_mpso_2d_mpo_uniform):
    def __init__(self,psi,H):
        super().__init__(psi,H)
        #objects needed to construct terms
        self.outerContract = dict()
        self.outerContract['bot'] = ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((-1,3,4,5),(-2,3,4,6),(6,5)),forder=(-2,-1),order=(4,3,5,6))
        self.outerContract['top']= ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((1,-2,4,5),(1,-3,4,6),(6,5)),forder=(-3,-2),order=(4,1,5,6))

    def getFixedPoints(self,d,Td):
        RR_d = dict()

        TT_d_bb = mpsu1Transfer_left_twoLayerWithMPSInsert_twoSite(self.psi.mps,self.psi.mpo,self.psi.T,'bb',Td)
        RR_d['bb'] = TT_d_bb.findRightEig()
        RR_d['bb'].norm_pairedCanon()
        RR_d['bb'] = RR_d['bb'].tensor
        del TT_d_bb

        TT_d_bt = mpsu1Transfer_left_twoLayerWithMPSInsert_twoSite(self.psi.mps,self.psi.mpo,self.psi.T,'bt',Td)
        RR_d['bt'] = TT_d_bt.findRightEig()
        RR_d['bt'].norm_pairedCanon()
        RR_d['bt'] = RR_d['bt'].tensor
        del TT_d_bt

        TT_d_tb = mpsu1Transfer_left_twoLayerWithMPSInsert_twoSite(self.psi.mps,self.psi.mpo,self.psi.T,'tb',Td)
        RR_d['tb'] = TT_d_tb.findRightEig()
        RR_d['tb'].norm_pairedCanon()
        RR_d['tb'] = RR_d['tb'].tensor
        del TT_d_tb

        TT_d_tt = mpsu1Transfer_left_twoLayerWithMPSInsert_twoSite(self.psi.mps,self.psi.mpo,self.psi.T,'tt',Td)
        RR_d['tt'] = TT_d_tt.findRightEig()
        RR_d['tt'].norm_pairedCanon()
        RR_d['tt'] = RR_d['tt'].tensor
        del TT_d_tt

        fp = dict()
        fp['RR_d'] = RR_d
        return fp
    def getOuterContracts(self,Td):
        outerContractDouble = dict()
        outerContractQuad = ncon([self.psi.mps,self.psi.mps,self.psi.mps.conj(),self.psi.mps.conj(),Td,self.psi.T.tensor],((-1,-3,9,10),(-5,-7,11,12),(-2,-4,9,15),(-6,-8,14,13),(15,10,14,11),(13,12)),forder=(-2,-4,-6,-8,-1,-3,-5,-7),order=(9,10,15,11,14,13,12))
        outerContractDouble['bb'] = ncon([outerContractQuad],((-2,3,-5,6,-1,3,-4,6)),forder=(-2,-5,-1,-4))
        outerContractDouble['bt']= ncon([outerContractQuad],((-2,3,4,-6,-1,3,4,-5)),forder=(-2,-6,-1,-5))
        outerContractDouble['tb']= ncon([outerContractQuad],((1,-3,-5,6,1,-2,-4,6)),forder=(-3,-5,-2,-4))
        outerContractDouble['tt']= ncon([outerContractQuad],((1,-3,4,-6,1,-2,4,-5)),forder=(-3,-6,-2,-5))
        fp = dict()
        fp['outerContractDouble'] = outerContractDouble
        return fp

    #building vertical left environment under above/below a two site unit cell (useful for quadrants)
    def buildTopEnvGeo_single(self,fixedPoints,outers,style,innerContract):
        return ncon([innerContract,fixedPoints['RR_d'][style],outers['outerContractDouble'][style],outers['outerContractDouble'][style]],((1,2,3,4,5,6),(-7,-8,5,6),(-9,1,-10,3),(-11,2,-12,4)),forder=(-9,-11,-10,-12,-7,-8),order=(1,3,4,2,5,6))
    def buildBotEnvGeo_single(self,fixedPoints,outers,style,innerContract):
        return ncon([innerContract,fixedPoints['RR_d'][style],outers['outerContractDouble'][style],outers['outerContractDouble'][style]],((1,2,3,4,5,6),(5,6,-7,-8),(1,-9,3,-10),(2,-11,4,-12)),forder=(-9,-11,-10,-12,-7,-8),order=(1,3,4,2,5,6))

    #forming wrapAround environment for left quadrants, 
    # given wrapAround environments env_bot, env_top which need to be acted 
    #on by Tb_inv['bot'],Tb_inv['top'] respectively
    def modifyLeftQuadrant_env(self,env_bot,env_top):
        env_bot = ncon([env_bot,self.psi.mpo,self.psi.mpo.conj()],((3,6,1,4,10,8),(2,5,1,4,-7,8),(2,5,3,6,-9,10)),forder=(-9,-7),order=(10,8,2,5,1,4,3,6))
        env_top = ncon([env_top,self.psi.mpo,self.psi.mpo.conj()],((3,6,1,4,10,8),(2,5,1,4,-7,8),(2,5,3,6,-9,10)),forder=(-9,-7),order=(10,8,2,5,1,4,3,6))
        env_bot = self.psi.Tb_inv['bot'].applyLeft(env_bot.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        env_top = self.psi.Tb_inv['top'].applyLeft(env_top.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        env_bot = ncon([env_bot,self.outerContract['bot'],self.outerContract['bot']],((-5,-6),(-1,-2),(-3,-4)),forder=(-1,-3,-2,-4,-5,-6))
        env_top = ncon([env_top,self.outerContract['top'],self.outerContract['top']],((-5,-6),(-1,-2),(-3,-4)),forder=(-1,-3,-2,-4,-5,-6))
        return env_bot + env_top

#terms with the Hamiltonian under site 1 of the mpo ('on centre')
class gradImplementation_mpso_2d_mpo_twoSite_twoBodyH_hori_site1(gradImplementation_mpso_2d_mpo_twoSite_twoBodyH_hori):
    def __init__(self,psi,H):
        super().__init__(psi,H)
        #objects needed to construct terms
        innerContract = ncon([self.psi.mpo,self.H,self.psi.mpo.conj()],((2,6,-1,-5,9,-10),(3,7,2,6),(3,7,-4,-8,9,-11)),forder=(-4,-8,-1,-5,-11,-10),order=(9,2,3,6,7))
        exp = ncon([innerContract,self.outerContract[self.style],self.outerContract[self.style],self.psi.R[self.style].tensor],((1,2,3,4,5,6),(1,3),(2,4),(5,6)),order=(1,3,2,4,5,6))
        self.h_tilde = (self.H.reshape(4,4)-exp*np.eye(4)).reshape(2,2,2,2)
        self.innerContract = ncon([self.psi.mpo,self.h_tilde,self.psi.mpo.conj()],((2,6,-1,-5,9,-10),(3,7,2,6),(3,7,-4,-8,9,-11)),forder=(-4,-8,-1,-5,-11,-10),order=(9,2,3,6,7))

    def getCentralTerms(self):
        return ncon([self.psi.mpo,self.H,self.outerContract[self.style],self.outerContract[self.style],self.psi.R[self.style].tensor],((2,6,1,5,-9,10),(-3,-7,2,6),(-4,1),(-8,5),(-11,10)),forder=(-3,-7,-4,-8,-9,-11),order=(10,5,6,1,2))

    def buildLeftEnv(self,H=None,wrapAround=False):
        if H is None:
            H = self.H
        env = ncon([self.psi.mpo,H,self.psi.mpo.conj(),self.outerContract[self.style],self.outerContract[self.style]],((2,6,1,5,9,-10),(3,7,2,6),(3,7,4,8,9,-11),(4,1),(8,5)),forder=(-11,-10),order=(9,2,6,3,7,1,4,5,8))
        env = self.psi.Tb_inv[self.style].applyRight(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        if wrapAround == True:
            env = ncon([env,self.outerContract[self.style],self.outerContract[self.style],self.psi.R[self.style].tensor],((-1,-2),(-5,-6),(-7,-8),(-3,-4)),forder=(-5,-7,-6,-8,-1,-2,-3,-4))
        return env

    def buildRightEnv(self,H=None):
        if H is None:
            H = self.H
        env = ncon([self.psi.mpo,H,self.psi.mpo.conj(),self.outerContract[self.style],self.outerContract[self.style],self.psi.R[self.style].tensor],((2,6,1,5,-9,10),(3,7,2,6),(3,7,4,8,-11,12),(4,1),(8,5),(12,10)),forder=(-11,-9),order=(10,12,6,7,2,3,5,8,1,4))
        env = self.psi.Tb_inv[self.style].applyLeft(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        env = ncon([env,self.outerContract[self.style],self.outerContract[self.style]],((-5,-6),(-1,-2),(-3,-4)),forder=(-1,-3,-2,-4,-5,-6))
        return env

#terms with the Hamiltonian under site 2 of the mpo ('off centre')
class gradImplementation_mpso_2d_mpo_twoSite_twoBodyH_hori_site2(gradImplementation_mpso_2d_mpo_twoSite_twoBodyH_hori):
    def __init__(self,psi,H):
        super().__init__(psi,H)
        #objects needed to construct terms
        innerContract = ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.mpo.conj()],((2,5,-1,-4,15,16),(9,13,-8,-12,16,-17),(6,10,5,9),(2,6,-3,-7,15,19),(10,13,-11,-14,19,-18)),forder=(-3,-7,-11,-14,-1,-4,-8,-12,-18,-17),order=(15,2,5,6,16,19,9,10,13))
        exp = ncon([innerContract,self.outerContract[self.style],self.outerContract[self.style],self.outerContract[self.style],self.outerContract[self.style],self.psi.R[self.style].tensor],((1,2,3,4,5,6,7,8,9,10),(1,5),(2,6),(3,7),(4,8),(9,10)),order=(1,5,2,6,3,7,4,8,9,10))
        self.h_tilde = (self.H.reshape(4,4)-exp*np.eye(4)).reshape(2,2,2,2)
        self.innerContract = ncon([self.psi.mpo,self.psi.mpo,self.h_tilde,self.psi.mpo.conj(),self.psi.mpo.conj()],((2,5,-1,-4,15,16),(9,13,-8,-12,16,-17),(6,10,5,9),(2,6,-3,-7,15,19),(10,13,-11,-14,19,-18)),forder=(-3,-7,-11,-14,-1,-4,-8,-12,-18,-17),order=(15,2,5,6,16,19,9,10,13))

    def getCentralTerms(self):
        grad = ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.outerContract[self.style],self.outerContract[self.style],self.outerContract[self.style],self.outerContract[self.style],self.psi.R[self.style].tensor],((-2,5,1,4,-15,16),(9,13,8,12,16,17),(-6,10,5,9),(10,13,11,14,-19,18),(-3,1),(-7,4),(11,8),(14,12),(18,17)),forder=(-2,-6,-3,-7,-15,-19),order=(18,17,13,9,10,12,14,8,11,16,5,4,1))
        grad += ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.outerContract[self.style],self.outerContract[self.style],self.outerContract[self.style],self.outerContract[self.style],self.psi.R[self.style].tensor],((2,5,1,4,15,16),(9,-13,8,12,16,17),(6,-10,5,9),(2,6,3,7,15,-19),(3,1),(7,4),(-11,8),(-14,12),(-18,17)),forder=(-10,-13,-11,-14,-19,-18),order=(15,2,5,6,1,3,4,7,16,9,8,12,17))
        return grad

    def buildLeftEnv(self,H=None,wrapAround=False):
        if H is None:
            H = self.H
        env = ncon([self.psi.mpo,self.psi.mpo,H,self.psi.mpo.conj(),self.psi.mpo.conj(),self.outerContract[self.style],self.outerContract[self.style],self.outerContract[self.style],self.outerContract[self.style]],((2,5,1,4,15,16),(9,13,8,12,16,-17),(6,10,5,9),(2,6,3,7,15,19),(10,13,11,14,19,-18),(3,1),(7,4),(11,8),(14,12)),forder=(-18,-17),order=(15,2,5,6,1,3,4,7,16,19,9,10,13,8,11,12,14))
        env = self.psi.Tb_inv[self.style].applyRight(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        if wrapAround == True:
            env = ncon([env,self.outerContract[self.style],self.outerContract[self.style],self.psi.R[self.style].tensor],((-1,-2),(-5,-6),(-7,-8),(-3,-4)),forder=(-5,-7,-6,-8,-1,-2,-3,-4))
        return env

    def buildRightEnv(self,H=None):
        if H is None:
            H = self.H
        env = ncon([self.psi.mpo,self.psi.mpo,H,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.R[self.style].tensor,self.outerContract[self.style],self.outerContract[self.style],self.outerContract[self.style],self.outerContract[self.style]],((2,5,1,4,-15,16),(9,13,8,12,16,17),(6,10,5,9),(2,6,3,7,-20,19),(10,13,11,14,19,18),(18,17),(3,1),(7,4),(11,8),(14,12)),forder=(-20,-15),order=(18,17,9,10,13,8,11,12,14,16,19,5,6,2,4,7,1,3))
        env = self.psi.Tb_inv[self.style].applyLeft(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        env = ncon([env,self.outerContract[self.style],self.outerContract[self.style]],((-5,-6),(-1,-2),(-3,-4)),forder=(-1,-3,-2,-4,-5,-6))
        return env

    #for building top env with a inner contract having 8 physical legs (upper left term in gradient)
    def buildTopEnvGeo_singleLeft(self,fixedPoints,outers,style,innerContract):
        return ncon([innerContract,self.psi.mpo,self.psi.mpo.conj(),fixedPoints['RR_d'][style],outers['outerContractDouble'][style],outers['outerContractDouble'][style],outers['outerContractDouble'][style],outers['outerContractDouble'][style]],((1,2,3,4,5,6,7,8,9,10),(14,17,13,16,-19,12),(14,17,15,18,-20,11),(11,12,9,10),(18,4,16,8),(15,3,13,7),(-21,2,-22,6),(-23,1,-24,5)),forder=(-23,-21,-24,-22,-20,-19),order=(11,12,14,17,9,10,18,4,16,8,15,3,13,7,2,6,1,5))
    def buildBotEnvGeo_singleLeft(self,fixedPoints,outers,style,innerContract):
        return ncon([innerContract,self.psi.mpo,self.psi.mpo.conj(),fixedPoints['RR_d'][style],outers['outerContractDouble'][style],outers['outerContractDouble'][style],outers['outerContractDouble'][style],outers['outerContractDouble'][style]],((1,2,3,4,5,6,7,8,9,10),(13,16,14,17,-19,12),(14,17,15,18,-20,11),(9,10,11,12),(4,18,8,16),(3,15,7,13),(2,-21,6,-22),(1,-23,5,-24)),forder=(-23,-21,-24,-22,-20,-19),order=(11,12,14,17,9,10,18,4,16,8,15,3,13,7,2,6,1,5))

class gradImplementation_mpso_2d_mpo_twoSite_twoBodyH_hori_site00(gradImplementation_mpso_2d_mpo_twoSite_twoBodyH_hori_site1):
    def __init__(self,psi,H):
        self.style = 'bot'
        super().__init__(psi,H)
    def buildTopEnvGeo(self,fixedPoints,outers):
        env = self.buildTopEnvGeo_single(fixedPoints,outers,'bb',self.innerContract)
        env += self.buildTopEnvGeo_single(fixedPoints,outers,'tb',self.innerContract)
        return env
    def buildBotEnvGeo(self,fixedPoints,outers):
        env = self.buildBotEnvGeo_single(fixedPoints,outers,'bb',self.innerContract)
        env += self.buildBotEnvGeo_single(fixedPoints,outers,'bt',self.innerContract)
        return env
    def buildTopEnvGeo_quadrants(self,fixedPoints,outers,leftEnv):
        innerContract = ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj()],((4,7),(2,5,-1,-4,7,-8),(2,5,-3,-6,4,-10)),forder=(-3,-6,-1,-4,-10,-8),order=(4,7,2,5))
        env = self.buildTopEnvGeo_single(fixedPoints,outers,'bb',innerContract)
        env += self.buildTopEnvGeo_single(fixedPoints,outers,'tb',innerContract)
        return env
    def buildBotEnvGeo_quadrants(self,fixedPoints,outers,leftEnv):
        innerContract = ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj()],((4,7),(2,5,-1,-4,7,-8),(2,5,-3,-6,4,-10)),forder=(-3,-6,-1,-4,-10,-8),order=(4,7,2,5))
        env = self.buildBotEnvGeo_single(fixedPoints,outers,'bb',innerContract)
        env += self.buildBotEnvGeo_single(fixedPoints,outers,'bt',innerContract)
        return env
    def buildRightEnvGeo_quadrants(self,fixedPoints,outers):
        env1 = self.buildTopEnvGeo_single(fixedPoints,outers,'bb',self.innerContract)
        env1 += self.buildBotEnvGeo_single(fixedPoints,outers,'bb',self.innerContract)
        env2 = self.buildTopEnvGeo_single(fixedPoints,outers,'tb',self.innerContract)
        env2 += self.buildBotEnvGeo_single(fixedPoints,outers,'bt',self.innerContract)
        return self.modifyLeftQuadrant_env(env1,env2)

class gradImplementation_mpso_2d_mpo_twoSite_twoBodyH_hori_site10(gradImplementation_mpso_2d_mpo_twoSite_twoBodyH_hori_site1):
    def __init__(self,psi,H):
        self.style = 'top'
        super().__init__(psi,H)
    def buildTopEnvGeo(self,fixedPoints,outers):
        env = self.buildTopEnvGeo_single(fixedPoints,outers,'tt',self.innerContract)
        env += self.buildTopEnvGeo_single(fixedPoints,outers,'bt',self.innerContract)
        return env
    def buildBotEnvGeo(self,fixedPoints,outers):
        env = self.buildBotEnvGeo_single(fixedPoints,outers,'tt',self.innerContract)
        env += self.buildBotEnvGeo_single(fixedPoints,outers,'tb',self.innerContract)
        return env
    def buildTopEnvGeo_quadrants(self,fixedPoints,outers,leftEnv):
        innerContract = ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj()],((4,7),(2,5,-1,-4,7,-8),(2,5,-3,-6,4,-10)),forder=(-3,-6,-1,-4,-10,-8),order=(4,7,2,5))
        env = self.buildTopEnvGeo_single(fixedPoints,outers,'tt',innerContract)
        env += self.buildTopEnvGeo_single(fixedPoints,outers,'bt',innerContract)
        return env
    def buildBotEnvGeo_quadrants(self,fixedPoints,outers,leftEnv,):
        innerContract = ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj()],((4,7),(2,5,-1,-4,7,-8),(2,5,-3,-6,4,-10)),forder=(-3,-6,-1,-4,-10,-8),order=(4,7,2,5))
        env = self.buildBotEnvGeo_single(fixedPoints,outers,'tt',innerContract)
        env += self.buildBotEnvGeo_single(fixedPoints,outers,'tb',innerContract)
        return env
    def buildRightEnvGeo_quadrants(self,fixedPoints,outers):
        env1 = self.buildTopEnvGeo_single(fixedPoints,outers,'bt',self.innerContract)
        env1 += self.buildBotEnvGeo_single(fixedPoints,outers,'tb',self.innerContract)
        env2 = self.buildTopEnvGeo_single(fixedPoints,outers,'tt',self.innerContract)
        env2 += self.buildBotEnvGeo_single(fixedPoints,outers,'tt',self.innerContract)
        return self.modifyLeftQuadrant_env(env1,env2)

class gradImplementation_mpso_2d_mpo_twoSite_twoBodyH_hori_site01(gradImplementation_mpso_2d_mpo_twoSite_twoBodyH_hori_site2):
    def __init__(self,psi,H):
        self.style = 'bot'
        super().__init__(psi,H)
    def buildTopEnvGeo(self,fixedPoints,outers):
        env = self.buildTopEnvGeo_singleLeft(fixedPoints,outers,'bb',self.innerContract)
        env += self.buildTopEnvGeo_singleLeft(fixedPoints,outers,'tb',self.innerContract)
        innerContract = ncon([self.innerContract,self.outerContract[self.style],self.outerContract[self.style]],((1,2,-3,-4,5,6,-7,-8,-9,-10),(1,5),(2,6)),forder=(-3,-4,-7,-8,-9,-10))
        env += self.buildTopEnvGeo_single(fixedPoints,outers,'bb',innerContract)
        env += self.buildTopEnvGeo_single(fixedPoints,outers,'tb',innerContract)
        return env
    def buildBotEnvGeo(self,fixedPoints,outers):
        env = self.buildBotEnvGeo_singleLeft(fixedPoints,outers,'bb',self.innerContract)
        env += self.buildBotEnvGeo_singleLeft(fixedPoints,outers,'bt',self.innerContract)
        innerContract = ncon([self.innerContract,self.outerContract[self.style],self.outerContract[self.style]],((1,2,-3,-4,5,6,-7,-8,-9,-10),(1,5),(2,6)),forder=(-3,-4,-7,-8,-9,-10))
        env += self.buildBotEnvGeo_single(fixedPoints,outers,'bb',innerContract)
        env += self.buildBotEnvGeo_single(fixedPoints,outers,'bt',innerContract)
        return env
    def buildTopEnvGeo_quadrants(self,fixedPoints,outers,leftEnv):
        innerContract = ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj()],((4,7),(2,5,-1,-4,7,-8),(2,5,-3,-6,4,-10)),forder=(-3,-6,-1,-4,-10,-8),order=(4,7,2,5))
        env = self.buildTopEnvGeo_single(fixedPoints,outers,'bb',innerContract)
        env += self.buildTopEnvGeo_single(fixedPoints,outers,'tb',innerContract)
        return env
    def buildBotEnvGeo_quadrants(self,fixedPoints,outers,leftEnv,):
        innerContract = ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj()],((4,7),(2,5,-1,-4,7,-8),(2,5,-3,-6,4,-10)),forder=(-3,-6,-1,-4,-10,-8),order=(4,7,2,5))
        env = self.buildBotEnvGeo_single(fixedPoints,outers,'bb',innerContract)
        env += self.buildBotEnvGeo_single(fixedPoints,outers,'bt',innerContract)
        return env
    def buildRightEnvGeo_quadrants(self,fixedPoints,outers):
        env1 = self.buildTopEnvGeo_singleLeft(fixedPoints,outers,'bb',self.innerContract)
        env1 += self.buildBotEnvGeo_singleLeft(fixedPoints,outers,'bb',self.innerContract)
        env2 = self.buildTopEnvGeo_singleLeft(fixedPoints,outers,'tb',self.innerContract)
        env2 += self.buildBotEnvGeo_singleLeft(fixedPoints,outers,'bt',self.innerContract)
        return self.modifyLeftQuadrant_env(env1,env2)

class gradImplementation_mpso_2d_mpo_twoSite_twoBodyH_hori_site11(gradImplementation_mpso_2d_mpo_twoSite_twoBodyH_hori_site2):
    def __init__(self,psi,H):
        self.style = 'top'
        super().__init__(psi,H)
    def buildTopEnvGeo(self,fixedPoints,outers):
        env = self.buildTopEnvGeo_singleLeft(fixedPoints,outers,'tt',self.innerContract)
        env += self.buildTopEnvGeo_singleLeft(fixedPoints,outers,'bt',self.innerContract)
        innerContract = ncon([self.innerContract,self.outerContract[self.style],self.outerContract[self.style]],((1,2,-3,-4,5,6,-7,-8,-9,-10),(1,5),(2,6)),forder=(-3,-4,-7,-8,-9,-10))
        env += self.buildTopEnvGeo_single(fixedPoints,outers,'tt',innerContract)
        env += self.buildTopEnvGeo_single(fixedPoints,outers,'bt',innerContract)
        return env
    def buildBotEnvGeo(self,fixedPoints,outers):
        env = self.buildBotEnvGeo_singleLeft(fixedPoints,outers,'tt',self.innerContract)
        env += self.buildBotEnvGeo_singleLeft(fixedPoints,outers,'tb',self.innerContract)
        innerContract = ncon([self.innerContract,self.outerContract[self.style],self.outerContract[self.style]],((1,2,-3,-4,5,6,-7,-8,-9,-10),(1,5),(2,6)),forder=(-3,-4,-7,-8,-9,-10))
        env += self.buildBotEnvGeo_single(fixedPoints,outers,'tt',innerContract)
        env += self.buildBotEnvGeo_single(fixedPoints,outers,'tb',innerContract)
        return env
    def buildTopEnvGeo_quadrants(self,fixedPoints,outers,leftEnv):
        innerContract = ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj()],((4,7),(2,5,-1,-4,7,-8),(2,5,-3,-6,4,-10)),forder=(-3,-6,-1,-4,-10,-8),order=(4,7,2,5))
        env = self.buildTopEnvGeo_single(fixedPoints,outers,'tt',innerContract)
        env += self.buildTopEnvGeo_single(fixedPoints,outers,'bt',innerContract)
        return env
    def buildBotEnvGeo_quadrants(self,fixedPoints,outers,leftEnv,):
        innerContract = ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj()],((4,7),(2,5,-1,-4,7,-8),(2,5,-3,-6,4,-10)),forder=(-3,-6,-1,-4,-10,-8),order=(4,7,2,5))
        env = self.buildBotEnvGeo_single(fixedPoints,outers,'tt',innerContract)
        env += self.buildBotEnvGeo_single(fixedPoints,outers,'tb',innerContract)
        return env
    def buildRightEnvGeo_quadrants(self,fixedPoints,outers):
        env1 = self.buildTopEnvGeo_singleLeft(fixedPoints,outers,'bt',self.innerContract)
        env1 += self.buildBotEnvGeo_singleLeft(fixedPoints,outers,'tb',self.innerContract)
        env2 = self.buildTopEnvGeo_singleLeft(fixedPoints,outers,'tt',self.innerContract)
        env2 += self.buildBotEnvGeo_singleLeft(fixedPoints,outers,'tt',self.innerContract)
        return self.modifyLeftQuadrant_env(env1,env2)
