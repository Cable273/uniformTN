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
class gradImplementation_mpso_2d_mps_uniform(ABC):
    def __init__(self,psi,H):
        self.psi = psi
        self.H = H

    #gets effective 1d Hamiltonians to compute mps gradient of 2d term
    #consists of two terms, effH_centre + effH_shifted
    @abstractmethod
    def getEffectiveH(self):
        pass

# -------------------------------------------------------------------------------------------------------------------------------------
#Concrete implementations
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
        centreEnv =  ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.R.tensor],((1,-10,5,6),(2,-11,6,7),(3,4,1,2),(3,-12,5,9),(4,-13,9,8),(8,7)),forder=(-12,-13,-10,-11),order=(5,1,3,6,9,2,4,7,8))
        H_eff_centre = ncon([centreEnv,outerContract],((1,-2,3,-4),(1,3)),forder=(-2,-4)) + ncon([centreEnv,outerContract],((-1,2,-3,4),(2,4)),forder=(-1,-3))
        #right half of plane (shifted)
        env =  ncon([self.H,self.psi.mpo,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo.conj(),outerContract,outerContract],((3,4,1,2),(1,10,5,6),(2,11,6,-7),(3,12,5,9),(4,13,9,-8),(12,10),(13,11)),forder=(-8,-7),order=(5,10,1,3,12,6,9,11,2,4,13))
        env = self.psi.Tb_inv.applyRight(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        H_eff_shifted = ncon([env,self.psi.mpo,self.psi.mpo.conj(),self.psi.R.tensor],((6,4),(2,-1,4,5),(2,-3,6,7),(7,5)),forder=(-3,-1),order=(4,6,2,5,7))
        return oneBodyH(H_eff_centre + H_eff_shifted)

class gradImplementation_mpso_2d_mps_uniform_twoBodyH_vert(gradImplementation_mpso_2d_mps_uniform):
    def getEffectiveH(self):
        H_eff_centre =  ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.RR.tensor],((1,-11,5,6),(2,-12,8,9),(3,4,1,2),(3,-13,5,7),(4,-14,8,10),(7,6,10,9)),forder=(-13,-14,-11,-12),order=(5,1,3,6,7,9,10,2,4,8))

        env = ncon([self.H,self.psi.mps,self.psi.mps,self.psi.mpo,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mps.conj(),self.psi.mps.conj(),self.psi.T.tensor],((3,7,2,6),(1,9,10),(5,10,11),(2,1,14,-15),(6,5,17,-18),(3,4,14,-16),(7,8,17,-19),(4,9,13),(8,13,12),(12,11)),forder=(-16,-15,-19,-18),order=(9,10,13,11,12,1,14,2,3,4,5,17,6,7,8))
        env = self.psi.Tb2_inv.applyRight(env.reshape(self.psi.D_mpo**4)).reshape(self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo)
        H_eff_shifted = ncon([self.psi.mpo,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo.conj(),env,self.psi.RR.tensor],((2,-1,7,9),(5,-4,11,13),(2,-3,8,10),(5,-6,12,14),(8,7,12,11),(10,9,14,13)),forder=(-3,-6,-1,-4),order=(13,14,5,9,10,2,8,7,12,11))

        physDim = H_eff_shifted.shape[0]
        return twoBodyH((H_eff_centre + H_eff_shifted).reshape(physDim**2,physDim**2))

class gradImplementation_mpso_2d_mps_uniform_plaquetteH(gradImplementation_mpso_2d_mps_uniform):
    def getEffectiveH(self):
        outerContractDouble = ncon([self.psi.mps,self.psi.mps,self.psi.mps.conj(),self.psi.mps.conj(),self.psi.T.tensor],((-1,5,6),(-2,6,7),(-3,5,9),(-4,9,8),(8,7)),forder=(-3,-4,-1,-2))
        H_eff =  ncon([self.psi.mpo,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo.conj(),self.H,self.psi.RR.tensor,outerContractDouble],((2,-1,17,18),(6,5,18,19),(3,-4,17,21),(7,8,21,20),(10,-9,22,23),(14,13,23,24),(11,-12,22,26),(15,16,26,25),(11,3,15,7,10,2,14,6),(25,24,20,19),(16,8,13,5)),forder=(-12,-4,-9,-1),order=(17,2,3,18,21,5,8,6,7,19,20,24,25,13,16,14,15,23,26,10,11,22))
        H_eff +=  ncon([self.psi.mpo,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo.conj(),self.H,self.psi.RR.tensor,outerContractDouble],((2,1,17,18),(6,-5,18,19),(3,4,17,21),(7,-8,21,20),(10,9,22,23),(14,-13,23,24),(11,12,22,26),(15,-16,26,25),(11,3,15,7,10,2,14,6),(25,24,20,19),(12,4,9,1)),forder=(-16,-8,-13,-5),order=(17,1,4,2,3,18,21,6,7,19,20,24,25,14,15,23,26,9,12,10,11,22))

        env = ncon([self.psi.mpo,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo.conj(),self.H,outerContractDouble,outerContractDouble],((2,1,17,18),(6,5,18,-19),(3,4,17,21),(7,8,21,-20),(10,9,22,23),(14,13,23,-24),(11,12,22,26),(15,16,26,-25),(11,3,15,7,10,2,14,6),(12,4,9,1),(16,8,13,5)),forder=(-25,-24,-20,-19),order=(17,22,1,4,9,12,2,3,10,11,18,21,23,26,5,8,13,16,6,7,14,15))
        env = self.psi.Tb2_inv.applyRight(env.reshape(self.psi.D_mpo**4)).reshape(self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo)
        H_eff += ncon([self.psi.mpo,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo.conj(),env,self.psi.RR.tensor],((2,-1,7,9),(5,-4,11,13),(2,-3,8,10),(5,-6,12,14),(8,7,12,11),(10,9,14,13)),forder=(-3,-6,-1,-4),order=(13,14,5,9,10,2,8,7,12,11))

        physDim = H_eff.shape[0]
        return twoBodyH(H_eff.reshape(physDim**2,physDim**2))

# -----------------------------
class gradImplementation_mpso_2d_mps_twoSite_square_oneBodyH(gradImplementation_mpso_2d_mps_uniform):
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

class gradImplementation_mpso_2d_mps_twoSite_square_twoBodyH_hori(gradImplementation_mpso_2d_mps_uniform):
    def getEffectiveH(self,site):
        if site == 1:
            outerContract= ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((-1,3,4,5),(-2,3,4,6),(6,5)),forder=(-2,-1),order=(4,3,5,6))
            tensorLabel = 'bot'
        elif site == 2:
            outerContract= ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((1,-2,4,5),(1,-3,4,6),(6,5)),forder=(-3,-2),order=(4,1,5,6))
            tensorLabel = 'top'

        innerContract_centre  = ncon([self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.R[tensorLabel].tensor],((2,6,-1,-5,9,10),(3,7,2,6),(3,7,-4,-8,9,11),(11,10)),forder=(-4,-8,-1,-5),order=(9,2,3,6,7,10,11))
        H_eff = ncon([innerContract_centre,outerContract],((-1,2,-3,4),(2,4)),forder=(-1,-3))
        H_eff += ncon([innerContract_centre,outerContract],((1,-2,3,-4),(1,3)),forder=(-2,-4))

        innerContract_offCentre  = ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.R[tensorLabel].tensor],((2,5,-1,-4,15,16),(9,13,-8,-12,16,17),(6,10,5,9),(2,6,-3,-7,15,19),(10,13,-11,-14,19,18),(18,17)),forder=(-3,-7,-11,-14,-1,-4,-8,-12),order=(15,2,5,6,16,19,9,10,13,17,18))
        H_eff += ncon([innerContract_offCentre,outerContract,outerContract,outerContract],((-1,2,3,4,-5,6,7,8),(2,6),(3,7),(4,8)),forder=(-1,-5))
        H_eff += ncon([innerContract_offCentre,outerContract,outerContract,outerContract],((1,-2,3,4,5,-6,7,8),(1,5),(3,7),(4,8)),forder=(-2,-6))
        H_eff += ncon([innerContract_offCentre,outerContract,outerContract,outerContract],((1,2,-3,4,5,6,-7,8),(1,5),(2,6),(4,8)),forder=(-3,-7))
        H_eff += ncon([innerContract_offCentre,outerContract,outerContract,outerContract],((1,2,3,-4,5,6,7,-8),(1,5),(2,6),(3,7)),forder=(-4,-8))

        leftEnv = ncon([self.psi.mpo,self.H,self.psi.mpo.conj(),outerContract,outerContract],((2,6,1,5,9,-10),(3,7,2,6),(3,7,4,8,9,-11),(4,1),(8,5)),forder=(-11,-10),order=(9,1,2,3,4,5,6,7,8))
        leftEnv += ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.mpo.conj(),outerContract,outerContract,outerContract,outerContract],((2,5,1,4,15,16),(9,13,8,12,16,-17),(6,10,5,9),(2,6,3,7,15,19),(10,13,11,14,19,-18),(3,1),(7,4),(11,8),(14,12)),forder=(-18,-17),order=(15,1,2,3,4,5,6,7,16,19,8,9,10,11,12,13,14))
        leftEnv = self.psi.Tb_inv[tensorLabel].applyRight(leftEnv.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        innerContract_centre  = ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj(),self.psi.R[tensorLabel].tensor],((9,7),(2,5,-1,-4,7,8),(2,5,-3,-6,9,10),(10,8)),forder=(-3,-6,-1,-4),order=(7,9,2,5,8,10))
        H_eff += ncon([innerContract_centre,outerContract],((-1,2,-3,4),(2,4)),forder=(-1,-3))
        H_eff += ncon([innerContract_centre,outerContract],((1,-2,3,-4),(1,3)),forder=(-2,-4))

        return oneBodyH(1/2*H_eff)

class gradImplementation_mpso_2d_mps_twoSite_square_twoBodyH_vert(gradImplementation_mpso_2d_mps_uniform):
    def getEffectiveH(self,site):
        if site == 1:
            outerContract= ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((-1,-3,5,6),(-2,-4,5,7),(7,6)),forder=(-2,-4,-1,-3),order=(5,6,7))
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
        physDim = H_eff.shape[0]
        return twoBodyH(1/2*H_eff.reshape(physDim**2,physDim**2))

# -----------------------------
class gradImplementation_mpso_2d_mps_twoSite_staircase_oneBodyH(gradImplementation_mpso_2d_mps_uniform):
    def getEffectiveH(self,site):
        pass

class gradImplementation_mpso_2d_mps_twoSite_staircase_twoBodyH_hori(gradImplementation_mpso_2d_mps_uniform):
    def getEffectiveH(self,site):
        outerContract = dict()
        outerContract['bot']= ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((-1,3,4,5),(-2,3,4,6),(6,5)),forder=(-2,-1),order=(4,3,5,6))
        outerContract['top']= ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((1,-2,4,5),(1,-3,4,6),(6,5)),forder=(-3,-2),order=(4,1,5,6))

        innerContract_centre = ncon([self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.R.tensor],((2,6,-1,-5,9,10),(3,7,2,6),(3,7,-4,-8,9,11),(11,10)),forder=(-4,-8,-1,-5),order=(9,2,3,6,7,10,11))
        innerContract_offCentre = ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.R.tensor],((2,5,-1,-4,15,16),(9,13,-8,-12,16,17),(6,10,5,9),(2,6,-3,-7,15,19),(10,13,-11,-14,19,18),(18,17)),forder=(-3,-7,-11,-14,-1,-4,-8,-12),order=(15,2,5,6,16,19,9,10,13,18,17))

        leftEnv = ncon([self.psi.mpo,self.H,self.psi.mpo.conj(),outerContract['bot'],outerContract['top']],((2,6,1,5,9,-10),(3,7,2,6),(3,7,4,8,9,-11),(4,1),(8,5)),forder=(-11,-10),order=(9,1,4,2,3,5,8,6,7))
        leftEnv += ncon([self.psi.mpo,self.psi.mpo.conj(),self.H,self.psi.mpo,self.psi.mpo.conj(),outerContract['bot'],outerContract['top'],outerContract['bot'],outerContract['top']],((2,5,1,4,15,16),(2,6,3,7,15,19),(6,10,5,9),(9,13,8,12,16,-17),(10,13,11,14,19,-18),(3,1),(7,4),(11,8),(14,12)),forder=(-18,-17),order=(15,2,1,3,4,7,5,6,16,19,8,11,9,10,13,12,14))
        leftEnv = self.psi.Tb_inv.applyRight(leftEnv.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        innerContract_right = ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj(),self.psi.R.tensor],((10,7),(2,5,-1,-4,7,8),(2,5,-3,-6,10,9),(9,8)),forder=(-3,-6,-1,-4),order=(10,7,2,5,8,9))

        if site == 1:
            H_eff = ncon([innerContract_centre,outerContract['top']],((-1,2,-3,4),(2,4)),forder=(-1,-3))
            H_eff += ncon([innerContract_right,outerContract['top']],((-1,2,-3,4),(2,4)),forder=(-1,-3))
            H_eff += ncon([innerContract_offCentre,outerContract['top'],outerContract['bot'],outerContract['top']],((-1,2,3,4,-5,6,7,8),(2,6),(3,7),(4,8)),forder=(-1,-5),order=(2,6,3,7,4,8))
            H_eff += ncon([innerContract_offCentre,outerContract['bot'],outerContract['top'],outerContract['top']],((1,2,-3,4,5,6,-7,8),(1,5),(2,6),(4,8)),forder=(-3,-7),order=(1,5,2,6,4,8))
        elif site == 2:
            H_eff = ncon([innerContract_centre,outerContract['bot']],((1,-2,3,-4),(1,3)),forder=(-2,-4))
            H_eff += ncon([innerContract_right,outerContract['bot']],((1,-2,3,-4),(1,3)),forder=(-2,-4))
            H_eff += ncon([innerContract_offCentre,outerContract['bot'],outerContract['bot'],outerContract['top']],((1,-2,3,4,5,-6,7,8),(1,5),(3,7),(4,8)),forder=(-2,-6),order=(1,5,3,7,4,8))
            H_eff += ncon([innerContract_offCentre,outerContract['bot'],outerContract['top'],outerContract['bot']],((1,2,3,-4,5,6,7,-8),(1,5),(2,6),(3,7)),forder=(-4,-8),order=(1,5,2,6,3,7))
        
        return oneBodyH(H_eff)


class gradImplementation_mpso_2d_mps_twoSite_staircase_twoBodyH_vert(gradImplementation_mpso_2d_mps_uniform):
    def getEffectiveH(self,site):
        outerContract = dict()
        outerContractDouble = dict()
        outerContract['bot']= ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((-1,3,4,5),(-2,3,4,6),(6,5)),forder=(-2,-1),order=(4,3,5,6))
        outerContract['top']= ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((1,-2,4,5),(1,-3,4,6),(6,5)),forder=(-3,-2),order=(4,1,5,6))
        outerContractDouble['square'] = ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((-1,-3,5,6),(-2,-4,5,7),(7,6)),forder=(-2,-4,-1,-3),order=(5,6,7))
        outerContractDouble['prong'] = ncon([self.psi.mps,self.psi.mps,self.psi.mps.conj(),self.psi.mps.conj(),self.psi.T.tensor],((1,-2,7,8),(-4,6,8,9),(1,-3,7,11),(-5,6,11,10),(10,9)),forder=(-3,-5,-2,-4),order=(7,1,8,11,6,9,10))

        if site == 1:
            H_eff = ncon([self.psi.mpo,self.psi.mpo.conj(),self.H,self.psi.mpo,self.psi.mpo.conj(),outerContract['bot'],self.psi.RR.tensor],((2,5,1,-4,15,16),(2,6,3,-7,15,17),(10,6,9,5),(9,13,-8,12,18,19),(10,13,-11,14,18,20),(3,1),(14,12,20,19,17,16)),forder=(-11,-7,-8,-4),order=(15,2,1,3,5,6,16,17,19,20,13,12,14,9,10,18))
            H_eff += ncon([self.psi.mpo,self.psi.mpo.conj(),self.H,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo.conj(),outerContractDouble['prong'],outerContractDouble['square'],self.psi.RR.tensor,np.eye(2)],((2,5,-1,4,21,22),(2,6,-3,7,21,23),(6,10,5,9),(9,13,8,12,24,25),(10,13,11,14,24,26),(16,19,15,18,22,27),(16,19,17,20,23,28),(7,11,4,8),(17,14,15,12),(20,18,28,27,26,25),(-29,-30)),forder=(-3,-29,-1,-30),order=[28,27,18,20,19,15,16,17,25,26,12,13,14,8,9,10,11,24,22,23,4,5,6,7,2,21])
            H_eff += ncon([self.psi.mpo,self.psi.mpo.conj(),self.H,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo.conj(),outerContract['bot'],outerContractDouble['prong'],self.psi.RR.tensor],((2,5,1,4,21,22),(2,6,3,7,21,23),(6,10,5,9),(9,13,8,-12,24,25),(10,13,11,-14,24,26),(16,19,-15,18,22,27),(16,19,-17,20,23,28),(3,1),(7,11,4,8),(20,18,28,27,26,25)),forder=(-17,-14,-15,-12),order=[21,2,1,3,4,5,6,7,22,23,16,19,27,28,18,20,25,26,13,8,9,10,11,24])

        elif site == 2:
            H_eff = ncon([self.psi.mpo,self.psi.mpo.conj(),self.H,self.psi.mpo,self.psi.mpo.conj(),outerContractDouble['square'],self.psi.RR.tensor,np.eye(2)],((2,5,-1,4,15,16),(2,6,-3,7,15,17),(10,6,9,5),(9,13,8,12,18,19),(10,13,11,14,18,20),(11,7,8,4),(14,12,20,19,17,16),(-21,-22)),forder=(-21,-3,-22,-1))
            H_eff += ncon([self.psi.mpo,self.psi.mpo.conj(),self.H,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo.conj(),outerContract['bot'],outerContractDouble['square'],self.psi.RR.tensor],((2,5,1,-4,21,22),(2,6,3,-7,21,23),(6,10,5,9),(9,13,-8,12,24,25),(10,13,-11,14,24,26),(16,19,15,18,22,27),(16,19,17,20,23,28),(3,1),(17,14,15,12),(20,18,28,27,26,25)),forder=(-7,-11,-4,-8),order=(28,27,18,19,20,15,16,17,22,23,5,6,1,2,3,21,25,26,13,12,14,9,10,24))

        leftEnv = ncon([self.psi.mpo,self.psi.mpo.conj(),self.H,self.psi.mpo,self.psi.mpo.conj(),outerContract['bot'],outerContractDouble['square']],((2,5,1,4,15,-16),(2,6,3,7,15,-17),(10,6,9,5),(9,13,8,-12,18,-19),(10,13,11,-14,18,-20),(3,1),(11,7,8,4)),forder=(-14,-12,-20,-19,-17,-16),order=(15,2,1,3,4,5,6,18,13,8,9,10,11))
        leftEnv += ncon([self.psi.mpo,self.psi.mpo.conj(),self.H,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo.conj(),outerContract['bot'],outerContractDouble['prong'],outerContractDouble['square']],((2,5,1,4,21,22),(2,6,3,7,21,23),(6,10,5,9),(9,13,8,12,24,-25),(10,13,11,14,24,-26),(16,19,15,-18,22,-28),(16,19,17,-20,23,-27),(3,1),(7,11,4,8),(17,14,15,12)),forder=(-20,-18,-27,-28,-26,-25),order=(21,2,1,3,4,5,6,7,22,23,16,19,15,17,24,9,10,13,8,11,12,14))
        leftEnv = self.psi.Tb2_inv.applyRight(leftEnv.reshape(self.psi.D_mpo**4*4)).reshape(2,2,self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo)
        if site == 1:
            H_eff += ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo.conj(),outerContractDouble['prong'],self.psi.RR.tensor],((21,22,16,15,14,13),(2,5,1,-4,13,17),(2,5,3,-6,14,18),(8,11,-7,10,15,19),(8,11,-9,12,16,20),(21,3,22,1),(12,10,20,19,18,17)),forder=(-9,-6,-7,-4))
        elif site == 2:
            H_eff += ncon([leftEnv,self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo.conj(),outerContractDouble['square'],self.psi.RR.tensor],((-21,-22,16,15,14,13),(2,5,-1,4,13,17),(2,5,-3,6,14,18),(8,11,7,10,15,19),(8,11,9,12,16,20),(9,6,7,4),(12,10,20,19,18,17)),forder=(-21,-3,-22,-1),order=(20,19,10,11,12,7,8,9,18,17,4,5,6,2,13,14,15,16))

        physDim = H_eff.shape[0]
        return twoBodyH(H_eff.reshape(physDim**2,physDim**2))

# -----------------------------
#mpso_2d bipartite and fourSite_sep ansatz (same code, different index mapping)
class gradImplementation_mpso_2d_mps_multipleTensors(gradImplementation_mpso_2d_mps_uniform):
    def __init__(self,psi,H,indexSetter):
        self.indexSetter = indexSetter
        super().__init__(psi,H)

def mpso_2d_mps_fourSite_sep_indexSetter(index):
    if index == 1:
        index1,index2,index3,index4 = 1,2,3,4
    elif index == 2:
        index1,index2,index3,index4 = 2,1,4,3
    elif index == 3:
        index1,index2,index3,index4 = 3,4,1,2
    elif index == 4:
        index1,index2,index3,index4 = 4,3,2,1
    return index1,index2,index3,index4

def mpso_2d_mps_bipartite_indexSetter(index):
    if index == 1:
        index1,index2,index3,index4 = 1,2,2,1
    elif index == 2:
        index1,index2,index3,index4 = 2,1,1,2
    return index1,index2,index3,index4

class gradImplementation_mpso_2d_mps_multipleTensors_oneBodyH(gradImplementation_mpso_2d_mps_multipleTensors):
    def getEffectiveH(self,index):
        index1,index2,index3,index4 = self.indexSetter(index)
        outerContract_1 =  ncon([self.psi.mps[index1],self.psi.mps[index1].conj(),self.psi.T[index3].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
        outerContract_2 =  ncon([self.psi.mps[index2],self.psi.mps[index2].conj(),self.psi.T[index4].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))

        #under current Hamiltonian
        H_eff_centre = ncon([self.psi.mpo[index1],self.H,self.psi.mpo[index1].conj(),self.psi.R[index2].tensor],((2,-1,5,6),(3,2),(3,-4,5,8),(8,6)),forder=(-4,-1),order=(5,2,3,6,8))

        #right half of plane (shifted)
        env_1 = ncon([self.psi.mpo[index1],self.H,self.psi.mpo[index1].conj(),outerContract_1],((2,1,5,-6),(3,2),(3,4,5,-7),(4,1)),forder=(-7,-6),order=(5,1,2,3,4))
        transfer = mpsu1Transfer_left_oneLayer(self.psi.mps[index2],self.psi.mpo[index2],self.psi.T[index4])
        env_1 = transfer.applyRight(env_1.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)

        env_2 = ncon([self.psi.mpo[index2],self.H,self.psi.mpo[index2].conj(),outerContract_2],((2,1,5,-6),(3,2),(3,4,5,-7),(4,1)),forder=(-7,-6),order=(5,1,2,3,4))

        env = env_1 + env_2
        env = self.psi.Tb_inv[index1].applyRight(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        H_eff_shifted = ncon([env,self.psi.mpo[index1],self.psi.mpo[index1].conj(),self.psi.R[index2].tensor],((6,4),(2,-1,4,5),(2,-3,6,7),(7,5)),forder=(-3,-1),order=(4,6,2,5,7))
        return oneBodyH(H_eff_centre + H_eff_shifted)

class gradImplementation_mpso_2d_mps_multipleTensors_twoBodyH_hori(gradImplementation_mpso_2d_mps_multipleTensors):
    def getEffectiveH(self,index):
        index1,index2,index3,index4 = self.indexSetter(index)
        outerContract_1 =  ncon([self.psi.mps[index1],self.psi.mps[index1].conj(),self.psi.T[index3].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
        outerContract_2 =  ncon([self.psi.mps[index2],self.psi.mps[index2].conj(),self.psi.T[index4].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
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

class gradImplementation_mpso_2d_mps_multipleTensors_twoBodyH_vert(gradImplementation_mpso_2d_mps_multipleTensors):
    def getEffectiveH(self,index):
        #under current Hamiltonian
        index1,index2,index3,index4 = self.indexSetter(index)
        H_eff_centre = ncon([self.psi.mpo[index1],self.psi.mpo[index3],self.H,self.psi.mpo[index1].conj(),self.psi.mpo[index3].conj(),self.psi.RR[index2].tensor],((1,-11,5,6),(2,-12,8,9),(3,4,1,2),(3,-13,5,7),(4,-14,8,10),(7,6,10,9)),forder=(-13,-14,-11,-12),order=(6,7,10,9,5,8,1,2,3,4))

        #right half of plane (shifted)
        env_1 = ncon([self.H,self.psi.mps[index1],self.psi.mps[index3],self.psi.mpo[index1],self.psi.mpo[index3],self.psi.mpo[index1].conj(),self.psi.mpo[index3].conj(),self.psi.mps[index1].conj(),self.psi.mps[index3].conj(),self.psi.T[index1].tensor],((3,7,2,6),(1,9,10),(5,10,11),(2,1,14,-15),(6,5,17,-18),(3,4,14,-16),(7,8,17,-19),(4,9,13),(8,13,12),(12,11)),forder=(-16,-15,-19,-18),order=(9,10,13,11,12,1,14,2,3,4,5,17,6,7,8))

        transfer = ncon([self.psi.mps[index2],self.psi.mps[index4],self.psi.mpo[index2],self.psi.mpo[index4],self.psi.mpo[index2].conj(),self.psi.mpo[index4].conj(),self.psi.mps[index2].conj(),self.psi.mps[index4].conj(),self.psi.T[index2].tensor],((1,7,8),(4,8,9),(2,1,-12,-14),(5,4,-16,-18),(2,3,-13,-15),(5,6,-17,-19),(3,7,11),(6,11,10),(10,9)),forder=(-13,-12,-17,-16,-15,-14,-19,-18),order=(7,8,9,10,11,1,2,3,4,5,6)).reshape(self.psi.D_mpo**4,self.psi.D_mpo**4)
        env_1 = np.dot(env_1.reshape(self.psi.D_mpo**4),transfer).reshape(self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo)
        env_2 = ncon([self.H,self.psi.mps[index2],self.psi.mps[index4],self.psi.mpo[index2],self.psi.mpo[index4],self.psi.mpo[index2].conj(),self.psi.mpo[index4].conj(),self.psi.mps[index2].conj(),self.psi.mps[index4].conj(),self.psi.T[index2].tensor],((3,7,2,6),(1,9,10),(5,10,11),(2,1,14,-15),(6,5,17,-18),(3,4,14,-16),(7,8,17,-19),(4,9,13),(8,13,12),(12,11)),forder=(-16,-15,-19,-18),order=(9,10,13,11,12,1,14,2,3,4,5,17,6,7,8))

        env = env_1 + env_2
        env = self.psi.Tb2_inv[index1].applyRight(env.reshape(self.psi.D_mpo**4)).reshape(self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo)
        H_eff_shifted = ncon([env,self.psi.mpo[index3],self.psi.mpo[index3].conj(),self.psi.mpo[index1],self.psi.mpo[index1].conj(),self.psi.RR[index2].tensor],((13,11,9,7),(2,-1,7,8),(2,-3,9,10),(5,-4,11,12),(5,-6,13,14),(14,12,10,8)),forder=(-6,-3,-4,-1),order=(7,9,2,11,13,5,8,10,12,14))
        physDim = H_eff_shifted.shape[0]
        return twoBodyH((H_eff_centre + H_eff_shifted).reshape(physDim**2,physDim**2))
# -----------------------------
