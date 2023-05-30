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

def gradFactory(psi,H):
    if type(psi) == uMPS_1d:
        return gradEvaluater_uniform_1d_oneSite(psi,H)
    elif type(psi) == uMPS_1d_left:
        return gradEvaluater_uniform_1d_oneSiteLeft(psi,H)
    elif type(psi) == uMPSU1_2d_left:
        return gradEvaluater_mpso_2d_uniform(psi,H)
    elif type(psi) == uMPSU1_2d_left_bipartite:
        return gradEvaluater_mpso_2d_bipartite(psi,H)
    elif type(psi) == uMPS_1d_left_bipartite:
        return gradEvaluater_bipartite_1d_left(psi,H)
    elif type(psi) == uMPS_1d_left_twoSite:
        return gradEvaluater_uniform_1d_twoSiteLeft(psi,H)

class gradEvaluater(ABC):
    def __init__(self,psi,H):
        self.psi = psi
        self.H = H
        self.H_imp = dict()
        for n in range(0,len(self.H.terms)):
            self.H_imp[n] = self.fetch_implementation(self.H.terms[n])

    @abstractmethod
    def eval(self):
        pass
    @abstractmethod
    def fetch_implementation(self):
        pass
    @abstractmethod
    def projectTDVP(self):
        pass

class gradEvaluater_uniform_1d(gradEvaluater):
    @abstractmethod
    def attachRight(self):
        pass
    @abstractmethod
    def attachLeft(self):
        pass

    def eval(self):
        self.grad = self.H_imp[0].getCentralTerms()
        for n in range(1,len(self.H.terms)):
            self.grad += self.H_imp[n].getCentralTerms()

        leftEnv = self.H_imp[0].buildLeftEnv()
        for n in range(1,len(self.H.terms)):
            leftEnv += self.H_imp[n].buildLeftEnv()
        self.grad += self.attachRight(leftEnv)

        rightEnv = self.H_imp[0].buildRightEnv()
        for n in range(1,len(self.H.terms)):
            rightEnv += self.H_imp[n].buildRightEnv()
        self.grad += self.attachLeft(rightEnv)

    def projectTDVP(self):
        self.grad = project_mpsTangentVector(self.grad,self.psi.mps,self.psi.R)

class gradEvaluater_mpso_2d_uniform(gradEvaluater):
    def fetch_implementation(self):
        pass
    def __init__(self,psi,H):
        self.psi = psi
        self.H = H
        self.gradA_evaluater = gradEvaluater_mpso_2d_mps_uniform(psi,H)
        self.gradB_evaluater = gradEvaluater_mpso_2d_mpo_uniform(psi,H)
    def eval(self,geo=True):
        self.gradA_evaluater.eval()
        self.gradB_evaluater.eval(geo=geo)
        self.grad = dict()
        self.grad['mps'] = self.gradA_evaluater.grad
        self.grad['mpo'] = self.gradB_evaluater.grad
    def projectTDVP(self):
        self.gradA_evaluater.projectTDVP()
        self.gradB_evaluater.projectTDVP()
        self.grad['mps'] = self.gradA_evaluater.grad
        self.grad['mpo'] = self.gradB_evaluater.grad

#temporary placeholder
class gradEvaluater_mpso_2d_bipartite(gradEvaluater):
    def __init__(self,psi,H):
        self.psi = psi
        self.H = H
    def fetch_implementation(self):
        pass
    def eval(self):
        pass
    def projectTDVP(self):
        pass

class gradEvaluater_mpso_2d_mps_uniform(gradEvaluater):
    #gradient of the mps tensor of 2d uniform MPSO ansatz
    #this may be constructed as the gradient of a uniform mps in 1d with an effective Hamiltonian
    #so just find effective Hamiltonians, then reuse gradEvaluater_uniform_1d to compute gradient terms
    def eval(self):
        effH= []
        for n in range(0,len(self.H.terms)):
            effH.append(self.H_imp[n].getEffectiveH())
        effH = localH(effH)
        #hacky
        eff_psi = copy.deepcopy(self.psi)
        eff_psi.D = self.psi.D_mps
        eff_psi.R = self.psi.T #abuse syntax of fixed points to reuse gradEvaluater_uniform_1d code (bad code..)

        eff_gradEval = gradEvaluater_uniform_1d_oneSiteLeft(eff_psi,effH)
        eff_gradEval.eval()
        self.grad = eff_gradEval.grad
    def projectTDVP(self):
        self.grad = project_mpsTangentVector(self.grad,self.psi.mps,self.psi.T)

class gradEvaluater_mpso_2d_mpo_uniform(gradEvaluater):
    @abstractmethod
    def attachBot(self):
        pass
    @abstractmethod
    def attachTop(self):
        pass
    def projectTDVP(self):
        self.grad = project_mpoTangentVector(self.grad,self.psi.mps,self.psi.mpo,self.psi.T,self.psi.R)
    def eval(self,geo=True,envTol=1e-5):
        #terms underneath Hamiltonian
        self.grad = np.zeros(self.psi.mpo.shape).astype(complex)
        for n in range(0,len(self.H.terms)):
            #terms under Hamiltonian
            self.grad += self.H_imp[n].getCentralTerms()
            #terms to left of H
            rightEnv = self.H_imp[n].buildRightEnv()
            self.grad += self.H_imp[n].attachLeftMax(rightEnv)
            #terms to right of H
            leftEnv = self.H_imp[n].buildLeftEnv()
            self.grad += self.H_imp[n].attachRightMax(leftEnv)

            # geometric sums...
            if geo is True:
                Td_matrix = np.eye(self.psi.D_mps**2)
                Td= Td_matrix.reshape(self.psi.D_mps,self.psi.D_mps,self.psi.D_mps,self.psi.D_mps)
                #tensors needed for geosum not dependent on d
                leftEnv_h_tilde = self.H_imp[n].buildLeftEnv(H=self.H_imp[n].h_tilde)
                for d in range(0,100):
                    gradRun = np.zeros(self.psi.mpo.shape).astype(complex)
                    #d dependant tensors needed
                    if d > 0:
                        Td_matrix = np.dot(Td_matrix,self.psi.Ta.matrix)
                        Td= Td_matrix.reshape(self.psi.D_mps,self.psi.D_mps,self.psi.D_mps,self.psi.D_mps)
                    rightFP_d = self.H_imp[n].getFixedPoints(d,Td) #bottleneck of algo
                    outers_d = self.H_imp[n].getOuterContracts(Td)

                    # terms below Hamiltonian
                    env = self.H_imp[n].buildTopEnvGeo(rightFP_d,outers_d)
                    # right quadrant lower
                    env += self.H_imp[n].buildTopEnvGeo_quadrants(rightFP_d,outers_d,leftEnv_h_tilde)
                    gradRun += self.attachBot(env)

                    # terms above Hamiltonian
                    env = self.H_imp[n].buildBotEnvGeo(rightFP_d,outers_d)
                    # right quadrant upper
                    env += self.H_imp[n].buildBotEnvGeo_quadrants(rightFP_d,outers_d,leftEnv_h_tilde)
                    gradRun += self.attachTop(env)

                    #left quadrants upper and lower
                    rightEnv = self.H_imp[n].buildRightEnvGeo_quadrants(rightFP_d,outers_d)
                    gradRun += self.H_imp[n].attachLeftSingle(rightEnv)

                    #check geometric sum decayed
                    mag = np.einsum('ijab,ijab',gradRun,gradRun.conj())
                    self.grad += gradRun
                    if np.abs(mag)<envTol:
                        break
                print("d: ",d,mag)
        self.grad = np.einsum('ijab->jiab',self.grad)

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

class gradImplementation_bipartite_1d(ABC):
    def __init__(self,psi,H,index1,index2):
        self.psi = psi
        self.H = H
        self.index1 = index1
        self.index2 = index2

    @abstractmethod
    def buildLeftEnv(self):
        pass
    @abstractmethod
    def buildRightEnv(self):
        pass
    @abstractmethod
    def getCentralTerms(self):
        pass


class gradImplementation_mpso_2d_mps_uniform(ABC):
    def __init__(self,psi,H):
        self.psi = psi
        self.H = H

    @abstractmethod
    def getEffectiveH(self):
        #gets effective 1d Hamiltonians to compute mps gradient of 2d term
        #consists of two terms, effH_centre + effH_shifted

        #1d grad under effH_centre is gradient terms under Hamiltonian in 2d
        #1d grad to left and right of effH_centre are grad terms in 2d below and above Hamiltonian respectively

        #1d grad under effH_shifted is gradient in line of H to the right in 2d
        #1d grad to left and right of effH_shifted are grad terms in the right lower and upper quadrants respectively
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

class gradEvaluater_mpso_2d_mpo_uniform(gradEvaluater_mpso_2d_mpo_uniform):
    def fetch_implementation(self,H):
        if type(H) == oneBodyH:
            return gradImplementation_mpso_2d_mpo_uniform_oneBodyH(self.psi,H.tensor)
        elif type(H) == twoBodyH_hori:
            return gradImplementation_mpso_2d_mpo_uniform_twoBodyH_hori(self.psi,H.tensor)
        elif type(H) == twoBodyH_vert:
            return gradImplementation_mpso_2d_mpo_uniform_twoBodyH_vert(self.psi,H.tensor)

    def attachTop(self,env):
        return ncon([env,self.psi.mps,self.psi.mpo,self.psi.mps.conj(),self.psi.T.tensor],((9,-8,7,4),(1,4,5),(-2,1,-6,7),(-3,9,10),(10,5)),forder=(-3,-2,-6,-8),order=(4,9,5,10,1,7))

    def attachBot(self,env):
        return ncon([env,self.psi.mps,self.psi.mpo,self.psi.mps.conj()],((9,-8,7,5),(1,4,5),(-2,1,-6,7),(-3,4,9)),forder=(-3,-2,-6,-8),order=(9,5,4,7,1))

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

# -------------------------------------------------------------------------------------------------------------------------------------

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

# -------------------------------------------------------------------------------------------------------------------------------------

class gradEvaluater_mpso_2d_mps_uniform(gradEvaluater_mpso_2d_mps_uniform):
    def fetch_implementation(self,H):
        if type(H) == oneBodyH:
            return gradImplementation_mpso_2d_mps_uniform_oneBodyH(self.psi,H.tensor)
        elif type(H) == twoBodyH_hori:
            return gradImplementation_mpso_2d_mps_uniform_twoBodyH_hori(self.psi,H.tensor)
        elif type(H) == twoBodyH_vert:
            return gradImplementation_mpso_2d_mps_uniform_twoBodyH_vert(self.psi,H.tensor)

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

# -------------------------------------------------------------------------------------------------------------------------------------

class gradEvaluater_uniform_1d_oneSite(gradEvaluater_uniform_1d):
    def fetch_implementation(self,H):
        if type(H) == oneBodyH:
            return gradImplementation_uniform_1d_oneSite_oneBodyH(self.psi,H.tensor)
        elif type(H) == twoBodyH or type(H) == twoBodyH_hori or type(H) == twoBodyH_vert:
            return gradImplementation_uniform_1d_oneSite_twoBodyH(self.psi,H.tensor)
            
    def attachRight(self,leftEnv):
        leftEnv = self.psi.Ta_inv.applyRight(leftEnv.reshape(self.psi.D**2)).reshape(self.psi.D,self.psi.D)
        return ncon([self.psi.mps,leftEnv,self.psi.R.tensor],((-1,2,3),(-4,2),(-5,3)),forder=(-1,-4,-5))
    def attachLeft(self,rightEnv):
        rightEnv = self.psi.Ta_inv.applyLeft(rightEnv.reshape(self.psi.D**2)).reshape(self.psi.D,self.psi.D)
        return ncon([self.psi.mps,self.psi.L.tensor,rightEnv],((-1,2,3),(-4,2),(-5,3)),forder=(-1,-4,-5))

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

# -------------------------------------------------------------------------------------------------------------------------------------

class gradEvaluater_uniform_1d_oneSiteLeft(gradEvaluater_uniform_1d):
    def fetch_implementation(self,H):
        if type(H) == oneBodyH:
            return gradImplementation_uniform_1d_oneSiteLeft_oneBodyH(self.psi,H.tensor)
        elif type(H) == twoBodyH or type(H) == twoBodyH_hori or type(H) == twoBodyH_vert:
            return gradImplementation_uniform_1d_oneSiteLeft_twoBodyH(self.psi,H.tensor)

    def attachRight(self,leftEnv):
        leftEnv = self.psi.Ta_inv.applyRight(leftEnv.reshape(self.psi.D**2)).reshape(self.psi.D,self.psi.D)
        return ncon([self.psi.mps,leftEnv,self.psi.R.tensor],((-1,2,3),(-4,2),(-5,3)),forder=(-1,-4,-5))
    def attachLeft(self,rightEnv):
        rightEnv = self.psi.Ta_inv.applyLeft(rightEnv.reshape(self.psi.D**2)).reshape(self.psi.D,self.psi.D)
        return ncon([self.psi.mps,rightEnv],((-1,-2,3),(-4,3)),forder=(-1,-2,-4))

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

# -------------------------------------------------------------------------------------------------------------------------------------
class gradEvaluater_bipartite_1d_left(gradEvaluater):
    #wrapper to find both gradients together for bipartite ansatz
    def __init__(self,psi,H):
        self.psi = psi
        self.H = H
    def eval(self):
        self.grad = dict()
        gradEvaluater1 = gradEvaluater_bipartite_1d_left_ind(self.psi,self.H,1)
        gradEvaluater2 = gradEvaluater_bipartite_1d_left_ind(self.psi,self.H,2)
        gradEvaluater1.eval()
        gradEvaluater2.eval()
        self.grad[1] = gradEvaluater1.grad
        self.grad[2] = gradEvaluater2.grad
    def fetch_implementation(self):
        pass
    def projectTDVP(self):
        self.grad[1] = project_mpsTangentVector(self.grad[1],self.psi.mps[1],self.psi.R[1])
        self.grad[2] = project_mpsTangentVector(self.grad[2],self.psi.mps[2],self.psi.R[2])

class gradEvaluater_bipartite_1d_left_ind(gradEvaluater_uniform_1d):
    #find an individual gradient of bipartite ansatz, d/dA_1 <H> or d/dA_2 <H> (index arg is which gradient)
    #can reuse gradEvaluater_uniform_1d eval code as same structure to find gradient
    def __init__(self,psi,H,index):
        if index == 1:
            self.index1 = 1
            self.index2 = 2
        elif index == 2:
            self.index1 = 2
            self.index2 = 1
        super().__init__(psi,H)

    def projectTDVP(self):
        self.grad = project_mpsTangentVector(self.grad,self.psi.mps[self.index1],self.psi.R[self.index1])

    def fetch_implementation(self,H):
        if type(H) == oneBodyH:
            return gradImplementation_bipartite_1d_left_oneBodyH(self.psi,H.tensor,self.index1,self.index2)
        elif type(H) == twoBodyH or type(H) == twoBodyH_hori or type(H) == twoBodyH_vert:
            return gradImplementation_bipartite_1d_left_twoBodyH(self.psi,H.tensor,self.index1,self.index2)

    def attachRight(self,leftEnv):
        leftEnv = self.psi.Ta_inv[self.index1].applyRight(leftEnv.reshape(self.psi.D**2)).reshape(self.psi.D,self.psi.D)
        return ncon([self.psi.mps[self.index1],leftEnv,self.psi.R[self.index2].tensor],((-1,2,3),(-4,2),(-5,3)),forder=(-1,-4,-5))
    def attachLeft(self,rightEnv):
        rightEnv = self.psi.Ta_inv[self.index2].applyLeft(rightEnv.reshape(self.psi.D**2)).reshape(self.psi.D,self.psi.D)
        return ncon([self.psi.mps[self.index1],rightEnv],((-1,-2,3),(-4,3)),forder=(-1,-2,-4))

class gradImplementation_bipartite_1d_left_oneBodyH(gradImplementation_bipartite_1d):
    def getCentralTerms(self):
        return 1/2*ncon([self.psi.mps[self.index1],self.H,self.psi.R[self.index2].tensor],((1,-3,4),(-2,1),(-5,4)),forder=(-2,-3,-5))
    def buildLeftEnv(self):
        env = ncon([self.psi.mps[self.index2],self.H,self.psi.mps[self.index2].conj()],((1,3,-4),(2,1),(2,3,-5)),forder=(-5,-4),order=(3,1,2))
        env += ncon([self.psi.mps[self.index1],self.H,self.psi.mps[self.index1].conj(),self.psi.mps[self.index2],self.psi.mps[self.index2].conj()],((1,3,4),(2,1),(2,3,5),(6,4,-7),(6,5,-8)),forder=(-8,-7),order=(3,1,2,4,5,6))
        return env/2
    def buildRightEnv(self):
        env = ncon([self.psi.mps[self.index2],self.H,self.psi.mps[self.index2].conj(),self.psi.R[self.index1].tensor],((1,-3,4),(2,1),(2,-6,5),(5,4)),forder=(-6,-3),order=(5,4,1,2))
        env += ncon([self.psi.mps[self.index1],self.H,self.psi.mps[self.index1].conj(),self.psi.R[self.index2].tensor,self.psi.mps[self.index2],self.psi.mps[self.index2].conj()],((1,3,4),(2,1),(2,6,5),(5,4),(7,-9,3),(7,-8,6)),forder=(-8,-9),order=(5,4,1,2,3,6,7))
        return env/2

class gradImplementation_bipartite_1d_left_twoBodyH(gradImplementation_bipartite_1d):
    def getCentralTerms(self):
        grad = ncon([self.psi.mps[self.index1],self.psi.mps[self.index2],self.H,self.psi.mps[self.index2].conj(),self.psi.R[self.index1].tensor],((1,-5,6),(3,6,7),(-2,4,1,3),(4,-9,8),(8,7)),forder=(-2,-5,-9),order=(8,7,3,4,6))
        grad += ncon([self.psi.mps[self.index2],self.psi.mps[self.index1],self.H,self.psi.mps[self.index2].conj(),self.psi.R[self.index2].tensor],((1,5,6),(3,6,7),(2,-4,1,3),(2,5,-9),(-8,7)),forder=(-4,-9,-8),order=(5,1,2,6,7))
        return grad/2
    def buildLeftEnv(self):
        env = ncon([self.psi.mps[self.index1],self.psi.mps[self.index2],self.H,self.psi.mps[self.index1].conj(),self.psi.mps[self.index2].conj()],((1,5,6),(3,6,-7),(2,4,1,3),(2,5,9),(4,9,-8)),forder=(-8,-7),order=(5,1,2,6,9,3,4))
        env += ncon([self.psi.mps[self.index2],self.psi.mps[self.index1],self.H,self.psi.mps[self.index2].conj(),self.psi.mps[self.index1].conj(),self.psi.mps[self.index2],self.psi.mps[self.index2].conj()],((1,5,6),(3,6,7),(2,4,1,3),(2,5,9),(4,9,8),(10,7,-12),(10,8,-11)),forder=(-11,-12),order=(5,1,2,6,9,3,4,7,8,10))
        return env/2
    def buildRightEnv(self):
        env = ncon([self.psi.mps[self.index2],self.psi.mps[self.index1],self.H,self.psi.mps[self.index2].conj(),self.psi.mps[self.index1].conj(),self.psi.R[self.index2].tensor],((1,-5,6),(3,6,7),(2,4,1,3),(2,-10,9),(4,9,8),(8,7)),forder=(-10,-5))
        env += ncon([self.psi.mps[self.index1],self.psi.mps[self.index2],self.H,self.psi.mps[self.index1].conj(),self.psi.mps[self.index2].conj(),self.psi.R[self.index1].tensor,self.psi.mps[self.index2],self.psi.mps[self.index2].conj()],((1,5,6),(3,6,7),(2,4,1,3),(2,10,9),(4,9,8),(8,7),(11,-13,5),(11,-12,10)),forder=(-12,-13))
        return env/2

# -------------------------------------------------------------------------------------------------------------------------------------
class gradEvaluater_uniform_1d_twoSiteLeft(gradEvaluater_uniform_1d):
    def fetch_implementation(self,H):
        if type(H) == oneBodyH:
            return gradImplementation_uniform_1d_twoSiteLeft_oneBodyH(self.psi,H.tensor)
        elif type(H) == twoBodyH or type(H) == twoBodyH_hori or type(H) == twoBodyH_vert:
            return gradImplementation_uniform_1d_twoSiteLeft_twoBodyH(self.psi,H.tensor)

    def attachRight(self,leftEnv):
        leftEnv = self.psi.Ta_inv.applyRight(leftEnv.reshape(self.psi.D**2)).reshape(self.psi.D,self.psi.D)
        return ncon([self.psi.mps,leftEnv,self.psi.R.tensor],((-1,-2,3,4),(-6,3),(-5,4)),forder=(-1,-2,-6,-5))
    def attachLeft(self,rightEnv):
        rightEnv = self.psi.Ta_inv.applyLeft(rightEnv.reshape(self.psi.D**2)).reshape(self.psi.D,self.psi.D)
        return ncon([self.psi.mps,rightEnv],((-1,-2,-3,4),(-5,4)),forder=(-1,-2,-3,-5))

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
