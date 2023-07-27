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
from gradImp_mps_1d import *
from gradImp_mpso_2d_mps import *
from gradImp_mpso_2d_mpo import *
from uniformTN_projectors import *

def gradFactory(psi,H):
    if type(psi) == uMPS_1d:
        return gradEvaluater_uniform_1d_oneSite(psi,H)

    elif type(psi) == uMPS_1d_left:
        return gradEvaluater_uniform_1d_oneSiteLeft(psi,H)

    elif type(psi) == uMPS_1d_left_twoSite:
        return gradEvaluater_uniform_1d_twoSiteLeft(psi,H)

    elif type(psi) == uMPS_1d_left_bipartite:
        return gradEvaluater_bipartite_1d_left(psi,H)

    elif type(psi) == uMPSU1_2d_left:
        return gradEvaluater_mpso_2d_uniform(psi,H)

    elif type(psi) == uMPSU1_2d_left_bipartite:
        return gradEvaluater_mpso_2d_bipartite(psi,H)

    elif type(psi) == uMPSU1_2d_left_twoSite_square:
        return gradEvaluater_mpso_2d_twoSite_square(psi,H)

    elif type(psi) == uMPSU1_2d_left_twoSite_staircase:
        return gradEvaluater_mpso_2d_twoSite_staircase(psi,H)

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

    def projectTangentSpace(self,metric):
        if metric == "euclid":
            self.projectTangentSpace_euclid()
        elif metric == "TDVP":
            self.projectTangentSpace_tdvp()
        else:
            print("ERROR: INVALID METRIC - skipping projection to tangent space")


# -------------------------------------------------------------------------------------------------------------------------------------
#1d uniform
#abstract
class gradEvaluater_uniform_1d(gradEvaluater):
    @abstractmethod
    def attachRight(self):
        pass
    @abstractmethod
    def attachLeft(self):
        pass

    def eval(self):
        self.grad = self.H_imp[0].getCentralTerms()
        for n in range(1,len(self.H_imp)):
            self.grad += self.H_imp[n].getCentralTerms()

        leftEnv = self.H_imp[0].buildLeftEnv()
        for n in range(1,len(self.H_imp)):
            leftEnv += self.H_imp[n].buildLeftEnv()
        self.grad += self.attachRight(leftEnv)

        rightEnv = self.H_imp[0].buildRightEnv()
        for n in range(1,len(self.H_imp)):
            rightEnv += self.H_imp[n].buildRightEnv()
        self.grad += self.attachLeft(rightEnv)

    def projectTangentSpace_euclid(self):
        self.grad = project_mps_euclid(self.grad,self.psi.mps)

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

    def projectTangentSpace_tdvp(self):
        self.grad = project_mps_tdvp_leftGauge(self.grad,self.psi.mps,self.psi.R)

class gradEvaluater_uniform_1d_twoSiteLeft(gradEvaluater_uniform_1d):
    def __init__(self,psi,H,H2=None):
        self.psi = psi
        #allow for freedom of having two different Hamiltonians which begin at site 1 of the unit cell or site 2
        #useful for 2d mpso mps two site unit cell gradient, which has two different effective H on each sublattice
        self.H_site1 = H
        if H2 is None:
            self.H_site2 = H
        else:
            self.H_site2 = H2
        self.H_imp = dict()
        for n in range(0,len(self.H_site1.terms)):
            self.H_imp[n] = self.fetch_implementation(self.H_site1.terms[n],self.H_site2.terms[n])

    def fetch_implementation(self,H_site1,H_site2):
        if type(H_site1) == oneBodyH and type(H_site2) == oneBodyH:
            imp1 =  gradImplementation_uniform_1d_twoSiteLeft_oneBodyH_site1(self.psi,H_site1.tensor)
            imp2 =  gradImplementation_uniform_1d_twoSiteLeft_oneBodyH_site2(self.psi,H_site2.tensor)
            return gradImplementation_uniform_1d_twoSiteLeft(self.psi,imp1,imp2)
        elif type(H_site1) == twoBodyH or type(H_site1) == twoBodyH_hori or type(H_site1) == twoBodyH_vert:
            if type(H_site2) == twoBodyH or type(H_site2) == twoBodyH_hori or type(H_site2) == twoBodyH_vert:
                imp1 =  gradImplementation_uniform_1d_twoSiteLeft_twoBodyH_site1(self.psi,H_site1.tensor)
                imp2 =  gradImplementation_uniform_1d_twoSiteLeft_twoBodyH_site2(self.psi,H_site2.tensor)
                return gradImplementation_uniform_1d_twoSiteLeft(self.psi,imp1,imp2)
            else:
                error = 1
        else:
            error = 1
        if error == 1:
            print("Error: Incompatible Hamiltonian types at differing sites for two site unit cell ansatz")
            print(type(H_site1))
            print(type(H_site2))
            return 1

    def projectTangentSpace_euclid(self):
        gradA = self.grad.reshape(4,self.psi.D,self.psi.D)
        A = self.psi.mps.reshape(4,self.psi.D,self.psi.D)
        self.grad =  project_mps_euclid(gradA,A).reshape(2,2,self.psi.D,self.psi.D)
    def projectTangentSpace_tdvp(self):
        gradA = self.grad.reshape(4,self.psi.D,self.psi.D)
        A = self.psi.mps.reshape(4,self.psi.D,self.psi.D)
        self.grad =  project_mps_tdvp_leftGauge(gradA,A,self.psi.R).reshape(2,2,self.psi.D,self.psi.D)

    def attachRight(self,leftEnv):
        leftEnv = self.psi.Ta_inv.applyRight(leftEnv.reshape(self.psi.D**2)).reshape(self.psi.D,self.psi.D)
        return ncon([self.psi.mps,leftEnv,self.psi.R.tensor],((-1,-2,3,4),(-6,3),(-5,4)),forder=(-1,-2,-6,-5))
    def attachLeft(self,rightEnv):
        rightEnv = self.psi.Ta_inv.applyLeft(rightEnv.reshape(self.psi.D**2)).reshape(self.psi.D,self.psi.D)
        return ncon([self.psi.mps,rightEnv],((-1,-2,-3,4),(-5,4)),forder=(-1,-2,-3,-5))

class gradEvaluater_bipartite_1d_left_ind(gradEvaluater_uniform_1d):
    #find an individual gradient of bipartite ansatz, d/dA_1 <H> or d/dA_2 <H> (index arg is which gradient)
    #can reuse gradEvaluater_uniform_1d eval code as same structure to find gradient
    def __init__(self,psi,H,H_index,grad_index):
        self.grad_index = grad_index
        self.H_index = H_index
        if grad_index == 1:
            self.index1 = 1
            self.index2 = 2
        elif grad_index == 2:
            self.index1 = 2
            self.index2 = 1
        super().__init__(psi,H)

    def fetch_implementation(self,H):
        if type(H) == oneBodyH:
            return gradImplementation_bipartite_1d_left_oneBodyH(self.psi,H.tensor,self.H_index,self.grad_index)
        elif type(H) == twoBodyH or type(H) == twoBodyH_hori or type(H) == twoBodyH_vert:
            return gradImplementation_bipartite_1d_left_twoBodyH(self.psi,H.tensor,self.H_index,self.grad_index)

    def projectTangentSpace_euclid(self):
        self.grad = project_mps_euclid(self.grad,self.psi.mps[self.index1])
    def projectTangentSpace_tdvp(self):
        self.grad = project_mps_tdvp_leftGauge(self.grad,self.psi.mps[self.index1],self.psi.R[self.index2])

    def attachRight(self,leftEnv):
        leftEnv = self.psi.Ta_inv[self.index1].applyRight(leftEnv.reshape(self.psi.D**2)).reshape(self.psi.D,self.psi.D)
        return ncon([self.psi.mps[self.index1],leftEnv,self.psi.R[self.index2].tensor],((-1,2,3),(-4,2),(-5,3)),forder=(-1,-4,-5))
    def attachLeft(self,rightEnv):
        rightEnv = self.psi.Ta_inv[self.index2].applyLeft(rightEnv.reshape(self.psi.D**2)).reshape(self.psi.D,self.psi.D)
        return ncon([self.psi.mps[self.index1],rightEnv],((-1,-2,3),(-4,3)),forder=(-1,-2,-4))
# -------------------------------------------------------------------------------------------------------------------------------------
#2d ansatz
class gradEvaluater_mpso_2d(gradEvaluater):
    def __init__(self,psi,H):
        self.psi = psi
        self.H = H
        self.grad = dict()
    def fetch_implementation(self):
        pass
    def copyEvaluaters(self):
        self.grad['mps'] = self.gradA_evaluater.grad
        self.grad['mpo'] = self.gradB_evaluater.grad
    def eval(self,geo=False):
        self.gradA_evaluater.eval()
        self.gradB_evaluater.eval(geo=geo)
        self.copyEvaluaters()
    def projectTangentSpace(self,metric):
        self.gradA_evaluater.projectTangentSpace(metric)
        self.gradB_evaluater.projectTangentSpace(metric)
        self.copyEvaluaters()

class gradEvaluater_mpso_2d_uniform(gradEvaluater_mpso_2d):
    def __init__(self,psi,H):
        super().__init__(psi,H)
        self.gradA_evaluater = gradEvaluater_mpso_2d_mps_uniform(psi,H)
        self.gradB_evaluater = gradEvaluater_mpso_2d_mpo_uniform(psi,H)
class gradEvaluater_mpso_2d_twoSite_square(gradEvaluater_mpso_2d):
    def __init__(self,psi,H):
        super().__init__(psi,H)
        self.gradA_evaluater = gradEvaluater_mpso_2d_mps_twoSite_square(psi,H)
        self.gradB_evaluater = gradEvaluater_mpso_2d_mpo_twoSite_square(psi,H)
class gradEvaluater_mpso_2d_twoSite_staircase(gradEvaluater_mpso_2d):
    def __init__(self,psi,H):
        super().__init__(psi,H)
        self.gradA_evaluater = gradEvaluater_mpso_2d_mps_twoSite_staircase(psi,H)
        self.gradB_evaluater = gradEvaluater_mpso_2d_mpo_twoSite_staircase(psi,H)
class gradEvaluater_mpso_2d_bipartite(gradEvaluater_mpso_2d):
    def __init__(self,psi,H):
        super().__init__(psi,H)
        self.gradA_evaluater = gradEvaluater_mpso_2d_mps_bipartite(psi,H)
        self.gradB_evaluater = gradEvaluater_mpso_2d_mpo_bipartite(psi,H)
    def copyEvaluaters(self):
        self.grad['mps1'] = self.gradA_evaluater.grad[1]
        self.grad['mps2'] = self.gradA_evaluater.grad[2]
        self.grad['mpo1'] = self.gradB_evaluater.grad[1]
        self.grad['mpo2'] = self.gradB_evaluater.grad[2]

class gradEvaluater_mpso_2d_mps(gradEvaluater):
    #gradient of the mps tensor of 2d MPSO ansatz
    #this may be constructed as the gradient of a mps in 1d with an effective Hamiltonian
    #so just find effective Hamiltonians, then reuse 1d code to compute gradient terms
    def __init__(self,psi,H):
        super().__init__(psi,H)
        #effective 1d mps
        self.eff_psi = copy.deepcopy(self.psi)
        self.eff_psi.D = self.psi.D_mps
        self.eff_psi.R = self.psi.T #abuse syntax of fixed points to reuse gradEvaluater_uniform_1d code (bad code..)
        #effective 1d gradEvaluater
        self.effEvaluater = self.getEffective_1d_evaluater()
    def eval(self):
        self.effEvaluater.eval()
        self.grad = self.effEvaluater.grad
    def projectTangentSpace(self,metric):
        self.effEvaluater.projectTangentSpace(metric)
        self.grad = self.effEvaluater.grad

    @abstractmethod
    def getEffective_1d_evaluater(self):
        pass

class gradEvaluater_mpso_2d_mps_uniform(gradEvaluater_mpso_2d_mps):
    def getEffective_1d_evaluater(self):
        effH= []
        for n in range(0,len(self.H.terms)):
            effH.append(self.H_imp[n].getEffectiveH())
        effH = localH(effH)
        return gradEvaluater_uniform_1d_oneSiteLeft(self.eff_psi,effH)

    def fetch_implementation(self,H):
        if type(H) == oneBodyH:
            return gradImplementation_mpso_2d_mps_uniform_oneBodyH(self.psi,H.tensor)
        elif type(H) == twoBodyH_hori:
            return gradImplementation_mpso_2d_mps_uniform_twoBodyH_hori(self.psi,H.tensor)
        elif type(H) == twoBodyH_vert:
            return gradImplementation_mpso_2d_mps_uniform_twoBodyH_vert(self.psi,H.tensor)

class gradEvaluater_mpso_2d_mps_twoSite(gradEvaluater_mpso_2d_mps):
    def getEffective_1d_evaluater(self):
        effH1 = [] #Hamiltonian acting on first site
        effH2 = [] #"" on second site
        for n in range(0,len(self.H.terms)):
            effH1.append(self.H_imp[n].getEffectiveH(site=1))
            effH2.append(self.H_imp[n].getEffectiveH(site=2))
        effH1 = localH(effH1)
        effH2 = localH(effH2)
        return gradEvaluater_uniform_1d_twoSiteLeft(self.eff_psi,effH1,effH2)

class gradEvaluater_mpso_2d_mps_twoSite_square(gradEvaluater_mpso_2d_mps_twoSite):
    def fetch_implementation(self,H):
        if type(H) == oneBodyH:
            return gradImplementation_mpso_2d_mps_twoSite_square_oneBodyH(self.psi,H.tensor)
        elif type(H) == twoBodyH_hori:
            return gradImplementation_mpso_2d_mps_twoSite_square_twoBodyH_hori(self.psi,H.tensor)
        elif type(H) == twoBodyH_vert:
            return gradImplementation_mpso_2d_mps_twoSite_square_twoBodyH_vert(self.psi,H.tensor)

class gradEvaluater_mpso_2d_mps_twoSite_staircase(gradEvaluater_mpso_2d_mps_twoSite):
    def fetch_implementation(self,H):
        if type(H) == oneBodyH:
            return gradImplementation_mpso_2d_mps_twoSite_staircase_oneBodyH(self.psi,H.tensor)
        elif type(H) == twoBodyH_hori:
            return gradImplementation_mpso_2d_mps_twoSite_staircase_twoBodyH_hori(self.psi,H.tensor)
        elif type(H) == twoBodyH_vert:
            return gradImplementation_mpso_2d_mps_twoSite_staircase_twoBodyH_vert(self.psi,H.tensor)


class gradEvaluater_mpso_2d_mpo(gradEvaluater):
    @abstractmethod
    def wrapAroundLeft(self):
        pass

    @abstractmethod
    def wrapAroundRight(self):
        pass

    def gradMagnitude(self,grad):
        return np.einsum('ijab,ijab',grad,grad.conj())

    def eval_non_geo(self,H_term_index):
        n = H_term_index
        #terms under H
        grad = self.H_imp[n].getCentralTerms()
        #terms to left of H
        rightEnv = self.H_imp[n].wrapRightEnv(self.H_imp[n].buildRightEnv())
        grad += self.wrapAroundLeft(rightEnv)
        #terms to right of H
        leftEnv = self.H_imp[n].wrapLeftEnv(self.H_imp[n].buildLeftEnv())
        grad += self.wrapAroundRight(leftEnv)
        return grad

    #geometric sum
    def eval_geo(self,H_term_index,Td,rightFP_d,outers_d):
        n = H_term_index
        # terms below Hamiltonian
        env = self.H_imp[n].buildTopEnvGeo(rightFP_d,outers_d)
        #terms above Hamiltonian
        env += self.H_imp[n].buildBotEnvGeo(rightFP_d,outers_d)
        #right quadrant lower
        env += self.H_imp[n].buildTopEnvGeo_quadrants(rightFP_d,outers_d)
        #right quadrant upper
        env += self.H_imp[n].buildBotEnvGeo_quadrants(rightFP_d,outers_d)
        #left half of plane, upper and lower quadrants
        env += self.H_imp[n].buildRightEnvGeo_quadrants(rightFP_d,outers_d)
        return self.wrapAroundLeft(env)

    def eval(self,geo=True,envTol=1e-5,printEnv=True):
        self.grad = self.eval_non_geo(0)
        for n in range(0,len(self.H.terms)):
            if n > 0:
                self.grad += self.eval_non_geo(n)

            if geo is True:
                Td_matrix,Td = self.H_imp[n].init_mps_transfers()
                for d in range(0,100):
                    if d > 0:
                        Td_matrix,Td = self.H_imp[n].apply_mps_transfers(Td_matrix)
                    rightFP_d = self.H_imp[n].getFixedPoints(d,Td) #bottleneck of algo
                    outers_d = self.H_imp[n].getOuterContracts(d,Td)

                    gradRun = self.eval_geo(n,Td,rightFP_d,outers_d)

                    #check geometric sum decayed
                    mag = self.gradMagnitude(gradRun)
                    self.grad += gradRun
                    if np.abs(mag)<envTol:
                        break
                if printEnv is True:
                    print("d: ",d,mag)
            
class gradEvaluater_mpso_2d_mpo_uniform(gradEvaluater_mpso_2d_mpo):
    def fetch_implementation(self,H):
        if type(H) == oneBodyH:
            return gradImplementation_mpso_2d_mpo_uniform_oneBodyH(self.psi,H.tensor)
        elif type(H) == twoBodyH_hori:
            return gradImplementation_mpso_2d_mpo_uniform_twoBodyH_hori(self.psi,H.tensor)
        elif type(H) == twoBodyH_vert:
            return gradImplementation_mpso_2d_mpo_uniform_twoBodyH_vert(self.psi,H.tensor)

    def wrapAroundLeft(self,env):
        return ncon([self.psi.mpo,env],((-2,1,-4,5),(-3,1,-6,5)),forder=(-2,-3,-4,-6),order=(5,1))
    def wrapAroundRight(self,env):
        return ncon([self.psi.mpo,env],((-2,1,4,5),(-3,1,-6,4,-7,5)),forder=(-2,-3,-6,-7),order=(4,1,5))

    def projectTangentSpace_euclid(self):
        self.grad = project_mpo_euclid(self.grad,self.psi.mpo)
    def projectTangentSpace_tdvp(self):
        rho = ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1))
        self.grad = project_mpo_tdvp_leftGauge(self.grad,self.psi.mpo,self.psi.R,rho)

class gradEvaluater_mpso_2d_mpo_bipartite_ind(gradEvaluater_mpso_2d_mpo):
    #find an individual gradient of bipartite ansatz, d/dA_1 <H> or d/dA_2 <H> (index arg is which gradient)
    #can reuse gradEvaluater_uniform_1d eval code as same structure to find gradient
    def __init__(self,psi,H,H_index,grad_index):
        self.grad_index = grad_index
        self.H_index = H_index
        if grad_index == 1:
            self.index1 = 1
            self.index2 = 2
        elif grad_index == 2:
            self.index1 = 2
            self.index2 = 1
        super().__init__(psi,H)

    def fetch_implementation(self,H):
        if type(H) == oneBodyH:
            return gradImplementation_mpso_2d_mpo_bipartite_oneBodyH(self.psi,H.tensor,self.H_index,self.grad_index)
        elif type(H) == twoBodyH_hori: 
            return gradImplementation_mpso_2d_mpo_bipartite_twoBodyH_hori(self.psi,H.tensor,self.H_index,self.grad_index)
        elif type(H) == twoBodyH_vert: 
            return gradImplementation_mpso_2d_mpo_bipartite_twoBodyH_vert(self.psi,H.tensor,self.H_index,self.grad_index)

    def projectTangentSpace_euclid(self):
        self.grad = project_mpo_euclid(self.grad,self.psi.mpo[self.index1])
    def projectTangentSpace_tdvp(self):
        rho = ncon([self.psi.mps[self.index1],self.psi.mps[self.index1].conj(),self.psi.T[self.index2].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1))
        self.grad = project_mpo_tdvp_leftGauge(self.grad,self.psi.mpo[self.index1],self.psi.R[self.index2],rho)

    def wrapAroundLeft(self,env):
        return ncon([self.psi.mpo[self.index1],env],((-2,1,-4,5),(-3,1,-6,5)),forder=(-2,-3,-4,-6),order=(5,1))
    def wrapAroundRight(self,env):
        return ncon([self.psi.mpo[self.index1],env],((-2,1,4,5),(-3,1,-6,4,-7,5)),forder=(-2,-3,-6,-7),order=(4,1,5))

class gradEvaluater_mpso_2d_mpo_twoSite(gradEvaluater_mpso_2d_mpo):
    def wrapAroundLeft(self,env):
        return ncon([self.psi.mpo,env],((-2,-5,1,4,-7,8),(-3,-6,1,4,-9,8)),forder=(-2,-5,-3,-6,-7,-9),order=(8,1,4))
    def wrapAroundRight(self,env):
        return ncon([self.psi.mpo,env],((-2,-5,1,4,7,8),(-3,-6,1,4,-9,7,-10,8)),forder=(-2,-5,-3,-6,-9,-10),order=(7,8,1,4))
    def gradMagnitude(self,grad):
        return np.einsum('abcdef,abcdef',grad,grad.conj())

class gradEvaluater_mpso_2d_mpo_twoSite_staircase(gradEvaluater_mpso_2d_mpo_twoSite):
    def fetch_implementation(self,H):
        if type(H) == oneBodyH:
            imp1 = gradImplementation_mpso_2d_mpo_twoSite_staircase_oneBodyH_site1(self.psi,H.tensor)
            imp2 = gradImplementation_mpso_2d_mpo_twoSite_staircase_oneBodyH_site2(self.psi,H.tensor)
            return gradImplementation_mpso_2d_mpo_twoSite_staircase_wrapper(self.psi,imp1,imp2)
        elif type(H) == twoBodyH_hori:
            imp1 = gradImplementation_mpso_2d_mpo_twoSite_staircase_twoBodyH_hori_site1(self.psi,H.tensor)
            imp2 = gradImplementation_mpso_2d_mpo_twoSite_staircase_twoBodyH_hori_site2(self.psi,H.tensor)
            return gradImplementation_mpso_2d_mpo_twoSite_staircase_wrapper(self.psi,imp1,imp2)
        elif type(H) == twoBodyH_vert:
            imp1 = gradImplementation_mpso_2d_mpo_twoSite_staircase_twoBodyH_vert_site1(self.psi,H.tensor)
            imp2 = gradImplementation_mpso_2d_mpo_twoSite_staircase_twoBodyH_vert_site2(self.psi,H.tensor)
            return gradImplementation_mpso_2d_mpo_twoSite_staircase_wrapper(self.psi,imp1,imp2)

    def projectTangentSpace_euclid(self):
        gradB = self.grad.reshape(4,4,self.psi.D_mpo,self.psi.D_mpo)
        B = self.psi.mpo.reshape(4,4,self.psi.D_mpo,self.psi.D_mpo)
        self.grad = project_mpo_euclid(gradB,B).reshape(2,2,2,2,self.psi.D_mpo,self.psi.D_mpo)
    def projectTangentSpace_tdvp(self):
        gradB = self.grad.reshape(4,4,self.psi.D_mpo,self.psi.D_mpo)
        B = self.psi.mpo.reshape(4,4,self.psi.D_mpo,self.psi.D_mpo)
        rho1 = ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((-1,3,4,5),(-2,3,4,6),(6,5)),forder=(-2,-1))
        rho2 = ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((1,-2,4,5),(1,-3,4,6),(6,5)),forder=(-3,-2))
        rho = ncon([rho1,rho2],((-1,-2),(-3,-4)),forder=(-1,-3,-2,-4)).reshape(4,4)
        self.grad = project_mpo_tdvp_leftGauge(gradB,B,self.psi.R,rho).reshape(2,2,2,2,self.psi.D_mpo,self.psi.D_mpo)

class gradEvaluater_mpso_2d_mpo_twoSite_square_ind(gradEvaluater_mpso_2d_mpo_twoSite):
    def __init__(self,psi,H,siteLabel):
        self.siteLabel = siteLabel
        super().__init__(psi,H)

    def fetch_implementation(self,H):
        if type(H) == oneBodyH:
            if self.siteLabel == '00':
                return gradImplementation_mpso_2d_mpo_twoSite_square_oneBodyH_site00(self.psi,H.tensor)
            elif self.siteLabel == '01':
                return gradImplementation_mpso_2d_mpo_twoSite_square_oneBodyH_site01(self.psi,H.tensor)
            elif self.siteLabel == '10':
                return gradImplementation_mpso_2d_mpo_twoSite_square_oneBodyH_site10(self.psi,H.tensor)
            elif self.siteLabel == '11':
                return gradImplementation_mpso_2d_mpo_twoSite_square_oneBodyH_site11(self.psi,H.tensor)

        elif type(H) == twoBodyH_hori:
            if self.siteLabel == '00':
                return gradImplementation_mpso_2d_mpo_twoSite_square_twoBodyH_hori_site00(self.psi,H.tensor)
            elif self.siteLabel == '01':
                return gradImplementation_mpso_2d_mpo_twoSite_square_twoBodyH_hori_site01(self.psi,H.tensor)
            elif self.siteLabel == '10':
                return gradImplementation_mpso_2d_mpo_twoSite_square_twoBodyH_hori_site10(self.psi,H.tensor)
            elif self.siteLabel == '11':
                return gradImplementation_mpso_2d_mpo_twoSite_square_twoBodyH_hori_site11(self.psi,H.tensor)

        elif type(H) == twoBodyH_vert:
            if self.siteLabel == '00':
                return gradImplementation_mpso_2d_mpo_twoSite_square_twoBodyH_vert_site00(self.psi,H.tensor)
            elif self.siteLabel == '01':
                return gradImplementation_mpso_2d_mpo_twoSite_square_twoBodyH_vert_site01(self.psi,H.tensor)
            elif self.siteLabel == '10':
                return gradImplementation_mpso_2d_mpo_twoSite_square_twoBodyH_vert_site10(self.psi,H.tensor)
            elif self.siteLabel == '11':
                return gradImplementation_mpso_2d_mpo_twoSite_square_twoBodyH_vert_site11(self.psi,H.tensor)

# -------------------------------------------------------------------------------------------------------------------------------------
#bipartite Wrappers
class gradEvaluater_bipartite_1d_left(gradEvaluater):
    #wrapper to find both gradients together for bipartite ansatz
    def __init__(self,psi,H):
        self.psi = psi
        self.H = H
    def eval(self):
        self.grad = dict()
        #reuse gradEvaluater_uniform_1d code
        gradEvaluater_11 = gradEvaluater_bipartite_1d_left_ind(self.psi,self.H,H_index=1,grad_index=1)
        gradEvaluater_12 = gradEvaluater_bipartite_1d_left_ind(self.psi,self.H,H_index=1,grad_index=2)
        gradEvaluater_21 = gradEvaluater_bipartite_1d_left_ind(self.psi,self.H,H_index=2,grad_index=1)
        gradEvaluater_22 = gradEvaluater_bipartite_1d_left_ind(self.psi,self.H,H_index=2,grad_index=2)

        gradEvaluater_11.eval()
        gradEvaluater_12.eval()
        gradEvaluater_21.eval()
        gradEvaluater_22.eval()
        self.grad[1] = 1/2*(gradEvaluater_11.grad + gradEvaluater_21.grad)
        self.grad[2] = 1/2*(gradEvaluater_12.grad + gradEvaluater_22.grad)

    def fetch_implementation(self):
        pass
    def projectTangentSpace_euclid(self):
        self.grad[1] = project_mps_euclid(self.grad[1],self.psi.mps[1])
        self.grad[2] = project_mps_euclid(self.grad[2],self.psi.mps[2])
    def projectTangentSpace_tdvp(self):
        self.grad[1] = project_mps_tdvp_leftGauge(self.grad[1],self.psi.mps[1],self.psi.R[2])
        self.grad[2] = project_mps_tdvp_leftGauge(self.grad[2],self.psi.mps[2],self.psi.R[1])

class gradEvaluater_mpso_2d_mps_bipartite(gradEvaluater):
    def eval(self):
        effH_1 = []
        effH_2 = []
        for n in range(0,len(self.H.terms)):
            effH_1.append(self.H_imp[n].getEffectiveH(1))
            effH_2.append(self.H_imp[n].getEffectiveH(2))
        effH_1 = localH(effH_1)
        effH_2 = localH(effH_2)
        eff_psi = copy.deepcopy(self.psi)
        eff_psi.D = self.psi.D_mps
        eff_psi.R = self.psi.T

        #reuse 1d code here
        gradEvaluater_11 = gradEvaluater_bipartite_1d_left_ind(eff_psi,effH_1,H_index=1,grad_index=1)
        gradEvaluater_12 = gradEvaluater_bipartite_1d_left_ind(eff_psi,effH_1,H_index=1,grad_index=2)
        gradEvaluater_21 = gradEvaluater_bipartite_1d_left_ind(eff_psi,effH_2,H_index=2,grad_index=1)
        gradEvaluater_22 = gradEvaluater_bipartite_1d_left_ind(eff_psi,effH_2,H_index=2,grad_index=2)

        gradEvaluater_11.eval()
        gradEvaluater_12.eval()
        gradEvaluater_21.eval()
        gradEvaluater_22.eval()
        self.grad = dict()
        self.grad[1] = 1/2*(gradEvaluater_11.grad + gradEvaluater_21.grad)
        self.grad[2] = 1/2*(gradEvaluater_12.grad + gradEvaluater_22.grad)

    def projectTangentSpace_euclid(self):
        self.grad[1] = project_mps_euclid(self.grad[1],self.psi.mps[1])
        self.grad[2] = project_mps_euclid(self.grad[2],self.psi.mps[2])
    def projectTangentSpace_tdvp(self):
        self.grad[1] = project_mps_tdvp_leftGauge(self.grad[1],self.psi.mps[1],self.psi.T[2])
        self.grad[2] = project_mps_tdvp_leftGauge(self.grad[2],self.psi.mps[2],self.psi.T[1])

class gradEvaluater_mpso_2d_mps_bipartite(gradEvaluater_mpso_2d_mps_bipartite):
    def fetch_implementation(self,H):
        if type(H) == oneBodyH:
            return gradImplementation_mpso_2d_mps_bipartite_oneBodyH(self.psi,H.tensor)
        elif type(H) == twoBodyH_hori:
            return gradImplementation_mpso_2d_mps_bipartite_twoBodyH_hori(self.psi,H.tensor)
        elif type(H) == twoBodyH_vert:
            return gradImplementation_mpso_2d_mps_bipartite_twoBodyH_vert(self.psi,H.tensor)

#for mpso_2d ansatz with 4 individual gradients
class gradEvaluater_mpso_2d_mpo_twoSiteUnitCell_wrapper(gradEvaluater):
    def fetch_implementation(self):
        pass

    def eval(self,geo=True,envTol=1e-5,printEnv=False):
        #use eval_non_geo, eval_geo methods seperately rather than eval
        #so can just use one gradEvaluater to calc d dep fixed points, which are the same for all gradEvaluaters
        #so don't have to repeat the bottleneck of the algo 4 times for no reason
        self.grad_11 = self.gradEvaluater_11.eval_non_geo(0)
        self.grad_12 = self.gradEvaluater_12.eval_non_geo(0)
        self.grad_21 = self.gradEvaluater_21.eval_non_geo(0)
        self.grad_22 = self.gradEvaluater_22.eval_non_geo(0)
        for n in range(0,len(self.H.terms)):
            if n > 0:
                self.grad_11 += self.gradEvaluater_11.eval_non_geo(n)
                self.grad_12 += self.gradEvaluater_12.eval_non_geo(n)
                self.grad_21 += self.gradEvaluater_21.eval_non_geo(n)
                self.grad_22 += self.gradEvaluater_22.eval_non_geo(n)

            grad_11_breaker, grad_12_breaker, grad_21_breaker, grad_22_breaker = False,False,False,False
            d_11,d_12,d_21,d_22 = -1, -1 , -1, -1
            if geo is True:
                Td_matrix,Td = self.gradEvaluater_11.H_imp[n].init_mps_transfers()
                for d in range(0,100):
                    #d dependant tensors needed
                    if d > 0:
                        Td_matrix,Td = self.gradEvaluater_11.H_imp[n].apply_mps_transfers(Td_matrix)
                    #just use gradEvaluater_11 to calc d dep fixed points, which are the same for all gradEvaluaters
                    rightFP_d = self.gradEvaluater_11.H_imp[n].getFixedPoints(d,Td) #bottleneck of algo
                    outers_d = self.gradEvaluater_11.H_imp[n].getOuterContracts(d,Td)

                    if grad_11_breaker is False:
                        gradRun_11 = self.gradEvaluater_11.eval_geo(n,Td,rightFP_d,outers_d)
                        self.grad_11 += gradRun_11
                        d_11 += 1
                        mag_11 = self.gradEvaluater_11.gradMagnitude(gradRun_11)
                        if np.abs(mag_11)<envTol:
                            grad_11_breaker = True

                    if grad_12_breaker is False:
                        gradRun_12 = self.gradEvaluater_12.eval_geo(n,Td,rightFP_d,outers_d)
                        self.grad_12 += gradRun_12
                        d_12 += 1
                        mag_12 = self.gradEvaluater_12.gradMagnitude(gradRun_12)
                        if np.abs(mag_12)<envTol:
                            grad_12_breaker = True

                    if grad_21_breaker is False:
                        gradRun_21 = self.gradEvaluater_21.eval_geo(n,Td,rightFP_d,outers_d)
                        self.grad_21 += gradRun_21
                        d_21 += 1
                        mag_21 = self.gradEvaluater_21.gradMagnitude(gradRun_21)
                        if np.abs(mag_21)<envTol:
                            grad_21_breaker = True

                    if grad_22_breaker is False:
                        gradRun_22 = self.gradEvaluater_22.eval_geo(n,Td,rightFP_d,outers_d)
                        self.grad_22 += gradRun_22
                        d_22 += 1
                        mag_22 = self.gradEvaluater_22.gradMagnitude(gradRun_22)
                        if np.abs(mag_22)<envTol:
                            grad_22_breaker = True

                    if grad_11_breaker is True and grad_12_breaker is True and grad_21_breaker is True and grad_22_breaker is True:
                        break
                # if printEnv is True:
                print("d_11: ",d_11,mag_11)
                print("d_12: ",d_12,mag_12)
                print("d_21: ",d_21,mag_21)
                print("d_22: ",d_22,mag_22)
        self.buildTotalGrad()

class gradEvaluater_mpso_2d_mpo_twoSite_square(gradEvaluater_mpso_2d_mpo_twoSiteUnitCell_wrapper):
    def __init__(self,psi,H):
        self.psi = psi
        self.H = H
        self.gradEvaluater_11 = gradEvaluater_mpso_2d_mpo_twoSite_square_ind(self.psi,self.H,siteLabel='00')
        self.gradEvaluater_12 = gradEvaluater_mpso_2d_mpo_twoSite_square_ind(self.psi,self.H,siteLabel='01')
        self.gradEvaluater_21 = gradEvaluater_mpso_2d_mpo_twoSite_square_ind(self.psi,self.H,siteLabel='10')
        self.gradEvaluater_22 = gradEvaluater_mpso_2d_mpo_twoSite_square_ind(self.psi,self.H,siteLabel='11')

    def buildTotalGrad(self):
        self.grad = 1/4*(self.grad_11 + self.grad_12 + self.grad_21 + self.grad_22)

    def projectTangentSpace_euclid(self):
        gradB = self.grad.reshape(4,4,self.psi.D_mpo,self.psi.D_mpo)
        B = self.psi.mpo.reshape(4,4,self.psi.D_mpo,self.psi.D_mpo)
        self.grad = project_mpo_euclid(gradB,B).reshape(2,2,2,2,self.psi.D_mpo,self.psi.D_mpo)

class gradEvaluater_mpso_2d_mpo_bipartite(gradEvaluater_mpso_2d_mpo_twoSiteUnitCell_wrapper):
    def __init__(self,psi,H):
        self.psi = psi
        self.H = H
        self.gradEvaluater_11 = gradEvaluater_mpso_2d_mpo_bipartite_ind(self.psi,self.H,H_index=1,grad_index=1)
        self.gradEvaluater_12 = gradEvaluater_mpso_2d_mpo_bipartite_ind(self.psi,self.H,H_index=1,grad_index=2)
        self.gradEvaluater_21 = gradEvaluater_mpso_2d_mpo_bipartite_ind(self.psi,self.H,H_index=2,grad_index=1)
        self.gradEvaluater_22 = gradEvaluater_mpso_2d_mpo_bipartite_ind(self.psi,self.H,H_index=2,grad_index=2)
    
    def buildTotalGrad(self):
        self.grad = dict()
        self.grad[1] = 1/2*(self.grad_11 + self.grad_21)
        self.grad[2] = 1/2*(self.grad_12 + self.grad_22)

    def projectTangentSpace_euclid(self):
        self.grad[1] = project_mpo_euclid(self.grad[1],self.psi.mpo[1])
        self.grad[2] = project_mpo_euclid(self.grad[2],self.psi.mpo[2])
    def projectTangentSpace_tdvp(self):
        rho1 = ncon([self.psi.mps[1],self.psi.mps[1].conj(),self.psi.T[2].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1))
        rho2 = ncon([self.psi.mps[2],self.psi.mps[2].conj(),self.psi.T[1].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1))
        self.grad[1] = project_mpo_tdvp_leftGauge(self.grad[1],self.psi.mpo[1],self.psi.R[2],rho1)
        self.grad[2] = project_mpo_tdvp_leftGauge(self.grad[2],self.psi.mpo[2],self.psi.R[1],rho2)
