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
from uniformTN_gradImplementations import *

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

class gradEvaluater_uniform_1d_twoSiteLeft(gradEvaluater_uniform_1d):
    def fetch_implementation(self,H):
        if type(H) == oneBodyH:
            return gradImplementation_uniform_1d_twoSiteLeft_oneBodyH(self.psi,H.tensor)
        elif type(H) == twoBodyH or type(H) == twoBodyH_hori or type(H) == twoBodyH_vert:
            return gradImplementation_uniform_1d_twoSiteLeft_twoBodyH(self.psi,H.tensor)

    def projectTDVP(self):
        gradA = self.grad.reshape(4,self.psi.D,self.psi.D)
        A = self.psi.mps.reshape(4,self.psi.D,self.psi.D)
        grad =  project_mpsTangentVector(gradA,A,self.psi.R).reshape(2,2,self.psi.D,self.psi.D)
        self.grad = grad

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

    def projectTDVP(self):
        self.grad = project_mpsTangentVector(self.grad,self.psi.mps[self.index1],self.psi.R[self.index1])

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
        pass
    def eval(self,geo=True):
        self.gradA_evaluater.eval()
        self.gradB_evaluater.eval(geo=geo)
        self.copyEvaluaters()
    def projectTDVP(self):
        self.gradA_evaluater.projectTDVP()
        self.gradB_evaluater.projectTDVP()
        self.copyEvaluaters()

class gradEvaluater_mpso_2d_uniform(gradEvaluater_mpso_2d):
    def __init__(self,psi,H):
        super().__init__(psi,H)
        self.gradA_evaluater = gradEvaluater_mpso_2d_mps_uniform(psi,H)
        self.gradB_evaluater = gradEvaluater_mpso_2d_mpo_uniform(psi,H)
    def copyEvaluaters(self):
        self.grad['mps'] = self.gradA_evaluater.grad
        self.grad['mpo'] = self.gradB_evaluater.grad

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

class gradEvaluater_mpso_2d_mps_uniform(gradEvaluater):
    #gradient of the mps tensor of 2d uniform MPSO ansatz
    #this may be constructed as the gradient of a uniform mps in 1d with an effective Hamiltonian
    #so just find effective Hamiltonians, then reuse gradEvaluater_uniform_1d to compute gradient terms
    def eval(self):
        effH= []
        for n in range(0,len(self.H.terms)):
            effH.append(self.H_imp[n].getEffectiveH())
        effH = localH(effH)
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

    def eval_non_geo(self,H_term_index):
        n = H_term_index
        #terms under H
        grad = self.H_imp[n].getCentralTerms()
        #terms to left of H
        rightEnv = self.H_imp[n].buildRightEnv()
        grad += self.H_imp[n].attachLeftMax(rightEnv)
        #terms to right of H
        leftEnv = self.H_imp[n].buildLeftEnv()
        grad += self.H_imp[n].attachRightMax(leftEnv)
        return np.einsum('ijab->jiab',grad)

    #geometric sum
    def eval_geo(self,H_term_index,Td,rightFP_d,outers_d):
        n = H_term_index
        #necessary for quadrants
        leftEnv_h_tilde = self.H_imp[n].buildLeftEnv(H=self.H_imp[n].h_tilde)
        # terms below Hamiltonian
        env = self.H_imp[n].buildTopEnvGeo(rightFP_d,outers_d)
        #right quadrant lower
        env += self.H_imp[n].buildTopEnvGeo_quadrants(rightFP_d,outers_d,leftEnv_h_tilde)
        grad = self.attachBot(env)

        #terms above Hamiltonian
        env = self.H_imp[n].buildBotEnvGeo(rightFP_d,outers_d)
        #right quadrant upper
        env += self.H_imp[n].buildBotEnvGeo_quadrants(rightFP_d,outers_d,leftEnv_h_tilde)
        grad += self.attachTop(env)

        #left quadrants upper and lower
        rightEnv = self.H_imp[n].buildRightEnvGeo_quadrants(rightFP_d,outers_d)
        grad += self.H_imp[n].attachLeftSingle(rightEnv)
        return np.einsum('ijab->jiab',grad)


    def eval(self,geo=True,envTol=1e-5,printEnv=False):
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
                    outers_d = self.H_imp[n].getOuterContracts(Td)

                    gradRun = self.eval_geo(n,Td,rightFP_d,outers_d)

                    #check geometric sum decayed
                    mag = np.einsum('ijab,ijab',gradRun,gradRun.conj())
                    self.grad += gradRun
                    if np.abs(mag)<envTol:
                        break
                if printEnv is True:
                    print("d: ",d,mag)
            
    def projectTDVP(self):
        self.grad = project_mpoTangentVector(self.grad,self.psi.mps,self.psi.mpo,self.psi.T,self.psi.R)

class gradEvaluater_mpso_2d_mpo_bipartite_ind(gradEvaluater_mpso_2d_mpo_uniform):
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

    def projectTDVP(self):
        self.grad = project_mpsTangentVector(self.grad,self.psi.mps[self.index1],self.psi.mpo[self.index1],self.psi.T[self.index1],self.psi.R[self.index1])

    def attachTop(self,env):
        return ncon([env,self.psi.mps[self.index1],self.psi.mpo[self.index1],self.psi.mps[self.index1].conj(),self.psi.T[self.index2].tensor],((9,-8,7,4),(1,4,5),(-2,1,-6,7),(-3,9,10),(10,5)),forder=(-3,-2,-6,-8),order=(4,9,5,10,1,7))

    def attachBot(self,env):
        return ncon([env,self.psi.mps[self.index1],self.psi.mpo[self.index1],self.psi.mps[self.index1].conj()],((9,-8,7,5),(1,4,5),(-2,1,-6,7),(-3,4,9)),forder=(-3,-2,-6,-8),order=(9,5,4,7,1))

class gradEvaluater_mpso_2d_mps_uniform(gradEvaluater_mpso_2d_mps_uniform):
    def fetch_implementation(self,H):
        if type(H) == oneBodyH:
            return gradImplementation_mpso_2d_mps_uniform_oneBodyH(self.psi,H.tensor)
        elif type(H) == twoBodyH_hori:
            return gradImplementation_mpso_2d_mps_uniform_twoBodyH_hori(self.psi,H.tensor)
        elif type(H) == twoBodyH_vert:
            return gradImplementation_mpso_2d_mps_uniform_twoBodyH_vert(self.psi,H.tensor)

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
    def projectTDVP(self):
        self.grad[1] = project_mpsTangentVector(self.grad[1],self.psi.mps[1],self.psi.R[1])
        self.grad[2] = project_mpsTangentVector(self.grad[2],self.psi.mps[2],self.psi.R[2])

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
    def projectTDVP(self):
        self.grad[1] = project_mpsTangentVector(self.grad[1],self.psi.mps[1],self.psi.T[1])
        self.grad[2] = project_mpsTangentVector(self.grad[2],self.psi.mps[2],self.psi.T[2])

class gradEvaluater_mpso_2d_mpo_bipartite(gradEvaluater):
    def __init__(self,psi,H):
        self.psi = psi
        self.H = H

    def fetch_implementation(self):
        pass

    def eval(self,geo=True,envTol=1e-5,printEnv=False):
        self.grad = dict()
        gradEvaluater_11 = gradEvaluater_mpso_2d_mpo_bipartite_ind(self.psi,self.H,H_index=1,grad_index=1)
        gradEvaluater_12 = gradEvaluater_mpso_2d_mpo_bipartite_ind(self.psi,self.H,H_index=1,grad_index=2)
        gradEvaluater_21 = gradEvaluater_mpso_2d_mpo_bipartite_ind(self.psi,self.H,H_index=2,grad_index=1)
        gradEvaluater_22 = gradEvaluater_mpso_2d_mpo_bipartite_ind(self.psi,self.H,H_index=2,grad_index=2)

        grad_11 = gradEvaluater_11.eval_non_geo(0)
        grad_12 = gradEvaluater_12.eval_non_geo(0)
        grad_21 = gradEvaluater_21.eval_non_geo(0)
        grad_22 = gradEvaluater_22.eval_non_geo(0)
        for n in range(0,len(self.H.terms)):
            if n > 0:
                grad_11 += gradEvaluater_11.eval_non_geo(n)
                grad_12 += gradEvaluater_12.eval_non_geo(n)
                grad_21 += gradEvaluater_21.eval_non_geo(n)
                grad_22 += gradEvaluater_22.eval_non_geo(n)

            grad_11_breaker, grad_12_breaker, grad_21_breaker, grad_22_breaker = False,False,False,False
            d_11,d_12,d_21,d_22 = -1, -1 , -1, -1
            if geo is True:
                #just use one evaluater to calc d dep quantites (including fixed points)
                #they are the same for all, so no need to bother calculating fixed points 4 times...
                Td_matrix,Td = gradEvaluater_11.H_imp[n].init_mps_transfers()
                for d in range(0,100):
                    #d dependant tensors needed
                    if d > 0:
                        Td_matrix,Td = gradEvaluater_11.H_imp[n].apply_mps_transfers(Td_matrix)
                    rightFP_d = gradEvaluater_11.H_imp[n].getFixedPoints(d,Td) #bottleneck of algo
                    outers_d = gradEvaluater_11.H_imp[n].getOuterContracts(Td)

                    if grad_11_breaker is False:
                        gradRun_11 = gradEvaluater_11.eval_geo(n,Td,rightFP_d,outers_d)
                        grad_11 += gradRun_11
                        d_11 += 1
                        mag_11 = np.einsum('ijab,ijab',gradRun_11,gradRun_11.conj())
                        if np.abs(mag_11)<envTol:
                            grad_11_breaker = True

                    if grad_12_breaker is False:
                        gradRun_12 = gradEvaluater_12.eval_geo(n,Td,rightFP_d,outers_d)
                        grad_12 += gradRun_12
                        d_12 += 1
                        mag_12 = np.einsum('ijab,ijab',gradRun_12,gradRun_12.conj())
                        if np.abs(mag_12)<envTol:
                            grad_12_breaker = True

                    if grad_21_breaker is False:
                        gradRun_21 = gradEvaluater_21.eval_geo(n,Td,rightFP_d,outers_d)
                        grad_21 += gradRun_21
                        d_21 += 1
                        mag_21 = np.einsum('ijab,ijab',gradRun_21,gradRun_21.conj())
                        if np.abs(mag_21)<envTol:
                            grad_21_breaker = True

                    if grad_22_breaker is False:
                        gradRun_22 = gradEvaluater_22.eval_geo(n,Td,rightFP_d,outers_d)
                        grad_22 += gradRun_22
                        d_22 += 1
                        mag_22 = np.einsum('ijab,ijab',gradRun_22,gradRun_22.conj())
                        if np.abs(mag_22)<envTol:
                            grad_22_breaker = True

                    if grad_11_breaker is True and grad_12_breaker is True and grad_21_breaker is True and grad_22_breaker is True:
                        break
                if printEnv is True:
                    print("d_11: ",d_11,mag_11)
                    print("d_12: ",d_12,mag_12)
                    print("d_21: ",d_21,mag_21)
                    print("d_22: ",d_22,mag_22)
        self.grad[1] = 1/2*(grad_11 + grad_21)
        self.grad[2] = 1/2*(grad_12 + grad_22)

    def projectTDVP(self):
        self.grad[1] = project_mpoTangentVector(self.grad[1],self.psi.mps[1],self.psi.mpo[1],self.psi.T[1],self.psi.R[1])
        self.grad[2] = project_mpoTangentVector(self.grad[2],self.psi.mps[2],self.psi.mpo[2],self.psi.T[2],self.psi.R[2])

class gradEvaluater_mpso_2d_mps_bipartite(gradEvaluater_mpso_2d_mps_bipartite):
    def fetch_implementation(self,H):
        if type(H) == oneBodyH:
            return gradImplementation_mpso_2d_mps_bipartite_oneBodyH(self.psi,H.tensor)
        elif type(H) == twoBodyH_hori:
            return gradImplementation_mpso_2d_mps_bipartite_twoBodyH_hori(self.psi,H.tensor)
        elif type(H) == twoBodyH_vert:
            return gradImplementation_mpso_2d_mps_bipartite_twoBodyH_vert(self.psi,H.tensor)
