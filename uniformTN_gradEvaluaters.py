#!/usr/bin/env python# -*- coding: utf-8 -*-

from __future__ import division
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from scipy.sparse import linalg as sparse_linalg
from ncon import ncon
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

    elif type(psi) == uMPSU1_2d_left_NSite_block:
        return gradEvaluater_mpso_2d_uniform(psi,H)

    elif type(psi) == uMPSU1_2d_left_fourSite_sep:
        return gradEvaluater_mpso_2d_fourSite_sep(psi,H)

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
        elif type(H) == threeBodyH:
            return gradImplementation_uniform_1d_oneSiteLeft_threeBodyH(self.psi,H.tensor)

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
            self.index1,self.index2 = 1,2
        elif grad_index == 2:
            self.index1,self.index2 = 2,1
        super().__init__(psi,H)
    
    def fetch_implementation(self,H):
        if type(H) == oneBodyH:
            return gradImplementation_bipartite_1d_left_oneBodyH(self.psi,H.tensor,self.H_index,self.grad_index)
        elif type(H) == twoBodyH or type(H) == twoBodyH_hori or type(H) == twoBodyH_vert:
            return gradImplementation_bipartite_1d_left_twoBodyH(self.psi,H.tensor,self.H_index,self.grad_index)
        elif type(H) == threeBodyH:
            return gradImplementation_bipartite_1d_left_threeBodyH(self.psi,H.tensor,self.H_index,self.grad_index)

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

class gradEvaluater_mpso_2d_multipleTensors(gradEvaluater_mpso_2d):
    def copyEvaluaters(self):
        for n in range(1,self.psi.noTensors+1):
            self.grad['mps'+str(n)] = self.gradA_evaluater.grad[n]
            self.grad['mpo'+str(n)] = self.gradB_evaluater.grad[n]
class gradEvaluater_mpso_2d_bipartite(gradEvaluater_mpso_2d_multipleTensors):
    def __init__(self,psi,H):
        super().__init__(psi,H)
        self.gradA_evaluater = gradEvaluater_mpso_2d_mps_bipartite(psi,H)
        self.gradB_evaluater = gradEvaluater_mpso_2d_mpo_bipartite(psi,H)
class gradEvaluater_mpso_2d_fourSite_sep(gradEvaluater_mpso_2d_multipleTensors):
    def __init__(self,psi,H):
        super().__init__(psi,H)
        self.gradA_evaluater = gradEvaluater_mpso_2d_mps_fourSite_sep(psi,H)
        self.gradB_evaluater = gradEvaluater_mpso_2d_mpo_fourSite_sep(psi,H)
# -------------------------------------------------------------------------------------------------------------------------------------
#2d ansatz mps gradient
class gradEvaluater_mpso_2d_mps(gradEvaluater):
    #gradient of the mps tensor of 2d MPSO ansatz
    #this may be constructed as the gradient of a mps in 1d with an effective Hamiltonian
    #so just find effective Hamiltonians, then reuse 1d code to compute gradient terms
    def __init__(self,psi,H):
        super().__init__(psi,H)
        #effective 1d mps
        self.eff_psi = copy.deepcopy(self.psi)
        self.eff_psi.D = self.psi.D_mps
        self.eff_psi.R = self.psi.T 
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
        temp = effH.terms[0].matrix
        return gradEvaluater_uniform_1d_oneSiteLeft(self.eff_psi,effH)

    def fetch_implementation(self,H):
        if type(H) == oneBodyH:
            return gradImplementation_mpso_2d_mps_uniform_oneBodyH(self.psi,H.tensor)
        elif type(H) == twoBodyH_hori:
            return gradImplementation_mpso_2d_mps_uniform_twoBodyH_hori(self.psi,H.tensor)
        elif type(H) == twoBodyH_vert:
            return gradImplementation_mpso_2d_mps_uniform_twoBodyH_vert(self.psi,H.tensor)
        elif type(H) == plaquetteH:
            return gradImplementation_mpso_2d_mps_uniform_plaquetteH(self.psi,H.tensor)
        elif type(H) == cross2dH:
            return gradImplementation_mpso_2d_mps_uniform_cross2dH(self.psi,H.tensor)

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

# -------------------------------------------------------------------------------------------------------------------------------------
#2d ansatz mpo gradient
class gradEvaluater_mpso_2d_mpo(gradEvaluater):
    def gradMagnitude(self,grad):
        return np.einsum('ijab,ijab',grad,grad.conj())

    def eval_non_geo(self,H_term_index):
        n = H_term_index
        return self.H_imp[n].gradInline()

    #geometric sum
    def eval_geo(self,H_term_index,Td,rightFP_d,outers_d):
        n = H_term_index
        grad = self.H_imp[n].gradAbove(rightFP_d,outers_d)
        grad += self.H_imp[n].gradBelow(rightFP_d,outers_d)
        grad += self.H_imp[n].gradLeftQuadrants(rightFP_d,outers_d)
        return grad

    def eval(self,geo=True,envTol=1e-5,printEnv=True):
        self.grad = self.eval_non_geo(0).astype(complex)
        for n in range(0,len(self.H.terms)):
            if n > 0:
                self.grad += self.eval_non_geo(n)

            if geo is True:
                Td_matrix,Td = self.H_imp[n].d_dep.init_mps_transfers()
                for d in range(0,100):
                    if d > 0:
                        Td_matrix,Td = self.H_imp[n].d_dep.apply_mps_transfers(Td_matrix)
                    rightFP_d = self.H_imp[n].d_dep.getFixedPoints(d,Td) #bottleneck of algo
                    outers_d = self.H_imp[n].d_dep.getOuterContracts(d,Td)

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
        elif type(H) == plaquetteH:
            return gradImplementation_mpso_2d_mpo_uniform_plaquetteH(self.psi,H.tensor)
        elif type(H) == cross2dH:
            return gradImplementation_mpso_2d_mpo_uniform_cross2dH(self.psi,H.tensor)

    def projectTangentSpace_euclid(self):
        self.grad = project_mpo_euclid(self.grad,self.psi.mpo)
    def projectTangentSpace_tdvp(self):
        rho = ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1))
        self.grad = project_mpo_tdvp_leftGauge(self.grad,self.psi.mpo,self.psi.R,rho)

class gradEvaluater_mpso_2d_mpo_multipleTensors(gradEvaluater_mpso_2d_mpo):
    def gradMagnitude(self,grad):
        return np.einsum('uijab,uijab',grad,grad.conj())
    def projectTangentSpace_euclid(self):
        for n in range(1,self.psi.noTensors+1):
            self.grad[n] = project_mpo_euclid(self.grad[n],self.psi.mpo[n])
    def eval(self,geo=True,envTol=1e-5,printEnv=True):
        grad = dict()
        super().eval(geo,envTol,printEnv)
        for n in range(1,self.psi.noTensors+1):
            grad[n] = self.grad[n-1]/self.psi.noTensors
        self.grad = grad

class gradEvaluater_mpso_2d_mpo_bipartite(gradEvaluater_mpso_2d_mpo_multipleTensors):
    def fetch_implementation(self,H):
        if type(H) == oneBodyH:
            return gradImplementation_mpso_2d_mpo_bipartite_oneBodyH(self.psi,H.tensor)
        elif type(H) == twoBodyH_hori:
            return gradImplementation_mpso_2d_mpo_bipartite_twoBodyH_hori(self.psi,H.tensor)
        elif type(H) == twoBodyH_vert:
            return gradImplementation_mpso_2d_mpo_bipartite_twoBodyH_vert(self.psi,H.tensor)

    def projectTangentSpace_tdvp(self):
        rho1 = ncon([self.psi.mps[1],self.psi.mps[1].conj(),self.psi.T[2].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1))
        rho2 = ncon([self.psi.mps[2],self.psi.mps[2].conj(),self.psi.T[1].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1))
        self.grad[1] = project_mpo_tdvp_leftGauge(self.grad[1],self.psi.mpo[1],self.psi.R[2],rho1)
        self.grad[2] = project_mpo_tdvp_leftGauge(self.grad[2],self.psi.mpo[2],self.psi.R[1],rho2)

class gradEvaluater_mpso_2d_mpo_fourSite_sep(gradEvaluater_mpso_2d_mpo_multipleTensors):
    def fetch_implementation(self,H):
        if type(H) == oneBodyH:
            return gradImplementation_mpso_2d_mpo_fourSite_sep_oneBodyH(self.psi,H.tensor)
        elif type(H) == twoBodyH_hori:
            return gradImplementation_mpso_2d_mpo_fourSite_sep_twoBodyH_hori(self.psi,H.tensor)
        elif type(H) == twoBodyH_vert:
            return gradImplementation_mpso_2d_mpo_fourSite_sep_twoBodyH_vert(self.psi,H.tensor)
    def projectTangentSpace_tdvp(self):
        rho1 = ncon([self.psi.mps[1],self.psi.mps[1].conj(),self.psi.T[3].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1))
        rho2 = ncon([self.psi.mps[2],self.psi.mps[2].conj(),self.psi.T[4].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1))
        rho3 = ncon([self.psi.mps[3],self.psi.mps[3].conj(),self.psi.T[1].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1))
        rho4 = ncon([self.psi.mps[4],self.psi.mps[4].conj(),self.psi.T[2].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1))
        self.grad[1] = project_mpo_tdvp_leftGauge(self.grad[1],self.psi.mpo[1],self.psi.R[2],rho1)
        self.grad[2] = project_mpo_tdvp_leftGauge(self.grad[2],self.psi.mpo[2],self.psi.R[1],rho2)
        self.grad[3] = project_mpo_tdvp_leftGauge(self.grad[3],self.psi.mpo[3],self.psi.R[4],rho3)
        self.grad[4] = project_mpo_tdvp_leftGauge(self.grad[4],self.psi.mpo[4],self.psi.R[3],rho4)


class gradEvaluater_mpso_2d_mpo_twoSite(gradEvaluater_mpso_2d_mpo):
    def gradMagnitude(self,grad):
        return np.einsum('abcdef,abcdef',grad,grad.conj())
    def projectTangentSpace_euclid(self):
        gradB = self.grad.reshape(4,4,self.psi.D_mpo,self.psi.D_mpo)
        B = self.psi.mpo.reshape(4,4,self.psi.D_mpo,self.psi.D_mpo)
        self.grad = project_mpo_euclid(gradB,B).reshape(2,2,2,2,self.psi.D_mpo,self.psi.D_mpo)

class gradEvaluater_mpso_2d_mpo_twoSite_square(gradEvaluater_mpso_2d_mpo_twoSite):
    def fetch_implementation(self,H):
        if type(H) == oneBodyH:
            return gradImplementation_mpso_2d_mpo_twoSite_square_oneBodyH(self.psi,H.tensor)
        elif type(H) == twoBodyH_hori:
            return gradImplementation_mpso_2d_mpo_twoSite_square_twoBodyH_hori(self.psi,H.tensor)
        elif type(H) == twoBodyH_vert:
            return gradImplementation_mpso_2d_mpo_twoSite_square_twoBodyH_vert(self.psi,H.tensor)

    def eval(self,geo=True,envTol=1e-5,printEnv=True):
        super().eval(geo,envTol,printEnv)
        self.grad = self.grad/4 

    def projectTangentSpace_tdvp(self):
        print("twoSite_square TDVP projector NOT IMPLEMENTED: Doing nothing to project")
        pass

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

    def fetch_implementation(self,H):
        indexSetter = mpso_2d_mps_bipartite_indexSetter
        if type(H) == oneBodyH:
            return gradImplementation_mpso_2d_mps_multipleTensors_oneBodyH(self.psi,H.tensor,indexSetter = indexSetter)
        elif type(H) == twoBodyH_hori:
            return gradImplementation_mpso_2d_mps_multipleTensors_twoBodyH_hori(self.psi,H.tensor,indexSetter = indexSetter)
        elif type(H) == twoBodyH_vert:
            return gradImplementation_mpso_2d_mps_multipleTensors_twoBodyH_vert(self.psi,H.tensor,indexSetter = indexSetter)

class gradEvaluater_mpso_2d_mps_fourSite_sep(gradEvaluater):
    def effective_1d_bipartite_psi(self,index1,index2):
        eff_psi= copy.deepcopy(self.psi)
        eff_psi.D = self.psi.D_mps
        eff_psi.mps[1], eff_psi.mps[2] = self.psi.mps[index1], self.psi.mps[index2]
        eff_psi.R = dict()
        eff_psi.Ta[1], eff_psi.Ta[2] = self.psi.Ta[index1], self.psi.Ta[index2]
        eff_psi.R[1], eff_psi.R[2] = self.psi.T[index1], self.psi.T[index2]
        eff_psi.Ta_inv[1], eff_psi.Ta_inv[2] = self.psi.Ta_inv[index1], self.psi.Ta_inv[index2]
        return eff_psi

    def eval(self):
        effH_1,effH_2,effH_3,effH_4 = [],[],[],[]
        for n in range(0,len(self.H.terms)):
            effH_1.append(self.H_imp[n].getEffectiveH(1))
            effH_2.append(self.H_imp[n].getEffectiveH(2))
            effH_3.append(self.H_imp[n].getEffectiveH(3))
            effH_4.append(self.H_imp[n].getEffectiveH(4))
        effH_1, effH_2, effH_3, effH_4 = localH(effH_1), localH(effH_2), localH(effH_3), localH(effH_4)

        #reuse 1d code here
        eff_psi_13 = self.effective_1d_bipartite_psi(1,3)
        eff_psi_24 = self.effective_1d_bipartite_psi(2,4)

        gradEvaluater_11 = gradEvaluater_bipartite_1d_left_ind(eff_psi_13,effH_1,H_index=1,grad_index=1)
        gradEvaluater_13 = gradEvaluater_bipartite_1d_left_ind(eff_psi_13,effH_1,H_index=1,grad_index=2)
        gradEvaluater_31 = gradEvaluater_bipartite_1d_left_ind(eff_psi_13,effH_3,H_index=2,grad_index=1)
        gradEvaluater_33 = gradEvaluater_bipartite_1d_left_ind(eff_psi_13,effH_3,H_index=2,grad_index=2)

        gradEvaluater_22 = gradEvaluater_bipartite_1d_left_ind(eff_psi_24,effH_2,H_index=1,grad_index=1)
        gradEvaluater_24 = gradEvaluater_bipartite_1d_left_ind(eff_psi_24,effH_2,H_index=1,grad_index=2)
        gradEvaluater_42 = gradEvaluater_bipartite_1d_left_ind(eff_psi_24,effH_4,H_index=2,grad_index=1)
        gradEvaluater_44 = gradEvaluater_bipartite_1d_left_ind(eff_psi_24,effH_4,H_index=2,grad_index=2)

        gradEvaluater_11.eval()
        gradEvaluater_13.eval()
        gradEvaluater_31.eval()
        gradEvaluater_33.eval()
        gradEvaluater_22.eval()
        gradEvaluater_24.eval()
        gradEvaluater_42.eval()
        gradEvaluater_44.eval()

        self.grad = dict()
        self.grad[1] = 1/4*(gradEvaluater_11.grad + gradEvaluater_31.grad)
        self.grad[2] = 1/4*(gradEvaluater_22.grad + gradEvaluater_42.grad)
        self.grad[3] = 1/4*(gradEvaluater_33.grad + gradEvaluater_13.grad)
        self.grad[4] = 1/4*(gradEvaluater_44.grad + gradEvaluater_24.grad)

    def projectTangentSpace_euclid(self):
        self.grad[1] = project_mps_euclid(self.grad[1],self.psi.mps[1])
        self.grad[2] = project_mps_euclid(self.grad[2],self.psi.mps[2])
        self.grad[3] = project_mps_euclid(self.grad[3],self.psi.mps[3])
        self.grad[4] = project_mps_euclid(self.grad[4],self.psi.mps[4])
    def projectTangentSpace_tdvp(self):
        self.grad[1] = project_mps_tdvp_leftGauge(self.grad[1],self.psi.mps[1],self.psi.T[3])
        self.grad[2] = project_mps_tdvp_leftGauge(self.grad[2],self.psi.mps[2],self.psi.T[4])
        self.grad[3] = project_mps_tdvp_leftGauge(self.grad[3],self.psi.mps[3],self.psi.T[1])
        self.grad[4] = project_mps_tdvp_leftGauge(self.grad[4],self.psi.mps[4],self.psi.T[2])

    def fetch_implementation(self,H):
        indexSetter = mpso_2d_mps_fourSite_sep_indexSetter
        if type(H) == oneBodyH:
            return gradImplementation_mpso_2d_mps_multipleTensors_oneBodyH(self.psi,H.tensor,indexSetter=indexSetter)
        elif type(H) == twoBodyH_hori:
            return gradImplementation_mpso_2d_mps_multipleTensors_twoBodyH_hori(self.psi,H.tensor,indexSetter=indexSetter)
        elif type(H) == twoBodyH_vert:
            return gradImplementation_mpso_2d_mps_multipleTensors_twoBodyH_vert(self.psi,H.tensor,indexSetter=indexSetter)
