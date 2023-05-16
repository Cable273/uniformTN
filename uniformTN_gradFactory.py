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
from uniformTN_states import *

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

class gradEvaluater_uniform_1d(gradEvaluater):
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

    @abstractmethod
    def attachRight(self):
        pass
    @abstractmethod
    def attachLeft(self):
        pass

class gradImplementation_mpso_2d_mps(ABC):
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
    def buildTopEnv(self):
        pass
    @abstractmethod
    def buildBotEnv(self):
        pass
    @abstractmethod
    def getCentralTerms(self):
        pass
    @abstractmethod
    def attachRight(self):
        pass
    @abstractmethod
    def attachRightUpperBoundary(self):
        pass
    @abstractmethod
    def attachRightLowerBoundary(self):
        pass

class gradEvaluater_mpso_2d_mps(gradEvaluater):
    @abstractmethod
    def attachTop(self):
        pass
    @abstractmethod
    def attachBot(self):
        pass
    def eval(self):
        #terms underneath Hamiltonian
        self.grad = self.H_imp[0].getCentralTerms()
        print(self.H_imp[0].getCentralTerms())
        for n in range(1,len(self.H.terms)):
            self.grad += self.H_imp[n].getCentralTerms()

        #terms removed above H
        env = self.H_imp[0].buildBotEnv()
        for n in range(1,len(self.H.terms)):
            env += self.H_imp[n].buildBotEnv()
        self.grad += self.attachTop(env)

        #terms removed below H
        env = self.H_imp[0].buildTopEnv()
        for n in range(1,len(self.H.terms)):
            env += self.H_imp[n].buildTopEnv()
        self.grad += self.attachBot(env)

        #terms removed right of H
        #(terms removed left of H don't contribute (pull through condition))

        #must do horizontal terms seperate, 
        # as different length H's env don't add (different dimension environments)
        leftEnv = dict()
        for n in range(0,len(self.H.terms)):
            leftEnv[n] = self.H_imp[n].buildLeftEnv()
            self.grad += self.H_imp[n].attachRight(leftEnv[n])

        #corner quadrants
        for n in range(0,len(self.H.terms)):
            #right upper
            env = self.H_imp[n].attachRightUpperBoundary(leftEnv[n])
            self.grad += self.attachTop(env)
            #right lower
            env = self.H_imp[n].attachRightLowerBoundary(leftEnv[n])
            self.grad += self.attachBot(env)


class gradEvaluater_mpso_2d_mps_uniform(gradEvaluater_mpso_2d_mps):
    def fetch_implementation(self,H):
        if type(H) == oneBodyH:
            return gradImplementation_mpso_2d_mps_uniform_oneBodyH(self.psi,H.tensor)
        elif type(H) == twoBodyH_hori:
            return gradImplementation_mpso_2d_mps_uniform_twoBodyH_hori(self.psi,H.tensor)
        elif type(H) == twoBodyH_vert:
            return gradImplementation_mpso_2d_mps_uniform_twoBodyH_vert(self.psi,H.tensor)

    def attachTop(self,env):
        env = self.psi.Ta_inv.applyRight(env.reshape(self.psi.D_mps**2)).reshape(self.psi.D_mps,self.psi.D_mps)
        return ncon([self.psi.mps,env,self.psi.T.tensor],((-1,2,3),(-4,2),(-5,3)),forder=(-1,-4,-5))
    def attachBot(self,env):
        env = self.psi.Ta_inv.applyLeft(env.reshape(self.psi.D_mps**2)).reshape(self.psi.D_mps,self.psi.D_mps)
        return ncon([self.psi.mps,env],((-1,-2,3),(-4,3)),forder=(-1,-2,-4))

class gradImplementation_mpso_2d_mps_uniform_twoBodyH_hori(gradImplementation_mpso_2d_mps):
    def getCentralTerms(self):
        pass
    def buildLeftEnv(self):
        pass
    def buildRightEnv(self):
        pass
    def buildTopEnv(self):
        pass
    def buildBotEnv(self):
        pass
    def attachRight(self):
        pass
    def attachRightUpperBoundary(self):
        pass
    def attachRightLowerBoundary(self):
        pass

class gradImplementation_mpso_2d_mps_uniform_twoBodyH_vert(gradImplementation_mpso_2d_mps):
    def getCentralTerms(self):
        pass
    def buildLeftEnv(self):
        pass
    def buildRightEnv(self):
        pass
    def buildTopEnv(self):
        pass
    def buildBotEnv(self):
        pass
    def attachRight(self):
        pass
    def attachRightUpperBoundary(self):
        pass
    def attachRightLowerBoundary(self):
        pass

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
