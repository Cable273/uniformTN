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

class env_mpso:
    def __init__(self,psi,tensor,shape=None,label=None):
        self.psi = psi
        self.tensor = tensor
        self.D = self.psi.D_mpo
        self.label = label
        self.shape = shape
    def __add__(self,env2):
        newEnv = copy.deepcopy(self)
        newEnv.tensor = self.tensor + env2.tensor
        return newEnv
    def __sub__(self,env2):
        newEnv = copy.deepcopy(self)
        newEnv.tensor = self.tensor - env2.tensor
        return newEnv
    def print(self):
        print(self,self.shape,self.label)

class environmentGroup:
    def __init__(self,envs=None):
        if envs is None:
            self.envs = []
        else:
            self.envs = envs

    def add(self,newEnv):
        if isinstance(newEnv,environmentGroup):
            self.add_envGroup(newEnv)
        else:
            self.add_env(newEnv)
    def add_env(self,newEnv):
        added = 0
        for n in range(0,len(self.envs)):
            if type(self.envs[n]) == type(newEnv): 
                if self.envs[n].shape == newEnv.shape and self.envs[n].label == newEnv.label:
                    self.envs[n] += newEnv
                    added = 1
                    break
        if added == 0:
            self.envs.append(newEnv)
    def add_envGroup(self,newEnvGroup):
        for n in range(0,len(newEnvGroup.envs)):
            self.add_env(newEnvGroup.envs[n])

    def print(self):
        for n in range(0,len(self.envs)):
            self.envs[n].print()

# ---------------------------------------------------------------------------------------------------
#env_mpso_hori
class env_mpso_hori(env_mpso):
    @abstractmethod 
    def gradLeft(self):
        pass

class envGroup_mpso_hori(environmentGroup):
    def gradLeft(self,outers):
        grad = self.envs[0].gradLeft(outers)
        for n in range(1,len(self.envs)):
            grad += self.envs[n].gradLeft(outers)
        return grad

class env_mpso_hori_uniform(env_mpso_hori):
    def gradLeft(self,outers):
        env = self.psi.Tb_inv.applyLeft(self.tensor.reshape(self.D**2)).reshape(self.D,self.D)
        return ncon([env,self.psi.mpo,outers['len1']],((-6,5),(-2,1,-4,5),(-3,1)),forder=(-2,-3,-4,-6),order=(5,1))

class env_mpso_hori_twoSite_square(env_mpso_hori):
    def gradLeft(self,outers):
        env = self.psi.Tb_inv[self.label].applyLeft(self.tensor.reshape(self.D**2)).reshape(self.D,self.D)
        return ncon([env,self.psi.mpo,outers['len1'][self.label],outers['len1'][self.label]],((-9,8),(-2,-5,1,4,-7,8),(-3,1),(-6,4)),forder=(-2,-5,-3,-6,-7,-9),order=(8,4,1))

class env_mpso_hori_multipleTensors(env_mpso_hori):
    def gradLeft_siteLabels(self,siteLabels,outers):
        grad = np.zeros(np.append([self.psi.noTensors],self.psi.mpo[1].shape)).astype(complex)
        env = self.psi.Tb_inv[siteLabels[0]].applyLeft(self.tensor.reshape(self.D**2)).reshape(self.D,self.D)
        grad[siteLabels[1]-1] = ncon([env,self.psi.mpo[siteLabels[1]],outers['len1'][siteLabels[1]]],((-6,5),(-2,1,-4,5),(-3,1)),forder=(-2,-3,-4,-6),order=(5,1))
        #extra site to left
        env = ncon([env,self.psi.mpo[siteLabels[1]],self.psi.mpo[siteLabels[1]].conj(),outers['len1'][siteLabels[1]]],((7,5),(2,1,-4,5),(2,3,-6,7),(3,1)),forder=(-6,-4),order=(5,7,2,1,3))
        grad[siteLabels[0]-1] = ncon([env,self.psi.mpo[siteLabels[0]],outers['len1'][siteLabels[0]]],((-6,5),(-2,1,-4,5),(-3,1)),forder=(-2,-3,-4,-6),order=(5,1))
        return grad

class env_mpso_hori_bipartite(env_mpso_hori_multipleTensors):
    def gradLeft(self,outers):
        if self.label == 1:
            return self.gradLeft_siteLabels([1,2],outers)
        elif self.label == 2:
            return self.gradLeft_siteLabels([2,1],outers)

class env_mpso_hori_fourSite_sep(env_mpso_hori_multipleTensors):
    def gradLeft(self,outers):
        if self.label == 1:
            return self.gradLeft_siteLabels([1,2,3,4],outers)
        elif self.label == 2:
            return self.gradLeft_siteLabels([2,1,4,3],outers)
        elif self.label == 3:
            return self.gradLeft_siteLabels([3,4,1,2],outers)
        elif self.label == 4:
            return self.gradLeft_siteLabels([4,3,2,1],outers)

# ---------------------------------------------------------------------------------------------------
#env_mpso_vert
class env_mpso_vert(env_mpso):
    def __init__(self,psi,tensor,shape,label=None):
        super().__init__(psi,tensor,shape,label)
        self.eval = self.setEval()

    @abstractmethod 
    def setEval(self):
        pass
    @abstractmethod 
    def gradCentre(self):
        pass
    @abstractmethod 
    def gradAbove(self):
        pass
    @abstractmethod 
    def gradBelow(self):
        pass
    @abstractmethod 
    def build_hori(self):
        pass

class envGroup_mpso_vert(environmentGroup):
    def gradCentre(self,outers):
        grad = self.envs[0].gradCentre(outers)
        for n in range(1,len(self.envs)):
            grad += self.envs[n].gradCentre(outers)
        return grad

    def gradAbove(self,fixedPoints,outers):
        grad = self.envs[0].gradAbove(fixedPoints,outers)
        for n in range(1,len(self.envs)):
            grad += self.envs[n].gradAbove(fixedPoints,outers)
        return grad

    def gradBelow(self,fixedPoints,outers):
        grad = self.envs[0].gradBelow(fixedPoints,outers)
        for n in range(1,len(self.envs)):
            grad += self.envs[n].gradBelow(fixedPoints,outers)
        return grad

    def build_hori(self,fixedPoints,outers):
        envs = envGroup_mpso_hori()
        for n in range(0,len(self.envs)):
            envs.add(self.envs[n].build_hori(fixedPoints,outers))
        return envs

class env_mpso_vert_uniform(env_mpso_vert):
    def setEval(self):
        if self.shape == [1,1]:
            return env_mpso_vert_uniform_eval_1x1(self.psi)
        elif self.shape == [1,2]:
            return env_mpso_vert_uniform_eval_1x2(self.psi)
        elif self.shape == [2,1]:
            return env_mpso_vert_uniform_eval_2x1(self.psi)
        elif self.shape == [2,2]:
            return env_mpso_vert_uniform_eval_2x2(self.psi)
        elif self.shape == [3,3]:
            return env_mpso_vert_uniform_eval_3x3(self.psi)
        elif self.shape == [3,2]:
            return env_mpso_vert_uniform_eval_3x2(self.psi)
        elif self.shape == [3,1]:
            return env_mpso_vert_uniform_eval_3x1(self.psi)

    def gradCentre(self,outers):
        return self.eval.gradCentre(self.tensor,outers)
    def gradAbove(self,fixedPoints,outers):
        return self.eval.gradAbove(self.tensor,fixedPoints,outers)
    def gradBelow(self,fixedPoints,outers):
        return self.eval.gradBelow(self.tensor,fixedPoints,outers)
    def build_hori(self,fixedPoints,outers):
        grad = self.gradAbove(fixedPoints,outers) + self.gradBelow(fixedPoints,outers)
        return env_mpso_hori_uniform(self.psi, ncon([grad,self.psi.mpo.conj()],((1,2,-3,4),(1,2,-5,4)),forder=(-5,-3),order=(4,1,2)) )

class env_mpso_vert_twoSite_square(env_mpso_vert):
    def setEval(self):
        if self.shape == [1,1]:
            if self.label == 'bot':
                return env_mpso_vert_twoSite_square_eval_1x1_envBot(self.psi)
            elif self.label == 'top':
                return env_mpso_vert_twoSite_square_eval_1x1_envTop(self.psi)
        elif self.shape == [1,2]:
            if self.label == 'bot':
                return env_mpso_vert_twoSite_square_eval_1x2_envBot(self.psi)
            elif self.label == 'top':
                return env_mpso_vert_twoSite_square_eval_1x2_envTop(self.psi)
        elif self.shape == [2,1]:
            if self.label == 'square':
                return env_mpso_vert_twoSite_square_eval_2x1_envSquare(self.psi)
            elif self.label == 'prong':
                return env_mpso_vert_twoSite_square_eval_2x1_envProng(self.psi)

    def gradCentre(self,outers):
        return self.eval.gradCentre(self.tensor,outers)
    def gradAbove(self,fixedPoints,outers):
        return self.eval.gradAbove('bot',self.tensor,fixedPoints,outers)  + self.eval.gradAbove('top',self.tensor,fixedPoints,outers) 
    def gradBelow(self,fixedPoints,outers):
        return self.eval.gradBelow('bot',self.tensor,fixedPoints,outers) + self.eval.gradBelow('top',self.tensor,fixedPoints,outers)
    def build_hori(self,fixedPoints,outers):
        gradBot = self.eval.gradAbove('bot',self.tensor,fixedPoints,outers) + self.eval.gradBelow('bot',self.tensor,fixedPoints,outers)
        gradTop = self.eval.gradAbove('top',self.tensor,fixedPoints,outers) + self.eval.gradBelow('top',self.tensor,fixedPoints,outers)
        envBot = ncon([gradBot,self.psi.mpo.conj()],((1,2,3,4,-5,6),(1,2,3,4,-7,6)),forder=(-7,-5),order=(6,2,4,1,3))
        envTop = ncon([gradTop,self.psi.mpo.conj()],((1,2,3,4,-5,6),(1,2,3,4,-7,6)),forder=(-7,-5),order=(6,2,4,1,3))
        env = envGroup_mpso_hori()
        env.add(env_mpso_hori_twoSite_square(self.psi,envBot,label='bot'))
        env.add(env_mpso_hori_twoSite_square(self.psi,envTop,label='top'))
        return env

class env_mpso_vert_multipleTensors(env_mpso_vert):
    def gradCentre(self,outers):
        grad = np.zeros(np.append([self.psi.noTensors],self.psi.mpo[1].shape)).astype(complex)
        for n in range(1,self.psi.noTensors+1):
            grad[n-1] = self.eval.gradCentre_tensor(n, self.tensor,outers)
        return grad
    def gradAbove(self,fixedPoints,outers):
        grad = np.zeros(np.append([self.psi.noTensors],self.psi.mpo[1].shape)).astype(complex)
        for n in range(1,self.psi.noTensors+1):
            grad[n-1] = self.eval.gradAbove_tensor(n, self.tensor,fixedPoints,outers)
        return grad
    def gradBelow(self,fixedPoints,outers):
        grad = np.zeros(np.append([self.psi.noTensors],self.psi.mpo[1].shape)).astype(complex)
        for n in range(1,self.psi.noTensors+1):
            grad[n-1] = self.eval.gradBelow_tensor(n, self.tensor,fixedPoints,outers)
        return grad
    def build_hori(self,fixedPoints,outers):
        grad = self.gradAbove(fixedPoints,outers) + self.gradBelow(fixedPoints,outers)
        envs_hori = envGroup_mpso_hori()
        for n in range(1,self.psi.noTensors+1):
            envs_hori.add(env_mpso_hori_bipartite(self.psi, ncon([grad[n-1],self.psi.mpo[n].conj()],((1,2,-3,4),(1,2,-5,4)),forder=(-5,-3),order=(4,1)), label = n ))
        return envs_hori

class env_mpso_vert_bipartite(env_mpso_vert_multipleTensors):
    def setEval(self):
        if self.label == 1:
            if self.shape == [1,1]:
                return env_mpso_vert_bipartite_eval_1x1_env1(self.psi)
            elif self.shape == [1,2]:
                return env_mpso_vert_bipartite_eval_1x2_env1(self.psi)
            elif self.shape == [2,1]:
                return env_mpso_vert_bipartite_eval_2x1_env1(self.psi)
        elif self.label == 2:
            if self.shape == [1,1]:
                return env_mpso_vert_bipartite_eval_1x1_env2(self.psi)
            elif self.shape == [1,2]:
                return env_mpso_vert_bipartite_eval_1x2_env2(self.psi)
            elif self.shape == [2,1]:
                return env_mpso_vert_bipartite_eval_2x1_env2(self.psi)

class env_mpso_vert_fourSite_sep(env_mpso_vert_multipleTensors):
    def setEval(self):
        if self.label == 1:
            if self.shape == [1,1]:
                return env_mpso_vert_fourSite_sep_eval_1x1_env1(self.psi)
            elif self.shape == [1,2]:
                return env_mpso_vert_fourSite_sep_eval_1x2_env1(self.psi)
            elif self.shape == [2,1]:
                return env_mpso_vert_fourSite_sep_eval_2x1_env1(self.psi)
        elif self.label == 2:
            if self.shape == [1,1]:
                return env_mpso_vert_fourSite_sep_eval_1x1_env2(self.psi)
            elif self.shape == [1,2]:
                return env_mpso_vert_fourSite_sep_eval_1x2_env2(self.psi)
            elif self.shape == [2,1]:
                return env_mpso_vert_fourSite_sep_eval_2x1_env2(self.psi)
        elif self.label == 3:
            if self.shape == [1,1]:
                return env_mpso_vert_fourSite_sep_eval_1x1_env3(self.psi)
            elif self.shape == [1,2]:
                return env_mpso_vert_fourSite_sep_eval_1x2_env3(self.psi)
            elif self.shape == [2,1]:
                return env_mpso_vert_fourSite_sep_eval_2x1_env3(self.psi)
        elif self.label == 4:
            if self.shape == [1,1]:
                return env_mpso_vert_fourSite_sep_eval_1x1_env4(self.psi)
            elif self.shape == [1,2]:
                return env_mpso_vert_fourSite_sep_eval_1x2_env4(self.psi)
            elif self.shape == [2,1]:
                return env_mpso_vert_fourSite_sep_eval_2x1_env4(self.psi)

# ---------------------------------------------------------------------------------------------------
#gradEvals for env_mpso_vert
class env_gradEval:
    def __init__(self,psi):
        self.psi = psi 
# ---------------------------------------------------------------------------------------------------
#gradEvals for env_mpso_vert uniform
class env_mpso_vert_uniform_eval_1x1(env_gradEval):
    def gradCentre(self,tensor,outers):
        return ncon([tensor,self.psi.R.tensor,outers['len1']],((-2,1,-4,6),(-5,6),(-3,1)),forder=(-2,-3,-4,-5),order=(6,1))
    def gradAbove(self,tensor,fixedPoints,outers):
        return ncon([tensor,self.psi.mpo.conj(),self.psi.mpo,fixedPoints,outers],((5,4,7,9),(5,6,7,8),(-2,1,-12,11),(8,9,-10,11),(6,-3,4,1)),forder=(-2,-3,-12,-10),order=(7,5,4,6,8,9,11,1))
    def gradBelow(self,tensor,fixedPoints,outers):
        return ncon([tensor,self.psi.mpo.conj(),self.psi.mpo,fixedPoints,outers],((2,1,7,8),(2,3,7,9),(-5,4,-12,10),(-11,10,9,8),(-6,3,4,1)),forder=(-5,-6,-12,-11),order=(7,2,1,3,9,8,10,4))

class env_mpso_vert_uniform_eval_1x2(env_gradEval):
    def gradCentre(self,tensor,outers):
        return ncon([tensor,self.psi.mpo.conj(),self.psi.R.tensor,outers['len1'],outers['len1']],((-2,5,1,4,-7,10),(5,6,-8,9),(9,10),(-3,1),(6,4)),forder=(-2,-3,-7,-8),order=(9,10,5,4,6,1))
    def gradAbove(self,tensor,fixedPoints,outers):
        return ncon([tensor,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,fixedPoints,outers,outers],((5,11,4,10,13,16),(5,6,13,14),(11,12,14,15),(8,7,19,18),(8,9,-20,17),(-2,1,-21,19),(15,16,17,18),(6,-3,4,1),(12,9,10,7)),forder=(-2,-3,-21,-20),order=(13,5,4,6,14,11,10,12,15,16,17,18,8,7,9,19,1))
    def gradBelow(self,tensor,fixedPoints,outers):
        return ncon([tensor,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,fixedPoints,outers,outers],((2,8,1,7,13,15),(2,3,13,14),(8,9,14,16),(11,10,19,17),(11,12,-21,18),(-5,4,-20,19),(18,17,16,15),(-6,3,4,1),(12,9,10,7)),forder=(-5,-6,-20,-21),order=(13,2,1,3,15,8,7,9,15,16,17,18,11,10,12,19,4))

class env_mpso_vert_uniform_eval_2x2(env_gradEval):
    def gradCentre(self,tensor,outers):
        grad = ncon([tensor,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.RR.tensor,outers['len2'],outers['len2']],((8,11,-2,5,7,10,1,4,17,20,-13,16),(5,16,-14,15),(8,9,17,18),(11,12,18,19),(19,20,15,16),(9,-3,7,1),(12,16,10,4)),forder=(-2,-3,-13,-14))
        grad += ncon([tensor,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.RR.tensor,outers['len2'],outers['len2']],((-8,11,2,5,7,10,1,4,-17,20,13,16),(2,3,13,14),(5,16,14,15),(11,12,-18,19),(19,20,15,16),(-9,3,7,1),(12,16,10,4)),forder=(-8,-9,-17,-18))
        return grad
    def gradAbove(self,tensor,fixedPoints,outers):
        return ncon([tensor,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,fixedPoints['upper'],outers['upper'],outers['upper']],((8,17,5,14,7,16,4,13,28,30,24,26),(8,9,28,29),(17,18,29,31),(5,6,24,25),(14,15,25,27),(11,10,20,22),(11,12,-21,23),(-2,1,-19,20),(31,30,27,26,23,22),(9,6,-3,7,4,1),(18,15,12,16,13,10)),forder=(-2,-3,-19,-21),order=(28,8,24,5,4,6,7,9,25,29,14,17,13,15,16,18,31,30,27,26,22,23,11,10,12,20,1))
    def gradBelow(self,tensor,fixedPoints,outers):
        return ncon([tensor,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,fixedPoints['upper'],outers['upper'],outers['upper']],((5,14,2,11,4,13,1,10,23,25,19,21),(5,6,23,24),(14,15,24,26),(2,3,19,20),(11,12,20,22),(17,16,28,30),(17,18,-29,31),(-8,7,-27,28),(31,30,26,25,22,21),(-9,6,3,7,4,1),(18,15,12,16,13,10)),forder=(-8,-9,-27,-29),order=(19,2,23,5,1,3,4,6,20,24,11,14,10,12,13,15,21,22,25,26,30,31,17,16,18,28,7))

class env_mpso_vert_uniform_eval_2x1(env_gradEval):
    def gradCentre(self,tensor,outers):
        grad = ncon([tensor,self.psi.mpo.conj(),self.psi.RR.tensor,outers['len2']],((2,-5,1,4,7,9,-10,12),(2,3,7,8),(8,9,-11,12),(3,-6,1,4)),forder=(-5,-6,-10,-11),order=(7,2,1,3,8,9,12,4))
        grad += ncon([tensor,self.psi.mpo.conj(),self.psi.RR.tensor,outers['len2']],((-2,5,1,4,-7,9,10,12),(5,6,10,11),(-8,9,11,12),(-3,6,1,4)),forder=(-2,-3,-7,-8),order=(10,5,4,6,11,12,9,1))
        return grad
    def gradAbove(self,tensor,fixedPoints,outers):
        return ncon([tensor,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo,fixedPoints['upper'],outers['upper']],((8,5,7,4,16,17,13,14),(8,9,16,18),(5,6,13,15),(-2,1,-10,11),(18,17,15,14,-12,11),(9,6,-3,7,4,1)),forder=(-2,-3,-10,-12),order=(16,8,18,17,13,5,14,15,4,6,7,9,11,1))
    def gradBelow(self,tensor,fixedPoints,outers):
        return ncon([tensor,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo,fixedPoints['lower'],outers['lower']],((5,2,4,1,13,14,10,11),(2,3,10,12),(5,6,13,15),(-8,7,-16,17),(-18,17,15,14,12,11),(-9,6,3,7,4,1)),forder=(-8,-9,-16,-18),order=(10,2,11,12,13,5,14,15,1,3,4,6,17,7))

class env_mpso_vert_uniform_eval_3x3(env_gradEval):
    def gradCentre(self,tensor,outers):
        grad = ncon([tensor,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.RRR.tensor,outers['len3'],outers['len3'],outers['len3']],((20,23,26,11,14,17,-2,5,8,19,22,25,10,13,16,1,4,7,38,42,33,37,-28,32),(5,6,-29,30),(8,9,30,31),(11,12,33,34),(14,15,34,35),(17,18,35,36),(20,21,38,39),(23,24,39,40),(26,27,40,41),(41,42,36,37,31,32),(21,12,-3,19,10,1),(24,15,6,22,13,4),(27,18,9,25,16,7)),forder=(-2,-3,-28,-29),order=(7,16,25,4,13,22,1,10,19,32,37,42,31,8,9,36,17,18,41,26,27,30,5,6,35,14,15,40,23,24,34,11,12,33,39,20,21,38))
        grad += ncon([tensor,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.RRR.tensor,outers['len3'],outers['len3'],outers['len3']],((20,23,26,-11,14,17,2,5,8,19,22,25,10,13,16,1,4,7,38,42,-33,37,28,32),(2,3,28,29),(5,6,29,30),(8,9,30,31),(14,15,-34,35),(17,18,35,36),(20,21,38,39),(23,24,39,40),(26,27,40,41),(41,42,36,37,31,32),(21,-12,3,19,10,1),(24,15,6,22,13,4),(27,18,9,25,16,7)),forder=(-11,-12,-33,-34))
        grad += ncon([tensor,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.RRR.tensor,outers['len3'],outers['len3'],outers['len3']],((-20,23,26,11,14,17,2,5,8,19,22,25,10,13,16,1,4,7,-38,42,33,37,28,32),(2,3,28,29),(5,6,29,30),(8,9,30,31),(11,12,33,34),(14,15,34,35),(17,18,35,36),(23,24,-39,40),(26,27,40,41),(41,42,36,37,31,32),(-21,12,3,19,10,1),(24,15,6,22,13,4),(27,18,9,25,16,7)),forder=(-20,-21,-38,-39))
        return grad
    def gradAbove(self,tensor,fixedPoints,outers):
        return np.zeros(self.psi.mpo.shape).astype(complex)
    def gradBelow(self,tensor,fixedPoints,outers):
        return np.zeros(self.psi.mpo.shape).astype(complex)

class env_mpso_vert_uniform_eval_3x2(env_gradEval):
    def gradCentre(self,tensor,outers):
        grad = ncon([tensor,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.RRR.tensor,outers['len3'],outers['len3']],((14,17,8,11,-2,5,13,16,7,10,1,4,27,30,23,26,-19,22),(5,6,-20,21),(8,9,23,24),(11,12,24,25),(14,15,27,28),(17,18,28,29),(29,30,25,26,21,22),(15,9,-3,13,7,1),(18,12,6,16,10,4)),forder=(-2,-3,-19,-20))
        grad += ncon([tensor,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.RRR.tensor,outers['len3'],outers['len3']],((14,17,-8,11,2,5,13,16,7,10,1,4,27,30,-23,26,19,22),(2,3,19,20),(5,6,20,21),(11,12,-24,25),(14,15,27,28),(17,18,28,29),(29,30,25,26,21,22),(15,-9,3,13,7,1),(18,12,6,16,10,4)),forder=(-8,-9,-23,-24))
        grad += ncon([tensor,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.RRR.tensor,outers['len3'],outers['len3']],((-14,17,8,11,2,5,13,16,7,10,1,4,-27,30,23,26,19,22),(2,3,19,20),(5,6,20,21),(8,9,23,24),(11,12,24,25),(17,18,-28,29),(29,30,25,26,21,22),(-15,9,3,13,7,1),(18,12,6,16,10,4)),forder=(-14,-15,-27,-28))
        return grad
    def gradAbove(self,tensor,fixedPoints,outers):
        return np.zeros(self.psi.mpo.shape).astype(complex)
    def gradBelow(self,tensor,fixedPoints,outers):
        return np.zeros(self.psi.mpo.shape).astype(complex)

class env_mpso_vert_uniform_eval_3x1(env_gradEval):
    def gradCentre(self,tensor,outers):
        grad = ncon([tensor,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.RRR.tensor,outers['len3']],((8,5,-2,7,4,1,16,18,13,15,-10,12),(5,6,13,14),(8,9,16,17),(17,18,14,15,-11,12),(9,6,-3,7,4,1)),forder=(-2,-3,-10,-11))
        grad += ncon([tensor,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.RRR.tensor,outers['len3']],((8,-5,2,7,4,1,16,18,-13,15,10,12),(2,3,10,11),(8,9,16,17),(17,18,-14,15,11,12),(9,-6,3,7,4,1)),forder=(-5,-6,-13,-14))
        grad += ncon([tensor,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.RRR.tensor,outers['len3']],((-8,5,2,7,4,1,-16,18,13,15,10,12),(2,3,10,11),(5,6,13,14),(-17,18,14,15,11,12),(-9,6,3,7,4,1)),forder=(-8,-9,-16,-17))
        return grad
    def gradAbove(self,tensor,fixedPoints,outers):
        return np.zeros(self.psi.mpo.shape).astype(complex)
    def gradBelow(self,tensor,fixedPoints,outers):
        return np.zeros(self.psi.mpo.shape).astype(complex)
# ---------------------------------------------------------------------------------------------------
#gradEvals twoSite_square
class env_mpso_vert_twoSite_square_eval_1x1(env_gradEval):
    def gradCentre_ind(self,style,tensor,outers):
        return ncon([tensor,self.psi.R[style].tensor,outers[style],outers[style]],((-2,-5,1,4,-7,8),(-9,8),(-3,1),(-6,4)),forder=(-2,-5,-3,-6,-7,-9),order=(8,4,1))
    def gradAbove_ind(self,style,tensor,fixedPoints,outers):
        return ncon([tensor,self.psi.mpo.conj(),self.psi.mpo,fixedPoints[style],outers[style],outers[style]],((5,11,4,10,16,17),(5,11,6,12,16,18),(-2,-8,1,7,-13,14),(18,17,-15,14),(6,-3,4,1),(12,-9,10,7)),forder=(-2,-8,-3,-9,-13,-15),order=(16,5,11,4,6,10,12,17,18,14,7,1))
    def gradBelow_ind(self,style,tensor,fixedPoints,outers):
        return ncon([tensor,self.psi.mpo.conj(),self.psi.mpo,fixedPoints[style],outers[style],outers[style]],((2,5,1,4,13,14),(2,5,3,6,13,15),(-8,-11,7,10,-16,17),(-18,17,15,14),(-9,3,7,1),(-12,6,10,4)),forder=(-8,-11,-9,-12,-16,-18),order=(13,2,5,1,3,4,6,14,15,17,10,7))

class env_mpso_vert_twoSite_square_eval_1x2(env_gradEval):
    def gradCentre_ind(self,style,tensor,outers):
        return ncon([tensor,self.psi.mpo.conj(),self.psi.R[style].tensor,outers[style],outers[style],outers[style],outers[style]],((-2,-5,8,11,1,4,7,10,-13,14),(8,11,9,12,-16,15),(15,14),(-3,1),(-6,4),(9,7),(12,10)),forder=(-2,-5,-3,-6,-13,-16),order=(15,14,11,8,12,10,9,7,1,4))
    def gradAbove_ind(self,style,tensor,fixedPoints,outers):
        return ncon([tensor,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,fixedPoints[style],outers[style],outers[style],outers[style],outers[style]],((5,11,17,23,4,10,16,22,30,32),(5,11,6,12,30,31),(17,23,18,24,31,33),(14,20,13,19,26,28),(14,20,15,21,-27,29),(-2,-8,1,7,-25,26),(33,32,29,28),(6,-3,4,1),(12,-9,10,7),(18,15,16,13),(24,21,22,19)),forder=(-2,-8,-3,-9,-25,-27),order=(30,5,11,4,6,10,12,31,17,23,16,18,22,24,33,32,28,28,14,20,19,21,13,15,26,7,1))
    def gradBelow_ind(self,style,tensor,fixedPoints,outers):
        return ncon([tensor,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,fixedPoints[style],outers[style],outers[style],outers[style],outers[style]],((2,8,14,20,1,7,13,19,25,27),(2,8,3,9,25,26),(14,20,15,21,26,28),(17,23,16,22,30,32),(17,23,18,24,-31,33),(-5,-11,4,10,-29,30),(33,32,28,27),(-6,3,4,1),(-12,9,10,7),(18,15,16,13),(24,21,22,19)),forder=(-5,-11,-6,-12,-29,-31),order=(25,2,8,1,3,7,9,26,14,20,13,15,19,21,27,28,32,33,17,23,22,24,16,18,30,10,4))

class env_mpso_vert_twoSite_square_eval_2x1(env_gradEval):
    def gradCentre_ind(self,style,tensor,outers):
        grad = ncon([tensor,self.psi.mpo.conj(),self.psi.RR[style].tensor,outers[style],outers[style]],((5,11,-2,-8,4,10,1,7,16,17,-13,14),(5,11,6,12,16,18),(18,17,-15,14),(6,-3,4,1),(12,-9,10,7)),forder=(-2,-8,-3,-9,-13,-15),order=(16,5,11,4,6,10,12,18,17,14,7,1))
        grad += ncon([tensor,self.psi.mpo.conj(),self.psi.RR[style].tensor,outers[style],outers[style]],((-5,-11,2,8,4,10,1,7,-16,17,13,14),(2,8,3,9,13,15),(-18,17,15,14),(-6,3,4,1),(-12,9,10,7)),forder=(-5,-11,-6,-12,-16,-18),order=(13,2,8,1,3,7,9,14,15,17,10,4))
        return grad
    def gradAbove_ind(self,style,tensor,fixedPoints,outers):
        return ncon([tensor,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo,fixedPoints[style],outers[style],outers[style]],((14,17,8,11,13,16,7,10,19,20,22,23),(14,17,15,18,19,21),(8,11,9,12,22,24),(-2,-5,1,4,-25,26),(21,20,24,23,-27,26),(15,9,-3,13,7,1),(18,12,-6,16,10,4)),forder=(-2,-5,-3,-6,-25,-27),order=(19,14,17,13,15,16,18,20,21,23,24,8,11,7,9,10,12,22,26,1,4))
    def gradBelow_ind(self,style,tensor,fixedPoints,outers):
        return ncon([tensor,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.mpo,fixedPoints[style],outers[style],outers[style]],((5,14,2,11,4,13,1,10,22,23,19,20),(5,14,6,15,22,24),(2,11,3,12,19,21),(-8,-17,7,16,-25,26),(-27,26,24,23,21,20),(-9,6,3,7,4,1),(-18,15,12,16,13,10)),forder=(-8,-17,-9,-18,-25,-27),order=(19,2,11,1,3,10,12,20,21,23,24,5,14,4,6,13,15,22,26,16,7))

# ---------------------------------------------------------------------------------------------------
#gradEvals multiple Tensors
class env_mpso_vert_multipleTensors_eval_1x1(env_gradEval):
    def gradCentre_ind(self,tensor,fixedPoints,outers):
        return ncon([tensor,fixedPoints,outers],((-2,1,-4,6),(-5,6),(-3,1)),forder=(-2,-3,-4,-5),order=(6,1))
    def gradAbove_ind(self,envLabels,gradLabels,tensor,fixedPoints,outers):
        return ncon([tensor,self.psi.mpo[envLabels[0]].conj(),self.psi.mpo[gradLabels[0]],fixedPoints,outers],((5,4,7,9),(5,6,7,8),(-2,1,-12,11),(8,9,-10,11),(6,-3,4,1)),forder=(-2,-3,-12,-10),order=(7,5,4,6,8,9,11,1))
    def gradBelow_ind(self,envLabels,gradLabels,tensor,fixedPoints,outers):
        return ncon([tensor,self.psi.mpo[envLabels[0]].conj(),self.psi.mpo[gradLabels[0]],fixedPoints,outers],((2,1,7,8),(2,3,7,9),(-5,4,-12,10),(-11,10,9,8),(-6,3,4,1)),forder=(-5,-6,-12,-11),order=(7,2,1,3,9,8,10,4))

class env_mpso_vert_multipleTensors_eval_1x2(env_gradEval):
    def gradCentre_ind(self,envLabels,tensor,fixedPoints,outers):
        return ncon([tensor,self.psi.mpo[envLabels[0]].conj(),fixedPoints,outers[0],outers[1]],((-2,5,1,4,-7,10),(5,6,-8,9),(9,10),(-3,1),(6,4)),forder=(-2,-3,-7,-8),order=(9,10,5,4,6,1))
    def gradAbove_ind(self,envLabels,gradLabels,tensor,fixedPoints,outers):
        return ncon([tensor,self.psi.mpo[envLabels[0]].conj(),self.psi.mpo[envLabels[1]].conj(),self.psi.mpo[gradLabels[1]],self.psi.mpo[gradLabels[1]].conj(),self.psi.mpo[gradLabels[0]],fixedPoints,outers[0],outers[1]],((5,11,4,10,13,16),(5,6,13,14),(11,12,14,15),(8,7,19,18),(8,9,-20,17),(-2,1,-21,19),(15,16,17,18),(6,-3,4,1),(12,9,10,7)),forder=(-2,-3,-21,-20),order=(13,5,4,6,14,11,10,12,15,16,17,18,8,7,9,19,1))
    def gradBelow_ind(self,envLabels,gradLabels,tensor,fixedPoints,outers):
        return ncon([tensor,self.psi.mpo[envLabels[0]].conj(),self.psi.mpo[envLabels[1]].conj(),self.psi.mpo[gradLabels[1]],self.psi.mpo[gradLabels[1]].conj(),self.psi.mpo[gradLabels[0]],fixedPoints,outers[0],outers[1]],((2,8,1,7,13,15),(2,3,13,14),(8,9,14,16),(11,10,19,17),(11,12,-21,18),(-5,4,-20,19),(18,17,16,15),(-6,3,4,1),(12,9,10,7)),forder=(-5,-6,-20,-21),order=(13,2,1,3,15,8,7,9,15,16,17,18,11,10,12,19,4))

class env_mpso_vert_multipleTensors_eval_2x1(env_gradEval):
    def gradCentre_ind_bot(self,envLabels,tensor,fixedPoints,outers):
        return ncon([tensor,self.psi.mpo[envLabels[0]].conj(),fixedPoints,outers],((-2,5,1,4,-7,9,10,12),(5,6,10,11),(-8,9,11,12),(-3,6,1,4)),forder=(-2,-3,-7,-8),order=(10,5,4,6,11,12,9,1))
    def gradCentre_ind_top(self,envLabels,tensor,fixedPoints,outers):
        return ncon([tensor,self.psi.mpo[envLabels[0]].conj(),fixedPoints,outers],((2,-5,1,4,7,9,-10,12),(2,3,7,8),(8,9,-11,12),(3,-6,1,4)),forder=(-5,-6,-10,-11),order=(7,2,1,3,8,9,12,4))
    def gradAbove_ind(self,envLabels,gradLabels,tensor,fixedPoints,outers):
        return ncon([tensor,self.psi.mpo[envLabels[0]].conj(),self.psi.mpo[envLabels[1]].conj(),self.psi.mpo[gradLabels[0]],fixedPoints,outers],((8,5,7,4,16,17,13,14),(8,9,16,18),(5,6,13,15),(-2,1,-10,11),(18,17,15,14,-12,11),(9,6,-3,7,4,1)),forder=(-2,-3,-10,-12),order=(16,8,18,17,13,5,14,15,4,6,7,9,11,1))
    def gradBelow_ind(self,envLabels,gradLabels,tensor,fixedPoints,outers):
        return ncon([tensor,self.psi.mpo[envLabels[1]].conj(),self.psi.mpo[envLabels[0]].conj(),self.psi.mpo[gradLabels[0]],fixedPoints,outers],((5,2,4,1,13,14,10,11),(2,3,10,12),(5,6,13,15),(-8,7,-16,17),(-18,17,15,14,12,11),(-9,6,3,7,4,1)),forder=(-8,-9,-16,-18),order=(10,2,11,12,13,5,14,15,1,3,4,6,17,7))
# ---------------------------------------------------------------------------------------------------
#twoSite_square
class env_mpso_vert_twoSite_square_eval_1x1_envBot(env_mpso_vert_twoSite_square_eval_1x1):
    def gradCentre(self,tensor,outers):
        return self.gradCentre_ind('bot',tensor,outers['len1'])
    def gradAbove(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 'bot':
            return self.gradAbove_ind('bb',tensor,fixedPoints['RR_d'],outers['outerContractDouble'])
        else:
            return self.gradAbove_ind('bt',tensor,fixedPoints['RR_d'],outers['outerContractDouble'])
        return grad
    def gradBelow(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 'bot':
            return self.gradBelow_ind('bb',tensor,fixedPoints['RR_d'],outers['outerContractDouble'])
        else:
            return self.gradBelow_ind('tb',tensor,fixedPoints['RR_d'],outers['outerContractDouble'])
        return grad

class env_mpso_vert_twoSite_square_eval_1x1_envTop(env_mpso_vert_twoSite_square_eval_1x1):
    def gradCentre(self,tensor,outers):
        return self.gradCentre_ind('top',tensor,outers['len1'])
    def gradAbove(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 'bot':
            return self.gradAbove_ind('tb',tensor,fixedPoints['RR_d'],outers['outerContractDouble'])
        else:
            return self.gradAbove_ind('tt',tensor,fixedPoints['RR_d'],outers['outerContractDouble'])
        return grad
    def gradBelow(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 'bot':
            return self.gradBelow_ind('bt',tensor,fixedPoints['RR_d'],outers['outerContractDouble'])
        else:
            return self.gradBelow_ind('tt',tensor,fixedPoints['RR_d'],outers['outerContractDouble'])
        return grad

class env_mpso_vert_twoSite_square_eval_1x2_envBot(env_mpso_vert_twoSite_square_eval_1x2):
    def gradCentre(self,tensor,outers):
        return self.gradCentre_ind('bot',tensor,outers['len1'])
    def gradAbove(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 'bot':
            return self.gradAbove_ind('bb',tensor,fixedPoints['RR_d'],outers['outerContractDouble'])
        else:
            return self.gradAbove_ind('bt',tensor,fixedPoints['RR_d'],outers['outerContractDouble'])
        return grad
    def gradBelow(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 'bot':
            return self.gradBelow_ind('bb',tensor,fixedPoints['RR_d'],outers['outerContractDouble'])
        else:
            return self.gradBelow_ind('tb',tensor,fixedPoints['RR_d'],outers['outerContractDouble'])
        return grad

class env_mpso_vert_twoSite_square_eval_1x2_envTop(env_mpso_vert_twoSite_square_eval_1x2):
    def gradCentre(self,tensor,outers):
        return self.gradCentre_ind('top',tensor,outers['len1'])
    def gradAbove(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 'bot':
            return self.gradAbove_ind('tb',tensor,fixedPoints['RR_d'],outers['outerContractDouble'])
        else:
            return self.gradAbove_ind('tt',tensor,fixedPoints['RR_d'],outers['outerContractDouble'])
        return grad
    def gradBelow(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 'bot':
            return self.gradBelow_ind('bt',tensor,fixedPoints['RR_d'],outers['outerContractDouble'])
        else:
            return self.gradBelow_ind('tt',tensor,fixedPoints['RR_d'],outers['outerContractDouble'])
        return grad

class env_mpso_vert_twoSite_square_eval_2x1_envSquare(env_mpso_vert_twoSite_square_eval_2x1):
    def gradCentre(self,tensor,outers):
        return self.gradCentre_ind('square',tensor,outers['len2'])
    def gradAbove(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 'bot':
            return self.gradAbove_ind('sb',tensor,fixedPoints['RRR_d_upper'],outers['outerContractTriple_upper'])
        else:
            return self.gradAbove_ind('st',tensor,fixedPoints['RRR_d_upper'],outers['outerContractTriple_upper'])
        return grad
    def gradBelow(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 'bot':
            return self.gradBelow_ind('bs',tensor,fixedPoints['RRR_d_lower'],outers['outerContractTriple_lower'])
        else:
            return self.gradBelow_ind('ts',tensor,fixedPoints['RRR_d_lower'],outers['outerContractTriple_lower'])
        return grad

class env_mpso_vert_twoSite_square_eval_2x1_envProng(env_mpso_vert_twoSite_square_eval_2x1):
    def gradCentre(self,tensor,outers):
        return self.gradCentre_ind('prong',tensor,outers['len2'])
    def gradAbove(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 'bot':
            return self.gradAbove_ind('pb',tensor,fixedPoints['RRR_d_upper'],outers['outerContractTriple_upper'])
        else:
            return self.gradAbove_ind('pt',tensor,fixedPoints['RRR_d_upper'],outers['outerContractTriple_upper'])
        return grad
    def gradBelow(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 'bot':
            return self.gradBelow_ind('bp',tensor,fixedPoints['RRR_d_lower'],outers['outerContractTriple_lower'])
        else:
            return self.gradBelow_ind('tp',tensor,fixedPoints['RRR_d_lower'],outers['outerContractTriple_lower'])
        return grad

# ---------------------------------------------------------------------------------------------------
#bipartite
class env_mpso_vert_bipartite_eval_1x1_env1(env_mpso_vert_multipleTensors_eval_1x1):
    def gradCentre_tensor(self,gradTensorLabel,tensor,outers):
        if gradTensorLabel == 1:
            return self.gradCentre_ind(tensor,self.psi.R[2].tensor,outers['len1'][1])
        elif gradTensorLabel == 2:
            return 0
    def gradAbove_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 1:
            return self.gradAbove_ind([1],[1],tensor,fixedPoints['RR_d_p1'][2],outers['outerContractDouble_p1'][1])
        elif gradTensorLabel == 2:
            return self.gradAbove_ind([1],[2],tensor,fixedPoints['RR_d'][2],outers['outerContractDouble'][1])
    def gradBelow_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 1:
            return self.gradBelow_ind([1],[1],tensor,fixedPoints['RR_d_p1'][2],outers['outerContractDouble_p1'][1])
        elif gradTensorLabel == 2:
            return self.gradBelow_ind([1],[2],tensor,fixedPoints['RR_d'][1],outers['outerContractDouble'][2])

class env_mpso_vert_bipartite_eval_1x1_env2(env_mpso_vert_multipleTensors_eval_1x1):
    def gradCentre_tensor(self,gradTensorLabel,tensor,outers):
        if gradTensorLabel == 2:
            return self.gradCentre_ind(tensor,self.psi.R[1].tensor,outers['len1'][2])
        else:
            return 0
    def gradAbove_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 1:
            return self.gradAbove_ind([2],[1],tensor,fixedPoints['RR_d'][1],outers['outerContractDouble'][2])
        elif gradTensorLabel == 2:
            return self.gradAbove_ind([2],[2],tensor,fixedPoints['RR_d_p1'][1],outers['outerContractDouble_p1'][2])
    def gradBelow_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 1:
            return self.gradBelow_ind([2],[1],tensor,fixedPoints['RR_d'][2],outers['outerContractDouble'][1])
        elif gradTensorLabel == 2:
            return self.gradBelow_ind([2],[2],tensor,fixedPoints['RR_d_p1'][1],outers['outerContractDouble_p1'][2])

class env_mpso_vert_bipartite_eval_1x2_env1(env_mpso_vert_multipleTensors_eval_1x2):
    def gradCentre_tensor(self,gradTensorLabel,tensor,outers):
        if gradTensorLabel == 1:
            return self.gradCentre_ind([2],tensor,self.psi.R[1].tensor,[outers['len1'][1],outers['len1'][2]])
        else:
            return 0
    def gradAbove_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 1:
            return self.gradAbove_ind([1,2],[1,2],tensor,fixedPoints['RR_d_p1'][1],[outers['outerContractDouble_p1'][1],outers['outerContractDouble_p1'][2]])
        else:
            return self.gradAbove_ind([1,2],[2,1],tensor,fixedPoints['RR_d'][1],[outers['outerContractDouble'][1],outers['outerContractDouble'][2]])
    def gradBelow_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 1:
            return self.gradBelow_ind([1,2],[1,2],tensor,fixedPoints['RR_d_p1'][1],[outers['outerContractDouble_p1'][1],outers['outerContractDouble_p1'][2]])
        else:
            return self.gradBelow_ind([1,2],[2,1],tensor,fixedPoints['RR_d'][2],[outers['outerContractDouble'][2],outers['outerContractDouble'][1]])

class env_mpso_vert_bipartite_eval_1x2_env2(env_mpso_vert_multipleTensors_eval_1x2):
    def gradCentre_tensor(self,gradTensorLabel,tensor,outers):
        if gradTensorLabel == 2:
            return self.gradCentre_ind([1],tensor,self.psi.R[2].tensor,[outers['len1'][2],outers['len1'][1]])
        else:
            return 0
    def gradAbove_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 1:
            return self.gradAbove_ind([2,1],[1,2],tensor,fixedPoints['RR_d'][2],[outers['outerContractDouble'][2],outers['outerContractDouble'][1]])
        else:
            return self.gradAbove_ind([2,1],[2,1],tensor,fixedPoints['RR_d_p1'][2],[outers['outerContractDouble_p1'][2],outers['outerContractDouble_p1'][1]])
    def gradBelow_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 1:
            return self.gradBelow_ind([2,1],[1,2],tensor,fixedPoints['RR_d'][1],[outers['outerContractDouble'][1],outers['outerContractDouble'][2]])
        else:
            return self.gradBelow_ind([2,1],[2,1],tensor,fixedPoints['RR_d_p1'][2],[outers['outerContractDouble_p1'][2],outers['outerContractDouble_p1'][1]])

class env_mpso_vert_bipartite_eval_2x1_env1(env_mpso_vert_multipleTensors_eval_2x1):
    def gradCentre_tensor(self,gradTensorLabel,tensor,outers):
        if gradTensorLabel == 1:
            return self.gradCentre_ind_bot([2],tensor,self.psi.RR[2].tensor,outers['len2'][1])
        else:
            return self.gradCentre_ind_top([1],tensor,self.psi.RR[2].tensor,outers['len2'][1])
    def gradAbove_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 1:
            return self.gradAbove_ind([1,2],[1],tensor,fixedPoints['RRR_d_upper'][2],outers['outerContractTriple_upper'][1])
        else:
            return self.gradAbove_ind([1,2],[2],tensor,fixedPoints['RRR_d_upper_p1'][2],outers['outerContractTriple_upper_p1'][1])
    def gradBelow_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 1:
            return self.gradBelow_ind([1,2],[1],tensor,fixedPoints['RRR_d_lower_p1'][2],outers['outerContractTriple_lower_p1'][1])
        else:
            return self.gradBelow_ind([1,2],[2],tensor,fixedPoints['RRR_d_lower'][1],outers['outerContractTriple_lower'][2])

class env_mpso_vert_bipartite_eval_2x1_env2(env_mpso_vert_multipleTensors_eval_2x1):
    def gradCentre_tensor(self,gradTensorLabel,tensor,outers):
        if gradTensorLabel == 1:
            return self.gradCentre_ind_top([2],tensor,self.psi.RR[1].tensor,outers['len2'][2])
        else:
            return self.gradCentre_ind_bot([1],tensor,self.psi.RR[1].tensor,outers['len2'][2])
    def gradAbove_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 1:
            return self.gradAbove_ind([2,1],[1],tensor,fixedPoints['RRR_d_upper_p1'][1],outers['outerContractTriple_upper_p1'][2])
        else:
            return self.gradAbove_ind([2,1],[2],tensor,fixedPoints['RRR_d_upper'][1],outers['outerContractTriple_upper'][2])
    def gradBelow_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 1:
            return self.gradBelow_ind([2,1],[1],tensor,fixedPoints['RRR_d_lower'][2],outers['outerContractTriple_lower'][1])
        else:
            return self.gradBelow_ind([2,1],[2],tensor,fixedPoints['RRR_d_lower_p1'][1],outers['outerContractTriple_lower_p1'][2])

# ---------------------------------------------------------------------------------------------------
#fourSite_sep
class env_mpso_vert_fourSite_sep_eval_1x1_env1(env_mpso_vert_multipleTensors_eval_1x1):
    def gradCentre_tensor(self,gradTensorLabel,tensor,outers):
        if gradTensorLabel == 1:
            return self.gradCentre_ind(tensor,self.psi.R[2].tensor,outers['len1'][1])
        else:
            return 0
    def gradAbove_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 1:
            return self.gradAbove_ind([1],[1],tensor,fixedPoints['RR_d_p1'][2],outers['outerContractDouble_p1'][1])
        if gradTensorLabel == 3:
            return self.gradAbove_ind([1],[3],tensor,fixedPoints['RR_d'][2],outers['outerContractDouble'][1])
        else:
            return 0
    def gradBelow_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 1:
            return self.gradBelow_ind([1],[1],tensor,fixedPoints['RR_d_p1'][2],outers['outerContractDouble_p1'][1])
        if gradTensorLabel == 3:
            return self.gradBelow_ind([1],[3],tensor,fixedPoints['RR_d'][4],outers['outerContractDouble'][3])
        else:
            return 0

class env_mpso_vert_fourSite_sep_eval_1x1_env2(env_mpso_vert_multipleTensors_eval_1x1):
    def gradCentre_tensor(self,gradTensorLabel,tensor,outers):
        if gradTensorLabel == 2:
            return self.gradCentre_ind(tensor,self.psi.R[1].tensor,outers['len1'][2])
        else:
            return 0
    def gradAbove_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 2:
            return self.gradAbove_ind([2],[2],tensor,fixedPoints['RR_d_p1'][1],outers['outerContractDouble_p1'][2])
        if gradTensorLabel == 4:
            return self.gradAbove_ind([2],[4],tensor,fixedPoints['RR_d'][1],outers['outerContractDouble'][2])
        else:
            return 0
    def gradBelow_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 2:
            return self.gradBelow_ind([2],[2],tensor,fixedPoints['RR_d_p1'][1],outers['outerContractDouble_p1'][2])
        elif gradTensorLabel == 4:
            return self.gradBelow_ind([2],[4],tensor,fixedPoints['RR_d'][3],outers['outerContractDouble'][4])
        else:
            return 0

class env_mpso_vert_fourSite_sep_eval_1x1_env3(env_mpso_vert_multipleTensors_eval_1x1):
    def gradCentre_tensor(self,gradTensorLabel,tensor,outers):
        if gradTensorLabel == 3:
            return self.gradCentre_ind(tensor,self.psi.R[4].tensor,outers['len1'][3])
        else:
            return 0
    def gradAbove_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 1:
            return self.gradAbove_ind([3],[1],tensor,fixedPoints['RR_d'][4],outers['outerContractDouble'][3])
        elif gradTensorLabel == 3:
            return self.gradAbove_ind([3],[3],tensor,fixedPoints['RR_d_p1'][4],outers['outerContractDouble_p1'][3])
        else:
            return 0
    def gradBelow_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 1:
            return self.gradBelow_ind([3],[1],tensor,fixedPoints['RR_d'][2],outers['outerContractDouble'][1])
        elif gradTensorLabel == 3:
            return self.gradBelow_ind([3],[3],tensor,fixedPoints['RR_d_p1'][4],outers['outerContractDouble_p1'][3])
        else:
            return 0

class env_mpso_vert_fourSite_sep_eval_1x1_env4(env_mpso_vert_multipleTensors_eval_1x1):
    def gradCentre_tensor(self,gradTensorLabel,tensor,outers):
        if gradTensorLabel == 4:
            return self.gradCentre_ind(tensor,self.psi.R[3].tensor,outers['len1'][4])
        else:
            return 0
    def gradAbove_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 2:
            return self.gradAbove_ind([4],[2],tensor,fixedPoints['RR_d'][3],outers['outerContractDouble'][4])
        elif gradTensorLabel == 4:
            return self.gradAbove_ind([4],[4],tensor,fixedPoints['RR_d_p1'][3],outers['outerContractDouble_p1'][4])
        else:
            return 0
    def gradBelow_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 2:
            return self.gradBelow_ind([4],[2],tensor,fixedPoints['RR_d'][1],outers['outerContractDouble'][2])
        elif gradTensorLabel == 4:
            return self.gradBelow_ind([4],[4],tensor,fixedPoints['RR_d_p1'][3],outers['outerContractDouble_p1'][4])
        else:
            return 0

class env_mpso_vert_fourSite_sep_eval_1x2_env1(env_mpso_vert_multipleTensors_eval_1x2):
    def gradCentre_tensor(self,gradTensorLabel,tensor,outers):
        if gradTensorLabel == 1:
            return self.gradCentre_ind([2],tensor,self.psi.R[1].tensor,[outers['len1'][1],outers['len1'][2]])
        else:
            return 0
    def gradAbove_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 1:
            return self.gradAbove_ind([1,2],[1,2],tensor,fixedPoints['RR_d_p1'][1],[outers['outerContractDouble_p1'][1],outers['outerContractDouble_p1'][2]])
        elif gradTensorLabel == 3:
            return self.gradAbove_ind([1,2],[3,4],tensor,fixedPoints['RR_d'][1],[outers['outerContractDouble'][1],outers['outerContractDouble'][2]])
        else:
            return 0
    def gradBelow_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 1:
            return self.gradBelow_ind([1,2],[1,2],tensor,fixedPoints['RR_d_p1'][1],[outers['outerContractDouble_p1'][1],outers['outerContractDouble_p1'][2]])
        elif gradTensorLabel == 3:
            return self.gradBelow_ind([1,2],[3,4],tensor,fixedPoints['RR_d'][3],[outers['outerContractDouble'][3],outers['outerContractDouble'][4]])
        else:
            return 0

class env_mpso_vert_fourSite_sep_eval_1x2_env2(env_mpso_vert_multipleTensors_eval_1x2):
    def gradCentre_tensor(self,gradTensorLabel,tensor,outers):
        if gradTensorLabel == 2:
            return self.gradCentre_ind([1],tensor,self.psi.R[2].tensor,[outers['len1'][2],outers['len1'][1]])
        else:
            return 0
    def gradAbove_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 2:
            return self.gradAbove_ind([2,1],[2,1],tensor,fixedPoints['RR_d_p1'][2],[outers['outerContractDouble_p1'][2],outers['outerContractDouble_p1'][1]])
        elif gradTensorLabel == 4:
            return self.gradAbove_ind([2,1],[4,3],tensor,fixedPoints['RR_d'][2],[outers['outerContractDouble'][2],outers['outerContractDouble'][1]])
        else:
            return 0
    def gradBelow_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 2:
            return self.gradBelow_ind([2,1],[2,1],tensor,fixedPoints['RR_d_p1'][2],[outers['outerContractDouble_p1'][2],outers['outerContractDouble_p1'][1]])
        elif gradTensorLabel == 4:
            return self.gradBelow_ind([1,2],[4,3],tensor,fixedPoints['RR_d'][4],[outers['outerContractDouble'][4],outers['outerContractDouble'][3]])
        else:
            return 0

class env_mpso_vert_fourSite_sep_eval_1x2_env3(env_mpso_vert_multipleTensors_eval_1x2):
    def gradCentre_tensor(self,gradTensorLabel,tensor,outers):
        if gradTensorLabel == 3:
            return self.gradCentre_ind([4],tensor,self.psi.R[3].tensor,[outers['len1'][3],outers['len1'][4]])
        else:
            return 0
    def gradAbove_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 1:
            return self.gradAbove_ind([3,4],[1,2],tensor,fixedPoints['RR_d'][3],[outers['outerContractDouble'][3],outers['outerContractDouble'][4]])
        elif gradTensorLabel == 3:
            return self.gradAbove_ind([3,4],[3,4],tensor,fixedPoints['RR_d_p1'][3],[outers['outerContractDouble_p1'][3],outers['outerContractDouble_p1'][4]])
        else:
            return 0
    def gradBelow_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 1:
            return self.gradBelow_ind([3,4],[1,2],tensor,fixedPoints['RR_d'][1],[outers['outerContractDouble'][1],outers['outerContractDouble'][2]])
        elif gradTensorLabel == 3:
            return self.gradBelow_ind([3,4],[3,4],tensor,fixedPoints['RR_d_p1'][3],[outers['outerContractDouble_p1'][3],outers['outerContractDouble_p1'][4]])
        else:
            return 0

class env_mpso_vert_fourSite_sep_eval_1x2_env4(env_mpso_vert_multipleTensors_eval_1x2):
    def gradCentre_tensor(self,gradTensorLabel,tensor,outers):
        if gradTensorLabel == 4:
            return self.gradCentre_ind([3],tensor,self.psi.R[4].tensor,[outers['len1'][4],outers['len1'][3]])
        else:
            return 0
    def gradAbove_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 2:
            return self.gradAbove_ind([4,3],[2,1],tensor,fixedPoints['RR_d'][4],[outers['outerContractDouble'][4],outers['outerContractDouble'][3]])
        elif gradTensorLabel == 4:
            return self.gradAbove_ind([4,3],[4,3],tensor,fixedPoints['RR_d_p1'][4],[outers['outerContractDouble_p1'][4],outers['outerContractDouble_p1'][3]])
        else:
            return 0
    def gradBelow_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 2:
            return self.gradBelow_ind([4,3],[2,1],tensor,fixedPoints['RR_d'][2],[outers['outerContractDouble'][2],outers['outerContractDouble'][1]])
        elif gradTensorLabel == 4:
            return self.gradBelow_ind([4,3],[4,3],tensor,fixedPoints['RR_d_p1'][4],[outers['outerContractDouble_p1'][4],outers['outerContractDouble_p1'][3]])
        else:
            return 0

class env_mpso_vert_fourSite_sep_eval_2x1_env1(env_mpso_vert_multipleTensors_eval_2x1):
    def gradCentre_tensor(self,gradTensorLabel,tensor,outers):
        if gradTensorLabel == 1:
            return self.gradCentre_ind_bot([3],tensor,self.psi.RR[2].tensor,outers['len2'][1])
        elif gradTensorLabel == 3:
            return self.gradCentre_ind_top([1],tensor,self.psi.RR[2].tensor,outers['len2'][1])
        else:
            return 0
    def gradAbove_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 1:
            return self.gradAbove_ind([1,3],[1],tensor,fixedPoints['RRR_d_upper'][2],outers['outerContractTriple_upper'][1])
        elif gradTensorLabel == 3:
            return self.gradAbove_ind([1,3],[3],tensor,fixedPoints['RRR_d_upper_p1'][2],outers['outerContractTriple_upper_p1'][1])
        else:
            return 0
    def gradBelow_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 1:
            return self.gradBelow_ind([1,3],[1],tensor,fixedPoints['RRR_d_lower_p1'][2],outers['outerContractTriple_lower_p1'][1])
        elif gradTensorLabel == 3:
            return self.gradBelow_ind([1,3],[3],tensor,fixedPoints['RRR_d_lower'][4],outers['outerContractTriple_lower'][3])
        else:
            return 0

class env_mpso_vert_fourSite_sep_eval_2x1_env2(env_mpso_vert_multipleTensors_eval_2x1):
    def gradCentre_tensor(self,gradTensorLabel,tensor,outers):
        if gradTensorLabel == 2:
            return self.gradCentre_ind_bot([4],tensor,self.psi.RR[1].tensor,outers['len2'][2])
        elif gradTensorLabel == 4:
            return self.gradCentre_ind_top([2],tensor,self.psi.RR[1].tensor,outers['len2'][2])
        else:
            return 0
    def gradAbove_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 2:
            return self.gradAbove_ind([2,4],[2],tensor,fixedPoints['RRR_d_upper'][1],outers['outerContractTriple_upper'][2])
        elif gradTensorLabel == 4:
            return self.gradAbove_ind([2,4],[4],tensor,fixedPoints['RRR_d_upper_p1'][1],outers['outerContractTriple_upper_p1'][2])
        else:
            return 0
    def gradBelow_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 2:
            return self.gradBelow_ind([2,4],[2],tensor,fixedPoints['RRR_d_lower_p1'][1],outers['outerContractTriple_lower_p1'][2])
        elif gradTensorLabel == 4:
            return self.gradBelow_ind([2,4],[4],tensor,fixedPoints['RRR_d_lower'][3],outers['outerContractTriple_lower'][4])
        else:
            return 0

class env_mpso_vert_fourSite_sep_eval_2x1_env3(env_mpso_vert_multipleTensors_eval_2x1):
    def gradCentre_tensor(self,gradTensorLabel,tensor,outers):
        if gradTensorLabel == 1:
            return self.gradCentre_ind_top([3],tensor,self.psi.RR[4].tensor,outers['len2'][3])
        elif gradTensorLabel == 3:
            return self.gradCentre_ind_bot([1],tensor,self.psi.RR[4].tensor,outers['len2'][3])
        else:
            return 0
    def gradAbove_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 1:
            return self.gradAbove_ind([3,1],[1],tensor,fixedPoints['RRR_d_upper_p1'][4],outers['outerContractTriple_upper_p1'][3])
        elif gradTensorLabel == 3:
            return self.gradAbove_ind([3,1],[3],tensor,fixedPoints['RRR_d_upper'][4],outers['outerContractTriple_upper'][3])
        else:
            return 0
    def gradBelow_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 1:
            return self.gradBelow_ind([3,1],[1],tensor,fixedPoints['RRR_d_lower'][2],outers['outerContractTriple_lower'][1])
        elif gradTensorLabel == 3:
            return self.gradBelow_ind([3,1],[3],tensor,fixedPoints['RRR_d_lower_p1'][4],outers['outerContractTriple_lower_p1'][3])
        else:
            return 0

class env_mpso_vert_fourSite_sep_eval_2x1_env4(env_mpso_vert_multipleTensors_eval_2x1):
    def gradCentre_tensor(self,gradTensorLabel,tensor,outers):
        if gradTensorLabel == 2:
            return self.gradCentre_ind_top([4],tensor,self.psi.RR[3].tensor,outers['len2'][4])
        elif gradTensorLabel == 4:
            return self.gradCentre_ind_bot([2],tensor,self.psi.RR[3].tensor,outers['len2'][4])
        else:
            return 0
    def gradAbove_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 2:
            return self.gradAbove_ind([4,2],[2],tensor,fixedPoints['RRR_d_upper_p1'][3],outers['outerContractTriple_upper_p1'][4])
        elif gradTensorLabel == 4:
            return self.gradAbove_ind([4,2],[4],tensor,fixedPoints['RRR_d_upper'][3],outers['outerContractTriple_upper'][4])
        else:
            return 0
    def gradBelow_tensor(self,gradTensorLabel,tensor,fixedPoints,outers):
        if gradTensorLabel == 2:
            return self.gradBelow_ind([4,2],[2],tensor,fixedPoints['RRR_d_lower'][1],outers['outerContractTriple_lower'][2])
        elif gradTensorLabel == 4:
            return self.gradBelow_ind([4,2],[4],tensor,fixedPoints['RRR_d_lower_p1'][3],outers['outerContractTriple_lower_p1'][4])
        else:
            return 0
