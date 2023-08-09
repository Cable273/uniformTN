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

from gradImp_mpso_2d_mpo_environments import *

class gradImplementation_mpso_2d_mpo(ABC):
    def __init__(self,psi,H):
        self.psi = psi
        self.H = H
        self.init_outerContracts()
        self.init_d_dep_quants()
        #build environments
        self.envs_vert_quadrantSeed = envGroup_mpso_vert()
        self.envs_vert = envGroup_mpso_vert()
        self.envs_hori = envGroup_mpso_hori()
        self.buildEnvs_vert()
        self.buildEnvs_hori()
            
    @abstractmethod
    def init_d_dep_quants(self):
        pass
    @abstractmethod
    def init_outerContracts(self):
        pass
    @abstractmethod
    def buildEnvs_vert(self):
        pass
    @abstractmethod
    def buildEnvs_hori(self):
        pass

    def gradInline(self):
        return self.envs_vert.gradCentre(self.outers) + self.envs_hori.gradLeft(self.outers)
    def gradAbove(self,fixedPoints_d,outers_d):
        return self.envs_vert.gradAbove(fixedPoints_d,outers_d)
    def gradBelow(self,fixedPoints_d,outers_d):
        return self.envs_vert.gradBelow(fixedPoints_d,outers_d)
    def gradLeftQuadrants(self,fixedPoints_d,outers_d):
        envs_hori = self.envs_vert_quadrantSeed.build_hori(fixedPoints_d,outers_d)
        return envs_hori.gradLeft(self.outers)

# -------------------------------------------------------------------------------------------------------------------------------------
#Uniform
class gradImplementation_mpso_2d_mpo_uniform_H_verticalLength_1(gradImplementation_mpso_2d_mpo):
    def init_outerContracts(self):
        self.outers = dict()
        self.outers['len1'] = ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
    def init_d_dep_quants(self):
        self.d_dep = d_dep_quants_uniform_verticalLength1(self.psi)

class gradImplementation_mpso_2d_mpo_uniform_H_verticalLength_2(gradImplementation_mpso_2d_mpo):
    def init_outerContracts(self):
        self.outers = dict()
        self.outers['len1'] =  ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
        self.outers['len2'] = ncon([self.psi.mps,self.psi.mps,self.psi.mps.conj(),self.psi.mps.conj(),self.psi.T.tensor],((-1,5,6),(-2,6,7),(-3,5,9),(-4,9,8),(8,7)),forder=(-3,-4,-1,-2))
    def init_d_dep_quants(self):
        self.d_dep = d_dep_quants_uniform_verticalLength2(self.psi)

class gradImplementation_mpso_2d_mpo_uniform_oneBodyH(gradImplementation_mpso_2d_mpo_uniform_H_verticalLength_1):
    def buildEnvs_vert(self):
        #terms under H
        env = ncon([self.psi.mpo,self.H],((2,-1,-4,-5),(-3,2)),forder=(-3,-1,-4,-5))
        self.envs_vert_quadrantSeed.add( env_mpso_vert_uniform(self.psi,env,shape=[1,1]) )
        self.envs_vert.add( env_mpso_vert_uniform(self.psi,env,shape=[1,1]) )
        #terms right of H
        env = ncon([self.psi.mpo.conj(),env,self.outers['len1']],((2,3,4,-6),(2,1,4,-5),(3,1)),forder=(-6,-5 ),order=(4,2,1,3))
        env = self.psi.Tb_inv.applyRight(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        env = ncon([self.psi.mpo,env],((-2,-1,3,-4),(-5,3)),forder=(-2,-1,-5,-4))
        self.envs_vert.add( env_mpso_vert_uniform(self.psi,env,shape=[1,1]) )
    def buildEnvs_hori(self):
        env = ncon([self.psi.mpo,self.psi.mpo.conj(),self.H,self.psi.R.tensor,self.outers['len1']],((2,1,-5,6),(3,4,-7,8),(3,2),(8,6),(4,1)),forder=(-7,-5),order=(8,6,1,2,3,4))
        self.envs_hori.add( env_mpso_hori_uniform(self.psi,env) )

class gradImplementation_mpso_2d_mpo_uniform_twoBodyH_hori(gradImplementation_mpso_2d_mpo_uniform_H_verticalLength_1):
    def buildEnvs_vert(self):
        #term under H (left side)
        env = ncon([self.psi.mpo,self.psi.mpo,self.H],((2,-1,-7,8),(5,-4,8,-9),(-3,-6,2,5)),forder=(-3,-6,-1,-4,-7,-9),order=(2,8,5))
        self.envs_vert_quadrantSeed.add( env_mpso_vert_uniform(self.psi,env,shape=[1,2]) )
        self.envs_vert.add( env_mpso_vert_uniform(self.psi,env,shape=[1,2]) )
        #term under H (right side)
        env = ncon([env,self.psi.mpo.conj(),self.outers['len1']],((2,-5,1,-4,6,-8),(2,3,6,-7),(3,1)),forder=(-5,-4,-7,-8),order=(6,2,1,3))
        self.envs_vert.add( env_mpso_vert_uniform(self.psi,env,shape=[1,1]) )
        #term right of H
        env = ncon([env,self.psi.mpo.conj(),self.outers['len1']],((2,1,4,-6),(2,3,4,-5),(3,1)),forder=(-5,-6),order=(4,2,1,3))
        env = self.psi.Tb_inv.applyRight(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        env = ncon([self.psi.mpo,env],((-2,-1,3,-4),(-5,3)),forder=(-2,-1,-5,-4))
        self.envs_vert.add( env_mpso_vert_uniform(self.psi,env,shape=[1,1]) )
    def buildEnvs_hori(self):
        env = ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.R.tensor,self.outers['len1'],self.outers['len1']],((2,1,-9,10),(6,5,10,11),(3,7,2,6),(3,4,-12,13),(7,8,13,14),(14,11),(4,1),(8,5)),forder=(-12,-9),order=(14,11,5,8,6,7,10,13,1,4,2,3))
        self.envs_hori.add( env_mpso_hori_uniform(self.psi,env) )

class gradImplementation_mpso_2d_mpo_uniform_twoBodyH_vert(gradImplementation_mpso_2d_mpo_uniform_H_verticalLength_2):
    def buildEnvs_vert(self):
        #term under H 
        env = ncon([self.psi.mpo,self.psi.mpo,self.H],((2,-1,-7,-8),(5,-4,-9,-10),(-3,-6,2,5)),forder=(-3,-6,-1,-4,-7,-8,-9,-10),order=(2,5))
        self.envs_vert_quadrantSeed.add( env_mpso_vert_uniform(self.psi,env,shape=[2,1]) )
        self.envs_vert.add( env_mpso_vert_uniform(self.psi,env,shape=[2,1]) )
        #term right of H
        env = ncon([env,self.psi.mpo.conj(),self.psi.mpo.conj(),self.outers['len2']],((5,2,4,1,10,-11,7,-8),(5,6,10,-12),(2,3,7,-9),(6,3,4,1)),forder=(-12,-11,-9,-8),order=(10,5,4,6,7,2,1,3))
        env = self.psi.Tb2_inv.applyRight(env.reshape(self.psi.D_mpo**4)).reshape(self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo)
        env = ncon([env,self.psi.mpo,self.psi.mpo],((-8,7,-6,5),(-2,-1,5,-9),(-4,-3,7,-10)),forder=(-4,-2,-3,-1,-8,-10,-6,-9))
        self.envs_vert.add( env_mpso_vert_uniform(self.psi,env,shape=[2,1]) )
    def buildEnvs_hori(self):
        envDouble = ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.RR.tensor,self.outers['len2']],((2,1,-9,10),(6,5,-11,12),(3,7,2,6),(3,4,-13,14),(7,8,-15,16),(14,10,16,12),(4,8,1,5)),forder=(-13,-9,-15,-11),order=(14,10,1,2,3,4,16,12,5,6,7,8))
        self.envs_hori.add( env_mpso_hori_uniform(self.psi,np.einsum('abcc->ab',envDouble)) )
        self.envs_hori.add( env_mpso_hori_uniform(self.psi,np.einsum('aabc->bc',envDouble)) )

# -----------------------------
#Bipartite
class gradImplementation_mpso_2d_mpo_bipartite(gradImplementation_mpso_2d_mpo):
    def buildEnvs_vert(self):
        self.buildEnvs_vert_siteLabels([1,2]) 
        self.buildEnvs_vert_siteLabels([2,1])
    def buildEnvs_hori(self):
        self.buildEnvs_hori_siteLabels([1,2]) 
        self.buildEnvs_hori_siteLabels([2,1])
        
class gradImplementation_mpso_2d_mpo_bipartite_H_verticalLength_1(gradImplementation_mpso_2d_mpo_bipartite):
    def init_outerContracts(self):
        self.outers = dict()
        self.outers['len1'] = dict()
        self.outers['len1'][1] =  ncon([self.psi.mps[1],self.psi.mps[1].conj(),self.psi.T[2].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
        self.outers['len1'][2] =  ncon([self.psi.mps[2],self.psi.mps[2].conj(),self.psi.T[1].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
    def init_d_dep_quants(self):
        self.d_dep = d_dep_quants_bipartite_verticalLength1(self.psi)

class gradImplementation_mpso_2d_mpo_bipartite_H_verticalLength_2(gradImplementation_mpso_2d_mpo_bipartite):
    def init_outerContracts(self):
        self.outers = dict()
        self.outers['len1'] = dict()
        self.outers['len2'] = dict()
        self.outers['len1'][1] =  ncon([self.psi.mps[1],self.psi.mps[1].conj(),self.psi.T[2].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
        self.outers['len1'][2] =  ncon([self.psi.mps[2],self.psi.mps[2].conj(),self.psi.T[1].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
        self.outers['len2'][1] = ncon([self.psi.mps[1],self.psi.mps[2],self.psi.mps[1].conj(),self.psi.mps[2].conj(),self.psi.T[1].tensor],((-1,5,6),(-2,6,7),(-3,5,9),(-4,9,8),(8,7)),forder=(-3,-4,-1,-2))
        self.outers['len2'][2] = ncon([self.psi.mps[2],self.psi.mps[1],self.psi.mps[2].conj(),self.psi.mps[1].conj(),self.psi.T[2].tensor],((-1,5,6),(-2,6,7),(-3,5,9),(-4,9,8),(8,7)),forder=(-3,-4,-1,-2))
    def init_d_dep_quants(self):
        self.d_dep = d_dep_quants_bipartite_verticalLength2(self.psi)

class gradImplementation_mpso_2d_mpo_bipartite_oneBodyH(gradImplementation_mpso_2d_mpo_bipartite_H_verticalLength_1):
    def buildEnvs_vert_siteLabels(self,site):
        #term under H
        env = ncon([self.psi.mpo[site[0]],self.H],((2,-1,-4,-5),(-3,2)),forder=(-3,-1,-4,-5))
        self.envs_vert_quadrantSeed.add( env_mpso_vert_bipartite(self.psi,env,shape=[1,1],label=site[0]))
        self.envs_vert.add( env_mpso_vert_bipartite(self.psi,env,shape=[1,1],label=site[0]))
        #term right of H
        env = ncon([self.psi.mpo[site[0]].conj(),env,self.outers['len1'][site[0]]],((2,3,4,-6),(2,1,4,-5),(3,1)),forder=(-6,-5 ),order=(4,2,1,3))
        env = self.psi.Tb_inv[site[1]].applyRight(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        env = ncon([self.psi.mpo[site[1]],env],((-2,-1,3,-4),(-5,3)),forder=(-2,-1,-5,-4))
        self.envs_vert.add( env_mpso_vert_bipartite(self.psi,env,shape=[1,1],label=site[1]) )
        #extra site (bipartite)
        env = ncon([self.psi.mpo[site[1]].conj(),env,self.outers['len1'][site[1]]],((2,3,4,-6),(2,1,4,-5),(3,1)),forder=(-6,-5 ),order=(4,2,1,3))
        env = ncon([self.psi.mpo[site[0]],env],((-2,-1,3,-4),(-5,3)),forder=(-2,-1,-5,-4))
        self.envs_vert.add( env_mpso_vert_bipartite(self.psi,env,shape=[1,1],label=site[0]) )
    def buildEnvs_hori_siteLabels(self,site):
        env = ncon([self.psi.mpo[site[0]],self.psi.mpo[site[0]].conj(),self.H,self.psi.R[site[1]].tensor,self.outers['len1'][site[0]]],((2,1,-5,6),(3,4,-7,8),(3,2),(8,6),(4,1)),forder=(-7,-5),order=(8,6,1,2,3,4))
        self.envs_hori.add( env_mpso_hori_bipartite(self.psi,env,label=site[0]) )

class gradImplementation_mpso_2d_mpo_bipartite_twoBodyH_hori(gradImplementation_mpso_2d_mpo_bipartite_H_verticalLength_1):
    def buildEnvs_vert_siteLabels(self,site):
        #term under H (left side)
        env = ncon([self.psi.mpo[site[0]],self.psi.mpo[site[1]],self.H],((2,-1,-7,8),(5,-4,8,-9),(-3,-6,2,5)),forder=(-3,-6,-1,-4,-7,-9),order=(2,8,5))
        self.envs_vert_quadrantSeed.add( env_mpso_vert_bipartite(self.psi,env,shape=[1,2],label=site[0]) )
        self.envs_vert.add( env_mpso_vert_bipartite(self.psi,env,shape=[1,2],label=site[0]) )
        #term under H (right side)
        env = ncon([env,self.psi.mpo[site[0]].conj(),self.outers['len1'][site[0]]],((2,-5,1,-4,6,-8),(2,3,6,-7),(3,1)),forder=(-5,-4,-7,-8),order=(6,2,1,3))
        self.envs_vert.add( env_mpso_vert_bipartite(self.psi,env,shape=[1,1],label=site[1]) )
        #term right of H
        env = ncon([env,self.psi.mpo[site[1]].conj(),self.outers['len1'][site[1]]],((2,1,4,-6),(2,3,4,-5),(3,1)),forder=(-5,-6),order=(4,2,1,3))
        env = self.psi.Tb_inv[site[0]].applyRight(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        env = ncon([self.psi.mpo[site[0]],env],((-2,-1,3,-4),(-5,3)),forder=(-2,-1,-5,-4))
        self.envs_vert.add( env_mpso_vert_bipartite(self.psi,env,shape=[1,1],label=site[0]) )
        #extra site
        env = ncon([env,self.psi.mpo[site[0]].conj(),self.outers['len1'][site[0]]],((2,1,4,-6),(2,3,4,-5),(3,1)),forder=(-5,-6),order=(4,2,1,3))
        env = ncon([self.psi.mpo[site[1]],env],((-2,-1,3,-4),(-5,3)),forder=(-2,-1,-5,-4))
        self.envs_vert.add( env_mpso_vert_bipartite(self.psi,env,shape=[1,1],label=site[1]) )
    def buildEnvs_hori_siteLabels(self,site):
        env = ncon([self.psi.mpo[site[0]],self.psi.mpo[site[1]],self.H,self.psi.mpo[site[0]].conj(),self.psi.mpo[site[1]].conj(),self.psi.R[site[0]].tensor,self.outers['len1'][site[0]],self.outers['len1'][site[1]]],((2,1,-9,10),(6,5,10,11),(3,7,2,6),(3,4,-12,13),(7,8,13,14),(14,11),(4,1),(8,5)),forder=(-12,-9),order=(14,11,5,8,6,7,10,13,1,4,2,3))
        self.envs_hori.add( env_mpso_hori_bipartite(self.psi,env,label=site[0]) )

class gradImplementation_mpso_2d_mpo_bipartite_twoBodyH_vert(gradImplementation_mpso_2d_mpo_bipartite_H_verticalLength_2):
    def buildEnvs_vert_siteLabels(self,site):
        #term under H 
        env = ncon([self.psi.mpo[site[0]],self.psi.mpo[site[1]],self.H],((2,-1,-7,-8),(5,-4,-9,-10),(-3,-6,2,5)),forder=(-3,-6,-1,-4,-7,-8,-9,-10),order=(2,5))
        self.envs_vert_quadrantSeed.add( env_mpso_vert_bipartite(self.psi,env,shape=[2,1],label=site[0]) )
        self.envs_vert.add( env_mpso_vert_bipartite(self.psi,env,shape=[2,1],label=site[0]) )
        #term right of H
        env = ncon([env,self.psi.mpo[site[0]].conj(),self.psi.mpo[site[1]].conj(),self.outers['len2'][site[0]]],((5,2,4,1,10,-11,7,-8),(5,6,10,-12),(2,3,7,-9),(6,3,4,1)),forder=(-12,-11,-9,-8),order=(10,5,4,6,7,2,1,3))
        env = self.psi.Tb2_inv[site[1]].applyRight(env.reshape(self.psi.D_mpo**4)).reshape(self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo)
        env = ncon([env,self.psi.mpo[site[0]],self.psi.mpo[site[1]]],((-8,7,-6,5),(-2,-1,5,-9),(-4,-3,7,-10)),forder=(-4,-2,-3,-1,-8,-10,-6,-9))
        self.envs_vert.add( env_mpso_vert_bipartite(self.psi,env,shape=[2,1],label=site[1]) )
        #extra site
        env = ncon([env,self.psi.mpo[site[1]].conj(),self.psi.mpo[site[0]].conj(),self.outers['len2'][site[1]]],((5,2,4,1,10,-11,7,-8),(5,6,10,-12),(2,3,7,-9),(6,3,4,1)),forder=(-12,-11,-9,-8),order=(10,5,4,6,7,2,1,3))
        env = ncon([env,self.psi.mpo[site[1]],self.psi.mpo[site[0]]],((-8,7,-6,5),(-2,-1,5,-9),(-4,-3,7,-10)),forder=(-4,-2,-3,-1,-8,-10,-6,-9))
        self.envs_vert.add( env_mpso_vert_bipartite(self.psi,env,shape=[2,1],label=site[0]) )
    def buildEnvs_hori_siteLabels(self,site):
        envDouble = ncon([self.psi.mpo[site[0]],self.psi.mpo[site[1]],self.H,self.psi.mpo[site[0]].conj(),self.psi.mpo[site[1]].conj(),self.psi.RR[site[1]].tensor,self.outers['len2'][site[0]]],((2,1,-9,10),(6,5,-11,12),(3,7,2,6),(3,4,-13,14),(7,8,-15,16),(14,10,16,12),(4,8,1,5)),forder=(-13,-9,-15,-11),order=(14,10,1,2,3,4,16,12,5,6,7,8))
        self.envs_hori.add( env_mpso_hori_bipartite(self.psi,np.einsum('abcc->ab',envDouble),label=site[0])) 
        self.envs_hori.add( env_mpso_hori_bipartite(self.psi,np.einsum('aabc->bc',envDouble),label=site[1])) 

# -----------------------------
#fourSite_sep
class gradImplementation_mpso_2d_mpo_fourSite_sep(gradImplementation_mpso_2d_mpo):
    def buildEnvs_vert(self):
        self.buildEnvs_vert_siteLabels([1,2,3,4]) 
        self.buildEnvs_vert_siteLabels([2,1,4,3]) 
        self.buildEnvs_vert_siteLabels([3,4,1,2]) 
        self.buildEnvs_vert_siteLabels([4,3,2,1]) 
    def buildEnvs_hori(self):
        self.buildEnvs_hori_siteLabels([1,2,3,4]) 
        self.buildEnvs_hori_siteLabels([2,1,4,3]) 
        self.buildEnvs_hori_siteLabels([3,4,1,2]) 
        self.buildEnvs_hori_siteLabels([4,3,2,1]) 
        
class gradImplementation_mpso_2d_mpo_fourSite_sep_H_verticalLength_1(gradImplementation_mpso_2d_mpo_fourSite_sep):
    def init_outerContracts(self):
        self.outers = dict()
        self.outers['len1'] = dict()
        self.outers['len1'][1] =  ncon([self.psi.mps[1],self.psi.mps[1].conj(),self.psi.T[3].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
        self.outers['len1'][2] =  ncon([self.psi.mps[2],self.psi.mps[2].conj(),self.psi.T[4].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
        self.outers['len1'][3] =  ncon([self.psi.mps[3],self.psi.mps[3].conj(),self.psi.T[1].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
        self.outers['len1'][4] =  ncon([self.psi.mps[4],self.psi.mps[4].conj(),self.psi.T[2].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
    def init_d_dep_quants(self):
        self.d_dep = d_dep_quants_fourSite_sep_verticalLength1(self.psi)

class gradImplementation_mpso_2d_mpo_fourSite_sep_H_verticalLength_2(gradImplementation_mpso_2d_mpo_fourSite_sep):
    def init_outerContracts(self):
        self.outers = dict()
        self.outers['len1'] = dict()
        self.outers['len2'] = dict()
        self.outers['len1'][1] =  ncon([self.psi.mps[1],self.psi.mps[1].conj(),self.psi.T[3].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
        self.outers['len1'][2] =  ncon([self.psi.mps[2],self.psi.mps[2].conj(),self.psi.T[4].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
        self.outers['len1'][3] =  ncon([self.psi.mps[3],self.psi.mps[3].conj(),self.psi.T[1].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
        self.outers['len1'][4] =  ncon([self.psi.mps[4],self.psi.mps[4].conj(),self.psi.T[2].tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
        self.outers['len2'][1] = ncon([self.psi.mps[1],self.psi.mps[3],self.psi.mps[1].conj(),self.psi.mps[3].conj(),self.psi.T[1].tensor],((-1,5,6),(-2,6,7),(-3,5,9),(-4,9,8),(8,7)),forder=(-3,-4,-1,-2))
        self.outers['len2'][2] = ncon([self.psi.mps[2],self.psi.mps[4],self.psi.mps[2].conj(),self.psi.mps[4].conj(),self.psi.T[2].tensor],((-1,5,6),(-2,6,7),(-3,5,9),(-4,9,8),(8,7)),forder=(-3,-4,-1,-2))
        self.outers['len2'][3] = ncon([self.psi.mps[3],self.psi.mps[1],self.psi.mps[3].conj(),self.psi.mps[1].conj(),self.psi.T[3].tensor],((-1,5,6),(-2,6,7),(-3,5,9),(-4,9,8),(8,7)),forder=(-3,-4,-1,-2))
        self.outers['len2'][4] = ncon([self.psi.mps[4],self.psi.mps[2],self.psi.mps[4].conj(),self.psi.mps[2].conj(),self.psi.T[4].tensor],((-1,5,6),(-2,6,7),(-3,5,9),(-4,9,8),(8,7)),forder=(-3,-4,-1,-2))
    def init_d_dep_quants(self):
        self.d_dep = d_dep_quants_bipartite_verticalLength2(self.psi)

class gradImplementation_mpso_2d_mpo_fourSite_sep_oneBodyH(gradImplementation_mpso_2d_mpo_fourSite_sep_H_verticalLength_1):
    def buildEnvs_vert_siteLabels(self,site):
        #term under H
        env = ncon([self.psi.mpo[site[0]],self.H],((2,-1,-4,-5),(-3,2)),forder=(-3,-1,-4,-5))
        self.envs_vert_quadrantSeed.add( env_mpso_vert_fourSite_sep(self.psi,env,shape=[1,1],label=site[0]))
        self.envs_vert.add( env_mpso_vert_fourSite_sep(self.psi,env,shape=[1,1],label=site[0]))
        #term right of H
        env = ncon([self.psi.mpo[site[0]].conj(),env,self.outers['len1'][site[0]]],((2,3,4,-6),(2,1,4,-5),(3,1)),forder=(-6,-5 ),order=(4,2,1,3))
        env = self.psi.Tb_inv[site[1]].applyRight(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        env = ncon([self.psi.mpo[site[1]],env],((-2,-1,3,-4),(-5,3)),forder=(-2,-1,-5,-4))
        self.envs_vert.add( env_mpso_vert_fourSite_sep(self.psi,env,shape=[1,1],label=site[1]) )
        #extra site 
        env = ncon([self.psi.mpo[site[1]].conj(),env,self.outers['len1'][site[1]]],((2,3,4,-6),(2,1,4,-5),(3,1)),forder=(-6,-5 ),order=(4,2,1,3))
        env = ncon([self.psi.mpo[site[0]],env],((-2,-1,3,-4),(-5,3)),forder=(-2,-1,-5,-4))
        self.envs_vert.add( env_mpso_vert_fourSite_sep(self.psi,env,shape=[1,1],label=site[0]) )
    def buildEnvs_hori_siteLabels(self,site):
        env = ncon([self.psi.mpo[site[0]],self.psi.mpo[site[0]].conj(),self.H,self.psi.R[site[1]].tensor,self.outers['len1'][site[0]]],((2,1,-5,6),(3,4,-7,8),(3,2),(8,6),(4,1)),forder=(-7,-5),order=(8,6,1,2,3,4))
        self.envs_hori.add( env_mpso_hori_fourSite_sep(self.psi,env,label=site[0]) )

class gradImplementation_mpso_2d_mpo_fourSite_sep_twoBodyH_hori(gradImplementation_mpso_2d_mpo_fourSite_sep_H_verticalLength_1):
    def buildEnvs_vert_siteLabels(self,site):
        #term under H (left side)
        env = ncon([self.psi.mpo[site[0]],self.psi.mpo[site[1]],self.H],((2,-1,-7,8),(5,-4,8,-9),(-3,-6,2,5)),forder=(-3,-6,-1,-4,-7,-9),order=(2,8,5))
        self.envs_vert_quadrantSeed.add( env_mpso_vert_fourSite_sep(self.psi,env,shape=[1,2],label=site[0]) )
        self.envs_vert.add( env_mpso_vert_fourSite_sep(self.psi,env,shape=[1,2],label=site[0]) )
        #term under H (right side)
        env = ncon([env,self.psi.mpo[site[0]].conj(),self.outers['len1'][site[0]]],((2,-5,1,-4,6,-8),(2,3,6,-7),(3,1)),forder=(-5,-4,-7,-8),order=(6,2,1,3))
        self.envs_vert.add( env_mpso_vert_fourSite_sep(self.psi,env,shape=[1,1],label=site[1]) )
        #term right of H
        env = ncon([env,self.psi.mpo[site[1]].conj(),self.outers['len1'][site[1]]],((2,1,4,-6),(2,3,4,-5),(3,1)),forder=(-5,-6),order=(4,2,1,3))
        env = self.psi.Tb_inv[site[0]].applyRight(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        env = ncon([self.psi.mpo[site[0]],env],((-2,-1,3,-4),(-5,3)),forder=(-2,-1,-5,-4))
        self.envs_vert.add( env_mpso_vert_fourSite_sep(self.psi,env,shape=[1,1],label=site[0]) )
        #extra site
        env = ncon([env,self.psi.mpo[site[0]].conj(),self.outers['len1'][site[0]]],((2,1,4,-6),(2,3,4,-5),(3,1)),forder=(-5,-6),order=(4,2,1,3))
        env = ncon([self.psi.mpo[site[1]],env],((-2,-1,3,-4),(-5,3)),forder=(-2,-1,-5,-4))
        self.envs_vert.add( env_mpso_vert_fourSite_sep(self.psi,env,shape=[1,1],label=site[1]) )
    def buildEnvs_hori_siteLabels(self,site):
        env = ncon([self.psi.mpo[site[0]],self.psi.mpo[site[1]],self.H,self.psi.mpo[site[0]].conj(),self.psi.mpo[site[1]].conj(),self.psi.R[site[0]].tensor,self.outers['len1'][site[0]],self.outers['len1'][site[1]]],((2,1,-9,10),(6,5,10,11),(3,7,2,6),(3,4,-12,13),(7,8,13,14),(14,11),(4,1),(8,5)),forder=(-12,-9),order=(14,11,5,8,6,7,10,13,1,4,2,3))
        self.envs_hori.add( env_mpso_hori_fourSite_sep(self.psi,env,label=site[0]) )

class gradImplementation_mpso_2d_mpo_fourSite_sep_twoBodyH_vert(gradImplementation_mpso_2d_mpo_fourSite_sep_H_verticalLength_2):
    def buildEnvs_vert_siteLabels(self,site):
        #term under H 
        env = ncon([self.psi.mpo[site[0]],self.psi.mpo[site[2]],self.H],((2,-1,-7,-8),(5,-4,-9,-10),(-3,-6,2,5)),forder=(-3,-6,-1,-4,-7,-8,-9,-10),order=(2,5))
        self.envs_vert_quadrantSeed.add( env_mpso_vert_fourSite_sep(self.psi,env,shape=[2,1],label=site[0]) )
        self.envs_vert.add( env_mpso_vert_fourSite_sep(self.psi,env,shape=[2,1],label=site[0]) )
        #term right of H
        env = ncon([env,self.psi.mpo[site[0]].conj(),self.psi.mpo[site[2]].conj(),self.outers['len2'][site[0]]],((5,2,4,1,10,-11,7,-8),(5,6,10,-12),(2,3,7,-9),(6,3,4,1)),forder=(-12,-11,-9,-8),order=(10,5,4,6,7,2,1,3))
        env = self.psi.Tb2_inv[site[1]].applyRight(env.reshape(self.psi.D_mpo**4)).reshape(self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo)
        env = ncon([env,self.psi.mpo[site[3]],self.psi.mpo[site[1]]],((-8,7,-6,5),(-2,-1,5,-9),(-4,-3,7,-10)),forder=(-4,-2,-3,-1,-8,-10,-6,-9))
        self.envs_vert.add( env_mpso_vert_fourSite_sep(self.psi,env,shape=[2,1],label=site[1]) )
        #extra site
        env = ncon([env,self.psi.mpo[site[1]].conj(),self.psi.mpo[site[3]].conj(),self.outers['len2'][site[1]]],((5,2,4,1,10,-11,7,-8),(5,6,10,-12),(2,3,7,-9),(6,3,4,1)),forder=(-12,-11,-9,-8),order=(10,5,4,6,7,2,1,3))
        env = ncon([env,self.psi.mpo[site[2]],self.psi.mpo[site[0]]],((-8,7,-6,5),(-2,-1,5,-9),(-4,-3,7,-10)),forder=(-4,-2,-3,-1,-8,-10,-6,-9))
        self.envs_vert.add( env_mpso_vert_fourSite_sep(self.psi,env,shape=[2,1],label=site[0]) )
    def buildEnvs_hori_siteLabels(self,site):
        envDouble = ncon([self.psi.mpo[site[0]],self.psi.mpo[site[2]],self.H,self.psi.mpo[site[0]].conj(),self.psi.mpo[site[2]].conj(),self.psi.RR[site[1]].tensor,self.outers['len2'][site[0]]],((2,1,-9,10),(6,5,-11,12),(3,7,2,6),(3,4,-13,14),(7,8,-15,16),(14,10,16,12),(4,8,1,5)),forder=(-13,-9,-15,-11),order=(14,10,1,2,3,4,16,12,5,6,7,8))
        self.envs_hori.add( env_mpso_hori_fourSite_sep(self.psi,np.einsum('abcc->ab',envDouble),label=site[0])) 
        self.envs_hori.add( env_mpso_hori_fourSite_sep(self.psi,np.einsum('aabc->bc',envDouble),label=site[2])) 

# -----------------------------
#twoSite_square
class gradImplementation_mpso_2d_mpo_twoSite_square_H_verticalLength_1(gradImplementation_mpso_2d_mpo):
    def buildEnvs_vert(self):
        self.buildEnvs_vert_outerType('bot')
        self.buildEnvs_vert_outerType('top')
    def buildEnvs_hori(self):
        self.buildEnvs_hori_outerType('bot')
        self.buildEnvs_hori_outerType('top')
    def init_outerContracts(self):
        self.outers = dict()
        self.outers['len1'] = dict()
        self.outers['len1']['bot'] = ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((-1,3,4,5),(-2,3,4,6),(6,5)),forder=(-2,-1),order=(4,3,5,6))
        self.outers['len1']['top']= ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((1,-2,4,5),(1,-3,4,6),(6,5)),forder=(-3,-2),order=(4,1,5,6))
    def init_d_dep_quants(self):
        self.d_dep = d_dep_quants_twoSite_square_verticalLength1(self.psi)

class gradImplementation_mpso_2d_mpo_twoSite_square_H_verticalLength_2(gradImplementation_mpso_2d_mpo):
    def buildEnvs_vert(self):
        self.buildEnvs_vert_outerType('square')
        self.buildEnvs_vert_outerType('prong')
    def buildEnvs_hori(self):
        self.buildEnvs_hori_outerType('square')
        self.buildEnvs_hori_outerType('prong')
    def init_outerContracts(self):
        self.outers = dict()
        self.outers['len1'] = dict()
        self.outers['len2'] = dict()
        self.outers['len1']['bot'] = ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((-1,3,4,5),(-2,3,4,6),(6,5)),forder=(-2,-1),order=(4,3,5,6))
        self.outers['len1']['top']= ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((1,-2,4,5),(1,-3,4,6),(6,5)),forder=(-3,-2),order=(4,1,5,6))
        self.outers['len2']['square'] = ncon([self.psi.mps,self.psi.mps.conj(),self.psi.T.tensor],((-1,-3,5,6),(-2,-4,5,7),(7,6)),forder=(-2,-4,-1,-3),order=(5,6,7))
        self.outers['len2']['prong'] = ncon([self.psi.mps,self.psi.mps,self.psi.mps.conj(),self.psi.mps.conj(),self.psi.T.tensor],((1,-2,7,8),(-4,6,8,9),(1,-3,7,11),(-5,6,11,10),(10,9)),forder=(-3,-5,-2,-4),order=(7,1,8,11,6,9,10))
    def init_d_dep_quants(self):
        self.d_dep = d_dep_quants_twoSite_square_verticalLength2(self.psi)

class gradImplementation_mpso_2d_mpo_twoSite_square_oneBodyH(gradImplementation_mpso_2d_mpo_twoSite_square_H_verticalLength_1):
    def buildEnvs_vert_outerType(self,outerType):
        #term under H 
        env = ncon([self.psi.mpo,self.H],((2,-5,-1,-4,-6,-7),(-3,2)),forder=(-3,-5,-1,-4,-6,-7))
        env += ncon([self.psi.mpo,self.H],((-2,4,-1,-3,-6,-7),(-5,4)),forder=(-2,-5,-1,-3,-6,-7))
        self.envs_vert_quadrantSeed.add( env_mpso_vert_twoSite_square(self.psi,env,shape=[1,1],label=outerType))
        self.envs_vert.add( env_mpso_vert_twoSite_square(self.psi,env,shape=[1,1],label=outerType))
        #term right of H
        env = ncon([env,self.psi.mpo.conj(),self.outers['len1'][outerType],self.outers['len1'][outerType]],((2,5,1,4,7,-8),(2,5,3,6,7,-9),(3,1),(6,4)),forder=(-9,-8),order=(7,2,5,1,3,4,6))
        env = self.psi.Tb_inv[outerType].applyRight(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        env = ncon([env,self.psi.mpo],((-7,5),(-2,-4,-1,-3,5,-6)),forder=(-2,-4,-1,-3,-7,-6))
        self.envs_vert.add( env_mpso_vert_twoSite_square(self.psi,env,shape=[1,1],label=outerType) )
    def buildEnvs_hori_outerType(self,outerType):
        env = ncon([self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.R[outerType].tensor,self.outers['len1'][outerType],self.outers['len1'][outerType]],((2,6,1,5,-8,9),(3,2),(3,6,4,7,-11,10),(10,9),(4,1),(7,5)),forder=(-11,-8),order=(10,9,6,5,7,2,3,1,4))
        env += ncon([self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.R[outerType].tensor,self.outers['len1'][outerType],self.outers['len1'][outerType]],((2,5,1,4,-8,9),(6,5),(2,6,3,7,-11,10),(10,9),(3,1),(7,4)),forder=(-11,-8),order=(10,9,5,6,4,7,2,1,3))
        self.envs_hori.add( env_mpso_hori_twoSite_square(self.psi,env,label=outerType) )

class gradImplementation_mpso_2d_mpo_twoSite_square_twoBodyH_hori(gradImplementation_mpso_2d_mpo_twoSite_square_H_verticalLength_1):
    def buildEnvs_vert_outerType(self,outerType):
        #term under H (at site 2 -ie underneath two mpo tensors)
        env_double = ncon([self.psi.mpo,self.psi.mpo,self.H],((-2,4,-1,-3,-11,12),(7,-10,-6,-9,12,-13),(-5,-8,4,7)),forder=(-2,-5,-8,-10,-1,-3,-6,-9,-11,-13),order=(4,12,7))
        self.envs_vert_quadrantSeed.add( env_mpso_vert_twoSite_square(self.psi,env_double,shape=[1,2],label=outerType) )
        self.envs_vert.add( env_mpso_vert_twoSite_square(self.psi,env_double,shape=[1,2],label=outerType) )
        #term under H (at site 1 -ie directly underneath)
        env_single = ncon([self.psi.mpo,self.H],((2,5,-1,-4,-7,-8),(-3,-6,2,5)),forder=(-3,-6,-1,-4,-7,-8))
        self.envs_vert_quadrantSeed.add( env_mpso_vert_twoSite_square(self.psi,env_single,shape=[1,1],label=outerType) )
        self.envs_vert.add( env_mpso_vert_twoSite_square(self.psi,env_single,shape=[1,1],label=outerType) )
        #term under H (right side)
        env = ncon([env_double,self.psi.mpo.conj(),self.outers['len1'][outerType],self.outers['len1'][outerType]],((1,2,-3,-4,5,6,-7,-8,9,-10),(1,2,11,12,9,-13),(11,5),(12,6)),forder=(-3,-4,-7,-8,-13,-10),order=(9,1,2,11,5,12,6))
        self.envs_vert.add( env_mpso_vert_twoSite_square(self.psi,env,shape=[1,1],label=outerType) )
        #term right of H
        env = env + env_single
        env = ncon([env,self.psi.mpo.conj(),self.outers['len1'][outerType],self.outers['len1'][outerType]],((2,5,1,4,7,-8),(2,5,3,6,7,-9),(3,1),(6,4)),forder=(-9,-8),order=(7,2,5,1,3,4,6))
        env = self.psi.Tb_inv[outerType].applyRight(env.reshape(self.psi.D_mpo**2)).reshape(self.psi.D_mpo,self.psi.D_mpo)
        env = ncon([env,self.psi.mpo],((-7,5),(-2,-4,-1,-3,5,-6)),forder=(-2,-4,-1,-3,-7,-6))
        self.envs_vert.add( env_mpso_vert_twoSite_square(self.psi,env,shape=[1,1],label=outerType) )
    def buildEnvs_hori_outerType(self,outerType):
        env = ncon([self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.R[outerType].tensor,self.outers['len1'][outerType],self.outers['len1'][outerType]],((2,6,1,5,-9,10),(3,7,2,6),(3,7,4,8,-12,11),(11,10),(4,1),(8,5)),forder=(-12,-9),order=(11,10,6,7,5,8,2,3,4,1))
        env += ncon([self.psi.mpo,self.psi.mpo,self.H,self.psi.mpo.conj(),self.psi.mpo.conj(),self.psi.R[outerType].tensor,self.outers['len1'][outerType],self.outers['len1'][outerType],self.outers['len1'][outerType],self.outers['len1'][outerType]],((2,5,1,4,-15,16),(9,13,8,12,16,17),(6,10,5,9),(2,6,3,7,-20,19),(10,13,11,14,19,18),(18,17),(3,1),(7,4),(11,8),(14,12)),forder=(-20,-15),order=(18,17,13,12,14,9,10,8,11,16,19,5,6,4,7,2,1,3))
        self.envs_hori.add( env_mpso_hori_twoSite_square(self.psi,env,label=outerType) )

class gradImplementation_mpso_2d_mpo_twoSite_square_twoBodyH_vert(gradImplementation_mpso_2d_mpo_twoSite_square_H_verticalLength_2):
    def buildEnvs_vert_outerType(self,outerType):
        #term under H 
        env = ncon([self.psi.mpo,self.psi.mpo,self.H],((2,-5,-1,-4,-11,-12),(7,-10,-6,-9,-13,-14),(-8,-3,7,2)),forder=(-8,-10,-3,-5,-6,-9,-1,-4,-13,-14,-11,-12))
        env += ncon([self.psi.mpo,self.psi.mpo,self.H],((-2,4,-1,-3,-11,-12),(-7,9,-6,-8,-13,-14),(-10,-5,9,4)),forder=(-7,-10,-2,-5,-6,-8,-1,-3,-13,-14,-11,-12))
        self.envs_vert_quadrantSeed.add( env_mpso_vert_twoSite_square(self.psi,env,shape=[2,1],label=outerType) )
        self.envs_vert.add( env_mpso_vert_twoSite_square(self.psi,env,shape=[2,1],label=outerType) )
        #term right of H 
        env = ncon([env,self.psi.mpo.conj(),self.psi.mpo.conj(),self.outers['len2'][outerType],self.outers['len2'][outerType]],((8,11,2,5,7,10,1,4,13,-14,16,-17),(8,11,9,12,13,-15),(2,5,3,6,16,-18),(9,3,7,1),(12,6,10,4)),forder=(-15,-14,-18,-17),order=(13,8,11,9,7,10,12,16,2,5,1,3,4,6))
        env = self.psi.Tb2_inv[outerType].applyRight(env.reshape(self.psi.D_mpo**4)).reshape(self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo,self.psi.D_mpo)
        env = ncon([env,self.psi.mpo,self.psi.mpo],((-12,11,-10,9),(-2,-6,-1,-5,9,-13),(-4,-8,-3,-7,11,-14)),forder=(-4,-8,-2,-6,-3,-7,-1,-5,-12,-14,-10,-13))
        self.envs_vert.add( env_mpso_vert_twoSite_square(self.psi,env,shape=[2,1],label=outerType) )
    def buildEnvs_hori_outerType(self,outerType):
        envDouble = ncon([self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo.conj(),self.H,self.psi.RR[outerType].tensor,self.outers['len2'][outerType],self.outers['len2'][outerType]],((2,6,1,5,-15,16),(3,6,4,7,-17,18),(9,13,8,12,-19,20),(10,13,11,14,-21,22),(10,3,9,2),(22,20,18,16),(11,4,8,1),(14,7,12,5)),forder=(-21,-19,-17,-15),order=(20,22,13,12,14,9,10,8,11,18,16,6,5,7,2,3,4,1))
        envDouble += ncon([self.psi.mpo,self.psi.mpo.conj(),self.psi.mpo,self.psi.mpo.conj(),self.H,self.psi.RR[outerType].tensor,self.outers['len2'][outerType],self.outers['len2'][outerType]],((2,5,1,4,-15,16),(2,6,3,7,-17,18),(9,12,8,11,-19,20),(9,13,10,14,-21,22),(13,6,12,5),(22,20,18,16),(10,3,8,1),(14,7,11,4)),forder=(-21,-19,-17,-15),order=(20,22,8,8,10,12,13,11,14,18,16,2,1,3,5,6,4,7))
        if outerType == 'square':
            self.envs_hori.add( env_mpso_hori_twoSite_square(self.psi,np.einsum('abcc->ab',envDouble),label='bot')) 
            self.envs_hori.add( env_mpso_hori_twoSite_square(self.psi,np.einsum('aabc->bc',envDouble),label='top')) 
        elif outerType == 'prong':
            self.envs_hori.add( env_mpso_hori_twoSite_square(self.psi,np.einsum('abcc->ab',envDouble),label='top')) 
            self.envs_hori.add( env_mpso_hori_twoSite_square(self.psi,np.einsum('aabc->bc',envDouble),label='bot')) 

# -----------------------------
class d_dep_quants(ABC):
    def __init__(self,psi):
        self.psi = psi

    @abstractmethod
    def init_mps_transfers(self):
        pass
    @abstractmethod
    def apply_mps_transfers(self):
        pass
    @abstractmethod
    def getFixedPoints(self):
        pass
    @abstractmethod
    def getOuterContracts(self):
        pass

class d_dep_quants_uniform(d_dep_quants):
    #(Ta)^d d dep transfer matrices, for geosum
    def init_mps_transfers(self):
        Td_matrix = np.eye(self.psi.D_mps**2)
        Td = Td_matrix.reshape(self.psi.D_mps,self.psi.D_mps,self.psi.D_mps,self.psi.D_mps)
        return Td_matrix,Td
    def apply_mps_transfers(self,Td_matrix):
        Td_matrix = np.dot(Td_matrix,self.psi.Ta.matrix)
        Td = Td_matrix.reshape(self.psi.D_mps,self.psi.D_mps,self.psi.D_mps,self.psi.D_mps)
        return Td_matrix,Td

class d_dep_quants_bipartite(d_dep_quants):
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

class d_dep_quants_uniform_verticalLength1(d_dep_quants_uniform):
    def getFixedPoints(self,d,Td):
        if d == 0:
            RR_d = self.psi.RR
        else:
            TT_d = mpsu1Transfer_left_twoLayerWithMpsInsert(self.psi.mps,self.psi.mpo,self.psi.T,Td)
            RR_d = TT_d.findRightEig()
            RR_d.norm_pairedCanon()
        return RR_d.tensor
    def getOuterContracts(self,d,Td):
        return ncon([self.psi.mps,self.psi.mps,self.psi.mps.conj(),self.psi.mps.conj(),Td,self.psi.T.tensor],((-1,5,6),(-2,7,8),(-3,5,11),(-4,10,9),(11,6,10,7),(9,8)),forder=(-3,-4,-1,-2),order=(5,6,11,7,10,8,9))

class d_dep_quants_uniform_verticalLength2(d_dep_quants_uniform):
    def getFixedPoints(self,d,Td):
        Twd_lower = mpsu1Transfer_left_threeLayerWithMpsInsert_lower(self.psi.mps,self.psi.mpo,self.psi.T,Td)
        R_Twd_lower = Twd_lower.findRightEig()
        R_Twd_lower.norm_pairedCanon()
        Twd_upper = mpsu1Transfer_left_threeLayerWithMpsInsert_upper(self.psi.mps,self.psi.mpo,self.psi.T,Td)
        R_Twd_upper = Twd_upper.findRightEig()
        R_Twd_upper.norm_pairedCanon()
        RRR = dict()
        RRR['lower'] = R_Twd_lower.tensor
        RRR['upper'] = R_Twd_upper.tensor
        return RRR

    def getOuterContracts(self,d,Td):
        outers = dict()
        outers['upper'] = ncon([self.psi.mps,self.psi.mps,self.psi.mps,self.psi.mps.conj(),self.psi.mps.conj(),self.psi.mps.conj(),Td,self.psi.T.tensor],((-1,7,8),(-3,8,9),(-5,10,11),(-2,7,15),(-4,15,14),(-6,13,12),(14,9,13,10),(12,11)),forder=(-2,-4,-6,-1,-3,-5),order=(7,8,15,9,14,10,13,11,12))
        outers['lower'] = ncon([self.psi.mps,self.psi.mps,self.psi.mps,self.psi.mps.conj(),self.psi.mps.conj(),self.psi.mps.conj(),Td,self.psi.T.tensor],((-1,7,8),(-3,9,10),(-5,10,11),(-2,7,15),(-4,14,13),(-6,13,12),(15,8,14,9),(12,11)),forder=(-2,-4,-6,-1,-3,-5),order=(7,8,15,9,14,10,13,11,12))
        return outers

class d_dep_quants_bipartite_verticalLength1(d_dep_quants_bipartite):
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

    def getOuterContracts(self,d,Td):
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

class d_dep_quants_bipartite_verticalLength2(d_dep_quants_bipartite):
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

    def getOuterContracts(self,d,Td):
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

class d_dep_quants_fourSite_sep(d_dep_quants):
    #d dep bipartite transfers for geosum
    def init_mps_transfers(self):
        Td_matrix = dict()
        Td = dict()
        Td_matrix[1] = np.eye(self.psi.D_mps**2)
        Td_matrix[2] = Td_matrix[1]
        Td_matrix[3] = Td_matrix[1]
        Td_matrix[4] = Td_matrix[1]
        Td[1] = Td_matrix[1].reshape(self.psi.D_mps,self.psi.D_mps,self.psi.D_mps,self.psi.D_mps)
        Td[2] = Td[1]
        Td[3] = Td[1]
        Td[4] = Td[1]
        return Td_matrix,Td
    def apply_mps_transfers(self,Td_matrix):
        Td_matrix[1] = np.dot(Td_matrix[1],self.psi.Ta[1].matrix)
        Td_matrix[2] = np.dot(Td_matrix[2],self.psi.Ta[2].matrix)
        Td_matrix[3] = np.dot(Td_matrix[3],self.psi.Ta[3].matrix)
        Td_matrix[4] = np.dot(Td_matrix[4],self.psi.Ta[4].matrix)
        Td = dict()
        Td[1] = Td_matrix[1].reshape(self.psi.D_mps,self.psi.D_mps,self.psi.D_mps,self.psi.D_mps)
        Td[2] = Td_matrix[2].reshape(self.psi.D_mps,self.psi.D_mps,self.psi.D_mps,self.psi.D_mps)
        Td[3] = Td_matrix[3].reshape(self.psi.D_mps,self.psi.D_mps,self.psi.D_mps,self.psi.D_mps)
        Td[4] = Td_matrix[4].reshape(self.psi.D_mps,self.psi.D_mps,self.psi.D_mps,self.psi.D_mps)
        return Td_matrix,Td

#TO DO
class d_dep_quants_fourSite_sep_verticalLength1(d_dep_quants_fourSite_sep):
    def getFixedPoints(self,d,Td):
        pass
    def getOuterContracts(self,d,Td):
        pass

class d_dep_quants_fourSite_sep_verticalLength2(d_dep_quants_fourSite_sep):
    def getFixedPoints(self,d,Td):
        pass
    def getOuterContracts(self,d,Td):
        pass

class d_dep_quants_twoSite_square_verticalLength1(d_dep_quants_uniform):
    def getFixedPoints(self,d,Td):
        RR_d = dict()

        TT_d_bb = mpsu1Transfer_left_twoLayerWithMPSInsert_twoSite_square(self.psi.mps,self.psi.mpo,self.psi.T,'bb',Td)
        RR_d['bb'] = TT_d_bb.findRightEig()
        RR_d['bb'].norm_pairedCanon()
        RR_d['bb'] = RR_d['bb'].tensor
        del TT_d_bb

        TT_d_bt = mpsu1Transfer_left_twoLayerWithMPSInsert_twoSite_square(self.psi.mps,self.psi.mpo,self.psi.T,'bt',Td)
        RR_d['bt'] = TT_d_bt.findRightEig()
        RR_d['bt'].norm_pairedCanon()
        RR_d['bt'] = RR_d['bt'].tensor
        del TT_d_bt

        TT_d_tb = mpsu1Transfer_left_twoLayerWithMPSInsert_twoSite_square(self.psi.mps,self.psi.mpo,self.psi.T,'tb',Td)
        RR_d['tb'] = TT_d_tb.findRightEig()
        RR_d['tb'].norm_pairedCanon()
        RR_d['tb'] = RR_d['tb'].tensor
        del TT_d_tb

        TT_d_tt = mpsu1Transfer_left_twoLayerWithMPSInsert_twoSite_square(self.psi.mps,self.psi.mpo,self.psi.T,'tt',Td)
        RR_d['tt'] = TT_d_tt.findRightEig()
        RR_d['tt'].norm_pairedCanon()
        RR_d['tt'] = RR_d['tt'].tensor
        del TT_d_tt

        fp = dict()
        fp['RR_d'] = RR_d
        return fp

    def getOuterContracts(self,d,Td):
        outerContractDouble = dict()
        outerContractQuad = ncon([self.psi.mps,self.psi.mps,self.psi.mps.conj(),self.psi.mps.conj(),Td,self.psi.T.tensor],((-1,-3,9,10),(-5,-7,11,12),(-2,-4,9,15),(-6,-8,14,13),(15,10,14,11),(13,12)),forder=(-2,-4,-6,-8,-1,-3,-5,-7),order=(9,10,15,11,14,13,12))
        outerContractDouble['bb'] = ncon([outerContractQuad],((-2,3,-5,6,-1,3,-4,6)),forder=(-2,-5,-1,-4))
        outerContractDouble['bt']= ncon([outerContractQuad],((-2,3,4,-6,-1,3,4,-5)),forder=(-2,-6,-1,-5))
        outerContractDouble['tb']= ncon([outerContractQuad],((1,-3,-5,6,1,-2,-4,6)),forder=(-3,-5,-2,-4))
        outerContractDouble['tt']= ncon([outerContractQuad],((1,-3,4,-6,1,-2,4,-5)),forder=(-3,-6,-2,-5))
        fp = dict()
        fp['outerContractDouble'] = outerContractDouble
        return fp

class d_dep_quants_twoSite_square_verticalLength2(d_dep_quants_uniform):
    def getFixedPoints(self,d,Td):
        RRR_d_lower = dict()
        RRR_d_upper = dict()

        transfer_sb = mpsu1Transfer_left_threeLayerWithMPSInsert_twoSite_upper(self.psi.mps,self.psi.mpo,self.psi.T,'sb',Td)
        RRR_d_upper['sb'] = transfer_sb.findRightEig()
        RRR_d_upper['sb'].norm_pairedCanon()
        RRR_d_upper['sb'] = RRR_d_upper['sb'].tensor
        del transfer_sb

        transfer_st = mpsu1Transfer_left_threeLayerWithMPSInsert_twoSite_upper(self.psi.mps,self.psi.mpo,self.psi.T,'st',Td)
        RRR_d_upper['st'] = transfer_st.findRightEig()
        RRR_d_upper['st'].norm_pairedCanon()
        RRR_d_upper['st'] = RRR_d_upper['st'].tensor
        del transfer_st

        transfer_pb = mpsu1Transfer_left_threeLayerWithMPSInsert_twoSite_upper(self.psi.mps,self.psi.mpo,self.psi.T,'pb',Td)
        RRR_d_upper['pb'] = transfer_pb.findRightEig()
        RRR_d_upper['pb'].norm_pairedCanon()
        RRR_d_upper['pb'] = RRR_d_upper['pb'].tensor
        del transfer_pb

        transfer_pt = mpsu1Transfer_left_threeLayerWithMPSInsert_twoSite_upper(self.psi.mps,self.psi.mpo,self.psi.T,'pt',Td)
        RRR_d_upper['pt'] = transfer_pt.findRightEig()
        RRR_d_upper['pt'].norm_pairedCanon()
        RRR_d_upper['pt'] = RRR_d_upper['pt'].tensor
        del transfer_pt

        transfer_bs = mpsu1Transfer_left_threeLayerWithMPSInsert_twoSite_lower(self.psi.mps,self.psi.mpo,self.psi.T,'bs',Td)
        RRR_d_lower['bs'] = transfer_bs.findRightEig()
        RRR_d_lower['bs'].norm_pairedCanon()
        RRR_d_lower['bs'] = RRR_d_lower['bs'].tensor
        del transfer_bs

        transfer_ts = mpsu1Transfer_left_threeLayerWithMPSInsert_twoSite_lower(self.psi.mps,self.psi.mpo,self.psi.T,'ts',Td)
        RRR_d_lower['ts'] = transfer_ts.findRightEig()
        RRR_d_lower['ts'].norm_pairedCanon()
        RRR_d_lower['ts'] = RRR_d_lower['ts'].tensor
        del transfer_ts

        transfer_bp = mpsu1Transfer_left_threeLayerWithMPSInsert_twoSite_lower(self.psi.mps,self.psi.mpo,self.psi.T,'bp',Td)
        RRR_d_lower['bp'] = transfer_bp.findRightEig()
        RRR_d_lower['bp'].norm_pairedCanon()
        RRR_d_lower['bp'] = RRR_d_lower['bp'].tensor
        del transfer_bp

        transfer_tp = mpsu1Transfer_left_threeLayerWithMPSInsert_twoSite_lower(self.psi.mps,self.psi.mpo,self.psi.T,'tp',Td)
        RRR_d_lower['tp'] = transfer_tp.findRightEig()
        RRR_d_lower['tp'].norm_pairedCanon()
        RRR_d_lower['tp'] = RRR_d_lower['tp'].tensor
        del transfer_tp

        fp = dict()
        fp['RRR_d_lower'] = RRR_d_lower
        fp['RRR_d_upper'] = RRR_d_upper
        return fp

    def getOuterContracts(self,d,Td):
        outerContractTriple_upper = dict()
        outerContractTriple_lower = dict()

        outerContractTriple_square_Quad = ncon([self.psi.mps,self.psi.mps,self.psi.mps.conj(),self.psi.mps.conj(),Td,self.psi.T.tensor],((-1,-3,9,10),(-5,-7,11,12),(-2,-4,9,15),(-6,-8,14,13),(15,10,14,11),(13,12)),forder=(-2,-4,-6,-8,-1,-3,-5,-7),order=(9,10,15,11,14,12,13))
        outerContractTriple_upper_prong_Quad = ncon([self.psi.mps,self.psi.mps,self.psi.mps,self.psi.mps.conj(),self.psi.mps.conj(),self.psi.mps.conj(),Td,self.psi.T.tensor],((1,-2,11,12),(-4,6,12,13),(-7,-9,14,15),(1,-3,11,19),(-5,6,19,18),(-8,-10,17,16),(18,13,17,14),(16,15)),forder=(-3,-5,-8,-10,-2,-4,-7,-9),order=(11,1,12,19,6,13,18,14,17,15,16))
        outerContractTriple_lower_prong_Quad = ncon([self.psi.mps,self.psi.mps,self.psi.mps,self.psi.mps.conj(),self.psi.mps.conj(),self.psi.mps.conj(),Td,self.psi.T.tensor],((-1,-3,11,12),(5,-6,13,14),(-8,10,14,15),(-2,-4,11,19),(5,-7,18,17),(-9,10,17,16),(19,12,18,13),(16,15)),forder=(-2,-4,-7,-9,-1,-3,-6,-8),order=(15,16,10,14,17,5,13,18,12,19,11))

        outerContractTriple_upper['sb'] = ncon([outerContractTriple_square_Quad],((-2,-4,-6,7,-1,-3,-5,7)),forder=(-2,-4,-6,-1,-3,-5))
        outerContractTriple_upper['st'] = ncon([outerContractTriple_square_Quad],((-2,-4,5,-7,-1,-3,5,-6)),forder=(-2,-4,-7,-1,-3,-6))
        outerContractTriple_upper['pb'] = ncon([outerContractTriple_upper_prong_Quad],((-2,-4,-6,7,-1,-3,-5,7)),forder=(-2,-4,-6,-1,-3,-5))
        outerContractTriple_upper['pt'] = ncon([outerContractTriple_upper_prong_Quad],((-2,-4,5,-7,-1,-3,5,-6)),forder=(-2,-4,-7,-1,-3,-6))
        outerContractTriple_lower['bs'] = ncon([outerContractTriple_square_Quad],((-2,3,-5,-7,-1,3,-4,-6)),forder=(-2,-5,-7,-1,-4,-6))
        outerContractTriple_lower['ts'] = ncon([outerContractTriple_square_Quad],((1,-3,-5,-7,1,-2,-4,-6)),forder=(-3,-5,-7,-2,-4,-6))
        outerContractTriple_lower['bp'] = ncon([outerContractTriple_lower_prong_Quad],((-2,3,-5,-7,-1,3,-4,-6)),forder=(-2,-5,-7,-1,-4,-6))
        outerContractTriple_lower['tp'] = ncon([outerContractTriple_lower_prong_Quad],((1,-3,-5,-7,1,-2,-4,-6)),forder=(-3,-5,-7,-2,-4,-6))

        fp = dict()
        fp['outerContractTriple_upper'] = outerContractTriple_upper
        fp['outerContractTriple_lower'] = outerContractTriple_lower
        return fp
