#!/usr/bin/env python# -*- coding: utf-8 -*-

from __future__ import division
import math
import numpy as np
import scipy as sp
from ncon import ncon
from uniformTN_states import *
import copy

class localH:
    def __init__(self,H_terms):
        self.terms = H_terms
    def exp(self,psi):
        E = 0
        for n in range(0,len(self.terms)):
            E += self.terms[n].exp(psi)
        return E
    def subtractExp(self,psi):
        self.subtractedTerms = dict()
        for n in range(0,len(self.terms)):
            self.subtractedTerms[n] = self.terms[n].subtractExp(psi)

    def __add__(self,H2):
        newObj = copy.deepcopy(self)
        added = []
        for n in range(0,len(self.terms)):
            for m in range(0,len(H2.terms)):
                if m not in added:
                    if type(self.terms[n]) is type(H2.terms[m]):
                        newObj.terms[n] += H2.terms[m]
                        added.append(m)

        #append terms that weren't added
        not_added = list( set(np.arange(0,len(H2.terms))) -set(added))
        for n in range(0,len(not_added)):
            newObj.terms.append(H2.terms[not_added[n]])
        return newObj

            
class localH_term:
    def __init__(self,H):
        self.matrix = H
        self.tensor = self.reshapeTensor(self.matrix)
    def subtractExp(self,psi):
        return self.reshapeTensor(self.matrix - self.exp(psi)*np.eye(np.size(self.matrix,axis=0)))

    def __add__(self,H2):
        if type(self) is type(H2):
            newObj = copy.deepcopy(self)
            newObj.__init__(self.matrix + H2.matrix)
            return newObj
        else:
            print("ERROR: adding different type of H ",type(self)," and ",type(H2))


    def exp(self,psi):
        if type(psi) == uMPSU1_2d_left:
            return self.exp_2d_left(psi)
        elif type(psi) == uMPSU1_2d_left_twoSite_square:
            return self.exp_2d_left_twoSite_square(psi)
        elif type(psi) == uMPSU1_2d_left_twoSite_staircase:
            return self.exp_2d_left_twoSite_staircase(psi)
        elif type(psi) == uMPSU1_2d_left_bipartite:
            return self.exp_2d_left_bipartite(psi)

        #blocked ansatz are same as uniform code but with effective hamiltonian
        elif type(psi) == uMPSU1_2d_left_NSite_block:
            return self.exp_2d_left(psi)

        elif type(psi) == uMPSU1_2d_left_fourSite_sep:
            return self.exp_2d_left_fourSite_sep(psi)

        elif type(psi) == uMPS_1d_left:
            return self.exp_1d_left(psi)
        elif type(psi) == uMPS_1d:
            return self.exp_1d(psi)
        elif type(psi) == uMPS_1d_left_bipartite:
            return self.exp_1d_left_bipartite(psi)
        elif type(psi) == uMPS_1d_left_twoSite:
            return self.exp_1d_left_twoSite(psi)

class oneBodyH(localH_term):
    def reshapeTensor(self,H_matrix):
        return H_matrix

    def exp_1d(self,psi):
        return np.real(ncon([psi.mps,self.tensor,psi.mps.conj(),psi.L.tensor,psi.R.tensor],((1,3,4),(2,1),(2,5,6),(5,3),(6,4))),order=(3,5,1,2,4,6))
    def exp_1d_left(self,psi):
        return np.real(ncon([psi.mps,self.tensor,psi.mps.conj(),psi.R.tensor],((1,3,4),(2,1),(2,3,5),(5,4)),order=(3,1,2,4,5)))
    def exp_2d_left(self,psi):
        return np.real(ncon([psi.mps,psi.mpo,self.tensor,psi.mpo.conj(),psi.mps.conj(),psi.R.tensor,psi.T.tensor],((1,8,9),(2,1,5,6),(3,2),(3,4,5,7),(4,8,10),(7,6),(10,9)),order=(8,9,10,5,2,3,7,6,1,4))) 

    def exp_1d_left_twoSite(self,psi):
        E = np.real(ncon([psi.mps,self.tensor,psi.mps.conj(),psi.R.tensor],((1,2,4,5),(3,1),(3,2,4,6),(6,5)),order=(4,1,3,2,5,6)))
        E += np.real(ncon([psi.mps,self.tensor,psi.mps.conj(),psi.R.tensor],((1,2,4,5),(3,2),(1,3,4,6),(6,5)),order=(4,1,2,3,5,6)))
        return E/2

    def exp_1d_left_bipartite_ind(self,mps1,R2):
        return np.real(ncon([mps1,self.tensor,mps1.conj(),R2.tensor],((1,3,4),(2,1),(2,3,5),(5,4)),order=(3,1,2,4,5)))
    def exp_1d_left_bipartite(self,psi):
        E = self.exp_1d_left_bipartite_ind(psi.mps[1],psi.R[2])
        E += self.exp_1d_left_bipartite_ind(psi.mps[2],psi.R[1])
        return E/2

    def exp_2d_left_twoSite_square(self,psi):
        #four terms
        outerContract = dict()
        outerContract['bot'] = ncon([psi.mps,psi.mps.conj(),psi.T.tensor],((-1,3,4,5),(-2,3,4,6),(6,5)),forder=(-2,-1),order=(4,3,5,6))
        outerContract['top']= ncon([psi.mps,psi.mps.conj(),psi.T.tensor],((1,-2,4,5),(1,-3,4,6),(6,5)),forder=(-3,-2),order=(4,1,5,6))

        E = ncon([psi.mpo,self.tensor,psi.mpo.conj(),outerContract['bot'],outerContract['bot'],psi.R['bot'].tensor],((2,6,1,5,8,9),(3,2),(3,6,4,7,8,10),(4,1),(7,5),(10,9)),order=(8,1,2,3,4,5,6,7,9,10))
        E += ncon([psi.mpo,self.tensor,psi.mpo.conj(),outerContract['top'],outerContract['top'],psi.R['top'].tensor],((2,6,1,5,8,9),(3,2),(3,6,4,7,8,10),(4,1),(7,5),(10,9)),order=(8,1,2,3,4,5,6,7,9,10))
        E += ncon([psi.mpo,self.tensor,psi.mpo.conj(),outerContract['bot'],outerContract['bot'],psi.R['bot'].tensor],((2,5,1,4,8,9),(6,5),(2,6,3,7,8,10),(3,1),(7,4),(10,9)),order=(8,1,2,3,4,5,6,7,9,10))
        E += ncon([psi.mpo,self.tensor,psi.mpo.conj(),outerContract['top'],outerContract['top'],psi.R['top'].tensor],((2,5,1,4,8,9),(6,5),(2,6,3,7,8,10),(3,1),(7,4),(10,9)),order=(8,1,2,3,4,5,6,7,9,10))
        return np.real(E/4)

    def exp_2d_left_twoSite_staircase(self,psi):
        #two terms
        outerContract = dict()
        outerContract['bot'] = ncon([psi.mps,psi.mps.conj(),psi.T.tensor],((-1,3,4,5),(-2,3,4,6),(6,5)),forder=(-2,-1),order=(4,3,5,6))
        outerContract['top']= ncon([psi.mps,psi.mps.conj(),psi.T.tensor],((1,-2,4,5),(1,-3,4,6),(6,5)),forder=(-3,-2),order=(4,1,5,6))

        E = ncon([psi.mpo,psi.mpo.conj(),self.tensor,outerContract['bot'],outerContract['top'],psi.R.tensor],((2,6,1,5,8,9),(3,6,4,7,8,10),(3,2),(4,1),(7,5),(10,9)),order=(8,6,2,3,1,4,5,7,9,10))
        E += ncon([psi.mpo,psi.mpo.conj(),self.tensor,outerContract['bot'],outerContract['top'],psi.R.tensor],((2,5,1,4,8,9),(2,6,3,7,8,10),(6,5),(3,1),(7,4),(10,9)),order=(8,2,5,6,1,3,4,7,9,10))
        return np.real(E/2)

    def exp_2d_left_bipartite(self,psi):
        E = ncon([psi.mps[1],psi.mpo[1],self.tensor,psi.mpo[1].conj(),psi.mps[1].conj(),psi.R[2].tensor,psi.T[2].tensor],((1,5,6),(2,1,8,9),(3,2),(3,4,8,10),(4,5,7),(10,9),(7,6)))
        E += ncon([psi.mps[2],psi.mpo[2],self.tensor,psi.mpo[2].conj(),psi.mps[2].conj(),psi.R[1].tensor,psi.T[1].tensor],((1,5,6),(2,1,8,9),(3,2),(3,4,8,10),(4,5,7),(10,9),(7,6)))
        return np.real(E/2)

    #2d left fourSite
    def exp_2d_left_fourSite_sep(self,psi):    
        E = ncon([psi.mps[1],psi.mpo[1],self.tensor,psi.mpo[1].conj(),psi.mps[1].conj(),psi.R[2].tensor,psi.T[3].tensor],((1,5,6),(2,1,8,9),(3,2),(3,4,8,10),(4,5,7),(10,9),(7,6)))
        E += ncon([psi.mps[2],psi.mpo[2],self.tensor,psi.mpo[2].conj(),psi.mps[2].conj(),psi.R[1].tensor,psi.T[4].tensor],((1,5,6),(2,1,8,9),(3,2),(3,4,8,10),(4,5,7),(10,9),(7,6)))
        E += ncon([psi.mps[3],psi.mpo[3],self.tensor,psi.mpo[3].conj(),psi.mps[3].conj(),psi.R[4].tensor,psi.T[1].tensor],((1,5,6),(2,1,8,9),(3,2),(3,4,8,10),(4,5,7),(10,9),(7,6)))
        E += ncon([psi.mps[4],psi.mpo[4],self.tensor,psi.mpo[4].conj(),psi.mps[4].conj(),psi.R[3].tensor,psi.T[2].tensor],((1,5,6),(2,1,8,9),(3,2),(3,4,8,10),(4,5,7),(10,9),(7,6)))
        return E/4

class twoBodyH(localH_term):
    def reshapeTensor(self,H_matrix):
        physDim = int(np.sqrt(H_matrix.shape[0]))
        return H_matrix.reshape([physDim,physDim,physDim,physDim])

    #1d
    def exp_1d(self,psi):
        return np.real(ncon([psi.mps,psi.mps,self.tensor,psi.mps.conj(),psi.mps.conj(),psi.L.tensor,psi.R.tensor],((1,5,6),(3,6,7),(2,4,1,3),(2,10,9),(4,9,8),(10,5),(8,7))))
    def exp_1d_left(self,psi):
        return np.real(ncon([psi.mps,psi.mps,self.tensor,psi.mps.conj(),psi.mps.conj(),psi.R.tensor],((1,5,6),(3,6,7),(2,4,1,3),(2,5,9),(4,9,8),(8,7))))

    def exp_1d_left_twoSite(self,psi):
        E = np.real(ncon([psi.mps,self.tensor,psi.mps.conj(),psi.R.tensor],((1,2,5,6),(3,4,1,2),(3,4,5,7),(7,6)),order=(5,1,3,2,4,6,7)))
        E += np.real(ncon([psi.mps,psi.mps,self.tensor,psi.mps.conj(),psi.mps.conj(),psi.R.tensor],((1,2,7,8),(3,4,8,9),(5,6,2,3),(1,5,7,11),(6,4,11,10),(10,9)),order=(7,1,2,5,8,11,3,6,4,9,10)))
        return E/2

    def exp_1d_left_bipartite_ind(self,mps1,mps2,R1):
        return np.real(ncon([mps1,mps2,self.tensor,mps1.conj(),mps2.conj(),R1.tensor],((1,5,6),(3,6,7),(2,4,1,3),(2,5,9),(4,9,8),(8,7))))
    def exp_1d_left_bipartite(self,psi):
        E = self.exp_1d_left_bipartite_ind(psi.mps[1],psi.mps[2],psi.R[1])
        E += self.exp_1d_left_bipartite_ind(psi.mps[2],psi.mps[1],psi.R[2])
        return E/2

class twoBodyH_hori(twoBodyH):

    def exp_2d_left(self,psi):
        centreContract = ncon([psi.mpo,psi.mpo,self.tensor,psi.mpo.conj(),psi.mpo.conj(),psi.R.tensor],((2,-1,9,10),(6,-5,10,11),(3,7,2,6),(3,-4,9,13),(7,-8,13,12),(12,11)),forder=(-4,-8,-1,-5),order=(9,2,3,10,13,6,7,11,12))
        outerContract = ncon([psi.mps,psi.mps.conj(),psi.T.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1))
        E = np.real(ncon([outerContract,centreContract,outerContract],((2,1),(2,3,1,4),(3,4)),order=(2,1,3,4)))
        return E

    #2d left biparite
    def exp_2d_left_bipartite_ind(self,mps1,mps2,mpo1,mpo2,T1,T2,R1):    
        centreContract = ncon([mpo1,mpo2,self.tensor,mpo1.conj(),mpo2.conj(),R1.tensor],((1,-10,5,6),(2,-11,6,7),(3,4,1,2),(3,-12,5,9),(4,-13,9,8),(8,7)),forder=(-12,-13,-10,-11),order=(5,6,9,7,8,1,2,3,4))
        outerContract1 = ncon([mps1,mps1.conj(),T2.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1))
        outerContract2 = ncon([mps2,mps2.conj(),T1.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1))
        E = np.real(ncon([outerContract1,centreContract,outerContract2],((2,1),(2,3,1,4),(3,4))))
        return E

    def exp_2d_left_bipartite(self,psi):
        E = self.exp_2d_left_bipartite_ind(psi.mps[1],psi.mps[2],psi.mpo[1],psi.mpo[2],psi.T[1],psi.T[2],psi.R[1])
        E += self.exp_2d_left_bipartite_ind(psi.mps[2],psi.mps[1],psi.mpo[2],psi.mpo[1],psi.T[2],psi.T[1],psi.R[2])
        return E/2

    def exp_2d_left_twoSite_square(self,psi):
        #four terms
        outerContract = dict()
        outerContract['bot'] = ncon([psi.mps,psi.mps.conj(),psi.T.tensor],((-1,3,4,5),(-2,3,4,6),(6,5)),forder=(-2,-1),order=(4,3,5,6))
        outerContract['top']= ncon([psi.mps,psi.mps.conj(),psi.T.tensor],((1,-2,4,5),(1,-3,4,6),(6,5)),forder=(-3,-2),order=(4,1,5,6))

        E = ncon([psi.mpo,self.tensor,psi.mpo.conj(),outerContract['bot'],outerContract['bot'],psi.R['bot'].tensor],((2,6,1,5,9,10),(3,7,2,6),(3,7,4,8,9,11),(4,1),(8,5),(11,10)),order=(9,1,2,3,4,5,6,7,8,10,11))
        E += ncon([psi.mpo,self.tensor,psi.mpo.conj(),outerContract['top'],outerContract['top'],psi.R['top'].tensor],((2,6,1,5,9,10),(3,7,2,6),(3,7,4,8,9,11),(4,1),(8,5),(11,10)),order=(9,1,2,3,4,5,6,7,8,10,11))
        E += ncon([psi.mpo,psi.mpo,self.tensor,psi.mpo.conj(),psi.mpo.conj(),outerContract['bot'],outerContract['bot'],outerContract['bot'],outerContract['bot'],psi.R['bot'].tensor],((2,5,1,4,15,16),(9,13,8,12,16,17),(6,10,5,9),(2,6,3,7,15,19),(10,13,11,14,19,18),(3,1),(7,4),(11,8),(14,12),(18,17)),order=(15,1,2,3,4,5,6,7,16,19,8,9,10,11,12,13,14,17,18))
        E += ncon([psi.mpo,psi.mpo,self.tensor,psi.mpo.conj(),psi.mpo.conj(),outerContract['top'],outerContract['top'],outerContract['top'],outerContract['top'],psi.R['top'].tensor],((2,5,1,4,15,16),(9,13,8,12,16,17),(6,10,5,9),(2,6,3,7,15,19),(10,13,11,14,19,18),(3,1),(7,4),(11,8),(14,12),(18,17)),order=(15,1,2,3,4,5,6,7,16,19,8,9,10,11,12,13,14,17,18))
        return np.real(E/4)

    def exp_2d_left_twoSite_staircase(self,psi):
        #two terms
        outerContract = dict()
        outerContract['bot'] = ncon([psi.mps,psi.mps.conj(),psi.T.tensor],((-1,3,4,5),(-2,3,4,6),(6,5)),forder=(-2,-1),order=(4,3,5,6))
        outerContract['top']= ncon([psi.mps,psi.mps.conj(),psi.T.tensor],((1,-2,4,5),(1,-3,4,6),(6,5)),forder=(-3,-2),order=(4,1,5,6))

        E = ncon([psi.mpo,self.tensor,psi.mpo.conj(),outerContract['bot'],outerContract['top'],psi.R.tensor],((2,6,1,5,9,10),(3,7,2,6),(3,7,4,8,9,11),(4,1),(8,5),(11,10)),order=(9,2,6,3,7,1,4,5,8,10,11))
        E += ncon([psi.mpo,psi.mpo.conj(),self.tensor,psi.mpo,psi.mpo.conj(),outerContract['bot'],outerContract['top'],outerContract['bot'],outerContract['top'],psi.R.tensor],((2,5,1,4,15,16),(2,6,3,7,15,19),(6,10,5,9),(9,13,8,12,16,17),(10,13,11,14,19,18),(3,1),(7,4),(11,8),(14,12),(18,17)),order=(15,2,1,3,4,7,5,6,16,19,13,9,10,8,11,12,14,17,18))
        return np.real(E/2)

    #2d left fourSite
    def exp_2d_left_fourSite_sep(self,psi):    
        E = self.exp_2d_left_bipartite_ind(psi.mps[1],psi.mps[2],psi.mpo[1],psi.mpo[2],psi.T[4],psi.T[3],psi.R[1])
        E += self.exp_2d_left_bipartite_ind(psi.mps[2],psi.mps[1],psi.mpo[2],psi.mpo[1],psi.T[3],psi.T[4],psi.R[2])
        E += self.exp_2d_left_bipartite_ind(psi.mps[3],psi.mps[4],psi.mpo[3],psi.mpo[4],psi.T[2],psi.T[1],psi.R[3])
        E += self.exp_2d_left_bipartite_ind(psi.mps[4],psi.mps[3],psi.mpo[4],psi.mpo[3],psi.T[1],psi.T[2],psi.R[4])
        return E/4

class twoBodyH_vert(twoBodyH):

    def exp_2d_left(self,psi):
        outerContract = ncon([psi.mps,psi.mps,psi.mps.conj(),psi.mps.conj(),psi.T.tensor],((-1,5,6),(-2,6,7),(-3,5,9),(-4,9,8),(8,7)),forder=(-3,-4,-1,-2),order=(5,6,9,7,8))
        innerContract = ncon([psi.mpo,psi.mpo,self.tensor,psi.mpo.conj(),psi.mpo.conj(),psi.RR.tensor],((2,-1,9,10),(6,-5,12,13),(3,7,2,6),(3,-4,9,11),(7,-8,12,14),(11,10,14,13)),forder=(-4,-8,-1,-5),order=(9,2,3,10,11,13,14,6,7,12))
        E = np.real(ncon([innerContract,outerContract],((1,2,3,4),(1,2,3,4))))
        return E

    def exp_2d_left_bipartite_ind(self,mps1,mps2,mpo1,mpo2,T1,RR2):
        outerContract = ncon([mps1,mps2,mps1.conj(),mps2.conj(),T1.tensor],((-1,5,6),(-2,6,7),(-3,5,9),(-4,9,8),(8,7)),forder=(-3,-4,-1,-2))
        innerContract = ncon([mpo1,mpo2,self.tensor,mpo1.conj(),mpo2.conj(),RR2.tensor],((1,-5,9,11),(2,-7,10,14),(3,4,1,2),(3,-6,9,12),(4,-8,10,13),(12,11,13,14)),forder=(-6,-8,-5,-7),order=(12,11,13,14,9,10,1,2,3,4))
        E = np.real(ncon([innerContract,outerContract],((1,2,3,4),(1,2,3,4))))
        return E

    def exp_2d_left_bipartite(self,psi):
        E =  self.exp_2d_left_bipartite_ind(psi.mps[1],psi.mps[2],psi.mpo[1],psi.mpo[2],psi.T[1],psi.RR[2])
        E +=  self.exp_2d_left_bipartite_ind(psi.mps[2],psi.mps[1],psi.mpo[2],psi.mpo[1],psi.T[2],psi.RR[1])
        return E/2

    def exp_2d_left_twoSite_square(self,psi):
        #four terms
        outerContract = dict()
        outerContract['square'] = ncon([psi.mps,psi.mps.conj(),psi.T.tensor],((-1,-3,5,6),(-2,-4,5,7),(7,6)),forder=(-2,-4,-1,-3),order=(5,6,7))
        outerContract['prong']= ncon([psi.mps,psi.mps,psi.mps.conj(),psi.mps.conj(),psi.T.tensor],((1,-2,7,8),(-4,6,8,9),(1,-3,7,11),(-5,6,11,10),(10,9)),forder=(-3,-5,-2,-4),order=(7,1,8,11,6,9,10))

        E = ncon([psi.mpo,psi.mpo.conj(),psi.mpo,psi.mpo.conj(),self.tensor,outerContract['square'],outerContract['square'],psi.RR['square'].tensor],((2,6,1,5,15,16),(3,6,4,7,15,17),(9,13,8,12,18,19),(10,13,11,14,18,20),(10,3,9,2),(11,4,8,1),(14,7,12,5),(20,19,17,16)),order=(15,1,2,3,4,5,6,7,16,17,19,20,12,13,14,8,9,10,11,18))
        E += ncon([psi.mpo,psi.mpo.conj(),psi.mpo,psi.mpo.conj(),self.tensor,outerContract['prong'],outerContract['prong'],psi.RR['prong'].tensor],((2,6,1,5,15,16),(3,6,4,7,15,17),(9,13,8,12,18,19),(10,13,11,14,18,20),(10,3,9,2),(11,4,8,1),(14,7,12,5),(20,19,17,16)),order=(15,1,2,3,4,5,6,7,16,17,19,20,12,13,14,8,9,10,11,18))
        E += ncon([psi.mpo,psi.mpo.conj(),psi.mpo,psi.mpo.conj(),self.tensor,outerContract['square'],outerContract['square'],psi.RR['square'].tensor],((2,5,1,4,15,16),(2,6,3,7,15,17),(9,12,8,11,18,19),(9,13,10,14,18,20),(13,6,12,5),(10,3,8,1),(14,7,11,4),(20,19,17,16)),order=(15,1,2,3,4,5,6,7,16,17,19,20,11,12,13,14,8,9,10,18))
        E += ncon([psi.mpo,psi.mpo.conj(),psi.mpo,psi.mpo.conj(),self.tensor,outerContract['prong'],outerContract['prong'],psi.RR['prong'].tensor],((2,5,1,4,15,16),(2,6,3,7,15,17),(9,12,8,11,18,19),(9,13,10,14,18,20),(13,6,12,5),(10,3,8,1),(14,7,11,4),(20,19,17,16)),order=(15,1,2,3,4,5,6,7,16,17,19,20,11,12,13,14,8,9,10,18))
        return np.real(E/4)

    def exp_2d_left_twoSite_staircase(self,psi):
        #two terms
        outerContract = dict()
        outerContractDouble = dict()
        outerContract['bot'] = ncon([psi.mps,psi.mps.conj(),psi.T.tensor],((-1,3,4,5),(-2,3,4,6),(6,5)),forder=(-2,-1),order=(4,3,5,6))
        outerContractDouble['square'] = ncon([psi.mps,psi.mps.conj(),psi.T.tensor],((-1,-3,5,6),(-2,-4,5,7),(7,6)),forder=(-2,-4,-1,-3),order=(5,6,7))
        outerContractDouble['prong'] = ncon([psi.mps,psi.mps,psi.mps.conj(),psi.mps.conj(),psi.T.tensor],((1,-2,7,8),(-4,6,8,9),(1,-3,7,11),(-5,6,11,10),(10,9)),forder=(-3,-5,-2,-4),order=(7,1,8,11,6,9,10))

        E = ncon([psi.mpo,psi.mpo.conj(),self.tensor,psi.mpo,psi.mpo.conj(),outerContract['bot'],outerContractDouble['square'],psi.RR.tensor],((2,5,1,4,15,16),(2,6,3,7,15,17),(10,6,9,5),(9,13,8,12,18,19),(10,13,11,14,18,20),(3,1),(11,7,8,4),(14,12,20,19,17,16)),order=(15,2,1,3,4,5,6,7,16,17,19,20,12,13,14,8,9,10,11,18))
        E += ncon([psi.mpo,psi.mpo.conj(),self.tensor,psi.mpo,psi.mpo.conj(),psi.mpo,psi.mpo.conj(),outerContract['bot'],outerContractDouble['prong'],outerContractDouble['square'],psi.RR.tensor],((2,5,1,4,21,22),(2,6,3,7,21,23),(6,10,5,9),(9,13,8,12,28,24),(10,13,11,14,28,25),(16,19,15,18,22,26),(16,19,17,20,23,27),(3,1),(7,11,4,8),(17,14,15,12),(20,18,27,26,25,24)),order=(25,24,13,28,8,9,10,11,12,14,26,27,18,19,20,15,16,17,22,23,2,4,5,6,7,1,3,21))
        return np.real(E/2)

    def exp_2d_left_fourSite_sep(self,psi):
        E =  self.exp_2d_left_bipartite_ind(psi.mps[1],psi.mps[3],psi.mpo[1],psi.mpo[3],psi.T[1],psi.RR[2])
        E +=  self.exp_2d_left_bipartite_ind(psi.mps[3],psi.mps[1],psi.mpo[3],psi.mpo[1],psi.T[3],psi.RR[4])
        E +=  self.exp_2d_left_bipartite_ind(psi.mps[2],psi.mps[4],psi.mpo[2],psi.mpo[4],psi.T[2],psi.RR[1])
        E +=  self.exp_2d_left_bipartite_ind(psi.mps[4],psi.mps[2],psi.mpo[4],psi.mpo[2],psi.T[4],psi.RR[3])
        return E/4


class plaquetteH(localH_term):
    def reshapeTensor(self,H_matrix):
        physDim = int(np.sqrt(np.sqrt(H_matrix.shape[0])))
        return H_matrix.reshape([physDim,physDim,physDim,physDim,physDim,physDim,physDim,physDim])
