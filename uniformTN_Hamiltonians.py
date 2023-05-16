#!/usr/bin/env python# -*- coding: utf-8 -*-

from __future__ import division
import math
import numpy as np
import scipy as sp
from ncon import ncon
from uniformTN_states import *

class localH:
    def __init__(self,H_terms):
        self.terms = H_terms
    def exp(self,psi):
        E = 0
        for n in range(0,len(self.terms)):
            E += self.terms[n].exp(psi)
        return E

class localH_term:
    def __init__(self,H):
        self.matrix = H
    def exp(self,psi):
        if type(psi) == uMPSU1_2d_left:
            return self.exp_2d_left(psi)
        elif type(psi) == uMPSU1_2d_left_bipartite:
            return self.exp_2d_left_bipartite(psi)
        elif type(psi) == uMPS_1d_left:
            return self.exp_1d_left(psi)
        elif type(psi) == uMPS_1d:
            return self.exp_1d(psi)

class oneBodyH(localH_term):
    def __init__(self,H):
        super().__init__(H)
        self.tensor = self.matrix.view()
        self.tensor.shape = np.array([2,2])
    def exp_1d(self,psi):
        return np.real(ncon([psi.mps,self.tensor,psi.mps.conj(),psi.L.tensor,psi.R.tensor],((1,3,4),(2,1),(2,5,6),(5,3),(6,4))),order=(3,5,1,2,4,6))
    def exp_1d_left(self,psi):
        return np.real(ncon([psi.mps,self.tensor,psi.mps.conj(),psi.R.tensor],((1,3,4),(2,1),(2,3,5),(5,4)),order=(3,1,2,4,5)))
    def exp_2d_left(self,psi):
        return np.real(ncon([psi.mps,psi.mpo,self.tensor,psi.mpo.conj(),psi.mps.conj(),psi.R.tensor,psi.T.tensor],((1,8,9),(2,1,5,6),(3,2),(3,4,5,7),(4,8,10),(7,6),(10,9)),order=(8,9,10,5,2,3,7,6,1,4))) 

class twoBodyH(localH_term):
    def __init__(self,H):
        super().__init__(H)
        self.tensor = self.matrix.view()
        self.tensor.shape = np.array([2,2,2,2])

    #1d
    def exp_1d(self,psi):
        return np.real(ncon([psi.mps,psi.mps,self.tensor,psi.mps.conj(),psi.mps.conj(),psi.L.tensor,psi.R.tensor],((1,5,6),(3,6,7),(2,4,1,3),(2,10,9),(4,9,8),(10,5),(8,7))))
    def exp_1d_left(self,psi):
        return np.real(ncon([psi.mps,psi.mps,self.tensor,psi.mps.conj(),psi.mps.conj(),psi.R.tensor],((1,5,6),(3,6,7),(2,4,1,3),(2,5,9),(4,9,8),(8,7))))


class twoBodyH_hori(twoBodyH):
    def exp_2d_left(self,psi):
        centreContract = ncon([psi.mpo,psi.mpo,self.tensor,psi.mpo.conj(),psi.mpo.conj(),psi.R.tensor],((1,-10,5,6),(2,-11,6,7),(3,4,1,2),(3,-12,5,9),(4,-13,9,8),(8,7)),forder=(-12,-13,-10,-11),order=(5,6,9,7,8,1,2,3,4))
        outerContract = ncon([psi.mps,psi.mps.conj(),psi.T.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1))
        E = np.real(ncon([outerContract,centreContract,outerContract],((2,1),(2,3,1,4),(3,4))))
        return E

    #2d left biparite
    def exp_2d_left_bipartite_ind(self,mps1,mps2,mpo1,mpo2,T1,T2,R1):    
        centreContract = ncon([mpo1,mpo2,self.tensor,mpo1.conj(),mpo2.conj(),R1.tensor],((1,-10,5,6),(2,-11,6,7),(3,4,1,2),(3,-12,5,9),(4,-13,9,8),(8,7)),forder=(-12,-13,-10,-11),order=(5,6,9,7,8,1,2,3,4))
        outerContract1 = ncon([mps1,mps1.conj(),T2.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1))
        outerContract2 = ncon([mps2,mps2.conj(),T1.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1))
        E = np.real(ncon([outerContract1,centreContract,outerContract2],((2,1),(2,3,1,4),(3,4))))
        return E

    def exp_2d_left_bipartite(self,psi):
        E = self.exp_2d_left_bipartite_ind(psi.mps1,psi.mps2,psi.mpo1,psi.mpo2,psi.T1,psi.T2,psi.R1)
        E += self.exp_2d_left_bipartite_ind(psi.mps2,psi.mps1,psi.mpo2,psi.mpo1,psi.T2,psi.T1,psi.R2)
        return E/2

class twoBodyH_vert(twoBodyH):
    def exp_2d_left(self,psi):
        outerContract = ncon([psi.mps,psi.mps,psi.mps.conj(),psi.mps.conj(),psi.T.tensor],((-1,5,6),(-2,6,7),(-3,5,9),(-4,9,8),(8,7)),forder=(-3,-4,-1,-2))
        innerContract = ncon([psi.mpo,psi.mpo,self.tensor,psi.mpo.conj(),psi.mpo.conj(),psi.RR.tensor],((1,-5,9,11),(2,-7,10,14),(3,4,1,2),(3,-6,9,12),(4,-8,10,13),(12,11,13,14)),forder=(-6,-8,-5,-7),order=(12,11,13,14,9,10,1,2,3,4))
        E = np.real(ncon([innerContract,outerContract],((1,2,3,4),(1,2,3,4))))
        return E

    def exp_2d_left_bipartite_ind(self,mps1,mps2,mpo1,mpo2,T1,RR2):
        outerContract = ncon([mps1,mps2,mps1.conj(),mps2.conj(),T1.tensor],((-1,5,6),(-2,6,7),(-3,5,9),(-4,9,8),(8,7)),forder=(-3,-4,-1,-2))
        innerContract = ncon([mpo1,mpo2,self.tensor,mpo1.conj(),mpo2.conj(),RR2.tensor],((1,-5,9,11),(2,-7,10,14),(3,4,1,2),(3,-6,9,12),(4,-8,10,13),(12,11,13,14)),forder=(-6,-8,-5,-7),order=(12,11,13,14,9,10,1,2,3,4))
        E = np.real(ncon([innerContract,outerContract],((1,2,3,4),(1,2,3,4))))
        return E

    def exp_2d_left_bipartite(self,psi):
        E =  self.exp_2d_left_bipartite_ind(psi.mps1,psi.mps2,psi.mpo1,psi.mpo2,psi.T1,psi.RR2)
        E +=  self.exp_2d_left_bipartite_ind(psi.mps2,psi.mps1,psi.mpo2,psi.mpo1,psi.T2,psi.RR1)
        return E/2

class plaquetteH(localH_term):
    def __init__(self,H):
        super().__init__(H)
        self.tensor = self.matrix.view()
        self.tensor.shape = np.array([2,2,2,2,2,2,2,2])
