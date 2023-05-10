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

def exp_1d_2body_left(twoBodyH,psi,R):
    return np.real(ncon([psi.mps,psi.mps,twoBodyH,psi.mps.conj(),psi.mps.conj(),R.tensor],((1,5,6),(3,6,7),(2,4,1,3),(2,5,9),(4,9,8),(8,7))))

def exp_1d_2body(twoBodyH,psi,L,R):
    return np.real(ncon([psi.mps,psi.mps,twoBodyH,psi.mps.conj(),psi.mps.conj(),L.tensor,R.tensor],((1,5,6),(3,6,7),(2,4,1,3),(2,10,9),(4,9,8),(10,5),(8,7))))

def exp_2d_1body_left(oneBodyH,psi,T,R):
    E = ncon([psi.mps,psi.mpo,oneBodyH,psi.mpo.conj(),psi.mps.conj(),R.tensor,T.tensor],((1,8,9),(2,1,5,6),(3,2),(3,4,5,7),(4,8,10),(7,6),(10,9)),order=(8,9,10,5,2,3,7,6,1,4))
    return np.real(E)

def exp_2d_2body_horizontal_left(twoBodyH,psi,T,R):    
    centreContract = ncon([psi.mpo,psi.mpo,twoBodyH,psi.mpo.conj(),psi.mpo.conj(),R.tensor],((1,-10,5,6),(2,-11,6,7),(3,4,1,2),(3,-12,5,9),(4,-13,9,8),(8,7)),forder=(-12,-13,-10,-11),order=(5,6,9,7,8,1,2,3,4))
    outerContract = ncon([psi.mps,psi.mps.conj(),T.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1))
    E = np.real(ncon([outerContract,centreContract,outerContract],((2,1),(2,3,1,4),(3,4))))
    return E
def exp_2d_2body_vertical_left(twoBodyH,psi,T,RR):
    outerContract = ncon([psi.mps,psi.mps,psi.mps.conj(),psi.mps.conj(),T.tensor],((-1,5,6),(-2,6,7),(-3,5,9),(-4,9,8),(8,7)),forder=(-3,-4,-1,-2))
    innerContract = ncon([psi.mpo,psi.mpo,twoBodyH,psi.mpo.conj(),psi.mpo.conj(),RR.tensor],((1,-5,9,11),(2,-7,10,14),(3,4,1,2),(3,-6,9,12),(4,-8,10,13),(12,11,13,14)),forder=(-6,-8,-5,-7),order=(12,11,13,14,9,10,1,2,3,4))
    E = np.real(ncon([innerContract,outerContract],((1,2,3,4),(1,2,3,4))))
    return E

def exp_2d_2body_horizontal_leftBip_ind(twoBodyH,mps1,mps2,mpo1,mpo2,T1,T2,R1):    
    centreContract = ncon([mpo1,mpo2,twoBodyH,mpo1.conj(),mpo2.conj(),R1.tensor],((1,-10,5,6),(2,-11,6,7),(3,4,1,2),(3,-12,5,9),(4,-13,9,8),(8,7)),forder=(-12,-13,-10,-11),order=(5,6,9,7,8,1,2,3,4))
    outerContract1 = ncon([mps1,mps1.conj(),T2.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1))
    outerContract2 = ncon([mps2,mps2.conj(),T1.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1))
    E = np.real(ncon([outerContract1,centreContract,outerContract2],((2,1),(2,3,1,4),(3,4))))
    return E
def exp_2d_2body_vertical_leftBip_ind(twoBodyH,mps1,mps2,mpo1,mpo2,T1,RR2):
    outerContract = ncon([mps1,mps2,mps1.conj(),mps2.conj(),T1.tensor],((-1,5,6),(-2,6,7),(-3,5,9),(-4,9,8),(8,7)),forder=(-3,-4,-1,-2))
    innerContract = ncon([mpo1,mpo2,twoBodyH,mpo1.conj(),mpo2.conj(),RR2.tensor],((1,-5,9,11),(2,-7,10,14),(3,4,1,2),(3,-6,9,12),(4,-8,10,13),(12,11,13,14)),forder=(-6,-8,-5,-7),order=(12,11,13,14,9,10,1,2,3,4))
    E = np.real(ncon([innerContract,outerContract],((1,2,3,4),(1,2,3,4))))
    return E

def exp_2d_2body_horizontal_leftBip(twoBodyH,psi,T1,T2,R1,R2):
    E = exp_2d_2body_horizontal_leftBip_ind(twoBodyH,psi.mps1,psi.mps2,psi.mpo1,psi.mpo2,T1,T2,R1)
    E += exp_2d_2body_horizontal_leftBip_ind(twoBodyH,psi.mps2,psi.mps1,psi.mpo2,psi.mpo1,T2,T1,R2)
    return E/2

def exp_2d_2body_vertical_leftBip(twoBodyH,psi,T1,T2,RR1,RR2):
    E =  exp_2d_2body_vertical_leftBip_ind(twoBodyH,psi.mps1,psi.mps2,psi.mpo1,psi.mpo2,T1,RR2)
    E +=  exp_2d_2body_vertical_leftBip_ind(twoBodyH,psi.mps2,psi.mps1,psi.mpo2,psi.mpo1,T2,RR1)
    return E/2
