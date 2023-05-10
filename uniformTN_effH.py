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

def vumps_1d_Hc(twoBodyH,psi,Hl,Hr):
    H = ncon([psi.Al,psi.Ar,twoBodyH,psi.Al.conj(),psi.Ar.conj()],((1,10,-5),(2,-6,7),(3,4,1,2),(3,10,-9),(4,-8,7)),forder=(-9,-8,-5,-6))
    H = H.astype(complex)
    H += ncon([Hl,np.eye(psi.D)],((-1,-2),(-4,-3)),forder=(-1,-4,-2,-3))
    H += ncon([np.eye(psi.D),Hr],((-1,-2),(-4,-3)),forder=(-1,-4,-2,-3))
    return H.reshape(psi.D**2,psi.D**2)
def vumps_1d_Hac(twoBodyH,psi,Hl,Hr):
    H = ncon([psi.Al,twoBodyH,psi.Al.conj(),np.eye(psi.D)],((1,6,-5),(3,-4,1,-2),(3,6,-7),(-8,-9)),forder=(-4,-7,-8,-2,-5,-9))
    H = H.astype(complex)
    H += ncon([psi.Ar,twoBodyH,psi.Ar.conj(),np.eye(psi.D)],((2,-5,6),(-3,4,-1,2),(4,-7,6),(-8,-9)),forder=(-3,-8,-7,-1,-9,-5))
    H += ncon([Hl,np.eye(2),np.eye(psi.D)],((-2,-1),(-4,-3),(-6,-5)),forder=(-4,-2,-6,-3,-1,-5))
    H += ncon([np.eye(psi.D),np.eye(2),Hr],((-2,-1),(-4,-3),(-6,-5)),forder=(-4,-2,-6,-3,-1,-5))
    return H.reshape(2*psi.D**2,2*psi.D**2)
def vumps_1d_leftEnv(twoBodyH,psi,Tl_inv):
    Hl = ncon([psi.Al,psi.Al,twoBodyH,psi.Al.conj(),psi.Al.conj()],((1,5,6),(2,6,-7),(3,4,1,2),(3,5,9),(4,9,-8)),forder=(-8,-7))
    Hl = Tl_inv.applyRight(Hl.reshape(psi.D**2)).reshape(psi.D,psi.D)
    return Hl
def vumps_1d_rightEnv(twoBodyH,psi,Tr_inv):
    Hr = ncon([psi.Ar,psi.Ar,twoBodyH,psi.Ar.conj(),psi.Ar.conj()],((1,-5,6),(2,6,7),(3,4,1,2),(3,-9,8),(4,8,7)),forder=(-9,-5))
    Hr = Tr_inv.applyLeft(Hr.reshape(psi.D**2)).reshape(psi.D,psi.D)
    return Hr
