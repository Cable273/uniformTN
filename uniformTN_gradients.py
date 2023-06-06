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

def grad_mps_1d(twoBodyH,A,L,R,T_inv):
    D = np.size(A,axis=1)
    #subtract off current exp - ensures gradient is zero when converged
    exp = np.real(ncon([A,A,twoBodyH,A.conj(),A.conj(),L.tensor,R.tensor],((1,5,6),(3,6,7),(2,4,1,3),(2,10,9),(4,9,8),(10,5),(8,7))))
    twoBodyH = (twoBodyH.reshape(4,4)-np.eye(4)*exp).reshape(2,2,2,2)

    grad = ncon([A,A,twoBodyH,A.conj(),L.tensor,R.tensor],((1,5,6),(3,6,7),(-2,4,1,3),(4,-9,8),(-10,5),(8,7)),forder=(-2,-10,-9),order=(8,7,3,4,6,5))
    grad += ncon([A,A,twoBodyH,A.conj(),L.tensor,R.tensor],((1,5,6),(3,6,7),(2,-4,1,3),(2,10,-9),(10,5),(-8,7)),forder=(-4,-9,-8),order=(10,5,1,2,6,7))

    leftEnv = ncon([A,A,twoBodyH,A.conj(),A.conj(),L.tensor],((1,5,6),(3,6,-7),(2,4,1,3),(2,10,9),(4,9,-8),(10,5)),forder=(-8,-7),order=(5,1,2,6,9,3,4))
    leftEnv = T_inv.applyRight(leftEnv.reshape(D**2)).reshape(D,D)
    grad += ncon([leftEnv,A,R.tensor],((-4,2),(-1,2,3),(-5,3)),forder=(-1,-4,-5))

    rightEnv = ncon([A,A,twoBodyH,A.conj(),A.conj(),R.tensor],((1,-5,6),(3,6,7),(2,4,1,3),(2,-10,9),(4,9,8),(8,7)),forder=(-10,-5))
    rightEnv = T_inv.applyLeft(rightEnv.reshape(D**2)).reshape(D,D)
    grad += ncon([L.tensor,A,rightEnv],((-4,2),(-1,2,3),(-5,3)),forder=(-1,-4,-5))
    # print(np.einsum('ijk,ijk',grad,grad.conj()))
    return grad

def grad_mps_1d_left(twoBodyH,A,R,T_inv):
    D = np.size(A,axis=1)
    #subtract off current exp - ensures gradient is zero when converged
    exp = np.real(ncon([A,A,twoBodyH,A.conj(),A.conj(),R.tensor],((1,5,6),(3,6,7),(2,4,1,3),(2,5,9),(4,9,8),(8,7))))
    twoBodyH = (twoBodyH.reshape(4,4)-np.eye(4)*exp).reshape(2,2,2,2)

    grad = ncon([A,A,twoBodyH,A.conj(),R.tensor],((1,-5,6),(3,6,7),(-2,4,1,3),(4,-9,8),(8,7)),forder=(-2,-5,-9),order=(8,7,3,4,6))
    grad += ncon([A,A,twoBodyH,A.conj(),R.tensor],((1,5,6),(3,6,7),(2,-4,1,3),(2,5,-9),(-8,7)),forder=(-4,-9,-8),order=(5,1,2,6,7))

    leftEnv = ncon([A,A,twoBodyH,A.conj(),A.conj()],((1,5,6),(3,6,-7),(2,4,1,3),(2,5,9),(4,9,-8)),forder=(-8,-7),order=(5,1,2,6,9,3,4))
    leftEnv = T_inv.applyRight(leftEnv.reshape(D**2)).reshape(D,D)
    grad += ncon([leftEnv,A,R.tensor],((-4,2),(-1,2,3),(-5,3)),forder=(-1,-4,-5))

    rightEnv = ncon([A,A,twoBodyH,A.conj(),A.conj(),R.tensor],((1,-5,6),(3,6,7),(2,4,1,3),(2,-10,9),(4,9,8),(8,7)),forder=(-10,-5))
    rightEnv = T_inv.applyLeft(rightEnv.reshape(D**2)).reshape(D,D)
    grad += ncon([A,rightEnv],((-1,-2,3),(-4,3)),forder=(-1,-2,-4))
    return grad

def grad_mps_horizontal(twoBodyH,psi,T,R,Ta_inv,Tw_inv,TDVP=False):
    grad = np.zeros(np.shape(psi.mps),dtype=complex)
    #terms where all contractions in W direction being complete
    centreContract = ncon([psi.mpo,psi.mpo,twoBodyH,psi.mpo.conj(),psi.mpo.conj(),R.tensor],((1,-10,5,6),(2,-11,6,7),(3,4,1,2),(3,-12,5,9),(4,-13,9,8),(8,7)),forder=(-12,-13,-10,-11),order=(5,6,9,7,8,1,2,3,4))
    #ring around horizontal interaction
    outerContract = ncon([psi.mps,psi.mps.conj(),T.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))

    #2 terms with mps tensor removed under local hamiltonian
    grad += ncon([centreContract,outerContract,psi.mps,T.tensor],((-3,4,1,2),(4,2),(1,-5,6),(-7,6)),forder=(-3,-5,-7))
    grad += ncon([centreContract,outerContract,psi.mps,T.tensor],((3,-4,1,2),(3,1),(2,-5,6),(-7,6)),forder=(-4,-5,-7))

    #4 terms above and below horizontal hamiltonian,
    # below 
    if TDVP is False:
        grad += ncon([centreContract,outerContract,psi.mps,psi.mps.conj(),T.tensor,Ta_inv.tensor,psi.mps],((3,4,1,2),(4,2),(1,8,9),(3,11,10),(10,9),(-12,7,11,8),(-5,-6,7)),forder=(-5,-6,-12),order=(9,10,4,2,1,3,8,11,7))
        grad += ncon([centreContract,outerContract,psi.mps,psi.mps.conj(),T.tensor,Ta_inv.tensor,psi.mps],((3,4,1,2),(3,1),(2,7,5),(4,8,6),(6,5),(-10,9,8,7),(-12,-11,9)),forder=(-12,-11,-10),order=(5,6,3,1,4,2,7,8,9))

    #above
    grad += ncon([centreContract,outerContract,psi.mps,psi.mps.conj(),Ta_inv.tensor,psi.mps,T.tensor],((3,4,1,2),(3,1),(2,6,7),(4,6,12),(12,7,-11,8),(-5,8,9),(-10,9)),forder=(-5,-11,-10),order=(6,4,2,3,1,7,12,8,9))
    grad += ncon([centreContract,outerContract,psi.mps,psi.mps.conj(),Ta_inv.tensor,psi.mps,T.tensor],((3,4,1,2),(4,2),(1,6,7),(3,6,12),(12,7,-11,8),(-5,8,9),(-10,9)),forder=(-5,-11,-10),order=(6,1,3,2,4,7,12,8,9))

    del centreContract
    centreContractLeft = ncon([twoBodyH,psi.mpo,psi.mpo,psi.mpo.conj(),psi.mpo.conj(),outerContract,outerContract],((3,4,1,2),(1,10,5,6),(2,11,6,-7),(3,12,5,9),(4,13,9,-8),(12,10),(13,11)),forder=(-8,-7),order=(5,10,1,3,12,6,9,11,2,4,13))
    expectation = np.real(ncon([centreContractLeft,R.tensor],((1,2),(1,2)))) #for left half terms
    centreContractLeft = Tw_inv.applyRight(centreContractLeft.reshape(psi.D_mpo**2)).reshape(psi.D_mpo,psi.D_mpo)
    centreContractLeft = ncon([centreContractLeft,psi.mpo,psi.mpo.conj(),R.tensor],((5,4),(2,-1,4,6),(2,-3,5,7),(7,6)),forder=(-3,-1),order=(5,4,2,7,6))
    #3 terms from right half (upper right quadrant, lower right quadrant, one in line with H)
    #right
    grad += ncon([centreContractLeft,psi.mps,T.tensor],((-2,1),(1,-3,4),(-5,4)),forder=(-2,-3,-5),order=(4,1))
    # upper right
    grad += ncon([centreContractLeft,psi.mps,psi.mps.conj(),Ta_inv.tensor,psi.mps,T.tensor],((2,1),(1,4,5),(2,4,10),(10,5,-9,6),(-3,6,7),(-8,7)),forder=(-3,-9,-8),order=(4,1,2,5,10,6,7))
    #lower right
    if TDVP is False:
        grad += ncon([centreContractLeft,psi.mps,psi.mps.conj(),T.tensor,Ta_inv.tensor,psi.mps],((3,2),(2,6,7),(3,9,8),(8,7),(-10,5,9,6),(-1,-4,5)),forder=(-1,-4,-10),order=(7,2,3,8,6,9,5))
    return grad

def grad_mps_vertical(twoBodyH,psi,T,RR,Ta_inv,Tw2_inv,TDVP=False):
    grad = np.zeros(np.shape(psi.mps),dtype=complex)

    centreContract = ncon([psi.mpo,psi.mpo,twoBodyH,psi.mpo.conj(),psi.mpo.conj(),RR.tensor],((1,-11,5,6),(2,-12,8,9),(3,4,1,2),(3,-13,5,7),(4,-14,8,10),(7,6,10,9)),forder=(-13,-14,-11,-12),order=(6,7,10,9,5,8,1,2,3,4))

    #2 terms with mps tensor removed under local hamiltonian
    grad += ncon([psi.mps,psi.mps,centreContract,psi.mps.conj(),T.tensor],((1,-5,6),(2,6,7),(-3,4,1,2),(4,-9,8),(8,7)),forder=(-3,-5,-9),order=(8,7,2,4,6,1))
    grad += ncon([psi.mps,psi.mps,centreContract,psi.mps.conj(),T.tensor],((1,5,6),(2,6,7),(3,-4,1,2),(3,5,-9),(-8,7)),forder=(-4,-9,-8),order=(5,1,3,5,2,7))

    #2 terms with mps removed vertical to local hamiltonian
    #above
    grad += ncon([psi.mps,psi.mps,centreContract,psi.mps.conj(),psi.mps.conj(),Ta_inv.tensor,psi.mps,T.tensor],((1,6,7),(2,7,8),(3,4,1,2),(3,6,14),(4,14,13),(13,8,-12,9),(-5,9,10),(-11,10)),forder=(-5,-12,-11),order=(6,1,3,7,14,2,4,8,13,9,10))
    #below
    if TDVP is False:
        grad += ncon([psi.mps,psi.mps,centreContract,psi.mps.conj(),psi.mps.conj(),T.tensor,Ta_inv.tensor,psi.mps],((1,8,9),(2,9,10),(3,4,1,2),(3,13,12),(4,12,11),(11,10),(-14,7,13,8),(-5,-6,7)),forder=(-5,-6,-14),order=(10,11,2,4,9,12,1,3,8,13,7))

    del centreContract

    #left vector
    centreContractLeft = ncon([twoBodyH,psi.mps,psi.mps,psi.mpo,psi.mpo,psi.mpo.conj(),psi.mpo.conj(),psi.mps.conj(),psi.mps.conj(),T.tensor],((3,7,2,6),(1,9,10),(5,10,11),(2,1,14,-15),(6,5,17,-18),(3,4,14,-16),(7,8,17,-19),(4,9,13),(8,13,12),(12,11)),forder=(-16,-15,-19,-18),order=(9,10,13,11,12,1,14,2,3,4,5,17,6,7,8))
    expectation = np.real(ncon([centreContractLeft,RR.tensor],((1,2,3,4),(1,2,3,4))))
    centreContractLeft = Tw2_inv.applyRight(centreContractLeft.reshape(psi.D_mpo**4)).reshape(psi.D_mpo,psi.D_mpo,psi.D_mpo,psi.D_mpo)
    # #reshape and do a dot product might be better for memory (big D)
    centreContractLeft = ncon([psi.mpo,psi.mpo,psi.mpo.conj(),psi.mpo.conj(),centreContractLeft,RR.tensor],((2,-1,7,9),(5,-4,11,13),(2,-3,8,10),(5,-6,12,14),(8,7,12,11),(10,9,14,13)),forder=(-3,-6,-1,-4),order=(13,14,5,9,10,2,8,7,12,11))
    #right
    grad += ncon([centreContractLeft,psi.mps,psi.mps,psi.mps.conj(),T.tensor],((-3,4,1,2),(1,-5,6),(2,6,7),(4,-9,8),(8,7)),forder=(-3,-5,-9),order=(8,7,6,1,2,4))
    grad += ncon([psi.mps,psi.mps,centreContractLeft,psi.mps.conj(),T.tensor],((1,5,6),(2,6,7),(3,-4,1,2),(3,5,-9),(-8,7)),forder=(-4,-9,-8),order=(7,6,5,1,2,3))
    #right up
    grad += ncon([psi.mps,psi.mps,centreContractLeft,psi.mps.conj(),psi.mps.conj(),Ta_inv.tensor,psi.mps,T.tensor],((1,6,7),(2,7,8),(3,4,1,2),(3,6,14),(4,14,13),(13,8,-12,9),(-5,9,10),(-11,10)),forder=(-5,-12,-11),order=(6,7,14,3,4,1,2,13,8,9,10))
    # right down
    if TDVP is False:
        grad += ncon([psi.mps,psi.mps,centreContractLeft,psi.mps.conj(),psi.mps.conj(),T.tensor,Ta_inv.tensor,psi.mps],((1,8,9),(2,9,10),(3,4,1,2),(3,13,12),(4,12,11),(11,10),(-14,7,13,8),(-5,-6,7)),forder=(-5,-6,-14),order=(10,11,9,12,1,2,3,4,8,13,7))

    return grad

def grad_mpu_horizontal(twoBodyH,psi,T,R,RR,Ta,Tw_inv,envTol=1e-5,TDVP=False):
    grad = np.zeros(np.shape(psi.mpo),dtype=complex)

    #ring around horizontal interaction
    outerContract = ncon([psi.mps,psi.mps.conj(),T.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
    #two terms with mpo removed under local hamiltonian
    grad += ncon([psi.mpo,psi.mpo,twoBodyH,psi.mpo.conj(),R.tensor,outerContract,outerContract],((2,1,-9,10),(6,5,10,11),(-3,7,2,6),(7,8,-13,12),(12,11),(-4,1),(8,5)),forder=(-4,-3,-9,-13),order=(12,11,5,8,6,7,10,1))
    grad += ncon([psi.mpo,psi.mpo,twoBodyH,psi.mpo.conj(),R.tensor,outerContract,outerContract],((2,1,9,10),(6,5,10,11),(3,-7,2,6),(3,4,9,-13),(-12,11),(4,1),(-8,5)),forder=(-8,-7,-13,-12),order=(9,1,2,3,4,10,5,6,11))

    #two terms with mpo removed in line with hamiltonian (left and right)
    centreContractLeft = ncon([psi.mpo,psi.mpo,twoBodyH,psi.mpo.conj(),psi.mpo.conj(),outerContract,outerContract],((2,1,9,10),(6,5,10,-11),(3,7,2,6),(3,4,9,13),(7,8,13,-12),(4,1),(8,5)),forder=(-12,-11),order=(9,1,2,3,4,10,13,6,7,5,8))
    exp = np.einsum('ij,ij',centreContractLeft,R.tensor)
    centreContractLeft = Tw_inv.applyRight(centreContractLeft.reshape(psi.D_mpo**2)).reshape(psi.D_mpo,psi.D_mpo)
    grad += ncon([centreContractLeft,psi.mpo,R.tensor,outerContract],((-6,4),(-2,1,4,5),(-7,5),(-3,1)),forder=(-3,-2,-6,-7),order=(4,5,1))

    if TDVP is False:
        centreContractRight = ncon([psi.mpo,psi.mpo,twoBodyH,psi.mpo.conj(),psi.mpo.conj(),R.tensor,outerContract,outerContract],((2,1,-9,10),(6,5,10,11),(3,7,2,6),(3,4,-14,13),(7,8,13,12),(12,11),(4,1),(8,5)),forder=(-14,-9),order=(11,12,5,6,7,8,10,13,2,3,1,4))
        centreContractRight = Tw_inv.applyLeft(centreContractRight.reshape(psi.D_mpo**2)).reshape(psi.D_mpo,psi.D_mpo)
        grad += ncon([psi.mpo,centreContractRight,outerContract],((-2,1,-4,5),(-6,5),(-3,1)),forder=(-3,-2,-4,-6),order=(5,1))
        del centreContractLeft

    #RRd geometric sums...
    h_tilde = (twoBodyH.reshape(4,4)-np.eye(4)*exp).reshape(2,2,2,2)
    innerContract = ncon([psi.mpo,psi.mpo,h_tilde,psi.mpo.conj(),psi.mpo.conj()],((2,-1,9,10),(6,-5,10,-11),(3,7,2,6),(3,-4,9,13),(7,-8,13,-12)),forder=(-4,-8,-1,-5,-12,-11),order=(9,2,3,10,13,6,7))
    for d in range(0,100):
        gradRun = np.zeros((2,2,psi.D_mpo,psi.D_mpo)).astype(complex)
        if d == 0:
            Td = np.eye(psi.D_mps**2)
            Td_tensor = Td.reshape(psi.D_mps,psi.D_mps,psi.D_mps,psi.D_mps)
            RR_d = RR
        else:
            Td = np.dot(Td,Ta.matrix)
            Td_tensor = Td.reshape(psi.D_mps,psi.D_mps,psi.D_mps,psi.D_mps)
            TT_d = mpsu1Transfer_left_twoLayerWithMpsInsert(psi.mps,psi.mpo,T,Td.reshape(psi.D_mps,psi.D_mps,psi.D_mps,psi.D_mps))
            RR_d = TT_d.findRightEig()
            RR_d.norm_pairedCanon()
            del TT_d

        #4 terms with mpo removed vertical to hamiltonian
        #new double ring around horizontal interaction
        outerContractDouble = ncon([psi.mps,psi.mps,psi.mps.conj(),psi.mps.conj(),Td_tensor,T.tensor],((-1,5,6),(-2,7,8),(-3,5,11),(-4,10,9),(11,6,10,7),(9,8)),forder=(-3,-4,-1,-2),order=(5,6,11,7,10,8,9))

        gradRun += ncon([innerContract,psi.mpo,psi.mpo,psi.mpo.conj(),RR_d.tensor,outerContractDouble,outerContractDouble],((1,2,3,4,5,6),(11,10,15,13),(-8,7,-17,15),(11,12,-16,14),(14,13,5,6),(-9,1,7,3),(12,2,10,4)),forder=(-9,-8,-17,-16),order=(5,6,13,14,15,4,2,10,12,3,1,7))
        gradRun += ncon([innerContract,psi.mpo,RR_d.tensor,outerContract,outerContractDouble],((1,2,3,4,5,6),(-8,7,-12,10),(-11,10,5,6),(1,3),(-9,2,7,4)),forder=(-9,-8,-12,-11),order=(5,6,10,1,3,2,4,7))
        gradRun += ncon([innerContract,psi.mpo,RR_d.tensor,outerContract,outerContractDouble],((1,2,3,4,5,6),(-8,7,-10,11),(5,6,-12,11),(1,3),(2,-9,4,7)),forder=(-9,-8,-10,-12),order=(5,6,11,1,3,2,4,7))
        gradRun += ncon([innerContract,psi.mpo,psi.mpo,psi.mpo.conj(),RR_d.tensor,outerContractDouble,outerContractDouble],((1,2,3,4,5,6),(11,10,14,15),(-8,7,-13,14),(11,12,-17,16),(5,6,16,15),(1,-9,3,7),(2,12,4,10)),forder=(-9,-8,-13,-17),order=(5,6,15,16,14,2,4,12,10,1,3,7))

        # #4 terms from 4 quadrants off diagonal
        centreContractLeft = ncon([innerContract,outerContract,outerContract],((1,2,3,4,-5,-6),(1,3),(2,4)),forder=(-5,-6))
        centreContractLeft = Tw_inv.applyRight(centreContractLeft.reshape(psi.D_mpo**2)).reshape(psi.D_mpo,psi.D_mpo)
        gradRun += ncon([psi.mpo,psi.mpo.conj(),psi.mpo,centreContractLeft,RR_d.tensor,outerContractDouble],((2,1,7,8),(2,3,9,10),(-5,4,-11,12),(9,7),(-13,12,10,8),(-6,3,4,1)),forder=(-6,-5,-11,-13),order=(8,10,2,7,9,12,1,3,4))
        gradRun += ncon([psi.mpo,psi.mpo,psi.mpo.conj(),centreContractLeft,RR_d.tensor,outerContractDouble],((-2,1,-7,8),(5,4,10,11),(5,6,12,13),(12,10),(13,11,-9,8),(6,-3,4,1)),forder=(-3,-2,-7,-9),order=(11,13,5,10,12,8,4,6,1))

        if TDVP is False:
            rightEnv = ncon([innerContract,RR_d.tensor,psi.mpo,psi.mpo,psi.mpo.conj(),psi.mpo.conj(),outerContractDouble,outerContractDouble],((1,2,3,4,5,6),(18,15,5,6),(11,10,14,15),(8,7,-13,14),(11,12,17,18),(8,9,-16,17),(9,1,7,3),(12,2,10,4)),forder=(-16,-13),order=(5,6,15,18,11,4,2,10,12,14,17,8,1,3,7,9))
            rightEnv += ncon([innerContract,RR_d.tensor,psi.mpo,psi.mpo,psi.mpo.conj(),psi.mpo.conj(),outerContractDouble,outerContractDouble],((1,2,3,4,5,6),(5,6,18,15),(11,10,14,15),(8,7,-13,14),(11,12,17,18),(8,9,-16,17),(1,9,3,7),(2,12,4,10)),forder=(-16,-13),order=(5,6,18,15,11,10,12,4,2,14,17,8,7,9,3,1))
            rightEnv = Tw_inv.applyLeft(rightEnv.reshape(psi.D_mpo**2)).reshape(psi.D_mpo,psi.D_mpo)
            gradRun += ncon([psi.mpo,rightEnv,outerContract],((-2,1,-4,5),(-6,5),(-3,1)),forder=(-3,-2,-4,-6),order=(5,1))

        mag = np.einsum('ijab,ijab',gradRun,gradRun.conj())
        grad += gradRun
        if np.abs(mag)<envTol:
            break
    print("(horizontal) d: ",d,mag)
    grad = np.einsum('ijab->jiab',grad)
    return grad

def grad_mpu_vertical(twoBodyH,psi,T,R,RR,Ta,Tw_inv,Tw2,Tw2_inv,envTol=1e-5,TDVP=False):
    grad = np.zeros(np.shape(psi.mpo),dtype=complex)

    #single ring in mps direction
    outerContractSingle = ncon([psi.mps,psi.mps.conj(),T.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
    #double ring in mps direction
    outerContract = ncon([psi.mps,psi.mps,psi.mps.conj(),psi.mps.conj(),T.tensor],((-1,5,6),(-2,6,7),(-3,5,9),(-4,9,8),(8,7)),forder=(-3,-4,-1,-2))
    #two terms with mpo removed under hamiltonian
    grad += ncon([psi.mpo,psi.mpo,twoBodyH,psi.mpo.conj(),RR.tensor,outerContract],((2,1,-9,10),(4,3,11,12),(-5,7,2,4),(7,8,11,14),(-13,10,14,12),(-6,8,1,3)),forder=(-6,-5,-9,-13),order=(12,14,11,4,7,3,8,10,2,1))
    grad += ncon([psi.mpo,psi.mpo,twoBodyH,psi.mpo.conj(),RR.tensor,outerContract],((2,1,9,10),(6,5,-11,12),(3,-7,2,6),(3,4,9,13),(13,10,-14,12),(4,-8,1,5)),forder=(-8,-7,-11,-14),order=(10,13,9,2,3,1,4,12,5))

    #4 terms with mpo removed horizontally in line with hamiltonian
    leftEnv = ncon([psi.mpo,psi.mpo,twoBodyH,psi.mpo.conj(),psi.mpo.conj(),outerContract],((2,1,9,-10),(6,5,11,-12),(3,7,2,6),(3,4,9,-13),(7,8,11,-14),(4,8,1,5)),forder=(-13,-10,-14,-12),order=(9,1,2,3,4,11,5,6,7,8))
    exp = np.einsum('abcd,abcd',leftEnv,RR.tensor)
    leftEnv = Tw2_inv.applyRight(leftEnv.reshape(psi.D_mpo**4)).reshape(psi.D_mpo,psi.D_mpo,psi.D_mpo,psi.D_mpo)
    grad += ncon([leftEnv,psi.mpo,psi.mpo,psi.mpo.conj(),RR.tensor,outerContract],((13,11,-9,7),(-2,1,7,8),(5,4,11,12),(5,6,13,14),(14,12,-10,8),(6,-3,4,1)),forder=(-3,-2,-9,-10),order=(14,12,5,4,6,8,1,13,11,7))
    grad += ncon([leftEnv,psi.mpo,psi.mpo.conj(),psi.mpo,RR.tensor,outerContract],((-10,9,8,7),(2,1,7,11),(2,3,8,12),(-5,4,9,13),(-14,13,12,11),(-6,3,4,1)),forder=(-6,-5,-10,-14),order=(11,12,2,1,3,7,8,9,13,4))

    if TDVP is False:
        rightEnv = ncon([psi.mpo,psi.mpo,twoBodyH,psi.mpo.conj(),psi.mpo.conj(),RR.tensor,outerContract],((2,1,-9,10),(6,5,-11,12),(3,7,2,6),(3,4,-13,14),(7,8,-15,16),(14,10,16,12),(4,8,1,5)),forder=(-13,-9,-15,-11),order=(12,16,5,6,7,8,10,14,1,2,3,4))
        #can OPTIMIZE with only applying Tw_inv
        rightEnv = Tw2_inv.applyLeft(rightEnv.reshape(psi.D_mpo**4)).reshape(psi.D_mpo,psi.D_mpo,psi.D_mpo,psi.D_mpo)
        rightEnv1 = np.einsum('abcc->ab',rightEnv)
        rightEnv2 = np.einsum('aabc->bc',rightEnv)
        grad += ncon([psi.mpo,rightEnv1,outerContractSingle],((-2,1,-4,5),(-6,5),(-3,1)),forder=(-3,-2,-4,-6),order=(5,1))
        grad += ncon([psi.mpo,rightEnv2,outerContractSingle],((-2,1,-4,5),(-6,5),(-3,1)),forder=(-3,-2,-4,-6),order=(5,1))

    #RRd geometric sums...
    #this is where big transfer matrices are necessary,D^12 !!
    #at most will have 2*D^12 arrays in memory at a time
    h_tilde = (twoBodyH.reshape(4,4)-np.eye(4)*exp).reshape(2,2,2,2)
    #redo 
    leftEnv = ncon([psi.mpo,psi.mpo,h_tilde,psi.mpo.conj(),psi.mpo.conj(),outerContract],((2,1,9,-10),(6,5,11,-12),(3,7,2,6),(3,4,9,-13),(7,8,11,-14),(4,8,1,5)),forder=(-13,-10,-14,-12),order=(9,1,2,3,4,11,5,6,7,8))
    leftEnv = Tw2_inv.applyRight(leftEnv.reshape(psi.D_mpo**4)).reshape(psi.D_mpo,psi.D_mpo,psi.D_mpo,psi.D_mpo)
    for d in range(0,100):
        gradRun = np.zeros((2,2,psi.D_mpo,psi.D_mpo)).astype(complex)
        if d == 0:
            Td = np.eye(psi.D_mps**2)
            Td_tensor = Td.reshape(psi.D_mps,psi.D_mps,psi.D_mps,psi.D_mps)
        else:
            Td = np.dot(Td,Ta.matrix)
            Td_tensor = Td.reshape(psi.D_mps,psi.D_mps,psi.D_mps,psi.D_mps)

        Twd_lower = mpsu1Transfer_left_threeLayerWithMpsInsert_lower(psi.mps,psi.mpo,T,Td_tensor)
        R_Twd_lower = Twd_lower.findRightEig()
        R_Twd_lower.norm_pairedCanon()
        del Twd_lower
        outercontractTriple_lower = ncon([psi.mps,psi.mps,psi.mps,psi.mps.conj(),psi.mps.conj(),psi.mps.conj(),Td_tensor,T.tensor],((-1,7,8),(-3,9,10),(-5,10,11),(-2,7,15),(-4,14,13),(-6,13,12),(15,8,14,9),(12,11)),forder=(-2,-4,-6,-1,-3,-5),order=(7,8,15,9,14,10,13,11,12))

        rightEnv = ncon([psi.mpo,psi.mpo,h_tilde,psi.mpo.conj(),psi.mpo.conj(),outercontractTriple_lower],((2,1,9,-10),(6,5,11,-12),(3,7,2,6),(3,4,9,-13),(7,8,11,-14),(-15,4,8,-16,1,5)),forder=(-15,-16,-13,-10,-14,-12),order=(9,2,3,11,7,6,1,4,5,8))
        #reshape and do dot product would be faster!!!! less RAM
        rightEnv = ncon([rightEnv,R_Twd_lower.tensor,psi.mpo],((-1,2,3,4,5,6),(-10,9,3,4,5,6),(-7,2,-8,9)),forder=(-1,-7,-10,-8),order=(3,4,5,6,9,2))
        gradRun += ncon([rightEnv],((-1,-7,-10,-8),),forder=(-1,-7,-8,-10))
        if TDVP is False:
            rightEnv = ncon([rightEnv,psi.mpo.conj()],((1,7,10,-8),(7,1,-2,10)),forder=(-2,-8),order=(10,1,7))
            rightEnv = Tw_inv.applyLeft(rightEnv.reshape(psi.D_mpo**2)).reshape(psi.D_mpo,psi.D_mpo)
            gradRun += ncon([psi.mpo,rightEnv,outerContractSingle],((-2,1,-4,5),(-6,5),(-3,1)),forder=(-3,-2,-4,-6),order=(5,1))

        gradRun += ncon([leftEnv,R_Twd_lower.tensor,psi.mpo,psi.mpo.conj(),psi.mpo,psi.mpo.conj(),psi.mpo,outercontractTriple_lower],((16,14,12,10),(-20,19,17,15,13,11),(2,1,10,11),(2,3,12,13),(5,4,14,15),(5,6,16,17),(-8,7,-18,19),(-9,6,3,7,4,1)),forder=(-9,-8,-18,-20),order=(10,12,2,14,16,5,1,3,4,6,11,13,15,17,19,7))

        #free up memory
        del R_Twd_lower
        del outercontractTriple_lower

        Twd_upper = mpsu1Transfer_left_threeLayerWithMpsInsert_upper(psi.mps,psi.mpo,T,Td_tensor)
        R_Twd_upper = Twd_upper.findRightEig()
        R_Twd_upper.norm_pairedCanon()
        del Twd_upper
        outercontractTriple_upper = ncon([psi.mps,psi.mps,psi.mps,psi.mps.conj(),psi.mps.conj(),psi.mps.conj(),Td_tensor,T.tensor],((-1,7,8),(-3,8,9),(-5,10,11),(-2,7,15),(-4,15,14),(-6,13,12),(14,9,13,10),(12,11)),forder=(-2,-4,-6,-1,-3,-5),order=(7,8,15,9,14,10,13,11,12))

        gradRun += ncon([leftEnv,R_Twd_upper.tensor,psi.mpo,psi.mpo,psi.mpo.conj(),psi.mpo,psi.mpo.conj(),outercontractTriple_upper],((19,17,15,13),(20,18,16,14,-12,11),(-2,1,-10,11),(5,4,13,14),(5,6,15,16),(8,7,17,18),(8,9,19,20),(9,6,-3,7,4,1)),forder=(-3,-2,-10,-12),order=(14,16,5,18,20,8,4,6,7,9,13,15,17,19,11,1))

        rightEnv = ncon([psi.mpo,psi.mpo,h_tilde,psi.mpo.conj(),psi.mpo.conj(),outercontractTriple_upper],((2,1,9,-10),(6,5,11,-12),(3,7,2,6),(3,4,9,-13),(7,8,11,-14),(4,8,-15,1,5,-16)),forder=(-15,-16,-13,-10,-14,-12),order=(9,2,3,11,7,6,1,4,5,8))
        #reshape and do dot product would be faster!!!! less RAM
        rightEnv = ncon([rightEnv,R_Twd_upper.tensor,psi.mpo],((-1,2,3,4,5,6),(3,4,5,6,-10,9),(-7,2,-8,9)),forder=(-1,-7,-10,-8),order=(3,4,5,6,9,2))
        gradRun += ncon([rightEnv],((-1,-7,-10,-8),),forder=(-1,-7,-8,-10))

        if TDVP is False:
            rightEnv = ncon([rightEnv,psi.mpo.conj()],((1,7,10,-8),(7,1,-2,10)),forder=(-2,-8),order=(10,1,7))
            rightEnv = Tw_inv.applyLeft(rightEnv.reshape(psi.D_mpo**2)).reshape(psi.D_mpo,psi.D_mpo)
            gradRun += ncon([psi.mpo,rightEnv,outerContractSingle],((-2,1,-4,5),(-6,5),(-3,1)),forder=(-3,-2,-4,-6),order=(5,1))


        mag = np.einsum('ijab,ijab',gradRun,gradRun.conj())
        grad += gradRun
        if np.abs(mag)<envTol or np.abs(mag)>1e4:
            break
    print("(vertical) d: ",d,mag)
    grad = np.einsum('ijab->jiab',grad)

    return grad

def grad_mps_horizontalBip(twoBodyH,mps1,mps2,mpo1,mpo2,T1,T2,R1,R2,Ta_12_inv,Ta_21_inv,Tw_12_inv,TDVP=False):
    D_mps = np.size(mps1,axis=1)
    D_mpo = np.size(mpo1,axis=2)
    grad_121 = np.zeros(np.shape(mps1),dtype=complex)
    grad_122 = np.zeros(np.shape(mps1),dtype=complex)

    centreContract = ncon([mpo1,mpo2,twoBodyH,mpo1.conj(),mpo2.conj(),R1.tensor],((1,-10,5,6),(2,-11,6,7),(3,4,1,2),(3,-12,5,9),(4,-13,9,8),(8,7)),forder=(-12,-13,-10,-11),order=(5,6,9,7,8,1,2,3,4))
    outerContract1 = ncon([mps1,mps1.conj(),T2.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(5,4,3))
    outerContract2 = ncon([mps2,mps2.conj(),T1.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(5,4,3))
    # #2 terms with mps tensor removed under local hamiltonian
    grad_121 += ncon([centreContract,outerContract2,mps1,T2.tensor],((-3,4,1,2),(4,2),(1,-5,6),(-7,6)),forder=(-3,-5,-7),order=(2,4,1,6))
    grad_122 += ncon([centreContract,outerContract1,mps2,T1.tensor],((3,-4,1,2),(3,1),(2,-5,6),(-7,6)),forder=(-4,-5,-7),order=(1,3,2,6))

    # #terms above and below horizontal hamiltonian,
    # #below 
    h_tl = ncon([centreContract,outerContract2,mps1,mps1.conj(),T2.tensor],((3,4,1,2),(4,2),(1,-5,6),(3,-7,8),(8,6)),forder=(-7,-5),order=(6,8,1,3,2,4))
    h_tl = Ta_12_inv.applyLeft(h_tl.reshape(D_mps**2)).reshape(D_mps,D_mps)
    grad_122 += ncon([h_tl,mps2],((-4,3),(-1,-2,3)),forder=(-1,-2,-4))
    h_tl = ncon([mps2,mps2.conj(),h_tl],((1,-2,3),(1,-4,5),(5,3)),forder=(-4,-2),order=(3,5,1))
    grad_121 += ncon([h_tl,mps1],((-4,3),(-1,-2,3)),forder=(-1,-2,-4))
    del h_tl

    h_tr = ncon([centreContract,outerContract1,mps2,mps2.conj(),T1.tensor],((3,4,1,2),(3,1),(2,-5,6),(4,-7,8),(8,6)),forder=(-7,-5),order=(6,8,2,4,1,3))
    h_tr = Ta_21_inv.applyLeft(h_tr.reshape(D_mps**2)).reshape(D_mps,D_mps)
    grad_121 += ncon([h_tr,mps1],((-4,3),(-1,-2,3)),forder=(-1,-2,-4))
    h_tr = ncon([mps1,mps1.conj(),h_tr],((1,-2,3),(1,-4,5),(5,3)),forder=(-4,-2),order=(3,5,1))
    grad_122 += ncon([h_tr,mps2],((-4,3),(-1,-2,3)),forder=(-1,-2,-4))
    del h_tr

    h_bl = ncon([centreContract,outerContract2,mps1,mps1.conj()],((3,4,1,2),(4,2),(1,5,-6),(3,5,-7)),forder=(-7,-6),order=(5,1,3,2,4))
    h_bl = Ta_21_inv.applyRight(h_bl.reshape(D_mps**2)).reshape(D_mps,D_mps)
    grad_122 += ncon([h_bl,mps2,T1.tensor],((-4,2),(-1,2,3),(-5,3)))
    h_bl = ncon([mps2,mps2.conj(),h_bl],((1,2,-3),(1,4,-5),(4,2)),forder=(-5,-3),order=(2,4,1))
    grad_121 += ncon([h_bl,mps1,T2.tensor],((-4,2),(-1,2,3),(-5,3)))
    del h_bl

    h_br = ncon([centreContract,outerContract1,mps2,mps2.conj()],((3,4,1,2),(3,1),(2,5,-6),(4,5,-7)),forder=(-7,-6),order=(5,2,4,1,3))
    h_br = Ta_12_inv.applyRight(h_br.reshape(D_mps**2)).reshape(D_mps,D_mps)
    grad_121 += ncon([h_br,mps1,T2.tensor],((-4,2),(-1,2,3),(-5,3)))
    h_br = ncon([mps1,mps1.conj(),h_br],((1,2,-3),(1,4,-5),(4,2)),forder=(-5,-3),order=(2,4,1))
    grad_122 += ncon([h_br,mps2,T1.tensor],((-4,2),(-1,2,3),(-5,3)))
    del h_br

    # #term in line with hamiltonian, to the right
    centreContractLeft = ncon([twoBodyH,mpo1,mpo2,mpo1.conj(),mpo2.conj(),outerContract1,outerContract2],((3,4,1,2),(1,10,5,6),(2,11,6,-7),(3,12,5,9),(4,13,9,-8),(12,10),(13,11)),forder=(-8,-7),order=(5,10,1,3,12,6,9,11,2,4,13))
    centreContractLeft = Tw_12_inv.applyRight(centreContractLeft.reshape(D_mpo**2)).reshape(D_mpo,D_mpo)
    grad_121 += ncon([mps1,mpo1,mpo1.conj(),centreContractLeft,R2.tensor,T2.tensor],((1,-4,5),(2,1,7,8),(2,-3,10,9),(10,7),(9,8),(-6,5)),forder=(-3,-4,-6),order=(7,10,2,8,9,1,5))

    # #upper / lower quadrants
    leftEnv = ncon([mps1,mpo1,mpo1.conj(),mps1.conj(),centreContractLeft,R2.tensor],((1,4,-5),(2,1,6,7),(2,3,8,9),(3,4,-10),(8,6),(9,7)),forder=(-10,-5),order=(6,8,2,7,9,4,1,3))
    leftEnv = Ta_21_inv.applyRight(leftEnv.reshape(D_mps**2)).reshape(D_mps,D_mps)
    grad_122 += ncon([leftEnv,mps2,T1.tensor],((-4,2),(-1,2,3),(-5,3)),forder=(-1,-4,-5))
    leftEnv = ncon([mps2,mps2.conj(),leftEnv],((1,2,-3),(1,4,-5),(4,2)),forder=(-5,-3),order=(2,4,1))
    grad_121 += ncon([leftEnv,mps1,T2.tensor],((-4,2),(-1,2,3),(-5,3)),forder=(-1,-4,-5))
    rightEnv = ncon([mps1,mpo1,mpo1.conj(),mps1.conj(),centreContractLeft,R2.tensor,T2.tensor],((1,-4,5),(2,1,6,7),(2,3,8,9),(3,-10,11),(8,6),(9,7),(11,5)),forder=(-10,-4),order=(6,8,2,7,9,5,11,1,3))
    rightEnv = Ta_12_inv.applyLeft(rightEnv.reshape(D_mps**2)).reshape(D_mps,D_mps)
    grad_122 += ncon([rightEnv,mps2],((-4,3),(-1,-2,3)),forder=(-1,-2,-4))
    rightEnv = ncon([mps2,mps2.conj(),rightEnv],((1,-2,3),(1,-4,5),(5,3)),forder=(-4,-2),order=(3,5,1))
    grad_121 += ncon([rightEnv,mps1],((-4,3),(-1,-2,3)),forder=(-1,-2,-4))

    # #now repeat, but with additional site added
    transfer = mpsu1Transfer_left_oneLayer(mps1,mpo1,T2)
    centreContractLeft = transfer.applyRight(centreContractLeft.reshape(D_mpo**2)).reshape(D_mpo,D_mpo)
    grad_122 += ncon([mps2,mpo2,mpo2.conj(),centreContractLeft,R1.tensor,T1.tensor],((1,-4,5),(2,1,7,8),(2,-3,10,9),(10,7),(9,8),(-6,5)),forder=(-3,-4,-6),order=(7,10,2,8,9,1,5))

    #upper / lower quadrants
    leftEnv = ncon([mps2,mpo2,mpo2.conj(),mps2.conj(),centreContractLeft,R1.tensor],((1,4,-5),(2,1,6,7),(2,3,8,9),(3,4,-10),(8,6),(9,7)),forder=(-10,-5),order=(6,8,2,7,9,4,1,3))
    leftEnv = Ta_12_inv.applyRight(leftEnv.reshape(D_mps**2)).reshape(D_mps,D_mps)
    grad_121 += ncon([leftEnv,mps1,T2.tensor],((-4,2),(-1,2,3),(-5,3)),forder=(-1,-4,-5))
    leftEnv = ncon([mps1,mps1.conj(),leftEnv],((1,2,-3),(1,4,-5),(4,2)),forder=(-5,-3),order=(2,4,1))
    grad_122 += ncon([leftEnv,mps2,T1.tensor],((-4,2),(-1,2,3),(-5,3)),forder=(-1,-4,-5))
    rightEnv = ncon([mps2,mpo2,mpo2.conj(),mps2.conj(),centreContractLeft,R1.tensor,T1.tensor],((1,-4,5),(2,1,6,7),(2,3,8,9),(3,-10,11),(8,6),(9,7),(11,5)),forder=(-10,-4),order=(6,8,2,7,9,5,11,1,3))
    rightEnv = Ta_21_inv.applyLeft(rightEnv.reshape(D_mps**2)).reshape(D_mps,D_mps)
    grad_121 += ncon([rightEnv,mps1],((-4,3),(-1,-2,3)),forder=(-1,-2,-4))
    rightEnv = ncon([mps1,mps1.conj(),rightEnv],((1,-2,3),(1,-4,5),(5,3)),forder=(-4,-2),order=(3,5,1))
    grad_122 += ncon([rightEnv,mps2],((-4,3),(-1,-2,3)),forder=(-1,-2,-4))

    return grad_121,grad_122

def grad_mps_verticalBip(twoBodyH,mps1,mps2,mpo1,mpo2,T1,T2,RR1,RR2,Ta_12_inv,Ta_21_inv,Tw2_21_inv,TDVP=False):
    D_mps = np.size(mps1,axis=1)
    D_mpo = np.size(mpo1,axis=2)
    grad_121 = np.zeros(np.shape(mps1),dtype=complex)
    grad_122 = np.zeros(np.shape(mps1),dtype=complex)

    centreContract = ncon([mpo1,mpo2,twoBodyH,mpo1.conj(),mpo2.conj(),RR2.tensor],((1,-11,5,6),(2,-12,8,9),(3,4,1,2),(3,-13,5,7),(4,-14,8,10),(7,6,10,9)),forder=(-13,-14,-11,-12),order=(6,7,10,9,5,8,1,2,3,4))
    # 2 terms with mps tensor removed under local hamiltonian
    grad_121 += ncon([mps1,mps2,centreContract,mps2.conj(),T1.tensor],((1,-5,6),(2,6,7),(-3,4,1,2),(4,-9,8),(8,7)),forder=(-3,-5,-9),order=(8,7,2,4,6,1))
    grad_122 += ncon([mps1,mps2,centreContract,mps1.conj(),T1.tensor],((1,5,6),(2,6,7),(3,-4,1,2),(3,5,-9),(-8,7)),forder=(-4,-9,-8),order=(5,1,3,5,2,7))

    # #2 terms with mps removed vertical to local hamiltonian
    # #above
    leftEnv = ncon([mps1,mps2,centreContract,mps1.conj(),mps2.conj()],((1,5,6),(2,6,-7),(3,4,1,2),(3,5,9),(4,9,-8)),forder=(-8,-7),order=(5,1,3,6,9,2,4))
    leftEnv = Ta_12_inv.applyRight(leftEnv.reshape(D_mps**2)).reshape(D_mps,D_mps)
    grad_121 += ncon([leftEnv,mps1,T2.tensor],((-4,2),(-1,2,3),(-5,3)),forder=(-1,-4,-5))
    leftEnv = ncon([mps1,mps1.conj(),leftEnv],((1,2,-3),(1,4,-5),(4,2)),forder=(-5,-3),order=(2,4,1))
    grad_122 += ncon([leftEnv,mps2,T1.tensor],((-4,2),(-1,2,3),(-5,3)),forder=(-1,-4,-5))
    #below
    rightEnv = ncon([mps1,mps2,centreContract,mps1.conj(),mps2.conj(),T1.tensor],((1,-5,6),(2,6,7),(3,4,1,2),(3,-10,9),(4,9,8),(8,7)),forder=(-10,-5),order=(7,8,2,4,6,9,1,3))
    rightEnv = Ta_12_inv.applyLeft(rightEnv.reshape(D_mps**2)).reshape(D_mps,D_mps)
    grad_122 += ncon([mps2,rightEnv],((-1,-2,3),(-4,3)),forder=(-1,-2,-4))
    rightEnv = ncon([mps2,mps2.conj(),rightEnv],((1,-2,3),(1,-4,5),(5,3)),forder=(-4,-2),order=(5,3,1))
    grad_121 += ncon([mps1,rightEnv],((-1,-2,3),(-4,3)),forder=(-1,-2,-4))
    del centreContract

    centreContractLeft = ncon([twoBodyH,mps1,mps2,mpo1,mpo2,mpo1.conj(),mpo2.conj(),mps1.conj(),mps2.conj(),T1.tensor],((3,7,2,6),(1,9,10),(5,10,11),(2,1,14,-15),(6,5,17,-18),(3,4,14,-16),(7,8,17,-19),(4,9,13),(8,13,12),(12,11)),forder=(-16,-15,-19,-18),order=(9,10,13,11,12,1,14,2,3,4,5,17,6,7,8))
    centreContractLeft = Tw2_21_inv.applyRight(centreContractLeft.reshape(D_mpo**4)).reshape(D_mpo,D_mpo,D_mpo,D_mpo)
    #reshape and do a dot product might be better for memory (big D)
    Env = ncon([mpo2,mpo1,mpo2.conj(),mpo1.conj(),centreContractLeft,RR1.tensor],((2,-1,7,9),(5,-4,11,13),(2,-3,8,10),(5,-6,12,14),(8,7,12,11),(10,9,14,13)),forder=(-3,-6,-1,-4),order=(13,14,5,9,10,2,8,7,12,11))
    #right
    grad_121 += ncon([mps2,mps1,Env,mps2.conj(),T2.tensor],((1,5,6),(2,6,7),(3,-4,1,2),(3,5,-9),(-8,7)),forder=(-4,-9,-8),order=(5,1,3,5,2,7))
    grad_122 += ncon([mps2,mps1,Env,mps1.conj(),T2.tensor],((1,-5,6),(2,6,7),(-3,4,1,2),(4,-9,8),(8,7)),forder=(-3,-5,-9),order=(8,7,2,4,6,1))
    # #right above
    leftEnv = ncon([mps2,mps1,Env,mps2.conj(),mps1.conj()],((1,5,6),(2,6,-7),(3,4,1,2),(3,5,9),(4,9,-8)),forder=(-8,-7),order=(5,1,3,6,9,2,4))
    leftEnv = Ta_21_inv.applyRight(leftEnv.reshape(D_mps**2)).reshape(D_mps,D_mps)
    grad_122 += ncon([leftEnv,mps2,T1.tensor],((-4,2),(-1,2,3),(-5,3)),forder=(-1,-4,-5))
    leftEnv = ncon([mps2,mps2.conj(),leftEnv],((1,2,-3),(1,4,-5),(4,2)),forder=(-5,-3),order=(2,4,1))
    grad_121 += ncon([leftEnv,mps1,T2.tensor],((-4,2),(-1,2,3),(-5,3)),forder=(-1,-4,-5))
    #right below
    rightEnv = ncon([mps2,mps1,Env,mps2.conj(),mps1.conj(),T2.tensor],((1,-5,6),(2,6,7),(3,4,1,2),(3,-10,9),(4,9,8),(8,7)),forder=(-10,-5),order=(7,8,2,4,6,9,1,3))
    rightEnv = Ta_21_inv.applyLeft(rightEnv.reshape(D_mps**2)).reshape(D_mps,D_mps)
    grad_121 += ncon([mps1,rightEnv],((-1,-2,3),(-4,3)),forder=(-1,-2,-4))
    rightEnv = ncon([mps1,mps1.conj(),rightEnv],((1,-2,3),(1,-4,5),(5,3)),forder=(-4,-2),order=(3,5,1))
    grad_122 += ncon([mps2,rightEnv],((-1,-2,3),(-4,3)),forder=(-1,-2,-4))

    # #now repeat with extra site inserted
    transfer = ncon([mps2,mps1,mpo2,mpo1,mpo2.conj(),mpo1.conj(),mps2.conj(),mps1.conj(),T2.tensor],((1,7,8),(4,8,9),(2,1,-12,-14),(5,4,-16,-18),(2,3,-13,-15),(5,6,-17,-19),(3,7,11),(6,11,10),(10,9)),forder=(-13,-12,-17,-16,-15,-14,-19,-18),order=(7,8,9,10,11,1,2,3,4,5,6)).reshape(D_mpo**4,D_mpo**4)
    centreContractLeft = np.dot(centreContractLeft.reshape(D_mpo**4),transfer).reshape(D_mpo,D_mpo,D_mpo,D_mpo)
    Env = ncon([mpo1,mpo2,mpo1.conj(),mpo2.conj(),centreContractLeft,RR2.tensor],((2,-1,7,9),(5,-4,11,13),(2,-3,8,10),(5,-6,12,14),(8,7,12,11),(10,9,14,13)),forder=(-3,-6,-1,-4),order=(13,14,5,9,10,2,8,7,12,11))
    #right
    grad_122 += ncon([mps1,mps2,Env,mps1.conj(),T1.tensor],((1,5,6),(2,6,7),(3,-4,1,2),(3,5,-9),(-8,7)),forder=(-4,-9,-8),order=(5,1,3,5,2,7))
    grad_121 += ncon([mps1,mps2,Env,mps2.conj(),T1.tensor],((1,-5,6),(2,6,7),(-3,4,1,2),(4,-9,8),(8,7)),forder=(-3,-5,-9),order=(8,7,2,4,6,1))

    # #right above
    leftEnv = ncon([mps1,mps2,Env,mps1.conj(),mps2.conj()],((1,5,6),(2,6,-7),(3,4,1,2),(3,5,9),(4,9,-8)),forder=(-8,-7),order=(5,1,3,6,9,2,4))
    leftEnv = Ta_12_inv.applyRight(leftEnv.reshape(D_mps**2)).reshape(D_mps,D_mps)
    grad_121 += ncon([leftEnv,mps1,T2.tensor],((-4,2),(-1,2,3),(-5,3)),forder=(-1,-4,-5))
    leftEnv = ncon([mps1,mps1.conj(),leftEnv],((1,2,-3),(1,4,-5),(4,2)),forder=(-5,-3),order=(2,4,1))
    grad_122 += ncon([leftEnv,mps2,T1.tensor],((-4,2),(-1,2,3),(-5,3)),forder=(-1,-4,-5))
    #right below
    rightEnv = ncon([mps1,mps2,Env,mps1.conj(),mps2.conj(),T1.tensor],((1,-5,6),(2,6,7),(3,4,1,2),(3,-10,9),(4,9,8),(8,7)),forder=(-10,-5),order=(7,8,2,4,6,9,1,3))
    rightEnv = Ta_12_inv.applyLeft(rightEnv.reshape(D_mps**2)).reshape(D_mps,D_mps)
    grad_122 += ncon([mps2,rightEnv],((-1,-2,3),(-4,3)),forder=(-1,-2,-4))
    rightEnv = ncon([mps2,mps2.conj(),rightEnv],((1,-2,3),(1,-4,5),(5,3)),forder=(-4,-2),order=(3,5,1))
    grad_121 += ncon([mps1,rightEnv],((-1,-2,3),(-4,3)),forder=(-1,-2,-4))

    return grad_121,grad_122

def grad_mpu_horizontalBip(twoBodyH,mps1,mps2,mpo1,mpo2,T1,T2,R1,R2,RR1,RR2,Ta_12,Ta_21,Tw_12_inv,Tw_21_inv,envTol=1e-5,TDVP=False):
    D_mps = np.size(mps1,axis=1)
    D_mpo = np.size(mpo1,axis=2)
    grad_121 = np.zeros(np.shape(mpo1),dtype=complex)
    grad_122 = np.zeros(np.shape(mpo1),dtype=complex)

    #for additional sites
    T2_mod = fixedPoint(ncon([mps2,mps2.conj(),T1.tensor],((1,-2,3),(1,-4,5),(5,3)),forder=(-4,-2),order=(5,3,1)).reshape(D_mps**2),D_mps,2)
    T1_mod = fixedPoint(ncon([mps1,mps1.conj(),T2.tensor],((1,-2,3),(1,-4,5),(5,3)),forder=(-4,-2),order=(5,3,1)).reshape(D_mps**2),D_mps,2)

    #ring around horizontal interaction
    outerContract1 = ncon([mps1,mps1.conj(),T2.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
    outerContract2 = ncon([mps2,mps2.conj(),T1.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
    outerContract1_p1 = ncon([mps1,mps1.conj(),T2_mod.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
    outerContract2_p1 = ncon([mps2,mps2.conj(),T1_mod.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))

    #additional transfers needed (pseudo inverses)
    Tw_12_p1 = mpsu1Transfer_left_oneLayerBip(mps1,mps2,mpo1,mpo2,T1_mod,T2_mod)
    Tw_21_p1 = mpsu1Transfer_left_oneLayerBip(mps2,mps1,mpo2,mpo1,T2_mod,T1_mod)
    R_12_p1 = Tw_12_p1.findRightEig()
    R_12_p1.norm_pairedCanon()
    R_21_p1 = Tw_21_p1.findRightEig()
    R_21_p1.norm_pairedCanon()
    Tw_12_p1_inv = inverseTransfer_left(Tw_12_p1,R_12_p1.vector)
    Tw_21_p1_inv = inverseTransfer_left(Tw_21_p1,R_21_p1.vector)

    #two terms with mpo removed under local hamiltonian
    grad_121 += ncon([mpo1,mpo2,twoBodyH,mpo2.conj(),R1.tensor,outerContract1,outerContract2],((2,1,-9,10),(6,5,10,11),(-3,7,2,6),(7,8,-13,12),(12,11),(-4,1),(8,5)),forder=(-4,-3,-9,-13),order=(12,11,5,8,6,7,10,1))
    grad_122 += ncon([mpo1,mpo2,twoBodyH,mpo1.conj(),R1.tensor,outerContract1,outerContract2],((2,1,9,10),(6,5,10,11),(3,-7,2,6),(3,4,9,-13),(-12,11),(4,1),(-8,5)),forder=(-8,-7,-13,-12),order=(9,1,2,3,4,10,5,6,11))

    #two terms with mpo removed in line with hamiltonian (left and right)
    centreContractLeft = ncon([mpo1,mpo2,twoBodyH,mpo1.conj(),mpo2.conj(),outerContract1,outerContract2],((2,1,9,10),(6,5,10,-11),(3,7,2,6),(3,4,9,13),(7,8,13,-12),(4,1),(8,5)),forder=(-12,-11),order=(9,1,2,3,4,10,13,6,7,5,8))
    exp = np.real(np.einsum('ij,ij',centreContractLeft,R1.tensor))
    centreContractLeft = Tw_12_inv.applyRight(centreContractLeft.reshape(D_mpo**2)).reshape(D_mpo,D_mpo)
    grad_121 += ncon([centreContractLeft,mpo1,R2.tensor,outerContract1],((-6,4),(-2,1,4,5),(-7,5),(-3,1)),forder=(-3,-2,-6,-7),order=(4,5,1))
    transfer = mpsu1Transfer_left_oneLayer(mps1,mpo1,T2)
    centreContractLeft = transfer.applyRight(centreContractLeft.reshape(D_mpo**2)).reshape(D_mpo,D_mpo)
    grad_122 += ncon([centreContractLeft,mpo2,R1.tensor,outerContract2],((-6,4),(-2,1,4,5),(-7,5),(-3,1)),forder=(-3,-2,-6,-7),order=(4,5,1))
    del centreContractLeft

    centreContractRight = ncon([mpo1,mpo2,twoBodyH,mpo1.conj(),mpo2.conj(),R1.tensor,outerContract1,outerContract2],((2,1,-9,10),(6,5,10,11),(3,7,2,6),(3,4,-14,13),(7,8,13,12),(12,11),(4,1),(8,5)),forder=(-14,-9),order=(11,12,5,6,7,8,10,13,2,3,1,4))
    centreContractRight = Tw_12_inv.applyLeft(centreContractRight.reshape(D_mpo**2)).reshape(D_mpo,D_mpo)
    grad_122 += ncon([mpo2,centreContractRight,outerContract2],((-2,1,-4,5),(-6,5),(-3,1)),forder=(-3,-2,-4,-6),order=(5,1))
    transfer = mpsu1Transfer_left_oneLayer(mps2,mpo2,T1)
    centreContractRight = transfer.applyLeft(centreContractRight.reshape(D_mpo**2)).reshape(D_mpo,D_mpo)
    grad_121 += ncon([mpo1,centreContractRight,outerContract1],((-2,1,-4,5),(-6,5),(-3,1)),forder=(-3,-2,-4,-6),order=(5,1))
    del centreContractRight

    # #RRd geometric sums...
    h_tilde = (twoBodyH.reshape(4,4)-np.eye(4)*exp).reshape(2,2,2,2)
    for d in range(0,100):
        gradRun_121 = np.zeros(np.shape(mpo1),dtype=complex)
        gradRun_122 = np.zeros(np.shape(mpo1),dtype=complex)
        if d == 0:
            Td_12 = np.eye(D_mps**2).astype(complex)
            Td_21 = np.eye(D_mps**2).astype(complex)
            Td_12_tensor = Td_12.reshape(D_mps,D_mps,D_mps,D_mps)
            Td_21_tensor = Td_21.reshape(D_mps,D_mps,D_mps,D_mps)
        else:
            Td_12 = np.dot(Td_12,Ta_12.matrix)
            Td_21 = np.dot(Td_21,Ta_21.matrix)
            Td_12_tensor = Td_12.reshape(D_mps,D_mps,D_mps,D_mps)
            Td_21_tensor = Td_21.reshape(D_mps,D_mps,D_mps,D_mps)

        #get new fixed points (RRd_1, RRd_2, RRd_1+1, RRd_2+1)
        TT_d_12 = mpsu1Transfer_left_twoLayerWithMpsInsertBip(mps1,mps2,mpo1,mpo2,T1,T2,Td_12_tensor,Td_21_tensor)
        RR_d_1 = TT_d_12.findRightEig()
        RR_d_1.norm_pairedCanon()
        del TT_d_12

        TT_d_21 = mpsu1Transfer_left_twoLayerWithMpsInsertBip(mps2,mps1,mpo2,mpo1,T2,T1,Td_21_tensor,Td_12_tensor)
        RR_d_2 = TT_d_21.findRightEig()
        RR_d_2.norm_pairedCanon()
        del TT_d_21

        TT_d_12_p1 = mpsu1Transfer_left_twoLayerWithMpsInsertBip_plusOne(mps1,mps2,mpo1,mpo2,T1,T2,Td_12_tensor,Td_21_tensor)
        RR_d_1_p1 = TT_d_12_p1.findRightEig()
        RR_d_1_p1.norm_pairedCanon()
        del TT_d_12_p1

        TT_d_21_p1 = mpsu1Transfer_left_twoLayerWithMpsInsertBip_plusOne(mps2,mps1,mpo2,mpo1,T2,T1,Td_21_tensor,Td_12_tensor)
        RR_d_2_p1 = TT_d_21_p1.findRightEig()
        RR_d_2_p1.norm_pairedCanon()
        del TT_d_21_p1

        innerContract = ncon([mpo1,mpo2,h_tilde,mpo1.conj(),mpo2.conj()],((2,-1,9,10),(6,-5,10,-11),(3,7,2,6),(3,-4,9,13),(7,-8,13,-12)),forder=(-4,-8,-1,-5,-12,-11),order=(9,2,3,10,13,6,7))
        # #need several outer rings... 
        outerContract_double1 = ncon([mps1,mps2,mps1.conj(),mps2.conj(),Td_21_tensor,T1.tensor],((-1,5,6),(-3,7,8),(-2,5,11),(-4,10,9),(11,6,10,7),(9,8)),forder=(-2,-4,-1,-3),order=(5,6,11,7,10,8,9))
        outerContract_double2 = ncon([mps2,mps1,mps2.conj(),mps1.conj(),Td_12_tensor,T2.tensor],((-1,5,6),(-3,7,8),(-2,5,11),(-4,10,9),(11,6,10,7),(9,8)),forder=(-2,-4,-1,-3),order=(5,6,11,7,10,8,9))
        outerContract_double1_p1 = ncon([mps1,mps2,mps1,mps1.conj(),mps2.conj(),mps1.conj(),Td_21_tensor,T2.tensor],((-1,6,7),(3,8,9),(-4,9,10),(-2,6,14),(3,13,12),(-5,12,11),(14,7,13,8),(11,10)),forder=(-2,-5,-1,-4),order=(6,7,14,8,13,3,9,12,10,11))
        outerContract_double2_p1 = ncon([mps2,mps1,mps2,mps2.conj(),mps1.conj(),mps2.conj(),Td_12_tensor,T1.tensor],((-1,6,7),(3,8,9),(-4,9,10),(-2,6,14),(3,13,12),(-5,12,11),(14,7,13,8),(11,10)),forder=(-2,-5,-1,-4),order=(6,7,14,8,13,3,9,12,10,11))


        # #4 terms with mpo removed vertical to hamiltonian
        # #bottom left
        gradRun_121 += ncon([innerContract,mpo2,mpo1,mpo2.conj(),RR_d_1_p1.tensor,outerContract_double1_p1,outerContract_double2_p1],((1,2,3,4,5,6),(11,10,15,13),(-8,7,-17,15),(11,12,-16,14),(14,13,5,6),(-9,1,7,3),(12,2,10,4)),forder=(-9,-8,-17,-16),order=(5,6,13,14,15,4,2,10,12,3,1,7))
        temp = ncon([innerContract,mpo2,mpo1,mpo2.conj(),RR_d_1_p1.tensor,outerContract_double1_p1,outerContract_double2_p1],((1,2,3,4,5,6),(11,10,15,13),(-8,7,-17,15),(11,12,-16,14),(14,13,5,6),(-9,1,7,3),(12,2,10,4)),forder=(-9,-8,-17,-16),order=(5,6,13,14,15,4,2,10,12,3,1,7))
        gradRun_122 += ncon([innerContract,mpo1,mpo2,mpo1.conj(),RR_d_2.tensor,outerContract_double2,outerContract_double1],((1,2,3,4,5,6),(11,10,15,13),(-8,7,-17,15),(11,12,-16,14),(14,13,5,6),(-9,1,7,3),(12,2,10,4)),forder=(-9,-8,-17,-16),order=(5,6,13,14,15,4,2,10,12,3,1,7))
        #bottom right
        gradRun_121 += ncon([innerContract,mpo1,RR_d_2.tensor,outerContract1,outerContract_double1],((1,2,3,4,5,6),(-8,7,-12,10),(-11,10,5,6),(1,3),(-9,2,7,4)),forder=(-9,-8,-12,-11),order=(5,6,10,1,3,2,4,7))
        gradRun_122 += ncon([innerContract,mpo2,RR_d_1_p1.tensor,outerContract1,outerContract_double2_p1],((1,2,3,4,5,6),(-8,7,-12,10),(-11,10,5,6),(1,3),(-9,2,7,4)),forder=(-9,-8,-12,-11),order=(5,6,10,1,3,2,4,7))
        # top right
        gradRun_121 += ncon([innerContract,mpo1,RR_d_1.tensor,outerContract1_p1,outerContract_double2],((1,2,3,4,5,6),(-8,7,-10,11),(5,6,-12,11),(1,3),(2,-9,4,7)),forder=(-9,-8,-10,-12),order=(5,6,11,1,3,2,4,7))
        gradRun_122 += ncon([innerContract,mpo2,RR_d_1_p1.tensor,outerContract1,outerContract_double2_p1],((1,2,3,4,5,6),(-8,7,-10,11),(5,6,-12,11),(1,3),(2,-9,4,7)),forder=(-9,-8,-10,-12),order=(5,6,11,1,3,2,4,7))
        #top left
        gradRun_121 += ncon([innerContract,mpo2,mpo1,mpo2.conj(),RR_d_1_p1.tensor,outerContract_double1_p1,outerContract_double2_p1],((1,2,3,4,5,6),(11,10,14,15),(-8,7,-13,14),(11,12,-17,16),(5,6,16,15),(1,-9,3,7),(2,12,4,10)),forder=(-9,-8,-13,-17),order=(5,6,15,16,14,2,4,12,10,1,3,7))
        gradRun_122 += ncon([innerContract,mpo1,mpo2,mpo1.conj(),RR_d_1.tensor,outerContract_double1,outerContract_double2],((1,2,3,4,5,6),(11,10,14,15),(-8,7,-13,14),(11,12,-17,16),(5,6,16,15),(1,-9,3,7),(2,12,4,10)),forder=(-9,-8,-13,-17),order=(5,6,15,16,14,2,4,12,10,1,3,7))

        # #4 terms from 4 quadrants off diagonal 

        # #right half of plane
        centreContractLeft = ncon([innerContract,outerContract1,outerContract2],((1,2,3,4,-5,-6),(1,3),(2,4)),forder=(-5,-6))
        centreContractLeft = Tw_12_inv.applyRight(centreContractLeft.reshape(D_mpo**2)).reshape(D_mpo,D_mpo)
        # #right upper
        gradRun_121 += ncon([mpo1,mpo1,mpo1.conj(),centreContractLeft,RR_d_2_p1.tensor,outerContract_double1_p1],((-2,1,-7,8),(5,4,10,11),(5,6,12,13),(12,10),(13,11,-9,8),(6,-3,4,1)),forder=(-3,-2,-7,-9),order=(11,13,5,10,12,8,4,6,1))
        #right lower
        gradRun_121 += ncon([mpo1,mpo1.conj(),mpo1,centreContractLeft,RR_d_2_p1.tensor,outerContract_double1_p1],((2,1,7,8),(2,3,9,10),(-5,4,-11,12),(9,7),(-13,12,10,8),(-6,3,4,1)),forder=(-6,-5,-11,-13),order=(8,10,2,7,9,12,1,3,4))
        gradRun_122 += ncon([mpo1,mpo1.conj(),mpo2,centreContractLeft,RR_d_1.tensor,outerContract_double2],((2,1,7,8),(2,3,9,10),(-5,4,-11,12),(9,7),(-13,12,10,8),(-6,3,4,1)),forder=(-6,-5,-11,-13),order=(8,10,2,7,9,12,1,3,4))

        #extra site
        transfer = mpsu1Transfer_left_oneLayer(mps1,mpo1,T2)
        centreContractLeft = transfer.applyRight(centreContractLeft.reshape(D_mpo**2)).reshape(D_mpo,D_mpo)
        #right upper
        gradRun_122 += ncon([mpo2,mpo2,mpo2.conj(),centreContractLeft,RR_d_1_p1.tensor,outerContract_double2_p1],((-2,1,-7,8),(5,4,10,11),(5,6,12,13),(12,10),(13,11,-9,8),(6,-3,4,1)),forder=(-3,-2,-7,-9),order=(11,13,5,10,12,8,4,6,1))
        #right lower
        gradRun_121 += ncon([mpo2,mpo2.conj(),mpo1,centreContractLeft,RR_d_2.tensor,outerContract_double1],((2,1,7,8),(2,3,9,10),(-5,4,-11,12),(9,7),(-13,12,10,8),(-6,3,4,1)),forder=(-6,-5,-11,-13),order=(8,10,2,7,9,12,1,3,4))
        gradRun_122 += ncon([mpo2,mpo2.conj(),mpo2,centreContractLeft,RR_d_1_p1.tensor,outerContract_double2_p1],((2,1,7,8),(2,3,9,10),(-5,4,-11,12),(9,7),(-13,12,10,8),(-6,3,4,1)),forder=(-6,-5,-11,-13),order=(8,10,2,7,9,12,1,3,4))

        #right side terms with modified horizontal transfer matrix (Tw_12_p1)
        centreContractLeft = ncon([innerContract,outerContract1_p1,outerContract2_p1],((1,2,3,4,-5,-6),(1,3),(2,4)),forder=(-5,-6))
        centreContractLeft = Tw_12_p1_inv.applyRight(centreContractLeft.reshape(D_mpo**2)).reshape(D_mpo,D_mpo)
        #right upper
        gradRun_122 += ncon([mpo2,mpo1,mpo1.conj(),centreContractLeft,RR_d_2.tensor,outerContract_double1],((-2,1,-7,8),(5,4,10,11),(5,6,12,13),(12,10),(13,11,-9,8),(6,-3,4,1)),forder=(-3,-2,-7,-9),order=(11,13,5,10,12,8,4,6,1))
        #extra site
        transfer = mpsu1Transfer_left_oneLayer(mps1,mpo1,T2_mod)
        centreContractLeft = transfer.applyRight(centreContractLeft.reshape(D_mpo**2)).reshape(D_mpo,D_mpo)
        #right upper
        gradRun_121 += ncon([mpo1,mpo2,mpo2.conj(),centreContractLeft,RR_d_1.tensor,outerContract_double2],((-2,1,-7,8),(5,4,10,11),(5,6,12,13),(12,10),(13,11,-9,8),(6,-3,4,1)),forder=(-3,-2,-7,-9),order=(11,13,5,10,12,8,4,6,1))

        # #left half of plane
        # #left lower
        rightEnv = ncon([innerContract,RR_d_1_p1.tensor,mpo2,mpo1,mpo2.conj(),mpo1.conj(),outerContract_double1_p1,outerContract_double2_p1],((1,2,3,4,5,6),(18,15,5,6),(11,10,14,15),(8,7,-13,14),(11,12,17,18),(8,9,-16,17),(9,1,7,3),(12,2,10,4)),forder=(-16,-13),order=(5,6,15,18,11,4,2,10,12,14,17,8,1,3,7,9))
        rightEnv += ncon([innerContract,RR_d_1_p1.tensor,mpo2,mpo1,mpo2.conj(),mpo1.conj(),outerContract_double1_p1,outerContract_double2_p1],((1,2,3,4,5,6),(5,6,18,15),(11,10,14,15),(8,7,-13,14),(11,12,17,18),(8,9,-16,17),(1,9,3,7),(2,12,4,10)),forder=(-16,-13),order=(5,6,18,15,11,10,12,4,2,14,17,8,7,9,3,1))
        rightEnv = Tw_12_inv.applyLeft(rightEnv.reshape(D_mpo**2)).reshape(D_mpo,D_mpo)
        gradRun_122 += ncon([mpo2,rightEnv,outerContract2],((-2,1,-4,5),(-6,5),(-3,1)),forder=(-3,-2,-4,-6),order=(5,1))
        transfer = mpsu1Transfer_left_oneLayer(mps2,mpo2,T1)
        rightEnv = transfer.applyLeft(rightEnv.reshape(D_mpo**2)).reshape(D_mpo,D_mpo)
        gradRun_121 += ncon([mpo1,rightEnv,outerContract1],((-2,1,-4,5),(-6,5),(-3,1)),forder=(-3,-2,-4,-6),order=(5,1))

        #left upper
        rightEnv = ncon([innerContract,RR_d_1.tensor,mpo1,mpo2,mpo1.conj(),mpo2.conj(),outerContract_double1,outerContract_double2],((1,2,3,4,5,6),(5,6,18,15),(11,10,14,15),(8,7,-13,14),(11,12,17,18),(8,9,-16,17),(1,9,3,7),(2,12,4,10)),forder=(-16,-13),order=(5,6,18,15,11,10,12,4,2,14,17,8,7,9,3,1))
        rightEnv += ncon([innerContract,RR_d_2.tensor,mpo1,mpo2,mpo1.conj(),mpo2.conj(),outerContract_double2,outerContract_double1],((1,2,3,4,5,6),(18,15,5,6),(11,10,14,15),(8,7,-13,14),(11,12,17,18),(8,9,-16,17),(9,1,7,3),(12,2,10,4)),forder=(-16,-13),order=(5,6,15,18,11,4,2,10,12,14,17,8,1,3,7,9))
        rightEnv = Tw_21_inv.applyLeft(rightEnv.reshape(D_mpo**2)).reshape(D_mpo,D_mpo)
        gradRun_121 += ncon([mpo1,rightEnv,outerContract1],((-2,1,-4,5),(-6,5),(-3,1)),forder=(-3,-2,-4,-6),order=(5,1))
        transfer = mpsu1Transfer_left_oneLayer(mps1,mpo1,T2)
        rightEnv = transfer.applyLeft(rightEnv.reshape(D_mpo**2)).reshape(D_mpo,D_mpo)
        gradRun_122 += ncon([mpo2,rightEnv,outerContract2],((-2,1,-4,5),(-6,5),(-3,1)),forder=(-3,-2,-4,-6),order=(5,1))

        mag1 = np.real(np.einsum('ijab,ijab',gradRun_121,gradRun_121.conj()))
        mag2 = np.real(np.einsum('ijab,ijab',gradRun_122,gradRun_122.conj()))
        grad_121 += gradRun_121
        grad_122 += gradRun_122
        if np.abs(mag1)<envTol and np.abs(mag2)<envTol:
            break
    print("(horizontal) d: ",d,mag1,mag2)

    grad_121 = np.einsum('ijab->jiab',grad_121)
    grad_122 = np.einsum('ijab->jiab',grad_122)
    return grad_121,grad_122

def grad_mpu_verticalBip(twoBodyH,mps1,mps2,mpo1,mpo2,T1,T2,R1,R2,RR1,RR2,Ta_12,Ta_21,Tw_12_inv,Tw_21_inv,Tw2_12_inv,Tw2_21_inv,envTol=1e-5,TDVP=False):
    D_mps = np.size(mps1,axis=1)
    D_mpo = np.size(mpo1,axis=2)
    grad_121 = np.zeros(np.shape(mpo1),dtype=complex)
    grad_122 = np.zeros(np.shape(mpo1),dtype=complex)

    #for additional sites
    T2_mod = fixedPoint(ncon([mps2,mps2.conj(),T1.tensor],((1,-2,3),(1,-4,5),(5,3)),forder=(-4,-2),order=(5,3,1)).reshape(D_mps**2),D_mps,2)
    T1_mod = fixedPoint(ncon([mps1,mps1.conj(),T2.tensor],((1,-2,3),(1,-4,5),(5,3)),forder=(-4,-2),order=(5,3,1)).reshape(D_mps**2),D_mps,2)

    #additional transfers needed (pseudo inverses)
    #double
    Tw2_21_p1 = mpsu1Transfer_left_twoLayerBip(mps2,mps1,mpo2,mpo1,T2_mod,T1_mod)
    RR_2_p1 = Tw2_21_p1.findRightEig()
    RR_2_p1.norm_pairedCanon()
    Tw2_21_p1_inv = inverseTransfer_left(Tw2_21_p1,RR_2_p1.vector)
    #single
    Tw_12_p1 = mpsu1Transfer_left_oneLayerBip(mps1,mps2,mpo1,mpo2,T1_mod,T2_mod)
    R_12_p1 = Tw_12_p1.findRightEig()
    R_12_p1.norm_pairedCanon()
    Tw_12_p1_inv = inverseTransfer_left(Tw_12_p1,R_12_p1.vector)

    #single ring in mps direction
    outerContract1 = ncon([mps1,mps1.conj(),T2.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
    outerContract2 = ncon([mps2,mps2.conj(),T1.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
    outerContract1_p1 = ncon([mps1,mps1.conj(),T2_mod.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
    outerContract2_p1 = ncon([mps2,mps2.conj(),T1_mod.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1),order=(3,4,5))
    # #double ring in mps direction
    outerContract_double1 = ncon([mps1,mps2,mps1.conj(),mps2.conj(),T1.tensor],((-1,5,6),(-2,6,7),(-3,5,9),(-4,9,8),(8,7)),forder=(-3,-4,-1,-2))
    outerContract_double2 = ncon([mps2,mps1,mps2.conj(),mps1.conj(),T2.tensor],((-1,5,6),(-2,6,7),(-3,5,9),(-4,9,8),(8,7)),forder=(-3,-4,-1,-2))
    outerContract_double1_p1 = ncon([mps1,mps2,mps1.conj(),mps2.conj(),T1_mod.tensor],((-1,5,6),(-2,6,7),(-3,5,9),(-4,9,8),(8,7)),forder=(-3,-4,-1,-2))
    outerContract_double2_p2 = ncon([mps2,mps1,mps2.conj(),mps1.conj(),T2_mod.tensor],((-1,5,6),(-2,6,7),(-3,5,9),(-4,9,8),(8,7)),forder=(-3,-4,-1,-2))

    # #two terms with mpo removed under hamiltonian
    grad_121 += ncon([mpo1,mpo2,twoBodyH,mpo2.conj(),RR2.tensor,outerContract_double1],((2,1,-9,10),(4,3,11,12),(-5,7,2,4),(7,8,11,14),(-13,10,14,12),(-6,8,1,3)),forder=(-6,-5,-9,-13),order=(12,14,11,4,7,3,8,10,2,1))
    grad_122 += ncon([mpo1,mpo2,twoBodyH,mpo1.conj(),RR2.tensor,outerContract_double1],((2,1,9,10),(6,5,-11,12),(3,-7,2,6),(3,4,9,13),(13,10,-14,12),(4,-8,1,5)),forder=(-8,-7,-11,-14),order=(10,13,9,2,3,1,4,12,5))

    # #4 terms with mpo removed horizontally in line with hamiltonian
    leftEnv = ncon([mpo1,mpo2,twoBodyH,mpo1.conj(),mpo2.conj(),outerContract_double1],((2,1,9,-10),(6,5,11,-12),(3,7,2,6),(3,4,9,-13),(7,8,11,-14),(4,8,1,5)),forder=(-13,-10,-14,-12),order=(9,1,2,3,4,11,5,6,7,8))
    exp = np.einsum('abcd,abcd',leftEnv,RR2.tensor)
    leftEnv = Tw2_21_inv.applyRight(leftEnv.reshape(D_mpo**4)).reshape(D_mpo,D_mpo,D_mpo,D_mpo)
    grad_121 += ncon([leftEnv,mpo1,mpo2,mpo2.conj(),RR1.tensor,outerContract_double2],((13,11,-9,7),(-2,1,7,8),(5,4,11,12),(5,6,13,14),(14,12,-10,8),(6,-3,4,1)),forder=(-3,-2,-9,-10),order=(14,12,5,4,6,8,1,13,11,7))
    grad_122 += ncon([leftEnv,mpo1,mpo1.conj(),mpo2,RR1.tensor,outerContract_double2],((-10,9,8,7),(2,1,7,11),(2,3,8,12),(-5,4,9,13),(-14,13,12,11),(-6,3,4,1)),forder=(-6,-5,-10,-14),order=(11,12,2,1,3,7,8,9,13,4))

    # extra site
    leftEnv = ncon([mpo1,mpo1.conj(),mpo2,mpo2.conj(),leftEnv,outerContract_double2],((2,1,7,-8),(2,3,9,-10),(5,4,11,-12),(5,6,13,-14),(13,11,9,7),(6,3,4,1)),forder=(-14,-12,-10,-8),order=(7,9,2,11,13,5,1,3,4,6))

    grad_121 += ncon([leftEnv,mpo2,mpo2.conj(),mpo1,RR2.tensor,outerContract_double1],((-10,9,8,7),(2,1,7,11),(2,3,8,12),(-5,4,9,13),(-14,13,12,11),(-6,3,4,1)),forder=(-6,-5,-10,-14),order=(11,12,2,1,3,7,8,9,13,4))
    grad_122 += ncon([leftEnv,mpo2,mpo1,mpo1.conj(),RR2.tensor,outerContract_double1],((13,11,-9,7),(-2,1,7,8),(5,4,11,12),(5,6,13,14),(14,12,-10,8),(6,-3,4,1)),forder=(-3,-2,-9,-10),order=(14,12,5,4,6,8,1,13,11,7))

    rightEnv = ncon([mpo1,mpo2,twoBodyH,mpo1.conj(),mpo2.conj(),RR2.tensor,outerContract_double1],((2,1,-9,10),(6,5,-11,12),(3,7,2,6),(3,4,-13,14),(7,8,-15,16),(14,10,16,12),(4,8,1,5)),forder=(-13,-9,-15,-11),order=(12,16,5,6,7,8,10,14,1,2,3,4))
    rightEnv = Tw2_12_inv.applyLeft(rightEnv.reshape(D_mpo**4)).reshape(D_mpo,D_mpo,D_mpo,D_mpo)
    temp = ncon([mpo1,rightEnv,outerContract1],((-2,1,-4,5),(7,7,-6,5),(-3,1)),forder=(-3,-2,-4,-6),order=(7,5,1))
    grad_121 += ncon([mpo1,rightEnv,outerContract1],((-2,1,-4,5),(7,7,-6,5),(-3,1)),forder=(-3,-2,-4,-6),order=(7,5,1))
    grad_122 += ncon([mpo2,rightEnv,outerContract2_p1],((-2,1,-4,5),(-6,5,7,7),(-3,1)),forder=(-3,-2,-4,-6),order=(7,5,1))

    #extra site
    rightEnv = ncon([mpo1,mpo1.conj(),mpo2,mpo2.conj(),rightEnv,outerContract_double2],((2,1,-7,8),(2,3,-9,10),(5,4,-11,12),(5,6,-13,14),(14,12,10,8),(6,3,4,1)),forder=(-13,-11,-9,-7),order=(8,10,2,12,14,5,6,3,4,1))
    temp += ncon([mpo1,rightEnv,outerContract1_p1],((-2,1,-4,5),(-6,5,7,7),(-3,1)),forder=(-3,-2,-4,-6),order=(7,5,1))
    grad_121 += ncon([mpo1,rightEnv,outerContract1_p1],((-2,1,-4,5),(-6,5,7,7),(-3,1)),forder=(-3,-2,-4,-6),order=(7,5,1))
    grad_122 += ncon([mpo2,rightEnv,outerContract2],((-2,1,-4,5),(7,7,-6,5),(-3,1)),forder=(-3,-2,-4,-6),order=(7,5,1))

    #RRd geometric sums...
    #this is where big transfer matrices are necessary,D^12 !!
    #at most will have 2*D^12 arrays in memory at a time
    h_tilde = (twoBodyH.reshape(4,4)-np.eye(4)*exp).reshape(2,2,2,2)
    for d in range(0,100):
        gradRun_121 = np.zeros(np.shape(mpo1),dtype=complex)
        gradRun_122 = np.zeros(np.shape(mpo1),dtype=complex)
        if d == 0:
            Td_12 = np.eye(D_mps**2).astype(complex)
            Td_21 = np.eye(D_mps**2).astype(complex)
            Td_12_tensor = Td_12.reshape(D_mps,D_mps,D_mps,D_mps)
            Td_21_tensor = Td_21.reshape(D_mps,D_mps,D_mps,D_mps)
        else:
            Td_12 = np.dot(Td_12,Ta_12.matrix)
            Td_21 = np.dot(Td_21,Ta_21.matrix)
            Td_12_tensor = Td_12.reshape(D_mps,D_mps,D_mps,D_mps)
            Td_21_tensor = Td_21.reshape(D_mps,D_mps,D_mps,D_mps)

        #get new fixed points 
        TTT_d_u_12 = mpsu1Transfer_left_threeLayerWithMpsInsert_upperBip(mps1,mps2,mpo1,mpo2,T1,T2,Td_12_tensor,Td_21_tensor)
        RRR_d_u_1 = TTT_d_u_12.findRightEig()
        RRR_d_u_1.norm_pairedCanon()
        del TTT_d_u_12

        TTT_d_u_21 = mpsu1Transfer_left_threeLayerWithMpsInsert_upperBip(mps2,mps1,mpo2,mpo1,T2,T1,Td_21_tensor,Td_12_tensor)
        RRR_d_u_2 = TTT_d_u_21.findRightEig()
        RRR_d_u_2.norm_pairedCanon()
        del TTT_d_u_21

        TTT_d_u_12_p1 = mpsu1Transfer_left_threeLayerWithMpsInsert_upperBip_plusOne(mps1,mps2,mpo1,mpo2,T1,T2,Td_12_tensor,Td_21_tensor)
        RRR_d_u_1_p1 = TTT_d_u_12_p1.findRightEig()
        RRR_d_u_1_p1.norm_pairedCanon()
        del TTT_d_u_12_p1

        TTT_d_u_21_p1 = mpsu1Transfer_left_threeLayerWithMpsInsert_upperBip_plusOne(mps2,mps1,mpo2,mpo1,T2,T1,Td_21_tensor,Td_12_tensor)
        RRR_d_u_2_p1 = TTT_d_u_21_p1.findRightEig()
        RRR_d_u_2_p1.norm_pairedCanon()
        del TTT_d_u_21_p1

        TTT_d_l_12 = mpsu1Transfer_left_threeLayerWithMpsInsert_lowerBip(mps1,mps2,mpo1,mpo2,T1,T2,Td_12_tensor,Td_21_tensor)
        RRR_d_l_1 = TTT_d_l_12.findRightEig()
        RRR_d_l_1.norm_pairedCanon()
        del TTT_d_l_12

        TTT_d_l_21 = mpsu1Transfer_left_threeLayerWithMpsInsert_lowerBip(mps2,mps1,mpo2,mpo1,T2,T1,Td_21_tensor,Td_12_tensor)
        RRR_d_l_2 = TTT_d_l_21.findRightEig()
        RRR_d_l_2.norm_pairedCanon()
        del TTT_d_l_21

        TTT_d_l_12_p1 = mpsu1Transfer_left_threeLayerWithMpsInsert_lowerBip_plusOne(mps1,mps2,mpo1,mpo2,T1,T2,Td_12_tensor,Td_21_tensor)
        RRR_d_l_1_p1 = TTT_d_l_12_p1.findRightEig()
        RRR_d_l_1_p1.norm_pairedCanon()
        del TTT_d_l_12_p1

        TTT_d_l_21_p1 = mpsu1Transfer_left_threeLayerWithMpsInsert_lowerBip_plusOne(mps2,mps1,mpo2,mpo1,T2,T1,Td_21_tensor,Td_12_tensor)
        RRR_d_l_2_p1 = TTT_d_l_21_p1.findRightEig()
        RRR_d_l_2_p1.norm_pairedCanon()
        del TTT_d_l_21_p1

        innerContract = ncon([mpo1,mpo2,h_tilde,mpo1.conj(),mpo2.conj()],((2,-1,9,-10),(6,-5,11,-12),(3,7,2,6),(3,-4,9,-13),(7,-8,11,-14)),forder=(-4,-8,-1,-5,-13,-10,-14,-12),order=(9,2,3,11,6,7))
        outerContract_triple_upper1 = ncon([mps1,mps2,mps1,mps1.conj(),mps2.conj(),mps1.conj(),Td_12_tensor,T2.tensor],((-1,7,8),(-3,8,9),(-5,10,11),(-2,7,15),(-4,15,14),(-6,13,12),(14,9,13,10),(12,11)),forder=(-2,-4,-6,-1,-3,-5),order=(7,8,15,9,14,10,13,11,12))
        outerContract_triple_upper2 = ncon([mps2,mps1,mps2,mps2.conj(),mps1.conj(),mps2.conj(),Td_21_tensor,T1.tensor],((-1,7,8),(-3,8,9),(-5,10,11),(-2,7,15),(-4,15,14),(-6,13,12),(14,9,13,10),(12,11)),forder=(-2,-4,-6,-1,-3,-5),order=(7,8,15,9,14,10,13,11,12))
        outerContract_triple_lower1 = ncon([mps1,mps2,mps1,mps1.conj(),mps2.conj(),mps1.conj(),Td_21_tensor,T2.tensor],((-1,7,8),(-3,9,10),(-5,10,11),(-2,7,15),(-4,14,13),(-6,13,12),(15,8,14,9),(12,11)),forder=(-2,-4,-6,-1,-3,-5),order=(7,8,15,9,14,10,13,11,12))
        outerContract_triple_lower2 = ncon([mps2,mps1,mps2,mps2.conj(),mps1.conj(),mps2.conj(),Td_12_tensor,T1.tensor],((-1,7,8),(-3,9,10),(-5,10,11),(-2,7,15),(-4,14,13),(-6,13,12),(15,8,14,9),(12,11)),forder=(-2,-4,-6,-1,-3,-5),order=(7,8,15,9,14,10,13,11,12))

        outerContract_triple_upper1_p1 = ncon([mps1,mps2,mps1,mps2,mps1.conj(),mps2.conj(),mps1.conj(),mps2.conj(),Td_12_tensor,T1.tensor],((-1,8,9),(-3,9,10),(5,11,12),(-6,12,13),(-2,8,18),(-4,18,17),(5,16,15),(-7,15,14),(17,10,16,11),(14,13),),forder=(-2,-4,-7,-1,-3,-6),order=(8,9,18,10,17,11,16,5,12,15,13,14))
        outerContract_triple_upper2_p1 = ncon([mps2,mps1,mps2,mps1,mps2.conj(),mps1.conj(),mps2.conj(),mps1.conj(),Td_21_tensor,T2.tensor],((-1,8,9),(-3,9,10),(5,11,12),(-6,12,13),(-2,8,18),(-4,18,17),(5,16,15),(-7,15,14),(17,10,16,11),(14,13),),forder=(-2,-4,-7,-1,-3,-6),order=(8,9,18,10,17,11,16,5,12,15,13,14))
        outerContract_triple_lower1_p1 = ncon([mps1,mps2,mps1,mps2,mps1.conj(),mps2.conj(),mps1.conj(),mps2.conj(),Td_21_tensor,T1.tensor],((-1,8,9),(3,10,11),(-4,11,12),(-6,12,13),(-2,8,18),(3,17,16),(-5,16,15),(-7,15,14),(18,9,17,10),(14,13)),forder=(-2,-5,-7,-1,-4,-6),order=(8,9,18,10,17,3,11,16,12,15,13,14))
        outerContract_triple_lower2_p1 = ncon([mps2,mps1,mps2,mps1,mps2.conj(),mps1.conj(),mps2.conj(),mps1.conj(),Td_12_tensor,T2.tensor],((-1,8,9),(3,10,11),(-4,11,12),(-6,12,13),(-2,8,18),(3,17,16),(-5,16,15),(-7,15,14),(18,9,17,10),(14,13)),forder=(-2,-5,-7,-1,-4,-6),order=(8,9,18,10,17,3,11,16,12,15,13,14))

        #vertical about hamiltonian
        #above
        gradRun_121 += ncon([innerContract,mpo1,outerContract_triple_upper1,RRR_d_u_2.tensor],((1,2,3,4,5,6,7,8),(-10,9,-12,13),(1,2,-11,3,4,9),(5,6,7,8,-14,13)),forder=(-11,-10,-12,-14),order=(5,6,1,3,7,8,4,2,13,9))
        gradRun_122 += ncon([innerContract,mpo2,outerContract_triple_upper1_p1,RRR_d_u_2_p1.tensor],((1,2,3,4,5,6,7,8),(-10,9,-12,13),(1,2,-11,3,4,9),(5,6,7,8,-14,13)),forder=(-11,-10,-12,-14),order=(5,6,1,3,7,8,4,2,13,9))
        #below
        gradRun_121 += ncon([innerContract,mpo1,outerContract_triple_lower1_p1,RRR_d_l_2_p1.tensor],((1,2,3,4,5,6,7,8),(-10,9,-12,13),(-11,1,2,9,3,4),(-14,13,5,6,7,8)),forder=(-11,-10,-12,-14),order=(5,6,1,3,7,8,2,4,13,9))
        gradRun_122 += ncon([innerContract,mpo2,outerContract_triple_lower2,RRR_d_l_1.tensor],((1,2,3,4,5,6,7,8),(-10,9,-12,13),(-11,1,2,9,3,4),(-14,13,5,6,7,8)),forder=(-11,-10,-12,-14),order=(5,6,1,3,7,8,2,4,13,9))

        # #right half of plane
        leftEnv = ncon([mpo1,mpo2,h_tilde,mpo1.conj(),mpo2.conj(),outerContract_double1],((2,1,9,-10),(6,5,11,-12),(3,7,2,6),(3,4,9,-13),(7,8,11,-14),(4,8,1,5)),forder=(-13,-10,-14,-12),order=(9,1,2,3,4,11,5,6,7,8))
        leftEnv = Tw2_21_inv.applyRight(leftEnv.reshape(D_mpo**4)).reshape(D_mpo,D_mpo,D_mpo,D_mpo)
        #right upper
        gradRun_121 += ncon([leftEnv,mpo1,mpo1,mpo1.conj(),mpo2,mpo2.conj(),RRR_d_u_1_p1.tensor,outerContract_triple_upper2_p1],((19,17,15,13),(-2,1,-10,11),(5,4,13,14),(5,6,15,16),(8,7,17,18),(8,9,19,20),(20,18,16,14,-12,11),(9,6,-3,7,4,1)),forder=(-3,-2,-10,-12),order=(13,15,4,5,6,17,19,7,8,9,14,16,18,20,11))
        #right lower
        gradRun_121 += ncon([leftEnv,mpo1,mpo1.conj(),mpo2,mpo2.conj(),mpo1,RRR_d_l_2.tensor,outerContract_triple_lower1],((16,14,12,10),(2,1,10,11),(2,3,12,13),(5,4,14,15),(5,6,16,17),(-8,7,-18,19),(-20,19,17,15,13,11),(-9,6,3,7,4,1)),forder=(-9,-8,-18,-20),order=(10,12,1,2,3,14,16,4,5,6,11,13,15,17,19,7))
        gradRun_122 += ncon([leftEnv,mpo1,mpo1.conj(),mpo2,mpo2.conj(),mpo2,RRR_d_l_1_p1.tensor,outerContract_triple_lower2_p1],((16,14,12,10),(2,1,10,11),(2,3,12,13),(5,4,14,15),(5,6,16,17),(-8,7,-18,19),(-20,19,17,15,13,11),(-9,6,3,7,4,1)),forder=(-9,-8,-18,-20),order=(10,12,1,2,3,14,16,4,5,6,11,13,15,17,19,7))

        #extra site
        transfer = ncon([mps2,mps1,mpo2,mpo1,mpo2.conj(),mpo1.conj(),mps2.conj(),mps1.conj(),T2.tensor],((1,7,8),(4,8,9),(2,1,-12,-14),(5,4,-16,-18),(2,3,-13,-15),(5,6,-17,-19),(3,7,11),(6,11,10),(10,9)),forder=(-13,-12,-17,-16,-15,-14,-19,-18),order=(7,8,9,10,11,1,2,3,4,5,6)).reshape(D_mpo**4,D_mpo**4)
        leftEnv = np.dot(leftEnv.reshape(D_mpo**4),transfer).reshape(D_mpo,D_mpo,D_mpo,D_mpo)
        # right upper
        gradRun_122 += ncon([leftEnv,mpo2,mpo2,mpo2.conj(),mpo1,mpo1.conj(),RRR_d_u_2_p1.tensor,outerContract_triple_upper1_p1],((19,17,15,13),(-2,1,-10,11),(5,4,13,14),(5,6,15,16),(8,7,17,18),(8,9,19,20),(20,18,16,14,-12,11),(9,6,-3,7,4,1)),forder=(-3,-2,-10,-12),order=(13,15,4,5,6,17,19,7,8,9,14,16,18,20,11))
        #right lower
        gradRun_121 += ncon([leftEnv,mpo2,mpo2.conj(),mpo1,mpo1.conj(),mpo1,RRR_d_l_2_p1.tensor,outerContract_triple_lower1_p1],((16,14,12,10),(2,1,10,11),(2,3,12,13),(5,4,14,15),(5,6,16,17),(-8,7,-18,19),(-20,19,17,15,13,11),(-9,6,3,7,4,1)),forder=(-9,-8,-18,-20),order=(10,12,1,2,3,14,16,4,5,6,11,13,15,17,19,7))
        gradRun_122 += ncon([leftEnv,mpo2,mpo2.conj(),mpo1,mpo1.conj(),mpo2,RRR_d_l_1.tensor,outerContract_triple_lower2],((16,14,12,10),(2,1,10,11),(2,3,12,13),(5,4,14,15),(5,6,16,17),(-8,7,-18,19),(-20,19,17,15,13,11),(-9,6,3,7,4,1)),forder=(-9,-8,-18,-20),order=(10,12,1,2,3,14,16,4,5,6,11,13,15,17,19,7))

        #with shifted transfer matrix
        leftEnv = ncon([mpo1,mpo2,h_tilde,mpo1.conj(),mpo2.conj(),outerContract_double1_p1],((2,1,9,-10),(6,5,11,-12),(3,7,2,6),(3,4,9,-13),(7,8,11,-14),(4,8,1,5)),forder=(-13,-10,-14,-12),order=(9,1,2,3,4,11,5,6,7,8))
        leftEnv = Tw2_21_p1_inv.applyRight(leftEnv.reshape(D_mpo**4)).reshape(D_mpo,D_mpo,D_mpo,D_mpo)
        #right upper
        gradRun_122 += ncon([leftEnv,mpo2,mpo1,mpo1.conj(),mpo2,mpo2.conj(),RRR_d_u_1.tensor,outerContract_triple_upper2],((19,17,15,13),(-2,1,-10,11),(5,4,13,14),(5,6,15,16),(8,7,17,18),(8,9,19,20),(20,18,16,14,-12,11),(9,6,-3,7,4,1)),forder=(-3,-2,-10,-12),order=(13,15,4,5,6,17,19,7,8,9,14,16,18,20,11))
        #extra site
        transfer = ncon([mps2,mps1,mpo2,mpo1,mpo2.conj(),mpo1.conj(),mps2.conj(),mps1.conj(),T2_mod.tensor],((1,7,8),(4,8,9),(2,1,-12,-14),(5,4,-16,-18),(2,3,-13,-15),(5,6,-17,-19),(3,7,11),(6,11,10),(10,9)),forder=(-13,-12,-17,-16,-15,-14,-19,-18),order=(7,8,9,10,11,1,2,3,4,5,6)).reshape(D_mpo**4,D_mpo**4)
        leftEnv = np.dot(leftEnv.reshape(D_mpo**4),transfer).reshape(D_mpo,D_mpo,D_mpo,D_mpo)
        #right upper
        gradRun_121 += ncon([leftEnv,mpo1,mpo2,mpo2.conj(),mpo1,mpo1.conj(),RRR_d_u_2.tensor,outerContract_triple_upper1],((19,17,15,13),(-2,1,-10,11),(5,4,13,14),(5,6,15,16),(8,7,17,18),(8,9,19,20),(20,18,16,14,-12,11),(9,6,-3,7,4,1)),forder=(-3,-2,-10,-12),order=(13,15,4,5,6,17,19,7,8,9,14,16,18,20,11))

        #left half of plane
        #left upper
        rightEnv = ncon([innerContract,mpo2,mpo2.conj(),RRR_d_u_2_p1.tensor,outerContract_triple_upper1_p1],((1,2,3,4,5,6,7,8),(10,9,-12,13),(10,11,-14,15),(5,6,7,8,15,13),(1,2,11,3,4,9)),forder=(-14,-12),order=(5,6,1,3,7,8,2,4,15,13,9,10,11))
        #left lower
        rightEnv += ncon([innerContract,mpo2,mpo2.conj(),RRR_d_l_1.tensor,outerContract_triple_lower2],((1,2,3,4,5,6,7,8),(10,9,-12,13),(10,11,-14,15),(15,13,5,6,7,8),(11,1,2,9,3,4)),forder=(-14,-12),order=(7,8,2,4,5,6,3,1,13,15,9,10,11))
        rightEnv = Tw_21_inv.applyLeft(rightEnv.reshape(D_mpo**2)).reshape(D_mpo,D_mpo)
        gradRun_121 += ncon([mpo1,rightEnv,outerContract1],((-2,1,-4,5),(-6,5),(-3,1)),forder=(-3,-2,-4,-6),order=(5,1))
        transfer = mpsu1Transfer_left_oneLayer(mps1,mpo1,T2)
        rightEnv = transfer.applyLeft(rightEnv.reshape(D_mpo**2)).reshape(D_mpo,D_mpo)
        gradRun_122 += ncon([mpo2,rightEnv,outerContract2],((-2,1,-4,5),(-6,5),(-3,1)),forder=(-3,-2,-4,-6),order=(5,1))

        rightEnv = ncon([innerContract,mpo1,mpo1.conj(),RRR_d_u_2.tensor,outerContract_triple_upper1],((1,2,3,4,5,6,7,8),(10,9,-12,13),(10,11,-14,15),(5,6,7,8,15,13),(1,2,11,3,4,9)),forder=(-14,-12),order=(5,6,1,3,7,8,2,4,15,13,9,10,11))
        rightEnv += ncon([innerContract,mpo1,mpo1.conj(),RRR_d_l_2_p1.tensor,outerContract_triple_lower1_p1],((1,2,3,4,5,6,7,8),(10,9,-12,13),(10,11,-14,15),(15,13,5,6,7,8),(11,1,2,9,3,4)),forder=(-14,-12),order=(7,8,2,4,5,6,3,1,13,15,9,10,11))
        rightEnv = Tw_12_inv.applyLeft(rightEnv.reshape(D_mpo**2)).reshape(D_mpo,D_mpo)
        gradRun_122 += ncon([mpo2,rightEnv,outerContract2],((-2,1,-4,5),(-6,5),(-3,1)),forder=(-3,-2,-4,-6),order=(5,1))
        transfer = mpsu1Transfer_left_oneLayer(mps2,mpo2,T1)
        rightEnv = transfer.applyLeft(rightEnv.reshape(D_mpo**2)).reshape(D_mpo,D_mpo)
        gradRun_121 += ncon([mpo1,rightEnv,outerContract1],((-2,1,-4,5),(-6,5),(-3,1)),forder=(-3,-2,-4,-6),order=(5,1))

        mag1 = np.real(np.einsum('ijab,ijab',gradRun_121,gradRun_121.conj()))
        mag2 = np.real(np.einsum('ijab,ijab',gradRun_122,gradRun_122.conj()))
        grad_121 += gradRun_121
        grad_122 += gradRun_122
        if np.abs(mag1)<envTol and np.abs(mag2)<envTol:
            break
    print("(vertical) d: ",d,mag1,mag2)
    grad_121 = np.einsum('ijab->jiab',grad_121)
    grad_122 = np.einsum('ijab->jiab',grad_122)
    return grad_121,grad_122
