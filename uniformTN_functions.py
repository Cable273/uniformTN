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

def polarDecomp(M):
    U,S,Vd = sp.linalg.svd(M,full_matrices = False)
    return np.dot(U,Vd)

def randoUnitary(d1,d2,real=False):
    if real is True:
        A = np.random.uniform(-1,1,(d1,d2)) 
    else:
        A = np.random.uniform(-1,1,(d1,d2)) + 1j*np.random.uniform(-1,1,(d1,d2))
    return polarDecomp(A)

def project_mpsTangentVector(gradA,A,T):
    T_eigvector_inverse = np.linalg.inv(T.tensor)
    temp = gradA - ncon([gradA,A.conj(),A],((1,2,-3),(1,2,4),(-6,-5,4)),forder=(-6,-5,-3))
    Xa = np.einsum('iab,cb->iac',temp,T_eigvector_inverse)
    return Xa

def project_mpoTangentVector(grad_mpo,mps,mpo,T,R):
    D_mpo = np.size(mpo,axis=2)
    #Solve M A_tilde = c

    #LHS
    rho = ncon([mps,mps.conj(),T.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1))
    M = ncon([np.eye(2),np.eye(D_mpo),R.tensor,rho],((-3,-2),(-6,-7),(-5,-8),(-4,-1)),forder=(-4,-3,-5,-6,-2,-1,-7,-8)).reshape(4*D_mpo**2,4*D_mpo**2)
    M += ncon([np.eye(2),np.eye(D_mpo),R.tensor,rho],((-3,-2),(-5,-6),(-8,-7),(-4,-1)),forder=(-2,-1,-6,-7,-4,-3,-8,-5)).reshape(4*D_mpo**2,4*D_mpo**2)
    M = M.transpose()

    #RHS
    #lagrange multipliers (gauge fixing)
    lambda_2 = ncon([grad_mpo,mpo.conj()],((2,-1,4,5),(2,-3,4,5)),forder=(-1,-3))
    lambda_2 += -ncon([mpo,grad_mpo.conj()],((2,-1,4,5),(2,-3,4,5)),forder=(-1,-3))
    lambda_1 = ncon([grad_mpo,mpo.conj()],((2,1,3,-4),(2,1,3,-5)),forder=(-4,-5))
    lambda_1 += -ncon([mpo,grad_mpo.conj()],((2,1,3,-4),(2,1,3,-5)),forder=(-4,-5))

    c = ncon([grad_mpo,mpo.conj()],((2,-1,4,-5),(2,-3,4,-6)),forder=(-1,-3,-5,-6))
    c += -ncon([mpo,grad_mpo.conj()],((2,-1,4,-5),(2,-3,4,-6)),forder=(-1,-3,-5,-6))
    c += -ncon([rho,lambda_1],((-2,-1),(-3,-4)),forder=(-2,-1,-3,-4))
    c += -ncon([lambda_2,R.tensor],((-2,-1),(-3,-4)),forder=(-2,-1,-3,-4))
    c = c.reshape(4*D_mpo**2)

    M_inv = np.linalg.inv(M)
    A_tilde = np.dot(M_inv,c).reshape(2,2,D_mpo,D_mpo)
    Xb= np.einsum('jiab,iubv->juav',mpo,A_tilde)
    return Xb
