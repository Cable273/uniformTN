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

#projection to closest unitary, wrt frobenius norm (==euclidean metric for g(x,x))
def polarDecomp(M):
    U,S,Vd = sp.linalg.svd(M,full_matrices = False)
    return np.dot(U,Vd)

#tangent space projectors

#Riemannian gradient descent + euclid metric
def project_mps_euclid(gradA,A):
    return gradA - ncon([gradA,A.conj(),A],((1,2,-3),(1,2,4),(-6,-5,4)),forder=(-6,-5,-3))

def project_mps_tdvp_leftGauge(gradA,A,T):
    T_eigvector_inverse = np.linalg.inv(T.tensor)
    temp = gradA - ncon([gradA,A.conj(),A],((1,2,-3),(1,2,4),(-6,-5,4)),forder=(-6,-5,-3))
    Xa = np.einsum('iab,cb->iac',temp,T_eigvector_inverse)
    return Xa

def project_mpo_euclid(grad_mpo,mpo):
    physDim = np.size(mpo,axis=0)
    D = np.size(mpo,axis=2)
    gb_matrix = np.einsum('ijab->iajb',grad_mpo).reshape(physDim*D,physDim*D)
    B_matrix = np.einsum('ijab->iajb',mpo).reshape(physDim*D,physDim*D)
    M = np.dot(B_matrix.conj().transpose(),gb_matrix)
    AH_part = 1/2*(M-M.conj().transpose())
    return np.einsum('iajb->ijab',np.dot(B_matrix,AH_part).reshape(physDim,D,physDim,D))

def project_mpo_tdvp_leftGauge(grad_mpo,mpo,R,rho,tol=1e-5):
    physDim = np.size(mpo,axis=0)
    D = np.size(mpo,axis=2)
    #Solve M A_tilde = c

    #LHS
    #NEGLECTED GEO TERMS (prob negligible - add later if needed for a model)
    M = ncon([np.eye(physDim),np.eye(D),R.tensor,rho],((-3,-2),(-6,-7),(-5,-8),(-4,-1)),forder=(-4,-3,-5,-6,-2,-1,-7,-8)).reshape((physDim*D)**2,(physDim*D)**2)
    M += ncon([np.eye(physDim),np.eye(D),R.tensor,rho],((-3,-2),(-5,-6),(-8,-7),(-4,-1)),forder=(-2,-1,-6,-7,-4,-3,-8,-5)).reshape((physDim*D)**2,(physDim*D)**2)
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
    c = c.reshape((physDim*D)**2)

    M_inv = np.linalg.inv(M)
    A_tilde = np.dot(M_inv,c).reshape(physDim,physDim,D,D)
    Xb= np.einsum('jiab,iubv->juav',mpo,A_tilde)
    return Xb
