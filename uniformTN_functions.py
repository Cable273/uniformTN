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

def project_mpoTangentVector(grad_mpo,mps,mpo,T,R,tol=1e-10,geo=False,geoTol=1e-5):
    D_mpo = np.size(mpo,axis=2)
    D_mps = np.size(mps,axis=1)
    #Solve M A_tilde = c

    #LHS
    rho = ncon([mps,mps.conj(),T.tensor],((-1,3,4),(-2,3,5),(5,4)),forder=(-2,-1))
    M = ncon([np.eye(2),np.eye(D_mpo),R.tensor,rho],((-3,-2),(-6,-7),(-5,-8),(-4,-1)),forder=(-4,-3,-5,-6,-2,-1,-7,-8)).reshape(4*D_mpo**2,4*D_mpo**2)
    M += ncon([np.eye(2),np.eye(D_mpo),R.tensor,rho],((-3,-2),(-5,-6),(-8,-7),(-4,-1)),forder=(-2,-1,-6,-7,-4,-3,-8,-5)).reshape(4*D_mpo**2,4*D_mpo**2)

    #geoSum (off diagonal terms in still present in TDVP metric 
    #(X_B^+)_i (X_B)_j <d_i psi | d_j psi > , X_B  on different sites, but same vertical
    #terms are still present with gauge fixing - can often neglect 
    if geo is True:
        from uniformTN_transfers import mpsTransfer,mpsu1Transfer_left_twoLayerWithMpsInsert
        Ta = mpsTransfer(mps)
        Ta_d = np.eye(D_mps**2)
        Ta_d_limit = np.einsum('ab,cd->abcd',T.tensor,np.eye(D_mps)).reshape(D_mps**2,D_mps**2)
        for d in range(0,100):
            #two site 'density matrix' with Ta_d inserted in between
            rho_d = ncon([mps,mps.conj(),mps,mps.conj(),Ta_d.reshape(D_mps,D_mps,D_mps,D_mps),T.tensor],((-1,5,6),(-2,5,11),(-3,7,8),(-4,10,9),(11,6,10,7),(9,8)),forder=(-2,-4,-1,-3))
            TT_d = mpsu1Transfer_left_twoLayerWithMpsInsert(mps,mpo,T,Ta_d.reshape(D_mps,D_mps,D_mps,D_mps))
            RR_d = TT_d.findRightEig()
            RR_d.norm_pairedCanon()

            M += 2*ncon([RR_d.tensor,rho_d],((-5,-6,-7,-8),(-2,-4,-1,-3)),forder=(-2,-1,-5,-6,-4,-3,-7,-8)).reshape(4*D_mpo**2,4*D_mpo**2)
            M += 2*ncon([RR_d.tensor,rho_d],((-5,-6,-7,-8),(-2,-4,-1,-3)),forder=(-4,-3,-7,-8,-2,-1,-5,-6)).reshape(4*D_mpo**2,4*D_mpo**2)

            Ta_d = np.dot(Ta_d,Ta.matrix)
            diff = Ta_d_limit - Ta_d
            mag = np.abs(np.einsum('ij,ij',diff,diff.conj()))
            if mag < geoTol:
                break

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

    A_tilde =  sp.sparse.linalg.bicgstab(M,c,tol=tol)[0].reshape(2,2,D_mpo,D_mpo)
    Xb= np.einsum('jiab,iubv->juav',mpo,A_tilde)
    return Xb

def project_mpoTangentVector_staircase(grad_mpo,mps,mpo,T,R,tol=1e-10):
    D_mps = np.size(mps,axis=2)
    D_mpo = np.size(mpo,axis=4)

    rho1 = ncon([mps,mps.conj(),T.tensor],((-1,3,4,5),(-2,3,4,6),(6,5)),forder=(-2,-1))
    rho2 = ncon([mps,mps.conj(),T.tensor],((1,-2,4,5),(1,-3,4,6),(6,5)),forder=(-3,-2))
    rho = np.einsum('ab,cd->acbd',rho1,rho2).reshape(4,4)

    mpo = mpo.reshape(4,4,D_mpo,D_mpo)
    grad_mpo = grad_mpo.reshape(4,4,D_mpo,D_mpo)

    #Solve M A_tilde = c
    #LHS
    M = ncon([np.eye(4),np.eye(D_mpo),R.tensor,rho],((-3,-2),(-6,-7),(-5,-8),(-4,-1)),forder=(-4,-3,-5,-6,-2,-1,-7,-8)).reshape(16*D_mpo**2,16*D_mpo**2)
    M += ncon([np.eye(4),np.eye(D_mpo),R.tensor,rho],((-3,-2),(-5,-6),(-8,-7),(-4,-1)),forder=(-2,-1,-6,-7,-4,-3,-8,-5)).reshape(16*D_mpo**2,16*D_mpo**2)
    M = M.transpose()

    #NEGLECTED GEO TERMS (prob negligible - add later if needed for a model)

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
    c = c.reshape(16*D_mpo**2)

    A_tilde =  sp.sparse.linalg.bicgstab(M,c,tol=tol)[0].reshape(4,4,D_mpo,D_mpo)
    Xb= np.einsum('jiab,iubv->juav',mpo,A_tilde)
    Xb = Xb.reshape(2,2,2,2,D_mpo,D_mpo)
    return Xb
