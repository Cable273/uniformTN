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

def project_mpoTangentVector(gradW,mps,mpo,T,R):
    D_mps = np.size(mps,axis=1)
    D_mpo = np.size(mpo,axis=2)
    #mpo tangent vector
    I = np.eye(2)
    X = np.array([[0,1],[1,0]])
    Y = np.array([[0,1j],[-1j,0]])
    Z = np.array([[-1,0],[0,1]])
    expX = np.real(np.einsum('ijk,ujb,ui,bk',mps,mps.conj(),X,T.tensor))
    expY = np.real(np.einsum('ijk,ujb,ui,bk',mps,mps.conj(),Y,T.tensor))
    expZ = np.real(np.einsum('ijk,ujb,ui,bk',mps,mps.conj(),Z,T.tensor))

    Mr = np.einsum('ab,dc->dbac',np.eye(D_mpo),R.tensor).reshape(D_mpo**2,D_mpo**2)
    Mp = Mr + Mr.transpose()
    Mm = Mr - Mr.transpose()
    M = np.zeros((3,3,D_mpo**2,D_mpo**2),dtype=complex)
    M[0,0] = (1-expX**2)*Mp
    M[1,1] = (1-expY**2)*Mp
    M[2,2] = (1-expZ**2)*Mp
    M[0,1] = 1j*expZ*Mm
    M[1,0] = -1j*expZ*Mm
    M[0,2] = -1j*expY*Mm
    M[2,0] = 1j*expY*Mm
    M[1,2] = 1j*expX*Mm
    M[2,1] = -1j*expX*Mm
    M = np.einsum('abcd->acbd',M)
    M = M.reshape(3*D_mpo**2,3*D_mpo**2)

    Cx = 1j*(np.einsum('kjab,kiac,ji->bc',mpo,gradW.conj(),X)-expX*np.einsum('kjab,kjac->bc',mpo,gradW.conj()))
    Cx = Cx + Cx.conj().transpose()
    Cx = Cx - np.trace(Cx)*R.tensor
    Cy = 1j*(np.einsum('kjab,kiac,ji->bc',mpo,gradW.conj(),Y)-expY*np.einsum('kjab,kjac->bc',mpo,gradW.conj()))
    Cy = Cy + Cy.conj().transpose()
    Cy = Cy - np.trace(Cy)*R.tensor
    Cz = 1j*(np.einsum('kjab,kiac,ji->bc',mpo,gradW.conj(),Z)-expZ*np.einsum('kjab,kjac->bc',mpo,gradW.conj()))
    Cz = Cz + Cz.conj().transpose()
    Cz = Cz - np.trace(Cz)*R.tensor
    Cxyz = np.zeros((3,D_mpo**2),dtype=complex)
    Cxyz[0] = Cx.reshape(D_mpo**2)
    Cxyz[1] = Cy.reshape(D_mpo**2)
    Cxyz[2] = Cz.reshape(D_mpo**2)
    Cxyz = Cxyz.reshape(3*D_mpo**2)

    M_inv = np.linalg.inv(M)
    Hxyz = np.dot(M_inv,Cxyz)
    Hx = Hxyz[:D_mpo**2].reshape(D_mpo,D_mpo)
    Hy = Hxyz[D_mpo**2:2*D_mpo**2].reshape(D_mpo,D_mpo)
    Hz = Hxyz[2*D_mpo**2:].reshape(D_mpo,D_mpo)
    H0 = -expX*Hx-expY*Hy-expZ*Hz

    M_anti = 1j*(np.kron(I,H0)+np.kron(X,Hx)+np.kron(Y,Hy)+np.kron(Z,Hz))
    B = np.einsum('ijab->iajb',mpo).reshape(2*D_mpo,2*D_mpo)
    Xb = np.einsum('iajb->ijab',np.dot(B,M_anti).reshape(2,D_mpo,2,D_mpo))
    return Xb
