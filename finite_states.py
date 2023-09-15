#_newWay!/usr/bin/env python
# -*- c ding: utf-8 -*-

from __future__ import division
import math
import numpy as np
import scipy as sp
from progressbar import ProgressBar
from scipy.sparse import linalg as sparse_linalg
from ncon import ncon
import copy

from rw_functions import save_obj,load_obj

class finite_1dChains:
    def __init__(self,N,phsDim,D):
        self.N = N
        self.phsDim = phsDim
        self.D = D
        self.tensors = dict()

    def setTensor(self,site,tensor):
        self.tensors[site] = tensor

    def setTensor_bulk(self,tensor):
        for n in range(1,self.N-1):
            self.setTensor(n,tensor)

    def __mul__(self,scalar):
        newMPO = copy.deepcopy(self)
        newMPO.tensors[0] = scalar * newMPO.tensors[0]
        return newMPO

    def __sub__(self,obj2):
        return self + obj2*(-1)

    def conj(self):
        newObj = copy.deepcopy(self)
        for n in range(0,newObj.N):
            newObj.tensors[n] = newObj.tensors[n].conj()
        return newObj

    def print_shape(self):
        for n in range(0,self.N):
            print(self.tensors[n].shape)

class mpo_finite(finite_1dChains):
    def randoInit(self):
        for n in range(0,self.N):
            if n == 0 or n == self.N-1:
                randoTensor = np.random.uniform(-1,1,(self.phsDim,self.phsDim,self.D))
            else:
                randoTensor = np.random.uniform(-1,1,(self.phsDim,self.phsDim,self.D,self.D))
            self.setTensor(n,randoTensor)

    def __add__(self,mpo2):
        newMPO = copy.deepcopy(self)
        for n in range(0,self.N):
            if n == 0 or n == self.N-1:
                D = self.tensors[n].shape[2] + mpo2.tensors[n].shape[2]
                newMPO.tensors[n] = np.zeros((self.phsDim,self.phsDim,D)).astype(complex)
                newMPO.tensors[n][:,:,:self.tensors[n].shape[2]] = self.tensors[n]
                newMPO.tensors[n][:,:,self.tensors[n].shape[2]:] = mpo2.tensors[n]
            else:
                D_left = self.tensors[n].shape[2] + mpo2.tensors[n].shape[2]
                D_right = self.tensors[n].shape[3] + mpo2.tensors[n].shape[3]
                newMPO.tensors[n] = np.zeros((self.phsDim,self.phsDim,D_left,D_right)).astype(complex)
                newMPO.tensors[n][:,:,:self.tensors[n].shape[2],:self.tensors[n].shape[2]] = self.tensors[n]
                newMPO.tensors[n][:,:,self.tensors[n].shape[2]:,self.tensors[n].shape[2]:] = mpo2.tensors[n]
        return newMPO

    def transpose(self):
        newMPO = copy.deepcopy(self)
        for n in range(0,newMPO.N):
            if n == 0 or n == newMPO.N-1:
                newMPO.tensors[n] = np.einsum('ijk->jik',newMPO.tensors[n])
            else:
                newMPO.tensors[n] = np.einsum('ijuv->jiuv',newMPO.tensors[n])
        return newMPO

    def exp(self,psi):
        if self.N == psi.N and self.phsDim == psi.phsDim:
            L = ncon([psi.tensors[0],self.tensors[0],psi.tensors[0].conj()],((1,-3),(2,1,-4),(2,-5)),forder=(-5,-4,-3),order=(1,2))
            for n in range(1,self.N-1):
                L = ncon([L,psi.tensors[n],self.tensors[n],psi.tensors[n].conj()],((7,5,3),(1,3,-4),(2,1,5,-6),(2,7,-8)),forder=(-8,-6,-4),order=(3,5,7,1,2))
            n = self.N-1
            return np.real(ncon([L,psi.tensors[n],self.tensors[n],psi.tensors[n].conj()],((4,5,3),(1,3),(2,1,5),(2,4)),order=(3,5,4,1,2)))
        else:
            print("MPS and MPO not compatible with each other")
            return None
    def var(self,psi):
        H2 = self.mult(self)
        return H2.exp(psi)-(self.exp(psi))**2

    def mult(self,mpo):
        if self.phsDim != mpo.phsDim:
            print("ERROR: MPO physical dimensions not compatible")
        else:
            newMPO = mpo_finite(self.N,self.phsDim,self.D*mpo.D)
            for n in range(0,self.N):
                D = self.tensors[n].shape[2]*mpo.tensors[n].shape[2]
                if n == 0 or n == self.N-1:
                    newMPO.tensors[n] = ncon([self.tensors[n],mpo.tensors[n]],((2,-1,-4),(-3,2,-5)),forder=(-3,-1,-5,-4)).reshape(self.phsDim,self.phsDim,D)
                else:
                    newMPO.tensors[n] = ncon([self.tensors[n],mpo.tensors[n]],((2,-1,-4,-5),(-3,2,-6,-7)),forder=(-3,-1,-6,-4,-7,-5)).reshape(self.phsDim,self.phsDim,D,D)
        return newMPO
                

class mps_finite(finite_1dChains):
    def randoInit(self):
        for n in range(0,self.N):
            if n == 0 or n == self.N-1:
                randoTensor = np.random.uniform(-1,1,(self.phsDim,self.D))
            else:
                randoTensor = np.random.uniform(-1,1,(self.phsDim,self.D,self.D))
            self.setTensor(n,randoTensor)
        self.canonCentre = None

    def __add__(self,mps2):
        newMPS = copy.deepcopy(self)
        for n in range(0,self.N):
            if n == 0 or n == self.N-1:
                D = self.tensors[n].shape[1] + mps2.tensors[n].shape[1]
                newMPS.tensors[n] = np.zeros((self.phsDim,D)).astype(complex)
                newMPS.tensors[n][:,:self.tensors[n].shape[1]] = self.tensors[n]
                newMPS.tensors[n][:,self.tensors[n].shape[1]:] = mps2.tensors[n]
            else:
                D_left = self.tensors[n].shape[1] + mps2.tensors[n].shape[1]
                D_right = self.tensors[n].shape[2] + mps2.tensors[n].shape[2]
                newMPS.tensors[n] = np.zeros((self.phsDim,D_left,D_right)).astype(complex)
                newMPS.tensors[n][:,:self.tensors[n].shape[1],:self.tensors.shape[1]] = self.tensors[n]
                newMPS.tensors[n][:,self.tensors[n].shape[1]:,self.tensors[n].shape[1]:] = mps2.tensors[n]
        return newMPS

    def overlap(self,mps):
        if self.N == mps.N and self.phsDim == mps.phsDim:
            L = np.einsum('ij,ik->kj',self.tensors[0],mps.tensors[0].conj())
            for n in range(1,self.N-1):
                L = ncon([L,self.tensors[n],mps.tensors[n].conj()],((4,2),(1,2,-3),(1,4,-5)),forder=(-5,-3),order=(2,4,1))
            n = self.N-1
            return ncon([L,self.tensors[n],mps.tensors[n].conj()],((3,2),(1,2),(1,3)),order=(2,3,1))
        else:
            print("Two mps' not compatible with each other")
            return None

    def canonForm_left(self,norm=True):
        for n in range(0,self.N-1):
            if n == self.N-2:
                self.canon_sitePair(n,n+1,"right",norm = norm)
            else:
                self.canon_sitePair(n,n+1,"right")
        self.canonCentre = self.N-1

    def canonForm_right(self,norm=True):
        for n in range(self.N-2,-1,-1):
            if n == 0:
                self.canon_sitePair(n,n+1,"left",norm = norm)
            else:
                self.canon_sitePair(n,n+1,"left")
        self.canonCentre = 0

    def shiftCanonCentre(self,sVals_direction,norm=False):
        if self.canonCentre is None:
            print("Error: Cannot shift canonical centre - not in a canonical form")
        else:
            if sVals_direction == 'left':
                self.canon_sitePair(self.canonCentre-1,self.canonCentre,sVals_direction,norm=norm)
                self.canonCentre += -1
            elif sVals_direction == 'right':
                self.canon_sitePair(self.canonCentre,self.canonCentre+1,sVals_direction,norm=norm)
                self.canonCentre += 1

    def canon_sitePair(self,site1,site2,sVals_direction,norm=False):
        shape1 = self.tensors[site1].shape
        shape2 = self.tensors[site2].shape
        if len(self.tensors[site1].shape) == 3 and len(self.tensors[site2].shape) == 3:
            M = ncon([self.tensors[site1],self.tensors[site2]],((-1,-3,4),(-2,4,-5)),forder=(-1,-3,-2,-5)).reshape(shape1[0]*shape1[1],shape2[0]*shape2[2])
        elif len(self.tensors[site1].shape) == 2 and len(self.tensors[site2].shape) == 3:
            M = ncon([self.tensors[site1],self.tensors[site2]],((-1,3),(-2,3,-4)),forder=(-1,-2,-4)).reshape(shape1[0],shape2[0]*shape2[2])
        elif len(self.tensors[site1].shape) == 3 and len(self.tensors[site2].shape) == 2:
            M = ncon([self.tensors[site1],self.tensors[site2]],((-1,-3,4),(-2,4)),forder=(-1,-3,-2)).reshape(shape1[0]*shape1[1],shape2[0])
        elif len(self.tensors[site1].shape) == 2 and len(self.tensors[site2].shape) == 2:
            M = ncon([self.tensors[site1],self.tensors[site2]],((-1,3),(-2,3)),forder=(-1,-2))

        U,S,Vd = sp.linalg.svd(M,full_matrices=False)
        U,S,Vd = U[:,:self.D],S[:self.D],Vd[:self.D,:] #truncate
        if norm is True:
            S = S / np.sqrt(np.dot(S,S))

        if sVals_direction == "left":
            U = np.dot(U,np.diag(S))
        elif sVals_direction == "right":
            Vd = np.dot(np.diag(S),Vd)

        if len(self.tensors[site1].shape) == 2:
            self.tensors[site1] = U
        elif len(self.tensors[site1].shape) == 3:
            self.tensors[site1] = U.reshape(shape1[0],shape1[1],np.size(S))

        if len(self.tensors[site2].shape) == 2:
            self.tensors[site2] = Vd.transpose()
        elif len(self.tensors[site2].shape) == 3:
            self.tensors[site2] = np.einsum('ijk->jik',Vd.reshape(np.size(S),shape2[0],shape2[2]))
