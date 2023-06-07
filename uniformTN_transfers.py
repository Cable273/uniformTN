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

class transferMatrix(ABC):
    def findLeftEig(self,noEigs_lanczos=1):
        el,ul = sp.sparse.linalg.eigs(self.matrix.transpose(),which="LM",k=noEigs_lanczos)
        return fixedPoint(ul[:,np.argmin(np.abs(el-1))],self.D,self.noLegs)
    def findRightEig(self,noEigs_lanczos=1):
        er,ur = sp.sparse.linalg.eigs(self.matrix,which="LM",k=noEigs_lanczos)
        return fixedPoint(ur[:,np.argmin(np.abs(er-1))],self.D,self.noLegs)
    def genTensor(self):
        self.tensor = self.matrix.view()
        self.tensor.shape = np.ones(2*self.noLegs,dtype=int)*self.D

    @abstractmethod
    def applyLeft(self):
        pass
    @abstractmethod
    def applyRight(self):
        pass

class regularTransfer(transferMatrix):
    def applyLeft(self,psi):
        return np.dot(self.matrix,psi)
    def applyRight(self,psi):
        return np.dot(psi,self.matrix)

class inverseTransfer(transferMatrix):
    def __init__(self,transferMatrix,leftEig,rightEig):
        # pseudo inverse = (I - T + |R><L|)^{-1} - |R><L|
        self.D = transferMatrix.D
        self.noLegs = transferMatrix.noLegs
        #inverseMatrix = (I-T+|R><L|)
        self.inverseMatrix = np.eye(np.size(transferMatrix.matrix,axis=1))-transferMatrix.matrix + np.outer(rightEig,leftEig)
        self.proj = np.outer(rightEig,leftEig)
        self.Q = np.eye(np.size(transferMatrix.matrix,axis=1)) - self.proj
    def applyLeft(self,rightVector,tol=1e-10):
        return sp.sparse.linalg.bicgstab(self.inverseMatrix,np.dot(self.Q,rightVector),tol=tol)[0]
    def applyRight(self,leftVector,tol=1e-10):
        return sp.sparse.linalg.bicgstab(self.inverseMatrix.transpose(),np.dot(self.Q.transpose(),leftVector),tol=tol)[0]
    def genInverse(self):
        self.matrix = sp.linalg.inv(self.inverseMatrix)-self.proj

class inverseTransfer_left(inverseTransfer):
    def __init__(self,transferMatrix,rightEig):
        #left eigvector assumed to be product of identities
        leftEig = np.eye(transferMatrix.D)
        for index in range(2,transferMatrix.noLegs,2):
            leftEig = ncon([leftEig,np.eye(transferMatrix.D)],(-np.arange(1,index+1),(-index-1,-index-2)))
        leftEig = leftEig.reshape(transferMatrix.D**transferMatrix.noLegs)
        super().__init__(transferMatrix,leftEig,rightEig)
class inverseTransfer_right(inverseTransfer):
    def __init__(self,transferMatrix,leftEig):
        #right eigvector assumed to be product of identities
        rightEig = np.eye(transferMatrix.D)
        for index in range(2,transferMatrix.noLegs,2):
            rightEig = ncon([rightEig,np.eye(transferMatrix.D)],(-np.arange(1,index+1),(-index-1,-index-2)))
        rightEig = rightEig.reshape(transferMatrix.D**transferMatrix.noLegs)
        super().__init__(transferMatrix,leftEig,rightEig)


class fixedPoint:
    def __init__(self,vector,D,noLegs):
        #take a (D^{no legs}) array as input (equivalent to (D,D,D..D) array
        self.vector = vector
        #use view object to raise errors for copies
        self.tensor = self.vector.view()
        self.tensor.shape = np.ones(noLegs,dtype=int)*D
        self.D = D
        self.noLegs = noLegs
    def norm_pairedVector(self,eigPair):
        self.vector /= np.dot(self.vector,eigPair)
    def norm_pairedCanon(self): 
        #norm assuming eigPair is set of identities acting on pairs of indices
        #ie np.einsum('abcdef...,ab,cd,ef,...',tensor,I,I,I,....) = 1
        contractionIndices = np.vstack((np.arange(0,self.noLegs/2,dtype=int),np.arange(0,self.noLegs/2,dtype=int))).transpose().flatten()+1
        self.vector /= ncon([self.tensor],(contractionIndices))

class mpsTransfer(regularTransfer):
    def __init__(self,A):
        self.D = np.size(A,axis=1)
        #no legs on each side of tensor (tensor is equiv to D^noLegs,D^noLegs array)
        self.noLegs = 2
        self.matrix = np.einsum('ijk,iab->ajbk',A,A.conj()).reshape(self.D**2,self.D**2)
class mpsTransfer_twoSite(regularTransfer):
    def __init__(self,A):
        self.D = np.size(A,axis=2)
        #no legs on each side of tensor (tensor is equiv to D^noLegs,D^noLegs array)
        self.noLegs = 2
        self.matrix = np.einsum('ijab,ijcd->cadb',A,A.conj()).reshape(self.D**2,self.D**2)
class mpoTransfer_withPauli(regularTransfer):
    def __init__(self,W,pauli=np.eye(2)):
        self.D = np.size(W,axis=2)
        #no legs on each side of tensor (tensor is equiv to D^noLegs,D^noLegs array)
        self.noLegs = 2
        self.matrix = np.einsum('ijab,iucd,ju->cadb',W,W.conj(),pauli).reshape(self.D**2,self.D**2)/2
class mpsu1Transfer_left_oneLayer(regularTransfer):
    def __init__(self,A,W,T):
        self.D = np.size(W,axis=2)
        self.noLegs = 2
        self.matrix = ncon([A,W,W.conj(),A.conj(),T.tensor],((1,4,5),(2,1,-7,-9),(2,3,-8,-10),(3,4,6),(6,5)),forder=(-8,-7,-10,-9),order=(4,5,6,1,2,3)).reshape(self.D**2,self.D**2)
class mpsu1Transfer_left_twoLayer(regularTransfer):
    def __init__(self,A,W,T):
        self.D = np.size(W,axis=2)
        self.noLegs = 4
        self.matrix = ncon([A,A,W,W,W.conj(),W.conj(),A.conj(),A.conj(),T.tensor],((1,7,8),(4,8,9),(2,1,-12,-14),(5,4,-16,-18),(2,3,-13,-15),(5,6,-17,-19),(3,7,11),(6,11,10),(10,9)),forder=(-13,-12,-17,-16,-15,-14,-19,-18),order=(7,8,9,10,11,1,2,3,4,5,6)).reshape(self.D**4,self.D**4)
class mpsu1Transfer_left_twoLayerWithMpsInsert(regularTransfer):
    def __init__(self,A,W,T,T_insert):
        self.D = np.size(W,axis=2)
        self.noLegs = 4
        self.matrix = ncon([A,A,W,W,W.conj(),W.conj(),A.conj(),A.conj(),T.tensor,T_insert],((1,7,8),(4,9,10),(2,1,-14,-16),(5,4,-18,-20),(2,3,-15,-17),(5,6,-19,-21),(3,7,13),(6,12,11),(11,10),(13,8,12,9)),forder=(-15,-14,-19,-18,-17,-16,-21,-20),order=(7,8,13,9,12,10,11,1,2,3,4,5,6)).reshape(self.D**4,self.D**4)

class mpsu1Transfer_left_threeLayerWithMpsInsert_lower(regularTransfer):
    def __init__(self,A,W,T,T_insert):
        self.D = np.size(W,axis=2)
        self.noLegs = 6
        self.matrix = ncon([A,A,A,W,W,W,W.conj(),W.conj(),W.conj(),A.conj(),A.conj(),A.conj(),T.tensor,T_insert],((1,10,11),(4,12,13),(7,13,14),(2,1,-19,-21),(5,4,-23,-25),(8,7,-27,-29),(2,3,-20,-22),(5,6,-24,-26),(8,9,-28,-30),(3,10,18),(6,17,16),(9,16,15),(15,14),(18,11,17,12)),forder=(-20,-19,-24,-23,-28,-27,-22,-21,-26,-25,-30,-29),order=(10,11,18,12,17,13,16,14,15,1,2,3,4,5,6,7,8,9)).reshape(self.D**6,self.D**6)
class mpsu1Transfer_left_threeLayerWithMpsInsert_upper(regularTransfer):
    def __init__(self,A,W,T,T_insert):
        self.D = np.size(W,axis=2)
        self.noLegs = 6
        self.matrix = ncon([A,A,A,W,W,W,W.conj(),W.conj(),W.conj(),A.conj(),A.conj(),A.conj(),T.tensor,T_insert],((1,10,11),(4,11,12),(7,13,14),(2,1,-19,-21),(5,4,-23,-25),(8,7,-27,-29),(2,3,-20,-22),(5,6,-24,-26),(8,9,-28,-30),(3,10,18),(6,18,17),(9,16,15),(15,14),(17,12,16,13)),forder=(-20,-19,-24,-23,-28,-27,-22,-21,-26,-25,-30,-29),order=(10,11,18,12,17,13,16,14,15,1,2,3,4,5,6,7,8,9)).reshape(self.D**6,self.D**6)

#bipartite uniform 2d lattice
class mpsTransferBip(regularTransfer):
    def __init__(self,A,B):
        self.D = np.size(A,axis=1)
        #no legs on each side of tensor (tensor is equiv to D^noLegs,D^noLegs array)
        self.noLegs = 2
        self.matrix = ncon([A,B,A.conj(),B.conj()],((1,-3,4),(2,4,-5),(1,-6,7),(2,7,-8)),forder=(-6,-3,-8,-5),order=(1,4,7,2)).reshape(self.D**2,self.D**2)

class mpsu1Transfer_left_oneLayerBip(regularTransfer):
    def __init__(self,A1,A2,B1,B2,T1,T2):
        self.D = np.size(B1,axis=2)
        self.noLegs = 2
        matrix1 = ncon([A1,B1,B1.conj(),A1.conj(),T2.tensor],((1,4,5),(2,1,-7,-9),(2,3,-8,-10),(3,4,6),(6,5)),forder=(-8,-7,-10,-9),order=(4,5,6,1,2,3)).reshape(self.D**2,self.D**2)
        matrix2 = ncon([A2,B2,B2.conj(),A2.conj(),T1.tensor],((1,4,5),(2,1,-7,-9),(2,3,-8,-10),(3,4,6),(6,5)),forder=(-8,-7,-10,-9),order=(4,5,6,1,2,3)).reshape(self.D**2,self.D**2)
        self.matrix = np.dot(matrix1,matrix2)
        del matrix1
        del matrix2
class mpsu1Transfer_left_twoLayerBip(regularTransfer):
    def __init__(self,A1,A2,B1,B2,T1,T2):
        self.D = np.size(B1,axis=2)
        self.noLegs = 4
        matrix1 = ncon([A1,A2,B1,B2,B1.conj(),B2.conj(),A1.conj(),A2.conj(),T1.tensor],((1,7,8),(4,8,9),(2,1,-12,-14),(5,4,-16,-18),(2,3,-13,-15),(5,6,-17,-19),(3,7,11),(6,11,10),(10,9)),forder=(-13,-12,-17,-16,-15,-14,-19,-18),order=(7,8,9,10,11,1,2,3,4,5,6)).reshape(self.D**4,self.D**4)
        matrix2 = ncon([A2,A1,B2,B1,B2.conj(),B1.conj(),A2.conj(),A1.conj(),T2.tensor],((1,7,8),(4,8,9),(2,1,-12,-14),(5,4,-16,-18),(2,3,-13,-15),(5,6,-17,-19),(3,7,11),(6,11,10),(10,9)),forder=(-13,-12,-17,-16,-15,-14,-19,-18),order=(7,8,9,10,11,1,2,3,4,5,6)).reshape(self.D**4,self.D**4)
        self.matrix = np.dot(matrix1,matrix2)
        del matrix1
        del matrix2
class mpsu1Transfer_left_twoLayerWithMpsInsertBip(regularTransfer):
    def __init__(self,A1,A2,B1,B2,T1,T2,T_12,T_21):
        self.D = np.size(B1,axis=2)
        self.noLegs = 4
        matrix1 = ncon([A1,A2,B1,B2,B1.conj(),B2.conj(),A1.conj(),A2.conj(),T1.tensor,T_21],((1,7,8),(4,9,10),(2,1,-14,-16),(5,4,-18,-20),(2,3,-15,-17),(5,6,-19,-21),(3,7,13),(6,12,11),(11,10),(13,8,12,9)),forder=(-15,-14,-19,-18,-17,-16,-21,-20),order=(7,8,13,9,12,10,11,1,2,3,4,5,6)).reshape(self.D**4,self.D**4)
        matrix2 = ncon([A2,A1,B2,B1,B2.conj(),B1.conj(),A2.conj(),A1.conj(),T2.tensor,T_12],((1,7,8),(4,9,10),(2,1,-14,-16),(5,4,-18,-20),(2,3,-15,-17),(5,6,-19,-21),(3,7,13),(6,12,11),(11,10),(13,8,12,9)),forder=(-15,-14,-19,-18,-17,-16,-21,-20),order=(7,8,13,9,12,10,11,1,2,3,4,5,6)).reshape(self.D**4,self.D**4)
        self.matrix = np.dot(matrix1,matrix2)
        del matrix1
        del matrix2
class mpsu1Transfer_left_twoLayerWithMpsInsertBip_plusOne(regularTransfer):
    def __init__(self,A1,A2,B1,B2,T1,T2,T_12,T_21):
        self.D = np.size(B1,axis=2)
        self.noLegs = 4
        matrix1 = ncon([A1,B1,B1.conj(),A1.conj(),A2,A2.conj(),A1,B1,B1.conj(),A1.conj(),T_21,T2.tensor],((1,8,9),(2,1,-17,-18),(2,3,-19,-20),(3,8,16),(4,10,11),(4,15,14),(5,11,12),(6,5,-21,-22),(6,7,-23,-24),(7,14,13),(16,9,15,10),(13,12)),forder=(-19,-17,-23,-21,-20,-18,-24,-22),order=(8,9,16,10,15,4,11,14,12,13,1,2,3,5,6,7)).reshape(self.D**4,self.D**4)
        matrix2 = ncon([A2,B2,B2.conj(),A2.conj(),A1,A1.conj(),A2,B2,B2.conj(),A2.conj(),T_12,T1.tensor],((1,8,9),(2,1,-17,-18),(2,3,-19,-20),(3,8,16),(4,10,11),(4,15,14),(5,11,12),(6,5,-21,-22),(6,7,-23,-24),(7,14,13),(16,9,15,10),(13,12)),forder=(-19,-17,-23,-21,-20,-18,-24,-22),order=(8,9,16,10,15,4,11,14,12,13,1,2,3,5,6,7)).reshape(self.D**4,self.D**4)
        self.matrix = np.dot(matrix1,matrix2)
        del matrix1
        del matrix2
class mpsu1Transfer_left_threeLayerWithMpsInsert_upperBip(regularTransfer):
    def __init__(self,mps1,mps2,mpo1,mpo2,T1,T2,T_12,T_21):
        self.D = np.size(mpo1,axis=2)
        self.noLegs = 6
        matrix1 = ncon([mps1,mps2,mps1,mpo1,mpo2,mpo1,mpo1.conj(),mpo2.conj(),mpo1.conj(),mps1.conj(),mps2.conj(),mps1.conj(),T2.tensor,T_12],((1,10,11),(4,11,12),(7,13,14),(2,1,-19,-21),(5,4,-23,-25),(8,7,-27,-29),(2,3,-20,-22),(5,6,-24,-26),(8,9,-28,-30),(3,10,18),(6,18,17),(9,16,15),(15,14),(17,12,16,13)),forder=(-20,-19,-24,-23,-28,-27,-22,-21,-26,-25,-30,-29),order=(10,11,18,12,17,13,16,14,15,1,2,3,4,5,6,7,8,9)).reshape(self.D**6,self.D**6)
        matrix2 = ncon([mps2,mps1,mps2,mpo2,mpo1,mpo2,mpo2.conj(),mpo1.conj(),mpo2.conj(),mps2.conj(),mps1.conj(),mps2.conj(),T1.tensor,T_21],((1,10,11),(4,11,12),(7,13,14),(2,1,-19,-21),(5,4,-23,-25),(8,7,-27,-29),(2,3,-20,-22),(5,6,-24,-26),(8,9,-28,-30),(3,10,18),(6,18,17),(9,16,15),(15,14),(17,12,16,13)),forder=(-20,-19,-24,-23,-28,-27,-22,-21,-26,-25,-30,-29),order=(10,11,18,12,17,13,16,14,15,1,2,3,4,5,6,7,8,9)).reshape(self.D**6,self.D**6)
        self.matrix = np.dot(matrix1,matrix2)
        del matrix1
        del matrix2
class mpsu1Transfer_left_threeLayerWithMpsInsert_lowerBip(regularTransfer):
    def __init__(self,mps1,mps2,mpo1,mpo2,T1,T2,T_12,T_21):
        self.D = np.size(mpo1,axis=2)
        self.noLegs = 6
        matrix1 = ncon([mps1,mps2,mps1,mpo1,mpo2,mpo1,mpo1.conj(),mpo2.conj(),mpo1.conj(),mps1.conj(),mps2.conj(),mps1.conj(),T2.tensor,T_21],((1,10,11),(4,12,13),(7,13,14),(2,1,-19,-21),(5,4,-23,-25),(8,7,-27,-29),(2,3,-20,-22),(5,6,-24,-26),(8,9,-28,-30),(3,10,18),(6,17,16),(9,16,15),(15,14),(18,11,17,12)),forder=(-20,-19,-24,-23,-28,-27,-22,-21,-26,-25,-30,-29),order=(10,11,18,12,17,13,16,14,15,1,2,3,4,5,6,7,8,9)).reshape(self.D**6,self.D**6)
        matrix2 = ncon([mps2,mps1,mps2,mpo2,mpo1,mpo2,mpo2.conj(),mpo1.conj(),mpo2.conj(),mps2.conj(),mps1.conj(),mps2.conj(),T1.tensor,T_12],((1,10,11),(4,12,13),(7,13,14),(2,1,-19,-21),(5,4,-23,-25),(8,7,-27,-29),(2,3,-20,-22),(5,6,-24,-26),(8,9,-28,-30),(3,10,18),(6,17,16),(9,16,15),(15,14),(18,11,17,12)),forder=(-20,-19,-24,-23,-28,-27,-22,-21,-26,-25,-30,-29),order=(10,11,18,12,17,13,16,14,15,1,2,3,4,5,6,7,8,9)).reshape(self.D**6,self.D**6)
        self.matrix = np.dot(matrix1,matrix2)
        del matrix1
        del matrix2
class mpsu1Transfer_left_threeLayerWithMpsInsert_upperBip_plusOne(regularTransfer):
    def __init__(self,mps1,mps2,mpo1,mpo2,T1,T2,T_12,T_21):
        self.D = np.size(mpo1,axis=2)
        self.noLegs = 6
        matrix1 = ncon([mps1,mpo1,mpo1.conj(),mps1.conj(),mps2,mpo2,mpo2.conj(),mps2.conj(),mps1,mps1.conj(),mps2,mpo2,mpo2.conj(),mps2.conj(),T_12,T1.tensor],((1,11,12),(2,1,-22,-23),(2,3,-24,-25),(3,11,21),(4,12,13),(5,4,-26,-27),(5,6,-28,-29),(6,21,20),(7,14,15),(7,19,18),(8,15,16),(9,8,-30,-31),(9,10,-32,-33),(10,18,17),(20,13,19,14),(17,16)),forder=(-24,-22,-28,-26,-32,-30,-25,-23,-29,-27,-33,-31),order=(11,12,21,13,20,14,19,7,15,18,16,17,1,2,3,4,5,6,8,9,10)).reshape(self.D**6,self.D**6)
        matrix2 = ncon([mps2,mpo2,mpo2.conj(),mps2.conj(),mps1,mpo1,mpo1.conj(),mps1.conj(),mps2,mps2.conj(),mps1,mpo1,mpo1.conj(),mps1.conj(),T_21,T2.tensor],((1,11,12),(2,1,-22,-23),(2,3,-24,-25),(3,11,21),(4,12,13),(5,4,-26,-27),(5,6,-28,-29),(6,21,20),(7,14,15),(7,19,18),(8,15,16),(9,8,-30,-31),(9,10,-32,-33),(10,18,17),(20,13,19,14),(17,16)),forder=(-24,-22,-28,-26,-32,-30,-25,-23,-29,-27,-33,-31),order=(11,12,21,13,20,14,19,7,15,18,16,17,1,2,3,4,5,6,8,9,10)).reshape(self.D**6,self.D**6)
        self.matrix = np.dot(matrix1,matrix2)
        del matrix1
        del matrix2
class mpsu1Transfer_left_threeLayerWithMpsInsert_lowerBip_plusOne(regularTransfer):
    def __init__(self,mps1,mps2,mpo1,mpo2,T1,T2,T_12,T_21):
        self.D = np.size(mpo1,axis=2)
        self.noLegs = 6
        matrix1 = ncon([mps1,mpo1,mpo1.conj(),mps1.conj(),mps2,mps2.conj(),mps1,mpo1,mpo1.conj(),mps1.conj(),mps2,mpo2,mpo2.conj(),mps2.conj(),T_12,T1.tensor],((1,11,12),(2,1,-22,-23),(2,3,-24,-25),(3,11,21),(4,12,13),(4,21,20),(5,14,15),(6,5,-26,-27),(6,7,-28,-29),(7,19,18),(8,15,16),(9,8,-30,-31),(9,10,-32,-33),(10,18,17),(20,13,19,14),(17,16)),forder=(-24,-22,-28,-26,-32,-30,-25,-23,-29,-27,-33,-31),order=(11,12,21,13,20,14,19,15,18,16,17,1,2,3,5,6,7,8,9,10)).reshape(self.D**6,self.D**6)
        matrix2 = ncon([mps2,mpo2,mpo2.conj(),mps2.conj(),mps1,mps1.conj(),mps2,mpo2,mpo2.conj(),mps2.conj(),mps1,mpo1,mpo1.conj(),mps1.conj(),T_21,T2.tensor],((1,11,12),(2,1,-22,-23),(2,3,-24,-25),(3,11,21),(4,12,13),(4,21,20),(5,14,15),(6,5,-26,-27),(6,7,-28,-29),(7,19,18),(8,15,16),(9,8,-30,-31),(9,10,-32,-33),(10,18,17),(20,13,19,14),(17,16)),forder=(-24,-22,-28,-26,-32,-30,-25,-23,-29,-27,-33,-31),order=(11,12,21,13,20,14,19,15,18,16,17,1,2,3,5,6,7,8,9,10)).reshape(self.D**6,self.D**6)
        self.matrix = np.dot(matrix1,matrix2)
        del matrix1
        del matrix2
