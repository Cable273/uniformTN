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

#regular transfer matrix + 2 physical legs either side of matrix
class transferPlusTwoPhysical(regularTransfer):
    def genTensor(self):
        self.tensor = self.matrix.view()
        shape = np.append(np.array([2,2]),np.ones(self.noLegs,dtype=int)*self.D)
        self.tensor.shape = np.append(shape,shape)
    def findLeftEig(self,noEigs_lanczos=1):
        el,ul = sp.sparse.linalg.eigs(self.matrix.transpose(),which="LM",k=noEigs_lanczos)
        return fixedPointPlusTwoPhysical(ul[:,np.argmin(np.abs(el-1))],self.D,self.noLegs)
    def findRightEig(self,noEigs_lanczos=1):
        er,ur = sp.sparse.linalg.eigs(self.matrix,which="LM",k=noEigs_lanczos)
        return fixedPointPlusTwoPhysical(ur[:,np.argmin(np.abs(er-1))],self.D,self.noLegs)

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

class inverseTransfer_leftPlusTwoPhysical(inverseTransfer):
    def __init__(self,transferMatrix,rightEig):
        #left eigvector assumed to be product of identities
        leftEig = np.eye(2)
        for index in range(2,transferMatrix.noLegs+1,2):
            leftEig = ncon([leftEig,np.eye(transferMatrix.D)],(-np.arange(1,index+1),(-index-1,-index-2)))
        leftEig = leftEig.reshape(4*transferMatrix.D**transferMatrix.noLegs)
        super().__init__(transferMatrix,leftEig,rightEig)

class inverseTransfer_rightPlusTwoPhysical(inverseTransfer):
    def __init__(self,transferMatrix,leftEig):
        #right eigvector assumed to be product of identities
        rightEig = np.eye(2)
        for index in range(2,transferMatrix.noLegs+1,2):
            rightEig = ncon([rightEig,np.eye(transferMatrix.D)],(-np.arange(1,index+1),(-index-1,-index-2)))
        rightEig = rightEig.reshape(4*transferMatrix.D**transferMatrix.noLegs)
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

class fixedPointPlusTwoPhysical(fixedPoint):
    def __init__(self,vector,D,noLegs):
        self.vector = vector
        self.tensor = self.vector.view()
        self.tensor.shape = np.append(np.array([2,2]),np.ones(noLegs,dtype=int)*D)
        self.D = D
        self.noLegs = noLegs
    def norm_pairedCanon(self):
        #norm assuming eigPair is set of identities acting on pairs of indices
        #ie np.einsum('abcdef...,ab,cd,ef,...',tensor,I,I,I,....) = 1
        contractionIndices = np.vstack((np.arange(0,self.noLegs/2+1,dtype=int),np.arange(0,self.noLegs/2+1,dtype=int))).transpose().flatten()+1
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
class mpsTransferBip(regularTransfer):
    def __init__(self,A,B):
        self.D = np.size(A,axis=1)
        #no legs on each side of tensor (tensor is equiv to D^noLegs,D^noLegs array)
        self.noLegs = 2
        self.matrix = ncon([A,B,A.conj(),B.conj()],((1,-3,4),(2,4,-5),(1,-6,7),(2,7,-8)),forder=(-6,-3,-8,-5),order=(1,4,7,2)).reshape(self.D**2,self.D**2)

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
class mpsu1Transfer_left_oneLayer_twoSite_staircase(regularTransfer): #twoSite unit cell mpso
    def __init__(self,A,B,T):
        self.D = np.size(B,axis=4)
        self.noLegs = 2
        outerContract = dict()
        outerContract['bot']= ncon([A,A.conj(),T.tensor],((-1,3,4,5),(-2,3,4,6),(6,5)),forder=(-2,-1),order=(4,3,5,6))
        outerContract['top']= ncon([A,A.conj(),T.tensor],((1,-2,4,5),(1,-3,4,6),(6,5)),forder=(-3,-2),order=(4,1,5,6))
        self.matrix = ncon([B,B.conj(),outerContract['bot'],outerContract['top']],((2,5,1,4,-7,-8),(2,5,3,6,-9,-10),(3,1),(6,4)),forder=(-9,-7,-10,-8),order=(1,2,3,4,5,6)).reshape(self.D**2,self.D**2)
class mpsu1Transfer_left_oneLayer_twoSite_square(regularTransfer): #twoSite unit cell mpso
    def __init__(self,A,B,T,style):
        self.D = np.size(B,axis=4)
        self.noLegs = 2
        if style == "bot":
            outerContract= ncon([A,A.conj(),T.tensor],((-1,3,4,5),(-2,3,4,6),(6,5)),forder=(-2,-1),order=(4,3,5,6))
        elif style == "top":
            outerContract= ncon([A,A.conj(),T.tensor],((1,-2,4,5),(1,-3,4,6),(6,5)),forder=(-3,-2),order=(4,1,5,6))
        else:
            print("ERROR: mpsu1Transfer_left_oneLayer_twoSite style not valid")
            return 1
        self.matrix = ncon([B,B.conj(),outerContract,outerContract],((2,5,1,4,-7,-8),(2,5,3,6,-9,-10),(3,1),(6,4)),forder=(-9,-7,-10,-8),order=(1,2,3,4,5,6)).reshape(self.D**2,self.D**2)
class mpsu1Transfer_left_oneLayerBip(regularTransfer):
    def __init__(self,A1,A2,B1,B2,T1,T2):
        self.D = np.size(B1,axis=2)
        self.noLegs = 2
        matrix1 = ncon([A1,B1,B1.conj(),A1.conj(),T2.tensor],((1,4,5),(2,1,-7,-9),(2,3,-8,-10),(3,4,6),(6,5)),forder=(-8,-7,-10,-9),order=(4,5,6,1,2,3)).reshape(self.D**2,self.D**2)
        matrix2 = ncon([A2,B2,B2.conj(),A2.conj(),T1.tensor],((1,4,5),(2,1,-7,-9),(2,3,-8,-10),(3,4,6),(6,5)),forder=(-8,-7,-10,-9),order=(4,5,6,1,2,3)).reshape(self.D**2,self.D**2)
        self.matrix = np.dot(matrix1,matrix2)
        del matrix1
        del matrix2

class mpsu1Transfer_left_twoLayer(regularTransfer):
    def __init__(self,A,W,T):
        self.D = np.size(W,axis=2)
        self.noLegs = 4
        self.matrix = ncon([A,A,W,W,W.conj(),W.conj(),A.conj(),A.conj(),T.tensor],((1,7,8),(4,8,9),(2,1,-12,-14),(5,4,-16,-18),(2,3,-13,-15),(5,6,-17,-19),(3,7,11),(6,11,10),(10,9)),forder=(-13,-12,-17,-16,-15,-14,-19,-18),order=(7,8,9,10,11,1,2,3,4,5,6)).reshape(self.D**4,self.D**4)
class mpsu1Transfer_left_twoLayer_twoSite_staircase(transferPlusTwoPhysical): 
    def __init__(self,A,B,T):
        self.D = np.size(B,axis=4)
        self.noLegs = 4
        outerContract = dict()
        outerContract['square'] = ncon([A,A.conj(),T.tensor],((-1,-3,5,6),(-2,-4,5,7),(7,6)),forder=(-2,-4,-1,-3),order=(5,6,7))
        outerContract['prong'] = ncon([A,A,A.conj(),A.conj(),T.tensor],((1,-2,7,8),(-4,6,8,9),(1,-3,7,11),(-5,6,11,10),(10,9)),forder=(-3,-5,-2,-4),order=(7,1,8,11,6,9,10))
        innerContract = ncon([B,B.conj()],((2,5,-1,-4,-7,-8),(2,5,-3,-6,-9,-10)),forder=(-3,-6,-1,-4,-9,-7,-10,-8),order=(2,5))
        self.matrix = ncon([innerContract,innerContract,outerContract['square'],outerContract['prong']],((1,2,3,4,-5,-6,-7,-8),(9,-10,11,-12,-13,-14,-15,-16),(9,2,11,4),(-17,1,-18,3)),forder=(-17,-18,-13,-14,-5,-6,-10,-12,-15,-16,-7,-8)).reshape(4*self.D**4,4*self.D**4)
class mpsu1Transfer_left_twoLayer_twoSite_square(regularTransfer): #twoSite unit cell mpso
    def __init__(self,A,B,T,style):
        self.D = np.size(B,axis=4)
        self.noLegs = 4
        if style == "square":
            outerContract= ncon([A,A.conj(),T.tensor],((-1,-3,5,6),(-2,-4,5,7),(7,6)),forder=(-2,-4,-1,-3),order=(5,6,7))
        elif style == "prong":
            outerContract= ncon([A,A,A.conj(),A.conj(),T.tensor],((1,-2,7,8),(-4,6,8,9),(1,-3,7,11),(-5,6,11,10),(10,9)),forder=(-3,-5,-2,-4),order=(7,1,8,11,6,9,10))
        else:
            print("ERROR: mpsu1Transfer_left_twoLayer_twoSite style not valid")
            return 1
        innerContract = ncon([B,B.conj()],((2,5,-1,-4,-7,-8),(2,5,-3,-6,-9,-10)),forder=(-3,-6,-1,-4,-9,-7,-10,-8),order=(2,5))
        self.matrix = ncon([innerContract,innerContract,outerContract,outerContract],((1,2,3,4,-5,-6,-7,-8),(9,10,11,12,-13,-14,-15,-16),(9,1,11,3),(10,2,12,4)),forder=(-13,-14,-5,-6,-15,-16,-7,-8),order=(3,1,4,2,11,9,12,10)).reshape(self.D**4,self.D**4)
class mpsu1Transfer_left_twoLayerBip(regularTransfer):
    def __init__(self,A1,A2,B1,B2,T1,T2):
        self.D = np.size(B1,axis=2)
        self.noLegs = 4
        matrix1 = ncon([A1,A2,B1,B2,B1.conj(),B2.conj(),A1.conj(),A2.conj(),T1.tensor],((1,7,8),(4,8,9),(2,1,-12,-14),(5,4,-16,-18),(2,3,-13,-15),(5,6,-17,-19),(3,7,11),(6,11,10),(10,9)),forder=(-13,-12,-17,-16,-15,-14,-19,-18),order=(7,8,9,10,11,1,2,3,4,5,6)).reshape(self.D**4,self.D**4)
        matrix2 = ncon([A2,A1,B2,B1,B2.conj(),B1.conj(),A2.conj(),A1.conj(),T2.tensor],((1,7,8),(4,8,9),(2,1,-12,-14),(5,4,-16,-18),(2,3,-13,-15),(5,6,-17,-19),(3,7,11),(6,11,10),(10,9)),forder=(-13,-12,-17,-16,-15,-14,-19,-18),order=(7,8,9,10,11,1,2,3,4,5,6)).reshape(self.D**4,self.D**4)
        self.matrix = np.dot(matrix1,matrix2)
        del matrix1
        del matrix2

class mpsu1Transfer_left_twoLayerWithMpsInsert(regularTransfer):
    def __init__(self,A,W,T,T_insert):
        self.D = np.size(W,axis=2)
        self.noLegs = 4
        self.matrix = ncon([A,A,W,W,W.conj(),W.conj(),A.conj(),A.conj(),T.tensor,T_insert],((1,7,8),(4,9,10),(2,1,-14,-16),(5,4,-18,-20),(2,3,-15,-17),(5,6,-19,-21),(3,7,13),(6,12,11),(11,10),(13,8,12,9)),forder=(-15,-14,-19,-18,-17,-16,-21,-20),order=(7,8,13,9,12,10,11,1,2,3,4,5,6)).reshape(self.D**4,self.D**4)
class mpsu1Transfer_left_twoLayerWithMPSInsert_twoSite_square(regularTransfer): #twoSite unit cell mpso
    def __init__(self,A,B,T,style,Td):
        self.D = np.size(B,axis=4)
        self.noLegs = 4

        outerContractQuad = ncon([A,A,A.conj(),A.conj(),Td,T.tensor],((-1,-3,9,10),(-5,-7,11,12),(-2,-4,9,15),(-6,-8,14,13),(15,10,14,11),(13,12)),forder=(-2,-4,-6,-8,-1,-3,-5,-7),order=(9,10,15,11,14,13,12))
        if style == 'bb':
            outerContract= ncon([outerContractQuad],((-2,3,-5,6,-1,3,-4,6)),forder=(-2,-5,-1,-4))
        elif style == 'bt':
            outerContract= ncon([outerContractQuad],((-2,3,4,-6,-1,3,4,-5)),forder=(-2,-6,-1,-5))
        elif style == 'tb':
            outerContract= ncon([outerContractQuad],((1,-3,-5,6,1,-2,-4,6)),forder=(-3,-5,-2,-4))
        elif style == 'tt':
            outerContract= ncon([outerContractQuad],((1,-3,4,-6,1,-2,4,-5)),forder=(-3,-6,-2,-5))
        else:
            print("ERROR: mpsu1Transfer_left_twoLayerWithMpsInsert_twoSite style not valid")
            return 1
        innerContract = ncon([B,B.conj()],((2,5,-1,-4,-7,-8),(2,5,-3,-6,-9,-10)),forder=(-3,-6,-1,-4,-9,-7,-10,-8),order=(2,5))
        self.matrix = ncon([innerContract,innerContract,outerContract,outerContract],((1,2,3,4,-5,-6,-7,-8),(9,10,11,12,-13,-14,-15,-16),(9,1,11,3),(10,2,12,4)),forder=(-13,-14,-5,-6,-15,-16,-7,-8),order=(3,1,4,2,11,9,12,10)).reshape(self.D**4,self.D**4)
class mpsu1Transfer_left_twoLayerWithMPSInsert_twoSite_staircase_even(regularTransfer): #twoSite unit cell mpso
    def __init__(self,A,B,T,Td):
        self.D = np.size(B,axis=4)
        self.noLegs = 4

        outerContractQuad = ncon([A,A,A.conj(),A.conj(),Td,T.tensor],((-1,-3,9,10),(-5,-7,11,12),(-2,-4,9,15),(-6,-8,14,13),(15,10,14,11),(13,12)),forder=(-2,-4,-6,-8,-1,-3,-5,-7),order=(9,10,15,11,14,13,12))
        outerContractDouble = dict()
        outerContractDouble['bot']= ncon([outerContractQuad],((-2,3,-5,6,-1,3,-4,6)),forder=(-2,-5,-1,-4))
        outerContractDouble['top']= ncon([outerContractQuad],((1,-3,4,-6,1,-2,4,-5)),forder=(-3,-6,-2,-5))
        innerContract = ncon([B,B.conj()],((2,5,-1,-4,-7,-8),(2,5,-3,-6,-9,-10)),forder=(-3,-6,-1,-4,-9,-7,-10,-8),order=(2,5))
        self.matrix = ncon([innerContract,innerContract,outerContractDouble['bot'],outerContractDouble['top']],((1,2,3,4,-5,-6,-7,-8),(9,10,11,12,-13,-14,-15,-16),(9,1,11,3),(10,2,12,4)),forder=(-13,-14,-5,-6,-15,-16,-7,-8),order=(3,1,4,2,11,9,12,10)).reshape(self.D**4,self.D**4)
class mpsu1Transfer_left_twoLayerWithMPSInsert_twoSite_staircase_odd(transferPlusTwoPhysical): #twoSite unit cell mpso
    def __init__(self,A,B,T,Td,Td_m1):
        self.D = np.size(B,axis=4)
        self.noLegs = 4
        outerContractDouble = dict()

        outerContractQuad = ncon([A,A,A.conj(),A.conj(),Td,T.tensor],((-1,-3,9,10),(-5,-7,11,12),(-2,-4,9,15),(-6,-8,14,13),(15,10,14,11),(13,12)),forder=(-2,-4,-6,-8,-1,-3,-5,-7),order=(9,10,15,11,14,13,12))
        outerContractQuad_m1 = ncon([A,A,A.conj(),A.conj(),Td_m1,T.tensor],((-1,-3,9,10),(-5,-7,11,12),(-2,-4,9,15),(-6,-8,14,13),(15,10,14,11),(13,12)),forder=(-2,-4,-6,-8,-1,-3,-5,-7),order=(9,10,15,11,14,13,12))

        outerContractDouble['prong'] = ncon([outerContractQuad],((1,-3,-5,6,1,-2,-4,6)),forder=(-3,-5,-2,-4))
        outerContractDouble['square'] = ncon([outerContractQuad_m1],((-2,3,4,-6,-1,3,4,-5)),forder=(-2,-6,-1,-5))
        innerContract = ncon([B,B.conj()],((2,5,-1,-4,-7,-8),(2,5,-3,-6,-9,-10)),forder=(-3,-6,-1,-4,-9,-7,-10,-8),order=(2,5))
        self.matrix = ncon([innerContract,innerContract,outerContractDouble['prong'],outerContractDouble['square']],((1,2,3,4,-5,-6,-7,-8),(9,-10,11,-12,-13,-14,-15,-16),(-17,1,-18,3),(9,2,11,4)),forder=(-17,-18,-13,-14,-5,-6,-10,-12,-15,-16,-7,-8),order=(1,3,2,4,11,9)).reshape(4*self.D**4,4*self.D**4)
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
class mpsu1Transfer_left_threeLayerWithMPSInsert_twoSite_upper(regularTransfer):
    def __init__(self,mps,mpo,T,style,Td):
        self.D = np.size(mpo,axis=4)
        self.noLegs = 6

        outerContractTriple_square_Quad = ncon([mps,mps,mps.conj(),mps.conj(),Td,T.tensor],((-1,-3,9,10),(-5,-7,11,12),(-2,-4,9,15),(-6,-8,14,13),(15,10,14,11),(13,12)),forder=(-2,-4,-6,-8,-1,-3,-5,-7),order=(9,10,15,11,14,12,13))
        outerContractTriple_upper_prong_Quad = ncon([mps,mps,mps,mps.conj(),mps.conj(),mps.conj(),Td,T.tensor],((1,-2,11,12),(-4,6,12,13),(-7,-9,14,15),(1,-3,11,19),(-5,6,19,18),(-8,-10,17,16),(18,13,17,14),(16,15)),forder=(-3,-5,-8,-10,-2,-4,-7,-9),order=(11,1,12,19,6,13,18,14,17,15,16))
        if style == 'sb':
            outerContractTriple = ncon([outerContractTriple_square_Quad],((-2,-4,-6,7,-1,-3,-5,7)),forder=(-2,-4,-6,-1,-3,-5))
        elif style == 'st':
            outerContractTriple = ncon([outerContractTriple_square_Quad],((-2,-4,5,-7,-1,-3,5,-6)),forder=(-2,-4,-7,-1,-3,-6))
        elif style == 'pb':
            outerContractTriple = ncon([outerContractTriple_upper_prong_Quad],((-2,-4,-6,7,-1,-3,-5,7)),forder=(-2,-4,-6,-1,-3,-5))
        elif style == 'pt':
            outerContractTriple = ncon([outerContractTriple_upper_prong_Quad],((-2,-4,5,-7,-1,-3,5,-6)),forder=(-2,-4,-7,-1,-3,-6))
        innerContract = ncon([mpo,mpo.conj()],((2,5,-1,-4,-7,-8),(2,5,-3,-6,-9,-10)),forder=(-3,-6,-1,-4,-9,-7,-10,-8),order=(2,5))

        self.matrix = ncon([innerContract,innerContract,innerContract,outerContractTriple,outerContractTriple],((1,2,3,4,-5,-6,-7,-8),(9,10,11,12,-13,-14,-15,-16),(17,18,19,20,-21,-22,-23,-24),(17,9,1,19,11,3),(18,10,2,20,12,4)),forder=(-21,-22,-13,-14,-5,-6,-23,-24,-15,-16,-7,-8),order=(19,20,17,18,11,12,9,10,1,2,3,4)).reshape(self.D**6,self.D**6)
class mpsu1Transfer_left_threeLayerWithMPSInsert_twoSite_lower(regularTransfer):
    def __init__(self,mps,mpo,T,style,Td):
        self.D = np.size(mpo,axis=4)
        self.noLegs = 6

        outerContractTriple_square_Quad = ncon([mps,mps,mps.conj(),mps.conj(),Td,T.tensor],((-1,-3,9,10),(-5,-7,11,12),(-2,-4,9,15),(-6,-8,14,13),(15,10,14,11),(13,12)),forder=(-2,-4,-6,-8,-1,-3,-5,-7),order=(9,10,15,11,14,12,13))
        outerContractTriple_lower_prong_Quad = ncon([mps,mps,mps,mps.conj(),mps.conj(),mps.conj(),Td,T.tensor],((-1,-3,11,12),(5,-6,13,14),(-8,10,14,15),(-2,-4,11,19),(5,-7,18,17),(-9,10,17,16),(19,12,18,13),(16,15)),forder=(-2,-4,-7,-9,-1,-3,-6,-8),order=(15,16,10,14,17,5,13,18,12,19,11))
        if style == 'bs':
            outerContractTriple = ncon([outerContractTriple_square_Quad],((-2,3,-5,-7,-1,3,-4,-6)),forder=(-2,-5,-7,-1,-4,-6))
        elif style == 'ts':
            outerContractTriple = ncon([outerContractTriple_square_Quad],((1,-3,-5,-7,1,-2,-4,-6)),forder=(-3,-5,-7,-2,-4,-6))
        elif style == 'bp':
            outerContractTriple = ncon([outerContractTriple_lower_prong_Quad],((-2,3,-5,-7,-1,3,-4,-6)),forder=(-2,-5,-7,-1,-4,-6))
        elif style == 'tp':
            outerContractTriple = ncon([outerContractTriple_lower_prong_Quad],((1,-3,-5,-7,1,-2,-4,-6)),forder=(-3,-5,-7,-2,-4,-6))
        innerContract = ncon([mpo,mpo.conj()],((2,5,-1,-4,-7,-8),(2,5,-3,-6,-9,-10)),forder=(-3,-6,-1,-4,-9,-7,-10,-8))

        self.matrix = ncon([innerContract,innerContract,innerContract,outerContractTriple,outerContractTriple],((1,2,3,4,-5,-6,-7,-8),(9,10,11,12,-13,-14,-15,-16),(17,18,19,20,-21,-22,-23,-24),(17,9,1,19,11,3),(18,10,2,20,12,4)),forder=(-21,-22,-13,-14,-5,-6,-23,-24,-15,-16,-7,-8),order=(19,20,17,18,11,12,9,10,1,2,3,4)).reshape(self.D**6,self.D**6)
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

