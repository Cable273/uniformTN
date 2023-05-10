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

from uniformTN_transfers import inverseTransfer_left,mpsTransfer
from uniformTN_gradients import grad_mps_1d_left,grad_mps_1d
from uniformTN_gradients import *

def polarDecomp(M):
    U,S,Vd = sp.linalg.svd(M,full_matrices = False)
    return np.dot(U,Vd)

def randoUnitary(d1,d2):
    A = np.random.uniform(-1,1,(d1,d2)) + 1j*np.random.uniform(-1,1,(d1,d2))
    U,S,Vd = sp.linalg.svd(A,full_matrices = False)
    return np.dot(U,Vd)

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

class uMPS_1d:
    def __init__(self,D,mps=None):
        self.mps = mps
        self.D = D
    def updateTensors(self,mps,mpo):
        self.mps = mps
    def randoInit(self):
        #random tensor
        M = np.random.uniform(-1,1,(2,self.D,self.D))
        #normalise
        T = mpsTransfer(M)
        e,u = np.linalg.eig(T.matrix)
        e0 = e[np.argmax(np.abs(e))]
        M = M / np.sqrt(np.abs(e0))
        self.mps = M
    def gradDescent(self,twoBodyH,L,R,T_inv,learningRate):
        #grad descent
        gradA = grad_mps_1d(twoBodyH,self.mps,L,R,T_inv)
        A = self.mps - learningRate*gradA
        #norm
        T = mpsTransfer(A)
        e,u = np.linalg.eig(T.matrix)
        e0 = e[np.argmax(np.abs(e))]
        A = A / np.sqrt(np.abs(e0))
        self.mps = A

class uMPS_1d_left(uMPS_1d):
    def randoInit(self):
        #random left canon mps tensor
        self.mps = randoUnitary(2*self.D,self.D).reshape(2,self.D,self.D)
    #update with an iteration of gradient descent
    def gradDescent(self,twoBodyH,R,T_inv,learningRate,metric=False):
        #grad descent
        gradA = grad_mps_1d_left(twoBodyH,self.mps,R,T_inv,metric=metric)
        A = self.mps - learningRate*gradA
        #polar decomp to ensure left canon still
        A = polarDecomp(A.reshape(2*self.D,self.D)).reshape(2,self.D,self.D)
        self.mps = A

class uMPS_1d_centre(uMPS_1d_left):
    def randoInit(self):
        super().randoInit() #initialise as random uniform mps, left canon
        self.construct_centreGauge_tensors()

    def construct_centreGauge_tensors(self):
        #get left/right dom eigvectors L,R
        T = mpsTransfer(self.mps)
        R = T.findRightEig()
        R.norm_pairedCanon()

        #decompose L=l^+l, R=rr^+
        l = np.eye(self.D)
        U,S,Vd = sp.linalg.svd(R.tensor,full_matrices=False)
        r = np.dot(U,np.diag(np.sqrt(S)))
        #mixed canon form
        C = np.dot(l,r)
        U,S,Vd = sp.linalg.svd(C)
        Ac = ncon([self.mps,l,r],((-1,3,4),(-2,3),(4,-5)),forder=(-1,-2,-5))
        self.Ac = Ac
        self.C = C
        #find Ar,Al
        self.polarDecomp_bestCanon()

    def polarDecomp_bestCanon(self):
        Ac_l = self.Ac.reshape(2*self.D,self.D)
        Ac_r = np.einsum('ijk->jik',self.Ac).reshape(self.D,2*self.D)

        Al_notCanon = np.dot(Ac_l,self.C.conj().transpose())
        Ar_notCanon = np.dot(self.C.conj().transpose(),Ac_r)
        Ul,Sl,Vl = sp.linalg.svd(Al_notCanon,full_matrices=False)
        Ur,Sr,Vr = sp.linalg.svd(Ar_notCanon,full_matrices=False)
        self.Al = np.dot(Ul,Vl).reshape(2,self.D,self.D)
        self.Ar= np.einsum('jik->ijk',np.dot(Ur,Vr).reshape(self.D,2,self.D))

    def polarDecomp_bestCanon_stable(self):
        Ac_l = self.Ac.reshape(2*self.D,self.D)
        Ac_r = np.einsum('ijk->jik',self.Ac).reshape(self.D,2*self.D)
        U_ac_l,S_ac_l,Vd_ac_l = sp.linalg.svd(Ac_l,full_matrices=False)
        U_ac_l = np.dot(U_ac_l,Vd_ac_l)

        U_ac_r,S_ac_r,Vd_ac_r = sp.linalg.svd(Ac_r,full_matrices = False)
        U_ac_r = np.dot(U_ac_r,Vd_ac_r)

        Uc,Sc,Vdc = sp.linalg.svd(self.C,full_matrices = False)
        Uc_l = np.dot(Uc,Vdc)
        Uc_r = np.dot(Uc,Vdc)

        self.Al = np.dot(U_ac_l,Uc_l.conj().transpose()).reshape(2,self.D,self.D)
        self.Ar = np.einsum('jik->ijk',np.dot(Uc_r.conj().transpose(),U_ac_r).reshape(self.D,2,self.D))

    def vumps_update(self,beta,twoBodyH,H_ac,H_c,stable_polar=True):
        #regular eigensolver #replace with lanzos for bigger D
        e_ac,u_ac = sp.linalg.eigh(H_ac)
        e_c,u_c = sp.linalg.eigh(H_c)
        #imaginary time evolve to pick out gs with component along previous vector 
        #(like power method to deal with degeneracies, but only one param beta) 
        Ac0 = np.dot(u_ac.conj().transpose(),self.Ac.reshape(2*self.D**2))
        C0 = np.dot(u_c.conj().transpose(),self.C.reshape(self.D**2))
        Ac = np.multiply(Ac0,np.exp(-beta*(e_ac-e_ac[np.argmin(e_ac)]))) #shift spectrum so gs is E=0, avoids nans
        C = np.multiply(C0,np.exp(-beta*(e_c-e_c[np.argmin(e_c)])))
        Ac = Ac / np.power(np.vdot(Ac,Ac),0.5)
        C = C / np.power(np.vdot(C,C),0.5)
        self.Ac = np.dot(u_ac,Ac).reshape(2,self.D,self.D)
        self.C = np.dot(u_c,C).reshape(self.D,self.D)
        if stable_polar is True:
            self.polarDecomp_bestCanon_stable()
        else:
            self.polarDecomp_bestCanon()
        self.mps = self.Al


class uMPSU_2d:
    def __init__(self,D_mps,D_mpo,mps=None,mpo=None):
        self.mps= mps
        self.mpo= mpo
        self.D_mps = D_mps
        self.D_mpo = D_mpo
    def updateTensors(self,mps,mpo):
        self.mps = mps
        self.mpo = mpo
        
class uMPSU1_2d_left(uMPSU_2d):
    def randoInit(self):
        #random left canon mps tensor
        self.mps = randoUnitary(2*self.D_mps,self.D_mps).reshape(2,self.D_mps,self.D_mps)
        #random unitary mpo, left canon
        self.mpo = np.einsum('iajb->ijab',randoUnitary(2*self.D_mpo,2*self.D_mpo).reshape(2,self.D_mpo,2,self.D_mpo))

    def gradDescent(self,twoBodyH,Ta,Tw,Tw2,T,R,RR,learningRate,envTol=1e-5,TDVP=False):
        #inverses
        Ta_inv = inverseTransfer_left(Ta,T.vector)
        Tw_inv = inverseTransfer_left(Tw,R.vector)
        Tw2_inv = inverseTransfer_left(Tw2,RR.vector)
        Ta_inv.genInverse()
        Ta_inv.genTensor()

        gradA = grad_mps_horizontal(twoBodyH,self,T,R,Ta_inv,Tw_inv) + grad_mps_vertical(twoBodyH,self,T,RR,Ta_inv,Tw2_inv)
        gradW = grad_mpu_horizontal(twoBodyH,self,T,R,RR,Ta,Tw_inv,envTol=envTol,TDVP=TDVP) + grad_mpu_vertical(twoBodyH,self,T,R,RR,Ta,Tw_inv,Tw2,Tw2_inv,envTol=envTol,TDVP=TDVP)

        if TDVP is True:
            #map gradA to metric
            Xa = project_mpsTangentVector(gradA,self.mps,T)
            Xb = project_mpoTangentVector(gradW,self.mps,self.mpo,T,R)
            print(np.einsum('ijk,ijk',Xa,Xa.conj()))
            print(np.einsum('ijab,ijab',Xb,Xb.conj()))
            print(np.einsum('ijab,ijab',gradW,gradW.conj()))
            A = self.mps - learningRate*Xa
            W = self.mpo - learningRate*Xb
        else:
            print(np.einsum('ijk,ijk',gradA,gradA.conj()))
            print(np.einsum('ijab,ijab',gradW,gradW.conj()))
            A = self.mps - learningRate*gradA
            W = self.mpo - learningRate*gradW

        #polar decomp to ensure left canon still
        A = polarDecomp(A.reshape(2*self.D_mps,self.D_mps)).reshape(2,self.D_mps,self.D_mps)
        W = np.einsum('iajb->ijab',polarDecomp(np.einsum('ijab->iajb',W).reshape(2*self.D_mpo,2*self.D_mpo)).reshape(2,self.D_mpo,2,self.D_mpo))
        self.mps = A
        self.mpo = W

class uMPSU1_2d_left_bipartite(uMPSU_2d):
    def randoInit(self):
        self.mps1 = randoUnitary(2*self.D_mps,self.D_mps).reshape(2,self.D_mps,self.D_mps)
        self.mps2 = randoUnitary(2*self.D_mps,self.D_mps).reshape(2,self.D_mps,self.D_mps)
        self.mpo1 = np.einsum('iajb->ijab',randoUnitary(2*self.D_mpo,2*self.D_mpo).reshape(2,self.D_mpo,2,self.D_mpo))
        self.mpo2 = np.einsum('iajb->ijab',randoUnitary(2*self.D_mpo,2*self.D_mpo).reshape(2,self.D_mpo,2,self.D_mpo))

    def gradDescent(self,twoBodyH,Ta_12,Ta_21,Tw_12,Tw_21,Tw2_12,Tw2_21,T1,T2,R1,R2,RR1,RR2,learningRate,envTol=1e-5,TDVP=False):
        #inverses
        Ta_12_inv = inverseTransfer_left(Ta_12,T1.vector)
        Ta_21_inv = inverseTransfer_left(Ta_21,T2.vector)
        Tw_12_inv = inverseTransfer_left(Tw_12,R1.vector)
        Tw_21_inv = inverseTransfer_left(Tw_21,R2.vector)
        Tw2_12_inv = inverseTransfer_left(Tw2_12,RR1.vector)
        Tw2_21_inv = inverseTransfer_left(Tw2_21,RR2.vector)

        gradA_hori_121,gradA_hori_122 = grad_mps_horizontalBip(twoBodyH,self.mps1,self.mps2,self.mpo1,self.mpo2,T1,T2,R1,R2,Ta_12_inv,Ta_21_inv,Tw_12_inv)
        gradA_hori_212,gradA_hori_211 = grad_mps_horizontalBip(twoBodyH,self.mps2,self.mps1,self.mpo2,self.mpo1,T2,T1,R2,R1,Ta_21_inv,Ta_12_inv,Tw_21_inv)
        gradA_vert_121,gradA_vert_122 = grad_mps_verticalBip(twoBodyH,self.mps1,self.mps2,self.mpo1,self.mpo2,T1,T2,RR1,RR2,Ta_12_inv,Ta_21_inv,Tw2_21_inv)
        gradA_vert_212,gradA_vert_211 = grad_mps_verticalBip(twoBodyH,self.mps2,self.mps1,self.mpo2,self.mpo1,T2,T1,RR2,RR1,Ta_21_inv,Ta_12_inv,Tw2_12_inv)
        gradA_hori1 = 1/2*(gradA_hori_121 + gradA_hori_211)
        gradA_hori2 = 1/2*(gradA_hori_122 + gradA_hori_212)
        gradA_vert1 = 1/2*(gradA_vert_121 + gradA_vert_211)
        gradA_vert2 = 1/2*(gradA_vert_122 + gradA_vert_212)
        gradA1 = gradA_hori1 + gradA_vert1
        gradA2 = gradA_hori2 + gradA_vert2

        gradB_hori_121,gradB_hori_122 = grad_mpu_horizontalBip(twoBodyH,self.mps1,self.mps2,self.mpo1,self.mpo2,T1,T2,R1,R2,RR1,RR2,Ta_12,Ta_21,Tw_12_inv,Tw_21_inv,envTol=1e-5,TDVP=False)
        gradB_hori_212,gradB_hori_211 = grad_mpu_horizontalBip(twoBodyH,self.mps2,self.mps1,self.mpo2,self.mpo1,T2,T1,R2,R1,RR2,RR1,Ta_21,Ta_12,Tw_21_inv,Tw_12_inv,envTol=1e-5,TDVP=False)
        gradB_vert_121,gradB_vert_122 = grad_mpu_verticalBip(twoBodyH,self.mps1,self.mps2,self.mpo1,self.mpo2,T1,T2,R1,R2,RR1,RR2,Ta_12,Ta_21,Tw_12_inv,Tw_21_inv,Tw2_12_inv,Tw2_21_inv,envTol=1e-5,TDVP=False)
        gradB_vert_212,gradB_vert_211 = grad_mpu_verticalBip(twoBodyH,self.mps2,self.mps1,self.mpo2,self.mpo1,T2,T1,R2,R1,RR2,RR1,Ta_21,Ta_12,Tw_21_inv,Tw_12_inv,Tw2_21_inv,Tw2_12_inv,envTol=1e-5,TDVP=False)

        gradB_hori1 = 1/2*(gradB_hori_121 + gradB_hori_211)
        gradB_hori2 = 1/2*(gradB_hori_122 + gradB_hori_212)
        gradB_vert1 = 1/2*(gradB_vert_121 + gradB_vert_211)
        gradB_vert2 = 1/2*(gradB_vert_122 + gradB_vert_212)
        gradB1 = gradB_hori1 + gradB_vert1
        gradB2 = gradB_hori2 + gradB_vert2

        if TDVP is True:
            Xa1 = project_mpsTangentVector(gradA1,self.mps1,T2)
            Xa2 = project_mpsTangentVector(gradA2,self.mps2,T1)
            Xb1 = project_mpoTangentVector(gradB1,self.mps1,self.mpo1,T2,R2)
            Xb2 = project_mpoTangentVector(gradB2,self.mps2,self.mpo2,T1,R1)
            print(np.real(np.einsum('ijk,ijk',Xa1,Xa1.conj())))
            print(np.real(np.einsum('ijk,ijk',Xa2,Xa2.conj())))
            print(np.real(np.einsum('ijab,ijab',Xb1,Xb1.conj())))
            print(np.real(np.einsum('ijab,ijab',Xb2,Xb2.conj())))
            A1 = self.mps1 - learningRate*Xa1
            A2 = self.mps2 - learningRate*Xa2
            B1 = self.mpo1 - learningRate*Xb1
            B2 = self.mpo2 - learningRate*Xb2
        else:
            A1 = self.mps1 - learningRate*gradA1
            A2 = self.mps2 - learningRate*gradA2
            B1 = self.mpo1 - learningRate*gradB1
            B2 = self.mpo2 - learningRate*gradB2

        #polar decomps to project to closest unitaries
        A1 = polarDecomp(A1.reshape(2*self.D_mps,self.D_mps)).reshape(2,self.D_mps,self.D_mps)
        A2 = polarDecomp(A2.reshape(2*self.D_mps,self.D_mps)).reshape(2,self.D_mps,self.D_mps)
        B1 = np.einsum('iajb->ijab',polarDecomp(np.einsum('ijab->iajb',B1).reshape(2*self.D_mpo,2*self.D_mpo)).reshape(2,self.D_mpo,2,self.D_mpo))
        B2 = np.einsum('iajb->ijab',polarDecomp(np.einsum('ijab->iajb',B2).reshape(2*self.D_mpo,2*self.D_mpo)).reshape(2,self.D_mpo,2,self.D_mpo))

        self.mps1 = A1
        self.mps2 = A2
        self.mpo1 = B1
        self.mpo2 = B2
