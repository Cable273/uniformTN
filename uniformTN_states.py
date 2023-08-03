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
import copy

from uniformTN_transfers import *
from uniformTN_projectors import polarDecomp

from rw_functions import save_obj,load_obj

#for initialising states
def randoUnitary(d1,d2,real=False):
    if real is True:
        A = np.random.uniform(-1,1,(d1,d2)) 
    else:
        A = np.random.uniform(-1,1,(d1,d2)) + 1j*np.random.uniform(-1,1,(d1,d2))
    return polarDecomp(A)

class stateAnsatz(ABC):
    @abstractmethod
    def randoInit(self):
        pass
    @abstractmethod
    def shiftTensors(self):
        pass
    @abstractmethod
    def norm(self):
        pass
    @abstractmethod
    def get_transfers(self):
        pass
    @abstractmethod
    def get_fixedPoints(self):
        pass
    @abstractmethod
    def get_inverses(self):
        pass
    @abstractmethod
    def del_transfers(self):
        pass
    @abstractmethod
    def del_fixedPoints(self):
        pass
    @abstractmethod
    def del_inverses(self):
        pass

    @abstractmethod
    def save(self):
        pass
    @abstractmethod
    def load(self):
        pass

    def gradDescent(self,H,learningRate,projectionMetric=None):
        from uniformTN_gradEvaluaters import gradFactory
        #subtract current expectation value (so cost f=<H>/<psi|psi> => df/dx = d/dx <H-eI>)
        H_new = copy.deepcopy(H)
        for n in range(0,len(H_new.terms)):
            H_new.terms[n].tensor = H.terms[n].subtractExp(self)

        gradEvaluater = gradFactory(self,H_new)
        gradEvaluater.eval()

        if projectionMetric is not None:
            gradEvaluater.projectTangentSpace(projectionMetric)

        self.shiftTensors(-learningRate,gradEvaluater.grad)
        self.norm()

class uMPS_1d(stateAnsatz):
    def __init__(self,D,mps=None):
        self.mps = mps
        self.D = D
    def randoInit(self):
        #WLOG just initialise as left canonical (always possible due to gauge)
        self.mps = randoUnitary(2*self.D,self.D).reshape(2,self.D,self.D)
    def get_transfers(self):
        self.Ta = mpsTransfer(self.mps)
    def get_fixedPoints(self):
        self.R = self.Ta.findRightEig()
        self.L = self.Ta.findLeftEig()
        self.R.norm_pairedVector(self.L.vector)
    def get_inverses(self):
        self.Ta_inv = inverseTransfer(self.Ta,self.L.vector,self.R.vector)
    def del_transfers(self):
        del self.Ta
    def del_fixedPoints(self):
        del self.R
        del self.L
    def del_inverses(self):
        del self.Ta_inv
    def shiftTensors(self,coef,tensor):
        self.mps += coef*tensor
    def norm(self):
        T = mpsTransfer(self.mps)
        e,u = np.linalg.eig(T.matrix)
        e0 = e[np.argmax(np.abs(e))]
        self.mps = self.mps / np.sqrt(np.abs(e0))

    def save(self,filename):
        np.save(filename,self.mps)
    def load(self,filename):
        self.mps = np.load(filename)
        self.D = np.size(self.mps,axis=1)

class uMPS_1d_left(uMPS_1d):
    def randoInit(self):
        self.mps = randoUnitary(2*self.D,self.D).reshape(2,self.D,self.D)
    def get_fixedPoints(self):
        self.R = self.Ta.findRightEig()
        self.R.norm_pairedCanon()
    def get_inverses(self):
        self.Ta_inv = inverseTransfer_left(self.Ta,self.R.vector)
    def del_fixedPoints(self):
        del self.R
    def norm(self):
        self.mps = polarDecomp(self.mps.reshape(2*self.D,self.D)).reshape(2,self.D,self.D)

class uMPS_1d_left_twoSite(uMPS_1d_left):
    def randoInit(self):
        self.mps = randoUnitary(4*self.D,self.D).reshape(2,2,self.D,self.D)
    def get_transfers(self):
        self.Ta = mpsTransfer_twoSite(self.mps)
    def norm(self):
        self.mps = polarDecomp(self.mps.reshape(4*self.D,self.D)).reshape(2,2,self.D,self.D)
    def load(self,filename):
        super().load(filename)
        self.D = np.size(self.mps,axis=2)

class uMPS_1d_left_bipartite(uMPS_1d_left):
    def randoInit(self):
        self.mps = dict()
        self.mps[1] = randoUnitary(2*self.D,self.D).reshape(2,self.D,self.D)
        self.mps[2] = randoUnitary(2*self.D,self.D).reshape(2,self.D,self.D)
    def get_transfers(self):
        self.Ta = dict()
        self.Ta[1] = mpsTransferBip(self.mps[1],self.mps[2])
        self.Ta[2] = mpsTransferBip(self.mps[2],self.mps[1])
    def get_fixedPoints(self):
        self.R = dict()
        for n in range(1,len(self.Ta)+1):
            self.R[n] = self.Ta[n].findRightEig()
            self.R[n].norm_pairedCanon()
    def get_inverses(self):
        self.Ta_inv = dict()
        for n in range(1,len(self.Ta)+1):
            self.Ta_inv[n] = inverseTransfer_left(self.Ta[n],self.R[n].vector)
    def shiftTensors(self,coef,tensorDict):
        self.mps[1] += coef*tensorDict[1]
        self.mps[2] += coef*tensorDict[2]
    def norm(self):
        self.mps[1] = polarDecomp(self.mps[1].reshape(2*self.D,self.D)).reshape(2,self.D,self.D)
        self.mps[2] = polarDecomp(self.mps[2].reshape(2*self.D,self.D)).reshape(2,self.D,self.D)

    def save(self,filename):
        data = dict()
        data['mps1'] = self.mps[1]
        data['mps2'] = self.mps[2]
        save_obj(data,filename)
    def load(self,filename):
        data = load_obj(filename)
        self.mps[1] = data['mps1']
        self.mps[2] = data['mps2']
        self.D = np.size(self.mps[1],axis=1)

class uMPSU_2d(stateAnsatz):
    def __init__(self,D_mps,D_mpo,mps=None,mpo=None):
        self.mps= mps
        self.mpo= mpo
        self.D_mps = D_mps
        self.D_mpo = D_mpo
    def del_transfers(self):
        del self.Ta
        del self.Tb
        del self.Tb2
    def del_fixedPoints(self):
        del self.T
        del self.R
        del self.RR
    def del_inverses(self):
        del self.Ta_inv
        del self.Tb_inv
        del self.Tb2_inv
    def shiftTensors(self,coef,tensorDict):
        self.mps += coef*tensorDict['mps']
        self.mpo += coef*tensorDict['mpo']

    def save(self,filename):
        data = dict()
        data['mps'] = self.mps
        data['mpo'] = self.mpo
        save_obj(data,filename)
    def load(self,filename):
        data = load_obj(filename)
        self.mps = data['mps']
        self.mpo = data['mpo']
        self.D_mps = np.size(self.mps,axis=1)
        self.D_mpo = np.size(self.mpo,axis=2)
        
class uMPSU1_2d_left(uMPSU_2d):
    def randoInit(self):
        self.mps = randoUnitary(2*self.D_mps,self.D_mps).reshape(2,self.D_mps,self.D_mps)
        self.mpo = np.einsum('iajb->ijab',randoUnitary(2*self.D_mpo,2*self.D_mpo).reshape(2,self.D_mpo,2,self.D_mpo))
    def get_transfers(self):
        self.Ta = mpsTransfer(self.mps)
        T = self.Ta.findRightEig()
        T.norm_pairedCanon()
        self.Tb = mpsu1Transfer_left_oneLayer(self.mps,self.mpo,T)
        self.Tb2 = mpsu1Transfer_left_twoLayer(self.mps,self.mpo,T)
    def get_fixedPoints(self):
        self.T = self.Ta.findRightEig()
        self.R = self.Tb.findRightEig()
        self.RR = self.Tb2.findRightEig()
        self.T.norm_pairedCanon()
        self.R.norm_pairedCanon()
        self.RR.norm_pairedCanon()
    def get_inverses(self):
        self.Ta_inv = inverseTransfer_left(self.Ta,self.T.vector)
        self.Tb_inv = inverseTransfer_left(self.Tb,self.R.vector)
        self.Tb2_inv = inverseTransfer_left(self.Tb2,self.RR.vector)
        self.Ta_inv.genInverse()
        self.Ta_inv.genTensor()
    def norm(self):
        #polar decomp to ensure left canon still
        self.mps = polarDecomp(self.mps.reshape(2*self.D_mps,self.D_mps)).reshape(2,self.D_mps,self.D_mps)
        self.mpo = np.einsum('iajb->ijab',polarDecomp(np.einsum('ijab->iajb',self.mpo).reshape(2*self.D_mpo,2*self.D_mpo)).reshape(2,self.D_mpo,2,self.D_mpo))

class uMPSU1_2d_left_twoSite(uMPSU1_2d_left):
    def randoInit(self):
        self.mps = randoUnitary(4*self.D_mps,self.D_mps).reshape(2,2,self.D_mps,self.D_mps)
        self.mpo = np.einsum('iajb->ijab',randoUnitary(4*self.D_mpo,4*self.D_mpo).reshape(4,self.D_mpo,4,self.D_mpo)).reshape(2,2,2,2,self.D_mpo,self.D_mpo)
    def norm(self):
        #polar decomp to ensure left canon still
        self.mps = polarDecomp(self.mps.reshape(4*self.D_mps,self.D_mps)).reshape(2,2,self.D_mps,self.D_mps)
        self.mpo = np.einsum('iajb->ijab',polarDecomp(np.einsum('ijab->iajb',self.mpo.reshape(4,4,self.D_mpo,self.D_mpo)).reshape(4*self.D_mpo,4*self.D_mpo)).reshape(4,self.D_mpo,4,self.D_mpo)).reshape(2,2,2,2,self.D_mpo,self.D_mpo)

    def load(self,filename):
        super().load(filename)
        self.D_mps = np.size(self.mps,axis=2)
        self.D_mpo = np.size(self.mpo,axis=4)


class uMPSU1_2d_left_twoSite_staircase(uMPSU1_2d_left_twoSite):
    def get_transfers(self):
        self.Ta = mpsTransfer_twoSite(self.mps)
        T = self.Ta.findRightEig()
        T.norm_pairedCanon()
        self.Tb = mpsu1Transfer_left_oneLayer_twoSite_staircase(self.mps,self.mpo,T)
        self.Tb2 = mpsu1Transfer_left_twoLayer_twoSite_staircase(self.mps,self.mpo,T)
    def get_inverses(self):
        self.Ta_inv = inverseTransfer_left(self.Ta,self.T.vector)
        self.Tb_inv = inverseTransfer_left(self.Tb,self.R.vector)
        self.Tb2_inv = inverseTransfer_leftPlusTwoPhysical(self.Tb2,self.RR.vector)

class uMPSU1_2d_left_twoSite_square(uMPSU1_2d_left_twoSite):
    def get_transfers(self):
        self.Ta = mpsTransfer_twoSite(self.mps)
        self.Tb = dict()
        self.Tb2 = dict()
        T = self.Ta.findRightEig()
        T.norm_pairedCanon()
        #label refers to which side of the mps tensor is attached to the mpo underneath
        self.Tb['bot'] = mpsu1Transfer_left_oneLayer_twoSite_square(self.mps,self.mpo,T,"bot")
        self.Tb['top'] = mpsu1Transfer_left_oneLayer_twoSite_square(self.mps,self.mpo,T,"top")
        #label refers style of transfer (unit cells make a square vs prongs to either side)
        self.Tb2['square'] = mpsu1Transfer_left_twoLayer_twoSite_square(self.mps,self.mpo,T,"square")
        self.Tb2['prong'] = mpsu1Transfer_left_twoLayer_twoSite_square(self.mps,self.mpo,T,"prong")
    def get_fixedPoints(self):
        self.T = self.Ta.findRightEig()
        self.T.norm_pairedCanon()
        self.R = dict()
        self.RR = dict()
        self.R['bot'] = self.Tb['bot'].findRightEig()
        self.R['top'] = self.Tb['top'].findRightEig()
        self.RR['square'] = self.Tb2['square'].findRightEig()
        self.RR['prong'] = self.Tb2['prong'].findRightEig()
        self.R['bot'].norm_pairedCanon()
        self.R['top'].norm_pairedCanon()
        self.RR['square'].norm_pairedCanon()
        self.RR['prong'].norm_pairedCanon()
    def get_inverses(self):
        self.Ta_inv = inverseTransfer_left(self.Ta,self.T.vector)
        self.Tb_inv = dict()
        self.Tb2_inv = dict()
        self.Tb_inv['top'] = inverseTransfer_left(self.Tb['top'],self.R['top'].vector)
        self.Tb_inv['bot'] = inverseTransfer_left(self.Tb['bot'],self.R['bot'].vector)
        self.Tb2_inv['square'] = inverseTransfer_left(self.Tb2['square'],self.RR['square'].vector)
        self.Tb2_inv['prong'] = inverseTransfer_left(self.Tb2['prong'],self.RR['prong'].vector)

#multiple mps/mpo tensors (eg bipartite ansatz, 4 site sep etc)
class uMPSU1_2d_left_multipleSites(uMPSU_2d):
    def randoInit(self):
        self.mps = dict()
        self.mpo = dict()
        for n in range(1,self.noTensors+1):
            self.mps[n] = randoUnitary(2*self.D_mps,self.D_mps).reshape(2,self.D_mps,self.D_mps)
            self.mpo[n] = np.einsum('iajb->ijab',randoUnitary(2*self.D_mpo,2*self.D_mpo).reshape(2,self.D_mpo,2,self.D_mpo))
    def get_fixedPoints(self):
        self.T = dict()
        self.R = dict()
        self.RR = dict()
        for n in range(1,self.noTensors+1):
            self.T[n] = self.Ta[n].findRightEig()
            self.T[n].norm_pairedCanon()
            self.R[n] = self.Tb[n].findRightEig()
            self.R[n].norm_pairedCanon()
            self.RR[n] = self.Tb2[n].findRightEig()
            self.RR[n].norm_pairedCanon()
    def get_inverses(self):
        self.Ta_inv = dict()
        self.Tb_inv = dict()
        self.Tb2_inv = dict()
        for n in range(1,self.noTensors+1):
            self.Ta_inv[n] = inverseTransfer_left(self.Ta[n],self.T[n].vector)
            self.Tb_inv[n] = inverseTransfer_left(self.Tb[n],self.R[n].vector)
            self.Tb2_inv[n] = inverseTransfer_left(self.Tb2[n],self.RR[n].vector)
    def shiftTensors(self,coef,tensorDict):
        for n in range(1,self.noTensors+1):
            self.mps[n] += coef*tensorDict['mps'+str(n)]
            self.mpo[n] += coef*tensorDict['mpo'+str(n)]
    def norm(self):
        #polar decomps to project to closest unitaries
        for n in range(1,self.noTensors+1):
            self.mps[n] = polarDecomp(self.mps[n].reshape(2*self.D_mps,self.D_mps)).reshape(2,self.D_mps,self.D_mps)
            self.mpo[n] = np.einsum('iajb->ijab',polarDecomp(np.einsum('ijab->iajb',self.mpo[n]).reshape(2*self.D_mpo,2*self.D_mpo)).reshape(2,self.D_mpo,2,self.D_mpo))
    def save(self,filename):
        data = dict()
        for n in range(1,self.noTensors+1):
            data['mps'+str(n)] = self.mps[n]
            data['mpo'+str(n)] = self.mpo[n]
        save_obj(data,filename)
    def load(self,filename):
        data = load_obj(filename)
        for n in range(1,self.noTensors+1):
            self.mps[n] = data['mps'+str(n)]
            self.mpo[n] = data['mpo'+str(n)]
        self.D_mps = np.size(self.mps[1],axis=1)
        self.D_mpo = np.size(self.mpo[1],axis=2)

class uMPSU1_2d_left_bipartite(uMPSU1_2d_left_multipleSites):
    def __init__(self,D_mps,D_mpo,mps=None,mpo=None):
        self.noTensors = 2
        super().__init__(D_mps,D_mpo,mps=mps,mpo=mpo)
    def get_transfers(self):
        self.Ta = dict()
        self.Tb = dict()
        self.Tb2 = dict()
        self.Ta[1] = mpsTransferBip(self.mps[1],self.mps[2])
        self.Ta[2] = mpsTransferBip(self.mps[2],self.mps[1])
        T1 = self.Ta[1].findRightEig()
        T2 = self.Ta[2].findRightEig()
        T1.norm_pairedCanon()
        T2.norm_pairedCanon()
        self.Tb[1] = mpsu1Transfer_left_oneLayerBip(self.mps[1],self.mps[2],self.mpo[1],self.mpo[2],T1,T2)
        self.Tb[2] = mpsu1Transfer_left_oneLayerBip(self.mps[2],self.mps[1],self.mpo[2],self.mpo[1],T2,T1)
        self.Tb2[1] = mpsu1Transfer_left_twoLayerBip(self.mps[1],self.mps[2],self.mpo[1],self.mpo[2],T1,T2)
        self.Tb2[2] = mpsu1Transfer_left_twoLayerBip(self.mps[2],self.mps[1],self.mpo[2],self.mpo[1],T2,T1)

class uMPSU1_2d_left_fourSite_sep(uMPSU1_2d_left_multipleSites):
    def __init__(self,D_mps,D_mpo,mps=None,mpo=None):
        self.noTensors = 4
        super().__init__(D_mps,D_mpo,mps=mps,mpo=mpo)
    def get_transfers(self):
        self.Ta = dict()
        self.Tb = dict()
        self.Tb2 = dict()
        self.Ta[1] = mpsTransferBip(self.mps[1],self.mps[3])
        self.Ta[2] = mpsTransferBip(self.mps[2],self.mps[4])
        self.Ta[3] = mpsTransferBip(self.mps[3],self.mps[1])
        self.Ta[4] = mpsTransferBip(self.mps[4],self.mps[2])
        T = dict()
        for n in range(1,5):
            T[n] = self.Ta[n].findRightEig()
            T[n].norm_pairedCanon()
        self.Tb[1] = mpsu1Transfer_left_oneLayerBip(self.mps[1],self.mps[2],self.mpo[1],self.mpo[2],T[4],T[3])
        self.Tb[2] = mpsu1Transfer_left_oneLayerBip(self.mps[2],self.mps[1],self.mpo[2],self.mpo[1],T[3],T[4])
        self.Tb[3] = mpsu1Transfer_left_oneLayerBip(self.mps[3],self.mps[4],self.mpo[3],self.mpo[4],T[2],T[1])
        self.Tb[4] = mpsu1Transfer_left_oneLayerBip(self.mps[4],self.mps[3],self.mpo[4],self.mpo[3],T[1],T[2])

        self.Tb2[1] = mpsu1Transfer_left_twoLayer_fourSiteSep(self.mps[1],self.mps[2],self.mps[3],self.mps[4],self.mpo[1],self.mpo[2],self.mpo[3],self.mpo[4],T[1],T[2])
        self.Tb2[2] = mpsu1Transfer_left_twoLayer_fourSiteSep(self.mps[2],self.mps[1],self.mps[4],self.mps[3],self.mpo[2],self.mpo[1],self.mpo[4],self.mpo[3],T[2],T[1])
        self.Tb2[3] = mpsu1Transfer_left_twoLayer_fourSiteSep(self.mps[3],self.mps[4],self.mps[1],self.mps[2],self.mpo[3],self.mpo[4],self.mpo[1],self.mpo[2],T[3],T[4])
        self.Tb2[4] = mpsu1Transfer_left_twoLayer_fourSiteSep(self.mps[4],self.mps[3],self.mps[2],self.mps[1],self.mpo[4],self.mpo[3],self.mpo[2],self.mpo[1],T[4],T[3])

class uMPSU1_2d_left_fourSite_block(uMPSU1_2d_left):
    def randoInit(self):
        self.mps = randoUnitary(16*self.D_mps,self.D_mps).reshape(16,self.D_mps,self.D_mps)
        self.mpo = np.einsum('iajb->ijab',randoUnitary(16*self.D_mpo,16*self.D_mpo).reshape(16,self.D_mpo,16,self.D_mpo)).reshape(16,16,self.D_mpo,self.D_mpo)
    def norm(self):
        #polar decomp to ensure left canon still
        self.mps = polarDecomp(self.mps.reshape(16*self.D_mps,self.D_mps)).reshape(16,self.D_mps,self.D_mps)
        self.mpo = np.einsum('iajb->ijab',polarDecomp(np.einsum('ijab->iajb',self.mpo.reshape(16,16,self.D_mpo,self.D_mpo)).reshape(16*self.D_mpo,16*self.D_mpo)).reshape(16,self.D_mpo,16,self.D_mpo)).reshape(16,16,self.D_mpo,self.D_mpo)


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
