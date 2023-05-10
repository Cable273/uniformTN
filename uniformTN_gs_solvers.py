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

from uniformTN_transfers import inverseTransfer_left,inverseTransfer,inverseTransfer_right
from uniformTN_gradients import grad_mps_1d_left
from uniformTN_gradients import *
from uniformTN_exp import exp_1d_2body_left,exp_1d_2body,exp_2d_2body_horizontal_leftBip,exp_2d_2body_vertical_leftBip
import copy

def gradDescent_1d_left(psi0,twoBodyH,N_iter,learningRate,decay=0,tol=1e-15,envTol=1e-15,printE0=False,printDiff=False,metric=False):
    psi = copy.deepcopy(psi0)
    eDensity = np.zeros(N_iter)
    pbar=ProgressBar()
    for n in pbar(range(0,N_iter)):
        # fixed points
        T = mpsTransfer(psi.mps)
        R = T.findRightEig()
        R.norm_pairedCanon()
        T_inv = inverseTransfer_left(T,R.vector)
        del T

        eDensity[n] = exp_1d_2body_left(twoBodyH,psi,R)
        if printE0 is True:
            print(eDensity[n])
        if n>1:
            diff = np.abs(eDensity[n]-eDensity[n-1])
            if printDiff is True:
                print(diff)
            if diff < tol:
                break

        if n > 1:
            learningRate = learningRate/(1+decay*n)
        psi.gradDescent(twoBodyH,R,T_inv,learningRate,metric=metric)
    eDensity = eDensity[:n]
    return psi,eDensity

def gradDescent_2d_left(psi0,twoBodyH,N_iter,learningRate,decay=0,tol=1e-15,envTol=1e-5,printE0=False,printDiff=False,TDVP=False):
    psi = copy.deepcopy(psi0)
    eDensity = np.zeros(N_iter)
    pbar=ProgressBar()
    for n in pbar(range(0,N_iter)):
        print("\n")
        if TDVP is True: #gauge transform for TDVP
            Ta = mpsTransfer(psi.mps)
            T = Ta.findRightEig()
            #gauge transform so diagonal in Z basis
            rho = np.einsum('ija,ujb,ba->ui',psi.mps,psi.mps.conj(),T.tensor)
            e,u =np.linalg.eig(rho)
            #now gauge transform so diagonal along diagonal in YZ plane
            theta = -math.pi/4
            u2 = np.array([[np.cos(theta/2),1j*np.sin(theta/2)],[1j*np.sin(theta/2),np.cos(theta/2)]])
            u = np.dot(u,u2)
            psi.mps = np.einsum('ijk,ia->ajk',psi.mps,u)
            psi.mpo = np.einsum('ijab,kj->ikab',psi.mpo,u.conj().transpose())
        # fixed points
        Ta = mpsTransfer(psi.mps)
        T = Ta.findRightEig()
        T.norm_pairedCanon()
        Tw = mpsu1Transfer_left_oneLayer(psi.mps,psi.mpo,T)
        R = Tw.findRightEig()
        R.norm_pairedCanon()
        Tw2 = mpsu1Transfer_left_twoLayer(psi.mps,psi.mpo,T)
        RR = Tw2.findRightEig()
        RR.norm_pairedCanon()
        e,u = np.linalg.eig(Ta.matrix)

        eDensity[n] = exp_2d_2body_horizontal_left(twoBodyH,psi,T,R)+exp_2d_2body_vertical_left(twoBodyH,psi,T,RR)
        if printE0 is True:
            print(eDensity[n])
        if n>1:
            diff = np.abs(eDensity[n]-eDensity[n-1])
            if printDiff is True:
                print(diff)
            if diff < tol:
                break
        if n > 1:
            learningRate = learningRate/(1+decay*n)

        psi.gradDescent(twoBodyH,Ta,Tw,Tw2,T,R,RR,learningRate,envTol=envTol,TDVP=TDVP)
        print(np.sort(np.abs(e)))

    eDensity = eDensity[:n]
    return psi,eDensity

def gradDescent_2d_leftBip(psi0,twoBodyH,N_iter,learningRate,decay=0,tol=1e-15,envTol=1e-5,printE0=False,printDiff=False,TDVP=False):
    psi = copy.deepcopy(psi0)
    eDensity = np.zeros(N_iter)
    pbar=ProgressBar()
    for n in pbar(range(0,N_iter)):
        print("\n")
        if TDVP is True: #gauge transform for TDVP
            Ta_12 = mpsTransferBip(psi.mps1,psi.mps2)
            Ta_21 = mpsTransferBip(psi.mps2,psi.mps1)
            T1 = Ta_12.findRightEig()
            T2 = Ta_21.findRightEig()
            T1.norm_pairedCanon()
            T2.norm_pairedCanon()

            #gauge transform so diagonal along diagonal in YZ plane
            theta = math.pi/4
            u2 = np.array([[np.cos(theta/2),1j*np.sin(theta/2)],[1j*np.sin(theta/2),np.cos(theta/2)]])

            #gauge transform so diagonal in Z basis
            rho = np.einsum('ija,ujb,ba->ui',psi.mps1,psi.mps1.conj(),T2.tensor)
            e,u =np.linalg.eig(rho)
            u = np.dot(u,u2)
            psi.mps1 = np.einsum('ijk,ia->ajk',psi.mps1,u)
            psi.mpo1 = np.einsum('ijab,kj->ikab',psi.mpo1,u.conj().transpose())

            #gauge transform so diagonal in Z basis
            rho = np.einsum('ija,ujb,ba->ui',psi.mps2,psi.mps2.conj(),T1.tensor)
            e,u =np.linalg.eig(rho)
            u = np.dot(u,u2)
            psi.mps2 = np.einsum('ijk,ia->ajk',psi.mps2,u)
            psi.mpo2 = np.einsum('ijab,kj->ikab',psi.mpo2,u.conj().transpose())


        #fixed points
        Ta_12 = mpsTransferBip(psi.mps1,psi.mps2)
        Ta_21 = mpsTransferBip(psi.mps2,psi.mps1)
        T1 = Ta_12.findRightEig()
        T2 = Ta_21.findRightEig()
        T1.norm_pairedCanon()
        T2.norm_pairedCanon()

        Tw_12 = mpsu1Transfer_left_oneLayerBip(psi.mps1,psi.mps2,psi.mpo1,psi.mpo2,T1,T2)
        Tw_21 = mpsu1Transfer_left_oneLayerBip(psi.mps2,psi.mps1,psi.mpo2,psi.mpo1,T2,T1)
        R1 = Tw_12.findRightEig()
        R2 = Tw_21.findRightEig()
        R1.norm_pairedCanon()
        R2.norm_pairedCanon()

        Tw2_12 = mpsu1Transfer_left_twoLayerBip(psi.mps1,psi.mps2,psi.mpo1,psi.mpo2,T1,T2)
        Tw2_21 = mpsu1Transfer_left_twoLayerBip(psi.mps2,psi.mps1,psi.mpo2,psi.mpo1,T2,T1)
        RR1 = Tw2_12.findRightEig()
        RR2 = Tw2_21.findRightEig()
        RR1.norm_pairedCanon()
        RR2.norm_pairedCanon()

        eDensity[n] = exp_2d_2body_horizontal_leftBip(twoBodyH,psi,T1,T2,R1,R2)+exp_2d_2body_vertical_leftBip(twoBodyH,psi,T1,T2,RR1,RR2)
        if printE0 is True:
            print(eDensity[n])
        if n>1:
            diff = np.abs(eDensity[n]-eDensity[n-1])
            if printDiff is True:
                print(diff)
            if diff < tol:
                break
        if n > 1:
            learningRate = learningRate/(1+decay*n)

        psi.gradDescent(twoBodyH,Ta_12,Ta_21,Tw_12,Tw_21,Tw2_12,Tw2_21,T1,T2,R1,R2,RR1,RR2,learningRate,envTol=1e-5,TDVP=TDVP)

    eDensity = eDensity[:n]
    return psi,eDensity

def gradDescent_1d_uniform(psi0,twoBodyH,N_iter,learningRate,decay=0,tol=1e-15,envTol=1e-15,printE0=False,printDiff=False):
    psi = copy.deepcopy(psi0)
    eDensity = np.zeros(N_iter)
    pbar=ProgressBar()
    for n in pbar(range(0,N_iter)):
        # fixed points
        T = mpsTransfer(psi.mps)
        R = T.findRightEig()
        L = T.findLeftEig()
        R.norm_pairedVector(L.vector)
        T_inv = inverseTransfer(T,L.vector,R.vector)
        del T

        eDensity[n] = exp_1d_2body(twoBodyH,psi,L,R)
        if printE0 is True:
            print(eDensity[n])
        if n>1:
            diff = np.abs(eDensity[n]-eDensity[n-1])
            if printDiff is True:
                print(diff)
            if diff < tol:
                break

        if n > 1:
            learningRate = learningRate/(1+decay*n)
        psi.gradDescent(twoBodyH,L,R,T_inv,learningRate)
    eDensity = eDensity[:n]
    return psi,eDensity

def construct_Hc(twoBodyH,psi,Hl,Hr):
    H = ncon([psi.Al,psi.Ar,twoBodyH,psi.Al.conj(),psi.Ar.conj()],((1,10,-5),(2,-6,7),(3,4,1,2),(3,10,-9),(4,-8,7)),forder=(-9,-8,-5,-6))
    H = H.astype(complex)
    H += ncon([Hl,np.eye(psi.D)],((-1,-2),(-4,-3)),forder=(-1,-4,-2,-3))
    H += ncon([np.eye(psi.D),Hr],((-1,-2),(-4,-3)),forder=(-1,-4,-2,-3))
    return H.reshape(psi.D**2,psi.D**2)
def construct_Hac(twoBodyH,psi,Hl,Hr):
    H = ncon([psi.Al,twoBodyH,psi.Al.conj(),np.eye(psi.D)],((1,6,-5),(3,-4,1,-2),(3,6,-7),(-8,-9)),forder=(-4,-7,-8,-2,-5,-9))
    H = H.astype(complex)
    H += ncon([psi.Ar,twoBodyH,psi.Ar.conj(),np.eye(psi.D)],((2,-5,6),(-3,4,-1,2),(4,-7,6),(-8,-9)),forder=(-3,-8,-7,-1,-9,-5))
    H += ncon([Hl,np.eye(2),np.eye(psi.D)],((-2,-1),(-4,-3),(-6,-5)),forder=(-4,-2,-6,-3,-1,-5))
    H += ncon([np.eye(psi.D),np.eye(2),Hr],((-2,-1),(-4,-3),(-6,-5)),forder=(-4,-2,-6,-3,-1,-5))
    return H.reshape(2*psi.D**2,2*psi.D**2)
def construct_Hl(twoBodyH,psi,Tl_inv):
    Hl = ncon([psi.Al,psi.Al,twoBodyH,psi.Al.conj(),psi.Al.conj()],((1,5,6),(2,6,-7),(3,4,1,2),(3,5,9),(4,9,-8)),forder=(-8,-7))
    Hl = Tl_inv.applyRight(Hl.reshape(psi.D**2)).reshape(psi.D,psi.D)
    return Hl
def construct_Hr(twoBodyH,psi,Tr_inv):
    Hr = ncon([psi.Ar,psi.Ar,twoBodyH,psi.Ar.conj(),psi.Ar.conj()],((1,-5,6),(2,6,7),(3,4,1,2),(3,-9,8),(4,8,7)),forder=(-9,-5))
    Hr = Tr_inv.applyLeft(Hr.reshape(psi.D**2)).reshape(psi.D,psi.D)
    return Hr
def vumps_1d(psi0,twoBodyH,N_iter,beta,tol=1e-15,printE0=False,printDiff=False):
    psi = copy.deepcopy(psi0)
    eDensity = np.zeros(N_iter)
    pbar=ProgressBar()
    for n in pbar(range(0,N_iter)):
        Tl = mpsTransfer(psi.Al)
        Tr = mpsTransfer(psi.Ar)
        #approximate left/right eig with previous
        L= fixedPoint(np.dot(psi.C.conj().transpose(),psi.C).reshape(psi.D**2),psi.D,2)
        R= fixedPoint(np.dot(psi.C,psi.C.conj().transpose()).reshape(psi.D**2),psi.D,2)

        eDensity[n] = exp_1d_2body_left(twoBodyH,psi,R)
        if printE0 is True:
            print(eDensity[n])
        if n>1:
            diff = np.abs(eDensity[n]-eDensity[n-1])
            if printDiff is True:
                print(diff)
            if diff < tol:
                break

        Tl_inv = inverseTransfer_left(Tl,R.vector)
        Tr_inv = inverseTransfer_right(Tr,L.vector)
        Hl = construct_Hl(twoBodyH,psi,Tl_inv)
        Hr = construct_Hr(twoBodyH,psi,Tr_inv)
        H_ac = construct_Hac(twoBodyH,psi,Hl,Hr)
        H_c = construct_Hc(twoBodyH,psi,Hl,Hr)

        # if n<2:
            # psi.vumps_update(beta,twoBodyH,H_ac,H_c,stable_polar = False)
        # else:
        psi.vumps_update(beta,twoBodyH,H_ac,H_c,stable_polar = True)
    eDensity = eDensity[:n]
    return psi,eDensity
