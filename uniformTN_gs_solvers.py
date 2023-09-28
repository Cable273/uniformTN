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

from uniformTN_transfers import inverseTransfer_left,inverseTransfer,inverseTransfer_right
import copy

def gradDescent(psi0,H,N_iter,learningRate,decay=0,tol=1e-15,envTol=1e-15,printE0=False,printDiff=False,projectionMetric=None):
    psi = copy.deepcopy(psi0) #dont change state
    eDensity = np.zeros(N_iter)
    pbar=ProgressBar()
    for n in pbar(range(0,N_iter)):
        psi.get_transfers()
        psi.get_fixedPoints()
        psi.get_inverses()

        eDensity[n] = H.exp(psi)
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

        psi.gradDescent(H,learningRate,projectionMetric=projectionMetric)
    eDensity = eDensity[:n]
    return psi,eDensity

def vumps_1d(psi0,H,N_iter,beta,tol=1e-15,printE0=False,printDiff=False,stable_polar = True,stateConsistErrors=False):
    psi = copy.deepcopy(psi0) #dont change state
    eDensity = np.zeros(N_iter)
    leftError = np.zeros(N_iter)
    rightError = np.zeros(N_iter)

    pbar=ProgressBar()
    for n in pbar(range(0,N_iter)):
        #construct pseudo inverses for Al / Ar
        psi.state_left.get_transfers() #psi.state_left = uMPS_1d_left obj with mps = Al
        psi.state_left.get_fixedPoints()
        psi.state_left.get_inverses()

        psi.state_right.get_transfers() #psi.state_right = uMPS_1d_right obj with mps = Ar
        psi.state_right.get_fixedPoints()
        psi.state_right.get_inverses()

        #for energy, use expectation with state translationally invariant Al Al Al...
        #Because for intermediate steps, centre gauge state given by Ac,C,Al,Ar isn't consistent
        eDensity[n] = H.exp(psi.state_left)
        if printE0 is True:
            print(eDensity[n])
        if n>1:
            diff = np.abs(eDensity[n]-eDensity[n-1])
            if printDiff is True:
                print(diff)
            if diff < tol:
                break
        psi.vumps_update(H,beta,stable_polar = stable_polar)

        diff_left = np.einsum('ijk,ku->iju',psi.Al,psi.C)-psi.Ac
        diff_right = np.einsum('ijk,uj->iuk',psi.Ar,psi.C)-psi.Ac
        leftError[n] = np.real(np.einsum('ijk,ijk',diff_left,diff_left.conj()))
        rightError[n] = np.real(np.einsum('ijk,ijk',diff_right,diff_right.conj()))

    eDensity = eDensity[:n]
    leftError = leftError[:n]
    rightError = rightError[:n]
    if stateConsistErrors is False:
        return psi,eDensity
    else:
        return psi,eDensity,leftError,rightError
