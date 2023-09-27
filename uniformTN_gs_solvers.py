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

def vumps_1d(psi0,H,N_iter,beta,tol=1e-15,printE0=False,printDiff=False,stable_polar = True):
    psi = copy.deepcopy(psi0) #dont change state
    eDensity = np.zeros(N_iter)
    pbar=ProgressBar()

    #get original Ac,C,Al,Ar from A
    psi.get_transfers()
    psi.get_fixedPoints()
    psi.get_inverses()
    psi.get_canonForms_centre()
    for n in pbar(range(0,N_iter)):
        if n > 0:
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
        psi.vumps_update(H,beta,stable_polar = stable_polar)
    eDensity = eDensity[:n]
    return psi,eDensity
