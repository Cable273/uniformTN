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

# DMRG sweeping
def dmrg_sweep(psi,H,tol=1e-5):
    #right canon
    if psi.canonCentre is not psi.N-1:
        psi.canonForm_right()

    #build right environments
    expEnv_left = dict()
    expEnv_right = dict()
    pbar = ProgressBar()
    for n in pbar(range(psi.N-1,0,-1)):
        if n == psi.N-1:
            expEnv_right[n] = ncon([psi.tensors[n],H.tensors[n],psi.tensors[n].conj()],((1,-3),(2,1,-5),(2,-4)),forder=(-4,-5,-3))
        else:
            expEnv_right[n] = ncon([expEnv_right[n+1],psi.tensors[n],H.tensors[n],psi.tensors[n].conj()],((8,6,4),(1,-3,4),(2,1,-5,6),(2,-7,8)),forder=(-7,-5,-3),order=(8,6,4,1,2))

    energyCost = np.real(ncon([expEnv_right[1],psi.tensors[0],H.tensors[0],psi.tensors[0].conj()],((5,4,3),(1,3),(2,1,4),(2,5))))
    breaker = 0
    print(energyCost)

    counter = 1
    while breaker == 0:
        energyCostPrev = energyCost
        #left to right sweep
        D = expEnv_right[1].shape[0]
        H_eff = ncon([H.tensors[0],expEnv_right[1]],((-2,-1,4),(-5,4,-3)),forder=(-2,-5,-1,-3)).reshape(H.phsDim*D,H.phsDim*D)
        e,u = np.linalg.eigh(H_eff)
        psi.setTensor(0,u[:,0].reshape(psi.phsDim,D))
        psi.shiftCanonCentre('right')
        expEnv_left[0] = ncon([psi.tensors[0],H.tensors[0],psi.tensors[0].conj()],((1,-3),(2,1,-4),(2,-5)),forder=(-5,-4,-3))
        for m in range(1,psi.N-1):
            D_left = expEnv_left[m-1].shape[0]
            D_right = expEnv_right[m+1].shape[0]
            H_eff = ncon([expEnv_left[m-1],H.tensors[m],expEnv_right[m+1]],((-5,4,-3),(-2,-1,4,7),(-8,7,-6)),forder=(-2,-5,-8,-1,-3,-6)).reshape(H.phsDim*D_left*D_right,H.phsDim*D_left*D_right)
            e,u = np.linalg.eigh(H_eff)
            psi.setTensor(m,u[:,0].reshape(psi.phsDim,D_left,D_right))
            psi.shiftCanonCentre('right')
            expEnv_left[m] = ncon([expEnv_left[m-1],psi.tensors[m],H.tensors[m],psi.tensors[m].conj()],((5,4,3),(1,3,-6),(2,1,4,-7),(2,5,-8)),forder=(-8,-7,-6),order=(3,4,5,1,2))

        #right to left sweep
        m = psi.N-1
        D = expEnv_left[m-1].shape[0]
        H_eff = ncon([expEnv_left[m-1],H.tensors[m]],((-5,4,-3),(-2,-1,4)),forder=(-2,-5,-1,-3)).reshape(H.phsDim*D,H.phsDim*D)
        e,u = np.linalg.eigh(H_eff)
        psi.setTensor(m,u[:,0].reshape(psi.phsDim,D))
        psi.shiftCanonCentre('left')
        expEnv_right[m] = ncon([psi.tensors[m],H.tensors[m],psi.tensors[m].conj()],((1,-3),(2,1,-4),(2,-5)),forder=(-5,-4,-3))
        for m in range(psi.N-2,0,-1):
            D_left = expEnv_left[m-1].shape[0]
            D_right = expEnv_right[m+1].shape[0]
            H_eff = ncon([expEnv_left[m-1],H.tensors[m],expEnv_right[m+1]],((-5,4,-3),(-2,-1,4,7),(-8,7,-6)),forder=(-2,-5,-8,-1,-3,-6)).reshape(H.phsDim*D_left*D_right,H.phsDim*D_left*D_right)
            e,u = np.linalg.eigh(H_eff)
            psi.setTensor(m,u[:,0].reshape(psi.phsDim,D_left,D_right))
            psi.shiftCanonCentre('left')
            expEnv_right[m] = ncon([expEnv_right[m+1],psi.tensors[m],H.tensors[m],psi.tensors[m].conj()],((8,6,4),(1,-3,4),(2,1,-5,6),(2,-7,8)),forder=(-7,-5,-3),order=(4,6,8,1,2))

        energyCost = np.real(ncon([expEnv_right[1],psi.tensors[0],H.tensors[0],psi.tensors[0].conj()],((5,4,3),(1,3),(2,1,4),(2,5))))
        print(counter,energyCost)
        diff = np.abs(energyCost - energyCostPrev)
        counter += 1
        if diff < tol:
            breaker = 1
