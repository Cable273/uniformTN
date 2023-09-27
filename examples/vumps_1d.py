#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from ncon import ncon

from uniformTN_states import *
from uniformTN_Hamiltonians import *
from uniformTN_transfers import mpsTransfer,mpsu1Transfer_left_oneLayer,mpsu1Transfer_left_twoLayer,mpsu1Transfer_left_twoLayerWithMpsInsert,inverseTransfer,mpsu1Transfer_left_threeLayerWithMpsInsert_lower,mpsu1Transfer_left_threeLayerWithMpsInsert_upper
from uniformTN_transfers import inverseTransfer,inverseTransfer_left
from rw_functions import save_obj,load_obj

N_iter = 1000
beta = 1
learningRate = 0.1
tol = 1e-14
eDensity = np.zeros(N_iter)
D = 2

np.random.seed(1)

#heis
J = 1
X = np.array([[0,1],[1,0]])
Z = np.array([[-1,0],[0,1]])
Y = np.array([[0,1j],[-1j,0]])
heis = J/4*(-np.kron(X,X)-np.kron(Z,Z)+np.kron(Y,Y))
H = localH([twoBodyH(heis)])

psi_left = uMPS_1d_left(D)
psi_left.randoInit()

psi_centre = uMPS_1d_centre(D)
psi_centre.mps = psi_left.mps

from uniformTN_gs_solvers import vumps_1d,gradDescent
#vumps in centre gauge
psi_centre,eDensity = vumps_1d(psi_centre,H,N_iter,beta,stable_polar=True)
#gradDescent in left gauge (euclidean metric)
psi_left_euclid,eDensity_left_euclid = gradDescent(psi_left,H,N_iter,learningRate,projectionMetric = 'euclid')
#gradDescent in left gauge (tdvp metric) == imag time evolution with TDVP 
psi_left_tdvp,eDensity_left_tdvp = gradDescent(psi_left,H,N_iter,learningRate,projectionMetric = 'TDVP')

plt.plot(eDensity,label="VUMPS")
plt.plot(eDensity_left_euclid,label="Grad descent")
plt.plot(eDensity_left_tdvp,label="Imag time TDVP")
plt.xlabel("Iter")
plt.ylabel("E0")
plt.legend()
plt.show()
