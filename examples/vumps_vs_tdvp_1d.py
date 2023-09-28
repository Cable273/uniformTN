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
from rw_functions import save_obj,load_obj

N_iter = 1000
beta = 0.5 #for vumps
learningRate = 0.1 #for gradDescent/TDVP
D = 2 #bond dimension

#ising
J = 1
hz = 0.4
X = np.array([[0,1],[1,0]])
Z = np.array([[-1,0],[0,1]])
H = localH([twoBodyH(J*np.kron(Z,Z)),oneBodyH(hz*X)])

np.random.seed(1)

from uniformTN_gs_solvers import vumps_1d,gradDescent

#vumps in centre gauge
psi_centre = uMPS_1d_centre(D)
psi_centre.randoInit()
psi_centre,eDensity = vumps_1d(psi_centre,H,N_iter,beta,stable_polar=True)

#example of gradDescent / tdvp in left gauge
psi_left = uMPS_1d_left(D)
psi_left.randoInit()
#gradDescent in left gauge (euclidean metric)
psi_left_euclid,eDensity_left_euclid = gradDescent(psi_left,H,N_iter,learningRate,projectionMetric = 'euclid')
#gradDescent in left gauge (tdvp metric) == imag time evolution with TDVP in left gauge
psi_left_tdvp,eDensity_left_tdvp = gradDescent(psi_left,H,N_iter,learningRate,projectionMetric = 'TDVP')

plt.plot(eDensity,label="VUMPS")
plt.plot(eDensity_left_euclid,label="Grad descent")
plt.plot(eDensity_left_tdvp,label="Imag time TDVP")
plt.xlabel("Iter")
plt.ylabel("E0")
plt.legend()
plt.show()
