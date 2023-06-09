#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from uniformTN_states import *
from uniformTN_transfers import mpsTransfer,mpsu1Transfer_left_oneLayer,mpsu1Transfer_left_twoLayer,mpsu1Transfer_left_twoLayerWithMpsInsert,inverseTransfer,mpsu1Transfer_left_threeLayerWithMpsInsert_lower,mpsu1Transfer_left_threeLayerWithMpsInsert_upper
from uniformTN_transfers import inverseTransfer,inverseTransfer_left
from uniformTN_gradients import grad_mpu_horizontal,grad_mpu_vertical,grad_mps_horizontal,grad_mps_vertical
from uniformTN_gs_solvers import gradDescent
from uniformTN_Hamiltonians import localH,twoBodyH,twoBodyH_hori,twoBodyH_vert


N_iter = 1000
learningRate = 0.1
tol = 1e-10
eDensity = np.zeros(N_iter)
D = 2
D_mps = D
D_mpo = D
J =  1

X = np.array([[0,1],[1,0]])
Y = np.array([[0,1j],[-1j,0]])
Z = np.array([[-1,0],[0,1]])
I = np.eye(2)
heisTerm = J/4*(np.kron(X,X)+np.kron(Y,Y)+np.kron(Z,Z))
# H = localH([twoBodyH_hori(heisTerm),twoBodyH_vert(heisTerm)])
H = localH([twoBodyH(heisTerm)])

# psi = uMPS_1d_left(D)
# psi = uMPS_1d_left_bipartite(D)
psi = uMPS_1d_left_twoSite(D)
# psi = uMPSU1_2d_left(D_mps,D_mpo)
# psi = uMPSU1_2d_left_bipartite(D_mps,D_mpo)
psi.randoInit()
psi,eDensity = gradDescent(psi,H,N_iter,learningRate,printE0=True,decay=0,tol=tol,envTol=1e-5,TDVP=False)

plt.plot(eDensity)
plt.xlabel("Iter")
plt.ylabel("Energy Density")
plt.suptitle(r"$H=J/4\sum_{<i,j> } X_i X_j + Y_i Y_j + Z_i Z_j$, $J=$"+str(J)+r", $D=$"+str(D)+"\n"+r"$\epsilon=$"+str(eDensity[np.size(eDensity)-1]))
plt.show()
