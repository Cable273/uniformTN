#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from uniformTN_states import uMPSU1_2d_left,uMPS_1d_centre,uMPSU1_2d_left_bipartite
from uniformTN_transfers import mpsTransfer,mpsu1Transfer_left_oneLayer,mpsu1Transfer_left_twoLayer,mpsu1Transfer_left_twoLayerWithMpsInsert,inverseTransfer,mpsu1Transfer_left_threeLayerWithMpsInsert_lower,mpsu1Transfer_left_threeLayerWithMpsInsert_upper,mpsTransferBip,mpsu1Transfer_left_oneLayerBip,mpsu1Transfer_left_twoLayerBip
from uniformTN_transfers import inverseTransfer,inverseTransfer_left
from uniformTN_exp import exp_2d_2body_horizontal_leftBip,exp_2d_2body_vertical_leftBip
from uniformTN_gradients import grad_mpu_horizontal,grad_mpu_vertical,grad_mps_horizontal,grad_mps_vertical,grad_mps_horizontalBip,grad_mps_verticalBip,grad_mpu_horizontalBip,grad_mpu_verticalBip
from uniformTN_gs_solvers import gradDescent_1d_left,gradDescent_2d_left,gradDescent_2d_leftBip

N_iter = 1000
learningRate = 0.1
tol = 1e-10
D = 2
D_mps = D
D_mpo = D
J =  1

X = np.array([[0,1],[1,0]])
Y = np.array([[0,1j],[-1j,0]])
Z = np.array([[-1,0],[0,1]])
I = np.eye(2)
twoBodyH = J/4*(np.kron(X,X)+np.kron(Y,Y)+np.kron(Z,Z)).reshape(2,2,2,2)

psi = uMPSU1_2d_left_bipartite(D_mps,D_mpo)
psi.randoInit()
#init mpos as identity
B = np.einsum('ab,cd->cbad',np.eye(D_mpo),np.eye(D_mpo))
psi.mpo1 = B
psi.mpo2 = B

psi,eDensity = gradDescent_2d_leftBip(psi,twoBodyH,N_iter,learningRate,printE0=True,decay=0,tol=tol,envTol=1e-5,TDVP=False)
eDensity = np.append(eDensity0,eDensity)

plt.plot(eDensity)
plt.xlabel("Iter")
plt.ylabel("Energy Density")
plt.suptitle(r"$H=J/4\sum_{<i,j> } X_i X_j + Y_i Y_j + Z_i Z_j$, $J=$"+str(J)+r", $D=$"+str(D)+"\n"+r"$\epsilon=$"+str(eDensity[np.size(eDensity)-1]))
plt.show()

