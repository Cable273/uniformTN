#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from uniformTN_states import uMPS_1d_left_bipartite,uMPSU1_2d_left_bipartite
from uniformTN_gs_solvers import gradDescent
from uniformTN_Hamiltonians import localH,twoBodyH,twoBodyH_hori,twoBodyH_vert

J = 1 #AF Heis
N_iter = 1000
learningRate = 0.1
D = 2
D_mps = D
D_mpo = D

#construct heisenberg local Hamiltonian
X = np.array([[0,1],[1,0]])
Y = np.array([[0,1j],[-1j,0]])
Z = np.array([[-1,0],[0,1]])
heisTerm = J/4*(np.kron(X,X)+np.kron(Y,Y)+np.kron(Z,Z))

#find gs of heis in 1d
H = localH([twoBodyH(heisTerm)]) #initialise H
#use two site unit cell, left canonical ansatz (better at small bond dimension)
psi = uMPS_1d_left_bipartite(D) #use bipartite ansatz for AF Hamiltonian
psi.randoInit()
psi,eDensity_1d = gradDescent(psi,H,N_iter,learningRate,printE0=True,projectionMetric="TDVP")
#projectionMetric = how to project to the tangent space during gradient descent
#projectionMetric=None (default) is no projection
#projectionMetric="euclid" is projection with euclidean metric
#projectionMetric="TDVP" is projection induced by imaginary time TDVP (therefore == TDVP)
#TDVP is preferred, but can be unstable due to ill conditioned inverses. Use euclid if TDVP is unstable.

print("\n")
#now find gs of heis in 2d
#H needs to account for heis interactions in both horizontal and vertical directions
H = localH([twoBodyH_hori(heisTerm),twoBodyH_vert(heisTerm)]) 
psi = uMPSU1_2d_left_bipartite(D_mps,D_mpo) #use bipartite ansatz for AF Hamiltonian
psi.randoInit()
psi,eDensity_2d = gradDescent(psi,H,N_iter,learningRate,printE0=True,projectionMetric="TDVP")

plt.plot(eDensity_1d,label="1d")
plt.plot(eDensity_2d,label="2d")
plt.legend()
plt.xlabel("Iter")
plt.ylabel("Energy Density")
plt.suptitle(r"$H=J/4\sum_{<i,j> } X_i X_j + Y_i Y_j + Z_i Z_j$, $J=$"+str(J)+r", $D=$"+str(D))
plt.show()
