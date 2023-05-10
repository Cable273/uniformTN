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

from uniformTN_transfers import inverseTransfer_left,mpsTransfer
from uniformTN_states import *

def getAllTransfers(psi):
    if type(psi) is uMPS_1d:
        return transfers_1d(psi)

    elif type(psi) is uMPS_1d_left:
        return transfers_1d_left(psi)

    elif type(psi) is uMPSU1_2d_left:
        return transfers_2d_left(psi)

    elif type(psi) is uMPSU1_2d_left_bipartite:
        return transfers_2d_leftBip(psi)

class stateTransfers(ABC):
    @abstractmethod
    def get_fixedPoints(self):
        pass
    @abstractmethod
    def get_inverses(self):
        pass

class transfers_1d(stateTransfers):
    def __init__(self,psi):
        self.Ta = mpsTransfer(psi.mps)
    def get_fixedPoints(self):
        return fixedPoints_1d(self)
    def get_inverses(self,fixedPoints):
        return inverses_1d(self,fixedPoints)

class transfers_1d_left(transfers_1d):
    def get_fixedPoints(self):
        return fixedPoints_1d_left(self)
    def get_inverses(self,fixedPoints):
        return inverses_1d_left(self,fixedPoints)

class transfers_2d_left:
    def __init__(self,psi):
        self.Ta = mpsTransfer(psi.mps)
        T = self.Ta.findRightEig()
        T.norm_pairedCanon()
        self.Tb = mpsu1Transfer_left_oneLayer(psi.mps,psi.mpo,T)
        self.Tb2 = mpsu1Transfer_left_twoLayer(psi.mps,psi.mpo,T)
    def get_fixedPoints(self):
        return fixedPoints_2d_left(self)
    def get_inverses(self,fixedPoints):
        return inverses_2d_left(self,fixedPoints)

class fixedPoints_1d:
    def __init__(self,transfers):
        self.R = transfers.Ta.findRightEig()
        self.L = transfers.Ta.findLeftEig()
        self.R.norm_pairedVector(self.L.vector)
class fixedPoints_1d_left:
    def __init__(self,transfers):
        self.R = transfers.Ta.findRightEig()
        self.R.norm_pairedCanon()
class fixedPoints_2d_left:
    def __init__(self,transfers):
        self.T = transfers.Ta.findRightEig()
        self.R = transfers.Tb.findRightEig()
        self.RR = transfers.Tb2.findRightEig()
        self.T.norm_pairedCanon()
        self.R.norm_pairedCanon()
        self.RR.norm_pairedCanon()

class inverses_1d:
    def __init__(self,transfers,fixedPoints):
        self.Ta_inv = inverseTransfer(transfers.Ta,fixedPoints.L.vector,fixedPoints.R.vector)
class inverses_1d_left:
    def __init__(self,transfers,fixedPoints):
        self.Ta_inv = inverseTransfer_left(transfers.Ta,fixedPoints.R.vector)
class inverses_2d_left:
    def __init__(self,transfers,fixedPoints):
        self.Ta_inv = inverseTransfer_left(transfers.Ta,fixedPoints.T.vector)
        self.Tb_inv = inverseTransfer_left(transfers.Tb,fixedPoints.R.vector)
        self.Tb2_inv = inverseTransfer_left(transfers.Tb2,fixedPoints.RR.vector)
        self.Ta_inv.genInverse()
        self.Ta_inv.genTensor()
