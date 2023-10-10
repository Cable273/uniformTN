#!/usr/bin/env python# -*- coding: utf-8 -*-

from __future__ import division
import math
import numpy as np
import scipy as sp
from ncon import ncon
from uniformTN_states import *
import copy

from uniformTN_Hamiltonians import *

class map2EffH(ABC):
    def map(self,H):
        if isinstance(H,localH):
            return self.mapMany(H)
        else:
            return self.mapSingle(H)

    def mapMany(self,localH):
        H_eff = self.mapSingle(localH.terms[0])
        for n in range(1,len(localH.terms)):
            H_eff += self.mapSingle(localH.terms[n])
        return H_eff

    def mapSingle(self,localH_term):
        if type(localH_term) == oneBodyH:
            return self.map_oneBodyH(localH_term)
        elif type(localH_term) == twoBodyH_hori:
            return self.map_twoBodyH_hori(localH_term)
        elif type(localH_term) == twoBodyH_vert:
            return self.map_twoBodyH_vert(localH_term)
        elif type(localH_term) == plaquetteH:
            return self.map_plaquetteH(localH_term)


class map2EffH_2d_blockingSites_rect(map2EffH):
    def __init__(self,shape):
        self.shape = shape

    def map_oneBodyH(self,localH_term):
        physDim = localH_term.tensor.shape[0]
        #blocking single row
        if self.shape[0] == 1 or self.shape[1] == 1: 
            length_of_line = np.max(self.shape)
            effH_tensor = blockingSingleRow_2d_effH_alongRow_oneSiteH(localH_term.tensor,length_of_line)
            return localH([oneBodyH(effH_tensor)])

        elif self.shape == [2,2]:
            effH_tensor = ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-1,-5),(-2,-6),(-3,-7),(-4,-8)),forder=(-1,-2,-3,-4,-5,-6,-7,-8))
            effH_tensor += ncon([np.eye(physDim),localH_term.tensor,np.eye(physDim),np.eye(physDim)],((-1,-5),(-2,-6),(-3,-7),(-4,-8)),forder=(-1,-2,-3,-4,-5,-6,-7,-8))
            effH_tensor += ncon([np.eye(physDim),np.eye(physDim),localH_term.tensor,np.eye(physDim)],((-1,-5),(-2,-6),(-3,-7),(-4,-8)),forder=(-1,-2,-3,-4,-5,-6,-7,-8))
            effH_tensor += ncon([np.eye(physDim),np.eye(physDim),np.eye(physDim),localH_term.tensor],((-1,-5),(-2,-6),(-3,-7),(-4,-8)),forder=(-1,-2,-3,-4,-5,-6,-7,-8))
            return localH([oneBodyH(1/4*effH_tensor.reshape(physDim**4,physDim**4))])

        elif self.shape == [2,3]:
            effH_tensor = ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-1,-7),(-2,-8),(-3,-9),(-4,-10),(-5,-11),(-6,-12)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12))
            effH_tensor += ncon([np.eye(physDim),localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-1,-7),(-2,-8),(-3,-9),(-4,-10),(-5,-11),(-6,-12)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12))
            effH_tensor += ncon([np.eye(physDim),np.eye(physDim),localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-1,-7),(-2,-8),(-3,-9),(-4,-10),(-5,-11),(-6,-12)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12))
            effH_tensor += ncon([np.eye(physDim),np.eye(physDim),np.eye(physDim),localH_term.tensor,np.eye(physDim),np.eye(physDim)],((-1,-7),(-2,-8),(-3,-9),(-4,-10),(-5,-11),(-6,-12)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12))
            effH_tensor += ncon([np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),localH_term.tensor,np.eye(physDim)],((-1,-7),(-2,-8),(-3,-9),(-4,-10),(-5,-11),(-6,-12)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12))
            effH_tensor += ncon([np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),localH_term.tensor],((-1,-7),(-2,-8),(-3,-9),(-4,-10),(-5,-11),(-6,-12)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12))
            return localH([oneBodyH(1/6*effH_tensor.reshape(physDim**6,physDim**6))])


    def map_twoBodyH_hori(self,localH_term):
        physDim = localH_term.tensor.shape[0]

        #blocking single row
        if self.shape[0] == 1: #horizontal row
            length_of_line = np.max(self.shape)
            effH_oneSite = blockingSingleRow_2d_effH_alongRow_oneSiteH(localH_term.tensor,length_of_line)
            effH_twoSite = blockingSingleRow_2d_effH_alongRow_twoSiteH(localH_term.tensor,length_of_line)
            return localH([oneBodyH(effH_oneSite),twoBodyH_hori(effH_twoSite)])
        elif self.shape[1] == 1: #vertical row
            length_of_line = np.max(self.shape)
            effH_twoSite = blockingSingleRow_2d_effH_perpRow_twoSiteH(localH_term.tensor,length_of_line)
            return localH([twoBodyH_hori(effH_twoSite)])

        elif self.shape == [2,2]:
            # effH_oneSite = ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim)],((-1,-2,-5,-6),(-3,-7),(-4,-8)),forder=(-1,-2,-3,-4,-5,-6,-7,-8)).reshape(physDim**4,physDim**4)
            # effH_oneSite += ncon([np.eye(physDim),np.eye(physDim),localH_term.tensor],((-1,-5),(-2,-6),(-3,-4,-7,-8)),forder=(-1,-2,-3,-4,-5,-6,-7,-8)).reshape(physDim**4,physDim**4)
            effH_oneSite = ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim)],((-1,-3,-5,-7),(-2,-6),(-4,-8)),forder=(-1,-2,-3,-4,-5,-6,-7,-8)).reshape(physDim**4,physDim**4)
            effH_oneSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim)],((-2,-4,-6,-8),(-1,-5),(-3,-7)),forder=(-1,-2,-3,-4,-5,-6,-7,-8)).reshape(physDim**4,physDim**4)
            effH_oneSite = effH_oneSite/4

            # effH_twoSite = ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-2,-5,-10,-13),(-1,-9),(-3,-11),(-4,-12),(-6,-14),(-7,-15),(-8,-16)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16)).reshape(physDim**8,physDim**8)
            # effH_twoSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-4,-7,-12,-15),(-1,-9),(-2,-10),(-3,-11),(-5,-13),(-6,-14),(-8,-16)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16)).reshape(physDim**8,physDim**8)
            effH_twoSite = ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-3,-5,-11,-13),(-1,-9),(-2,-10),(-4,-12),(-6,-14),(-7,-15),(-8,-16)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16)).reshape(physDim**8,physDim**8)
            effH_twoSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-4,-6,-12,-14),(-1,-9),(-2,-10),(-3,-11),(-5,-13),(-7,-15),(-8,-16)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16)).reshape(physDim**8,physDim**8)
            effH_twoSite = effH_twoSite/4
            return localH([oneBodyH(effH_oneSite),twoBodyH_hori(effH_twoSite)])

        elif self.shape == [2,3]:
            effH_oneSite = ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-1,-3,-7,-9),(-2,-8),(-4,-10),(-5,-11),(-6,-12)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12)).reshape(physDim**6,physDim**6)
            effH_oneSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-3,-5,-9,-11),(-1,-7),(-2,-8),(-4,-10),(-6,-12)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12)).reshape(physDim**6,physDim**6)
            effH_oneSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-2,-4,-8,-10),(-1,-7),(-3,-9),(-5,-11),(-6,-12)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12)).reshape(physDim**6,physDim**6)
            effH_oneSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-4,-6,-10,-12),(-1,-7),(-2,-8),(-3,-9),(-5,-11)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12)).reshape(physDim**6,physDim**6)
            effH_oneSite = effH_oneSite / 6

            effH_twoSite = ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-5,-7,-17,-19),(-1,-13),(-2,-14),(-3,-15),(-4,-16),(-6,-18),(-8,-20),(-9,-21),(-10,-22),(-11,-23),(-12,-24)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24)).reshape(physDim**12,physDim**12)
            effH_twoSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-6,-8,-18,-20),(-1,-13),(-2,-14),(-3,-15),(-4,-16),(-5,-17),(-7,-19),(-9,-21),(-10,-22),(-11,-23),(-12,-24)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24)).reshape(physDim**12,physDim**12)
            effH_twoSite = effH_twoSite / 6
            return localH([oneBodyH(effH_oneSite),twoBodyH_hori(effH_twoSite)])

        elif self.shape == [2,4]:
            effH_oneSite = ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-1,-3,-9,-11),(-2,-10),(-4,-12),(-5,-13),(-6,-14),(-7,-15),(-8,-16)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16)).reshape(physDim**8,physDim**8)
            effH_oneSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-3,-5,-11,-13),(-1,-9),(-2,-10),(-4,-12),(-6,-14),(-7,-15),(-8,-16)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16)).reshape(physDim**8,physDim**8)
            effH_oneSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-5,-7,-13,-15),(-1,-9),(-2,-10),(-3,-11),(-4,-12),(-6,-14),(-8,-16)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16)).reshape(physDim**8,physDim**8)
            effH_oneSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-2,-4,-10,-12),(-1,-9),(-3,-11),(-5,-13),(-6,-14),(-7,-15),(-8,-16)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16)).reshape(physDim**8,physDim**8)
            effH_oneSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-4,-6,-12,-14),(-1,-9),(-2,-10),(-3,-11),(-5,-13),(-7,-15),(-8,-16)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16)).reshape(physDim**8,physDim**8)
            effH_oneSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-6,-8,-14,-16),(-1,-9),(-2,-10),(-3,-11),(-4,-12),(-5,-13),(-7,-15)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16)).reshape(physDim**8,physDim**8)
            effH_oneSite = effH_oneSite / 8

            effH_twoSite = ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-7,-9,-23,-25),(-1,-17),(-2,-18),(-3,-19),(-4,-20),(-5,-21),(-6,-22),(-8,-24),(-10,-26),(-11,-27),(-12,-28),(-13,-29),(-14,-30),(-15,-31),(-16,-32)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29,-30,-31,-32)).reshape(physDim**16,physDim**16)
            effH_twoSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-8,-10,-24,-26),(-1,-17),(-2,-18),(-3,-19),(-4,-20),(-5,-21),(-6,-22),(-7,-23),(-9,-25),(-11,-27),(-12,-28),(-13,-29),(-14,-30),(-15,-31),(-16,-32)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29,-30,-31,-32)).reshape(physDim**16,physDim**16)
            effH_twoSite = effH_twoSite / 8

        elif self.shape == [3,2]:
            effH_oneSite = ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-1,-4,-7,-10),(-2,-8),(-3,-9),(-5,-11),(-6,-12)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12)).reshape(physDim**6,physDim**6)
            effH_oneSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-2,-5,-8,-11),(-1,-7),(-3,-9),(-4,-10),(-6,-12)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12)).reshape(physDim**6,physDim**6)
            effH_oneSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-3,-6,-9,-12),(-1,-7),(-2,-8),(-4,-10),(-5,-11)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12)).reshape(physDim**6,physDim**6)
            effH_oneSite = effH_oneSite / 6

            effH_twoSite = ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-4,-7,-16,-19),(-1,-13),(-2,-14),(-3,-15),(-5,-17),(-6,-18),(-8,-20),(-9,-21),(-10,-22),(-11,-23),(-12,-24)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24)).reshape(physDim**12,physDim**12)
            effH_twoSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-5,-8,-17,-20),(-1,-13),(-2,-14),(-3,-15),(-4,-16),(-6,-18),(-7,-19),(-9,-21),(-10,-22),(-11,-23),(-12,-24)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24)).reshape(physDim**12,physDim**12)
            effH_twoSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-6,-9,-18,-21),(-1,-13),(-2,-14),(-3,-15),(-4,-16),(-5,-17),(-7,-19),(-8,-20),(-10,-22),(-11,-23),(-12,-24)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24)).reshape(physDim**12,physDim**12)
            effH_twoSite = effH_twoSite / 6

            return localH([oneBodyH(effH_oneSite),twoBodyH_hori(effH_twoSite)])

    def map_twoBodyH_vert(self,localH_term):
        physDim = localH_term.tensor.shape[0]
        if self.shape[0] == 1:  #Horizontal row
            length_of_line = np.max(self.shape)
            effH_twoSite = blockingSingleRow_2d_effH_perpRow_twoSiteH(localH_term.tensor,length_of_line)
            return localH([twoBodyH_vert(effH_twoSite)])
        elif self.shape[1] == 1: #vertical row
            length_of_line = np.max(self.shape)
            effH_oneSite = blockingSingleRow_2d_effH_alongRow_oneSiteH(localH_term.tensor,length_of_line)
            effH_twoSite = blockingSingleRow_2d_effH_alongRow_twoSiteH(localH_term.tensor,length_of_line)
            return localH([oneBodyH(effH_oneSite),twoBodyH_vert(effH_twoSite)])

        elif self.shape == [2,2]:
            # effH_oneSite = ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim)],((-1,-3,-5,-7),(-2,-6),(-4,-8)),forder=(-1,-2,-3,-4,-5,-6,-7,-8)).reshape(physDim**4,physDim**4)
            # effH_oneSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim)],((-2,-4,-6,-8),(-1,-5),(-3,-7)),forder=(-1,-2,-3,-4,-5,-6,-7,-8)).reshape(physDim**4,physDim**4)
            effH_oneSite = ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim)],((-1,-2,-5,-6),(-3,-7),(-4,-8)),forder=(-1,-2,-3,-4,-5,-6,-7,-8)).reshape(physDim**4,physDim**4)
            effH_oneSite += ncon([np.eye(physDim),np.eye(physDim),localH_term.tensor],((-1,-5),(-2,-6),(-3,-4,-7,-8)),forder=(-1,-2,-3,-4,-5,-6,-7,-8)).reshape(physDim**4,physDim**4)
            effH_oneSite = effH_oneSite/4

            # effH_twoSite = ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-3,-5,-11,-13),(-1,-9),(-2,-10),(-4,-12),(-6,-14),(-7,-15),(-8,-16)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16)).reshape(physDim**8,physDim**8)
            # effH_twoSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-4,-6,-12,-14),(-1,-9),(-2,-10),(-3,-11),(-5,-13),(-7,-15),(-8,-16)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16)).reshape(physDim**8,physDim**8)
            effH_twoSite = ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-2,-5,-10,-13),(-1,-9),(-3,-11),(-4,-12),(-6,-14),(-7,-15),(-8,-16)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16)).reshape(physDim**8,physDim**8)
            effH_twoSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-4,-7,-12,-15),(-1,-9),(-2,-10),(-3,-11),(-5,-13),(-6,-14),(-8,-16)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16)).reshape(physDim**8,physDim**8)
            effH_twoSite = effH_twoSite/4
            return localH([oneBodyH(effH_oneSite),twoBodyH_vert(effH_twoSite)])

        elif self.shape == [2,3]:
            effH_oneSite = ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-1,-2,-7,-8),(-3,-9),(-4,-10),(-5,-11),(-6,-12)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12)).reshape(physDim**6,physDim**6)
            effH_oneSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-3,-4,-9,-10),(-1,-7),(-2,-8),(-5,-11),(-6,-12)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12)).reshape(physDim**6,physDim**6)
            effH_oneSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-5,-6,-11,-12),(-1,-7),(-2,-8),(-3,-9),(-4,-10)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12)).reshape(physDim**6,physDim**6)
            effH_oneSite = effH_oneSite / 6

            effH_twoSite = ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-2,-7,-14,-19),(-1,-13),(-3,-15),(-4,-16),(-5,-17),(-6,-18),(-8,-20),(-9,-21),(-10,-22),(-11,-23),(-12,-24)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24)).reshape(physDim**12,physDim**12)
            effH_twoSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-4,-9,-16,-21),(-1,-13),(-2,-14),(-3,-15),(-5,-17),(-6,-18),(-7,-19),(-8,-20),(-10,-22),(-11,-23),(-12,-24)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24)).reshape(physDim**12,physDim**12)
            effH_twoSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-6,-11,-18,-23),(-1,-13),(-2,-14),(-3,-15),(-4,-16),(-5,-17),(-7,-19),(-8,-20),(-9,-21),(-10,-22),(-12,-24)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24)).reshape(physDim**12,physDim**12)
            effH_twoSite = effH_twoSite / 6
            return localH([oneBodyH(effH_oneSite),twoBodyH_vert(effH_twoSite)])

        elif self.shape == [2,4]:
            effH_oneSite = ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-1,-2,-9,-10),(-3,-11),(-4,-12),(-5,-13),(-6,-14),(-7,-15),(-8,-16)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16)).reshape(physDim**8,physDim**8)
            effH_oneSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-3,-4,-11,-12),(-1,-9),(-2,-10),(-5,-13),(-6,-14),(-7,-15),(-8,-16)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16)).reshape(physDim**8,physDim**8)
            effH_oneSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-5,-6,-13,-14),(-1,-9),(-2,-10),(-3,-11),(-4,-12),(-7,-15),(-8,-16)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16)).reshape(physDim**8,physDim**8)
            effH_oneSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-7,-8,-15,-16),(-1,-9),(-2,-10),(-3,-11),(-4,-12),(-5,-13),(-6,-14)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16)).reshape(physDim**8,physDim**8)
            effH_oneSite = effH_oneSite / 8

            effH_twoSite = ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-2,-9,-18,-25),(-1,-17),(-3-19),(-4,-20),(-5,-21),(-6,-22),(-7,-23),(-8,-24),(-10,-26),(-11,-27),(-12,-28),(-13,-29),(-14,-30),(-15,-31),(-16,-32)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29,-30,-31,-32)).reshape(physDim**16,physDim**16)
            effH_twoSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-4,-11,-20,-27),(-1,-17),(-2,-18),(-3,-19),(-5,-21),(-6,-22),(-7,-23),(-8,-24),(-9,-25),(-10,-26),(-12,-28),(-13,-29),(-14,-30),(-15,-31),(-16,-32)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29,-30,-31,-32)).reshape(physDim**16,physDim**16)
            effH_twoSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-6,-13,-22,-29),(-1,-17),(-2,-18),(-3,-19),(-4,-20),(-5,-21),(-7,-23),(-8,-24),(-9,-25),(-10,-26),(-11,-27),(-12,-28),(-14,-30),(-15,-31),(-16,-32)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29,-30,-31,-32)).reshape(physDim**16,physDim**16)
            effH_twoSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-8,-15,-24,-31),(-1,-17),(-2,-18),(-3,-19),(-4,-20),(-5,-21),(-6,-22),(-7,-23),(-9,-25),(-10,-26),(-11,-27),(-12,-28),(-13,-29),(-14,-30),(-16,-32)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24,-25,-26,-27,-28,-29,-30,-31,-32)).reshape(physDim**16,physDim**16)
            effH_twoSite = effH_twoSite / 8

            return localH([oneBodyH(effH_oneSite),twoBodyH_vert(effH_twoSite)])

        elif self.shape == [3,2]:
            effH_oneSite = ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-1,-2,-7,-8),(-3,-9),(-4,-10),(-5,-11),(-6,-12)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12)).reshape(physDim**6,physDim**6)
            effH_oneSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-2,-3,-8,-9),(-1,-7),(-4,-10),(-5,-11),(-6,-12)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12)).reshape(physDim**6,physDim**6)
            effH_oneSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-4,-5,-10,-11),(-1,-7),(-2,-8),(-3,-9),(-6,-12)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12)).reshape(physDim**6,physDim**6)
            effH_oneSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-5,-6,-11,-12),(-1,-7),(-2,-8),(-3,-9),(-4,-10)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12)).reshape(physDim**6,physDim**6)
            effH_oneSite = effH_oneSite / 6

            effH_twoSite = ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-3,-7,-15,-19),(-1,-13),(-2,-14),(-4,-16),(-5,-17),(-6,-18),(-8,-20),(-9,-21),(-10,-22),(-11,-23),(-12,-24)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24)).reshape(physDim**12,physDim**12)
            effH_twoSite += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-6,-10,-18,-22),(-1,-13),(-2,-14),(-3,-15),(-4,-16),(-5,-17),(-7,-19),(-8,-20),(-9,-21),(-11,-23),(-12,-24)),forder=(-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12,-13,-14,-15,-16,-17,-18,-19,-20,-21,-22,-23,-24)).reshape(physDim**12,physDim**12)
            effH_twoSite = effH_twoSite / 6

            return localH([oneBodyH(effH_oneSite),twoBodyH_vert(effH_twoSite)])

    def map_plaquetteH(self,localH_term):
        physDim = localH_term.tensor.shape[0]
        if self.shape == [1,2]:
            effH_twoBodyH_vert = ncon([localH_term.tensor],((-1,-2,-3,-4,-5,-6,-7,-8)),forder=(-1,-3,-2,-4,-5,-7,-6,-8)).reshape(physDim**4,physDim**4)
            effH_plaquetteH = ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-1,-2,-3,-4,-5,-6,-7,-8),(-9,-10),(-11,-12),(-13,-14),(-15,-16)),forder=(-9,-1,-11,-2,-3,-13,-4,-15,-10,-5,-12,-6,-7,-14,-8,-16)).reshape(physDim**8,physDim**8)
            effH_twoBodyH_vert = effH_twoBodyH_vert / 2
            effH_plaquetteH = effH_plaquetteH / 2
            return localH([twoBodyH_vert(effH_twoBodyH_vert),plaquetteH(effH_plaquetteH)])

        elif self.shape == [1,3]:
            effH_twoBodyH_vert = ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim)],((-1,-2,-3,-4,-5,-6,-7,-8),(-9,-10),(-11,-12)),forder=(-1,-3,-9,-2,-4,-11,-5,-7,-10,-6,-8,-12)).reshape(physDim**6,physDim**6)
            effH_twoBodyH_vert += ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim)],((-1,-2,-3,-4,-5,-6,-7,-8),(-9,-10),(-11,-12)),forder=(-9,-1,-3,-11,-2,-4,-10,-5,-7,-12,-6,-8)).reshape(physDim**6,physDim**6)
            effH_plaquetteH = ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-1,-2,-3,-4,-5,-6,-7,-8),(-9,-10),(-11,-12),(-13,-14),(-15,-16),(-17,-18),(-19,-20),(-21,-22),(-23,-24)),forder=(-9,-11,-1,-13,-15,-2,-3,-17,-19,-4,-21,-23,-10,-12,-5,-14,-16,-6,-7,-18,-20,-8,-22,-24)).reshape(physDim**12,physDim**12)
            effH_twoBodyH_vert = effH_twoBodyH_vert / 3
            effH_plaquetteH = effH_plaquetteH / 3
            return localH([twoBodyH_vert(effH_twoBodyH_vert),plaquetteH(effH_plaquetteH)])

        elif self.shape == [2,1]:
            effH_twoBodyH_hori = ncon([localH_term.tensor],((-1,-2,-3,-4,-5,-6,-7,-8)),forder=(-1,-2,-3,-4,-5,-6,-7,-8)).reshape(physDim**4,physDim**4)
            effH_plaquetteH = ncon([localH_term.tensor,np.eye(physDim),np.eye(physDim),np.eye(physDim),np.eye(physDim)],((-1,-2,-3,-4,-5,-6,-7,-8),(-9,-10),(-11,-12),(-13,-14),(-15,-16)),forder=(-9,-1,-2,-11,-13,-3,-4,-15,-10,-5,-6,-12,-14,-7,-8,-16)).reshape(physDim**8,physDim**8)
            effH_twoBodyH_hori = effH_twoBodyH_hori / 2
            effH_plaquetteH = effH_plaquetteH / 2
            return localH([twoBodyH_hori(effH_twoBodyH_hori),plaquetteH(effH_plaquetteH)])

#generic code for blocking a single row of arbitrary length
def blockingSingleRow_2d_effH_alongRow_oneSiteH(tensor,length_of_line):
        lenTensor = int(np.size(tensor.shape)/2)
        physDim = tensor.shape[0]

        bot_indices = -np.arange(1,2*length_of_line,2)
        top_indices = bot_indices - 1
        stacked_indices = np.vstack((bot_indices,top_indices))
        labels = [np.array(stacked_indices[:,:lenTensor].flatten())]
        identityLabels = list(stacked_indices[:,lenTensor:].transpose())
        for n in range(0,len(identityLabels)):
            labels.append(identityLabels[n])

        tensorList = [tensor]
        for n in range(0,length_of_line-lenTensor):
            tensorList.append(np.eye(physDim))

        tensorIndex = np.append([0],np.ones(length_of_line-lenTensor)).astype(int)
        from sympy.utilities.iterables import multiset_permutations
        tensorIndex_perms = list(multiset_permutations(tensorIndex))
            
        #change forder to account for different positions of H
        forder = np.append(bot_indices,top_indices)
        effH_oneSite = ncon(tensorList,labels,forder=forder)/length_of_line
        for n in range(1,len(tensorIndex_perms)):
            forder = []
            c = lenTensor
            for m in range(0,np.size(tensorIndex_perms[n],axis=0)):
                if tensorIndex_perms[n][m] == 0:
                    for k in range(0,lenTensor):
                        forder.append(bot_indices[k])
                else:
                    forder.append(bot_indices[c])
                    c += 1
            forder = np.append(np.array(forder),np.array(forder)-1)
            effH_oneSite += ncon(tensorList,labels,forder=forder)/length_of_line
        effH_oneSite = effH_oneSite.reshape(physDim**length_of_line,physDim**length_of_line)
        return effH_oneSite

def blockingSingleRow_2d_effH_alongRow_twoSiteH(tensor,length_of_line):
        lenTensor = int(np.size(tensor.shape)/2)
        physDim = tensor.shape[0]

        tensorList = [tensor]
        for n in range(0,2*length_of_line-2):
            tensorList.append(np.eye(physDim))

        bot_indices = -np.arange(1,2*length_of_line+1)
        top_indices = bot_indices - 2*length_of_line
        stacked_indices = np.vstack((bot_indices,top_indices))
        tensorlabels = [np.array(stacked_indices[:,:lenTensor].flatten())]
        identityLabels = list(stacked_indices[:,lenTensor:].transpose())
        labels = []
        labels.append(tensorlabels[0])
        for n in range(0,len(identityLabels)):
            labels.append(identityLabels[n])
        forder_bot = []
        forder_top = []
        c = 0
        for n in range(0,length_of_line-1):
            forder_bot.append(identityLabels[c][0])
            forder_top.append(identityLabels[c][1])
            c+=1
        for n in range(0,lenTensor):
            forder_bot.append(tensorlabels[0][n])
            forder_top.append(tensorlabels[0][n+lenTensor])
        for n in range(0,length_of_line-1):
            forder_bot.append(identityLabels[c][0])
            forder_top.append(identityLabels[c][1])
            c+=1
        forder = np.append(forder_bot,forder_top)
        effH_twoSite = ncon(tensorList,labels,forder=forder).reshape((physDim**length_of_line)**2,(physDim**length_of_line)**2)/length_of_line
        return effH_twoSite



def blockingSingleRow_2d_effH_perpRow_twoSiteH(tensor,length_of_line):
        lenTensor = int(np.size(tensor.shape)/2)
        physDim = tensor.shape[0]

        lenTensor = 2

        bot_indices = -np.arange(1,2*length_of_line+1)
        top_indices = bot_indices - 2*length_of_line
        stacked_indices = np.vstack((bot_indices,top_indices))
        tensorlabels = [np.array(stacked_indices[:,:lenTensor].flatten())]
        identityLabels = list(stacked_indices[:,lenTensor:].transpose())
        labels = []
        labels.append(tensorlabels[0])
        for n in range(0,len(identityLabels)):
            labels.append(identityLabels[n])

        #eff two site hamiltonian
        tensorList = [tensor]
        for n in range(0,2*length_of_line-2):
            tensorList.append(np.eye(physDim))


        for n in range(0,length_of_line):
            forder_bot = []
            forder_top = []
            c = 0
            for i in range(0,n):
                forder_bot.append(identityLabels[c][0])
                forder_top.append(identityLabels[c][1])
                c+=1
            forder_bot.append(tensorlabels[0][0])
            forder_top.append(tensorlabels[0][2])
            for i in range(0,length_of_line-1):
                forder_bot.append(identityLabels[c][0])
                forder_top.append(identityLabels[c][1])
                c+=1
            forder_bot.append(tensorlabels[0][1])
            forder_top.append(tensorlabels[0][3])
            for i in range(0,length_of_line-n-1):
                forder_bot.append(identityLabels[c][0])
                forder_top.append(identityLabels[c][1])
                c+=1
            forder = np.append(forder_bot,forder_top)

            if n == 0:
                effH_twoSite = ncon(tensorList,labels,forder=forder).reshape((physDim**length_of_line)**2,(physDim**length_of_line)**2)/length_of_line
            else:
                effH_twoSite += ncon(tensorList,labels,forder=forder).reshape((physDim**length_of_line)**2,(physDim**length_of_line)**2)/length_of_line
        return effH_twoSite
