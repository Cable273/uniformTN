#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division

def save_obj(obj, name ):
    import pickle
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    import pickle5 as pickle
    # import pickle
    with open(name + '.pkl', 'rb') as f:
        # return pickle.load(f,encoding='latin1')
        return pickle.load(f)

