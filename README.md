# uniformTN

Many body quantum library for riemmannian gradient descent / TDVP on various infinite, translationally invariant tensor network ansatz, in both 1d and 2d. Allows one to calculate ground states and time evolve states in the thermodynamic limit.

Implemented ansatz:
- 1d: 
 - uniform MPS (uniform, left gauge, right gauge, centre gauge), 
 - Bipartite MPS (left gauge), 
 - Two site unit cell MPS (left gauge)
- 2d: 
 - MPSO in left gauge (sequential circuits - subset of iPEPS), 
 - MPSO blocked sites (grouping together sites to enhance tensors physical leg dimension)
 - Bipartite MPSO (left gauge)
 - FourSite_sep MPSO (left gauge) (four seperate tensors in a 2x2 grid)

Implemented Hamiltonians:
 - 1d: oneBodyH, twoBodyH (nearest neighbour), threeBodyH (next nearest neighbour)
 - 2d: oneBodyH, twoBodyH_hori, twoBodyH_vert, plaquetteH (2x2), cross2dH (nearest neighbour in both vertical and horizontal)

Implemented tangent space projection metrics:
 - Euclid
 - TDVP

Example scripts for a handful of physics problems given in ./examples/

To install:
 - Clone the git repo and add the library to your PYTHONPATH
 - Ensure you have the following python dependencies installed:    
    - numpy
    - scipy
    - ncon (version 1.0)
    - einsum
    - progressbar
    - abc 
    - pickle
    - pickle5
