
collide.cpp and collide_kokkos.cpp

These are implementations of the Transient Adaptive Subcell (TASC) method for selecting collision partner pairs in DSMC codes. 
These were written as contributions to the open-source DSMC code SPARTA. Included are C++ and Kokkos versions.

__________________________________

electron_kernel_1D_PK.py

This is a PyKokkos (Python wrapper to Kokkos) kernel for my Python-driven electron Boltzmann solver.
This kernel updates the simulated particles by doing collision cross section and collision computations along with advection.

__________________________________

particletocell_3D.cu

This is a CUDA kernel for 3-D particle-to-cell mapping for PIC codes. 
The work is split into two kernels, one for atomic tracking for summing values from particles to subcell quantities and the second for summing values over subcells.
It assumes uniform rectangular grids with each dimension normalized to [0,1] indendently.
