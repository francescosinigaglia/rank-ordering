# Rank ordering code

This script takes as input two three-dimensional fields (described as 3D NumPy arrays) and rank-order the probability distribution (PDF) of one onto the PDF of other. In particular, one can perform just a "one-step" rank ordering, as well as an iterative rank ordering. The iterative rank ordering can be done by iteratively correcting the 3D P(k) with an isotropic kernel in Fourier space.

To perform the rank ordering in one step set:
niter = 1, correct power = True/False depending on whether you want/don't want to correct the P(k) with the kernel

To perform the iterative rank ordering set:
niter = N, where N is the desired number of iterations. In this case it is meaningless not to compensate for the error introduced in the P(k), so the correct_power parameter is automatically set = True. 

