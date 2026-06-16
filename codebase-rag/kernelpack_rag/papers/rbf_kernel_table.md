## RBF Kernel Families — Table 1

The choice of radial basis function phi(epsilon * r) affects interpolant
smoothness and conditioning. Key families:

- Gaussian (C-inf):       exp(-(epsilon*r)^2)
- Inv. Multiquadric (C-inf): (1 + (epsilon*r)^2)^(-1/2)
- Matern C2 (M2):         exp(-epsilon*r) * (1 + epsilon*r)
- Matern C4 (M4):         exp(-epsilon*r) * (3 + 3*epsilon*r + (epsilon*r)^2)
- Wendland C2 (W2):       (1 - epsilon*r)_+^4 * (4*epsilon*r + 1)
- Wendland C4 (W4):       (1 - epsilon*r)_+^6 * (35*(epsilon*r)^2 + 18*epsilon*r + 3)

epsilon is a shape parameter multiplying Euclidean distance r.
Larger epsilon = more peaked kernel. Smaller epsilon = flatter kernel.

The kernel matrix K has entries K_ij = phi(epsilon * ||x_i - x_j||).
Global interpolation solves K @ lambda = f_train for coefficients lambda,
then evaluates s(x) = sum_i lambda_i * phi(epsilon * ||x - x_i||).

Source: arXiv 2603.23074, Table 1