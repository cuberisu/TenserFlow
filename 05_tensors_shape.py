import tensorflow as tf

# Everything is Tensor
# Tensor Ranks, Shapes, and Types

3   # a rank 0 tensor; this is a scalar with shape []
[1., 2., 3.]    # a rank 1 tensor; this is a vector with shape [3]
[[1., 2., 3.], [4., 5., 6.]]    # a rank 2 tensor; a matrix with shape [2, 3]
[[[1., 2., 3.,], [7., 8., 9.]]]     # a rank 3 tensor with shape [2, 1, 3]