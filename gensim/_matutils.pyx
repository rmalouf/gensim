import numpy as np
cimport numpy as np

import cython

from libc.math cimport log


#*****************************************************************************80
#
## DIGAMMA calculates DIGAMMA ( X ) = d ( LOG ( GAMMA ( X ) ) ) / dX
#
#  Licensing:
#
#    This code is distributed under the GNU LGPL license.
#
#  Modified:
#
#    20 March 2016
#
#  Author:
#
#    Original FORTRAN77 version by Jose Bernardo.
#    Python version by John Burkardt.
#
#  Reference:
#
#    Jose Bernardo,
#    Algorithm AS 103:
#    Psi ( Digamma ) Function,
#    Applied Statistics,
#    Volume 25, Number 3, 1976, pages 315-317.

cdef inline double psi(double x) nogil:
    cdef double value = 0.0

    #  Reduce to DIGAMA(X + N).
    while (x < 8.5):
        value = value - 1.0 / x
        x = x + 1.0

    # Use Stirling's (actually de Moivre's) expansion.
    cdef double r = 1.0 / x
    value += log(x) - 0.5 * r
    r = r * r
    value = value \
            - r * (1.0 / 12.0 \
                   - r * (1.0 / 120.0 \
                          - r * (1.0 / 252.0 \
                                 - r * (1.0 / 240.0 \
                                        - r * (1.0 / 132.0)))))

    return value

@cython.boundscheck(False)
cdef void direxp_float_1d(float[:] alpha, float[:] result) nogil:

    cdef int I = alpha.shape[0]
    cdef int i
    cdef float psi_sum = 0.0

    for i in range(I):
        result[i] = psi(alpha[i])
        psi_sum += alpha[i]

    psi_sum =  psi(psi_sum)
    for i in range(I):
        result[i] -= psi_sum
    return

@cython.boundscheck(False)
cdef void direxp_float_2d(float[:,:] alpha, float[:,:] result):

    cdef float[:] sum = np.empty((alpha.shape[0]), dtype=np.float32)
    cdef int I = alpha.shape[0]
    cdef int J = alpha.shape[1]
    cdef int i, j
    cdef float psi_sum

    with nogil:
        for i in range(I):
            sum[i] = 0.0
            for j in range(J):
                result[i,j] = psi(alpha[i,j])
                sum[i] += alpha[i,j]

        for i in range(I):
            psi_sum =  psi(sum[i])
            for j in range(J):
                result[i,j] -= psi_sum

    return

@cython.boundscheck(False)
cdef void direxp_double_1d(double[:] alpha, double[:] result) nogil:

    cdef int I = alpha.shape[0]
    cdef int i
    cdef double psi_sum = 0.0

    for i in range(I):
        result[i] = psi(alpha[i])
        psi_sum += alpha[i]

    psi_sum =  psi(psi_sum)
    for i in range(I):
        result[i] -= psi_sum
    return

@cython.boundscheck(False)
cdef void direxp_double_2d(double[:,:] alpha, double[:,:] result):

    cdef double[:] sum = np.empty((alpha.shape[0]), dtype=np.float64)
    cdef int I = alpha.shape[0]
    cdef int J = alpha.shape[1]
    cdef int i, j
    cdef double psi_sum

    with nogil:
        for i in range(I):
            sum[i] = 0.0
            for j in range(J):
                result[i,j] = psi(alpha[i,j])
                sum[i] += alpha[i,j]

        for i in range(I):
            psi_sum =  psi(sum[i])
            for j in range(J):
                result[i,j] -= psi_sum

    return

@cython.profile(True)
cpdef dirichlet_expectation(np.ndarray alpha):
    """
    For a vector `theta~Dir(alpha)`, compute `E[log(theta)]`.

    """

    cdef np.ndarray result = np.zeros_like(alpha, dtype=alpha.dtype)
    if alpha.ndim == 1:
        if alpha.dtype == np.float32:
            direxp_float_1d(alpha, result)
        elif alpha.dtype == np.float64:
            direxp_double_1d(alpha, result)
        else:
            raise ValueError, alpha.dtype
    elif alpha.ndim == 2:
        if alpha.dtype == np.float32:
            direxp_float_2d(alpha, result)
        elif alpha.dtype == np.float64:
            direxp_double_2d(alpha, result)
        else:
            raise ValueError, alpha.dtype
    else:
        raise ValueError, alpha.ndim
    return result
