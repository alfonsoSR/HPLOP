# cython: language_level = 3
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: initializedcheck = False

cdef:

    double [:, :] A
    double [:] B
    double [:] Bp
    double [:] Bhat
    double [:] Bphat
    double [:] c
