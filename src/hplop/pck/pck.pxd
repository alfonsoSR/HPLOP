# cython: language_level = 3
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False

cdef struct Body:
    char * name
    int id
    double mu
    double R


cdef Body moon, sun, mercury, venus, earth
cdef Body mars, jupiter, saturn, uranus, neptune, pluto
