# cython: language_level = 3
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False

cimport numpy as cnp
from hplop.pck.pck cimport Body

cdef extern from "SpiceUsr.h" nogil:

    ctypedef char SpiceChar
    ctypedef double SpiceDouble
    ctypedef float SpiceFloat
    ctypedef int SpiceInt
    ctypedef const char ConstSpiceChar
    ctypedef const double ConstSpiceDouble
    ctypedef const float ConstSpiceFloat
    ctypedef const int ConstSpiceInt

    cdef void furnsh_c(ConstSpiceChar * file)
    cdef void kclear_c()
    cdef void str2et_c(ConstSpiceChar * epoch, SpiceDouble * et)
    cdef void pxform_c(
        ConstSpiceChar * old_frame,
        ConstSpiceChar * new_frame,
        SpiceDouble epoch,
        SpiceDouble matrix[3][3]
    )
    cdef void mxv_c(
        ConstSpiceDouble matrix[3][3],
        ConstSpiceDouble old_r[3],
        SpiceDouble new_r[3]
    )
    cdef void spkpos_c(
        ConstSpiceChar * target,
        SpiceDouble epoch,
        ConstSpiceChar * frame,
        ConstSpiceChar * abcorr,
        ConstSpiceChar * obs,
        SpiceDouble X_obs_targ[3],
        SpiceDouble * light_time
    )
    cdef double clight_c()
    cdef void spkgps_c(
        SpiceInt target,
        SpiceDouble epoch,
        ConstSpiceChar * frame,
        SpiceInt obs,
        SpiceDouble X_obs_targ[3],
        SpiceDouble * light_time
    )

cdef class motion_law:

    cdef:
        int limit
        double c

        double *_Clm
        double [:, ::1] Clm

        double *_Slm
        double [:, ::1] Slm

        double *norm_n1
        double *norm_n2
        double *norm_n1_n1

        double *_norm_n1_m
        double [:, ::1] norm_n1_m

        double *_norm_n2_m
        double [:, ::1] norm_n2_m

        cnp.ndarray ds_array
        double [:] ds

    cdef void harmonics(self, const int max_deg, double t,
                        double r_vec[3])

    # cdef void third_body(self, double t, double r_vec[3], int id, double mu)

    cdef void third_body(self, double t, double r_vec[3], Body body)

cdef class ENRKE:

    cdef double tol, ec, PI, TWOPI

    cdef void ENRKE_evaluate(self, double M, double * Eout)


cdef class Kepler(ENRKE):

    cdef:
        double a, e, Omega, i, omega, t0, T, sqrt_ec, NU
        cnp.ndarray state_array
        double [:] state_memview

    cdef void time2nu(self, double t, double * nu)

    cdef void _nu2state(self, double nu, double[:] state)
