# cython: language_level = 3
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: initializedcheck = False

from hplop.pck.pck cimport Body, moon, earth, sun, jupiter, mars, venus
from libc.stdlib cimport calloc, free
from libc.math cimport sqrt, sin, cos, atan2, M_PI as pi
from libc.stdio cimport printf
import numpy as np
from hplop.utils.utils import sqlite


cdef class motion_law:

    def __cinit__(
        self, int max_deg, str db_name = "grgm1200b",
        char *kernels_path = "spice/metak"
    ):
        """
        Motion law extension type constructor

        Input
        -----
        `max_deg` : int

            Truncation degree for the spherical harmonics expansion of the
            lunar gravity field.

        `db_name` : str, optional

            Name of the database containing the set of spherical harmonics'
            coefficients to be used. The name should belong to an SQL database
            stored at the default localhost-location. (Should add a further
            description of the DB format).
            Default is `grgm1200b`.

        `kpath` : char *, optional

            Relative (or absolute) path to the file where spice kernels
            to be used are listed.
            Default is `spice/metak`.
        """

        cdef int i, j, d, o, n = max_deg + 1
        cdef double c, s

        # Retrieve spherical harmonics coefficients from database

        self.limit = max_deg

        self._Clm = <double*>calloc(n * n, sizeof(double))
        if self._Clm == NULL:
            raise MemoryError('Memory allocation failed for self._Clm')
        self.Clm = <double[:n, :n]>self._Clm

        self._Slm = <double*>calloc(n * n, sizeof(double))
        if self._Slm == NULL:
            raise MemoryError('Memory allocation failed for self._Slm')
        self.Slm = <double[:n, :n]>self._Slm

        with sqlite(db_name) as db:

            db.execute(
                """
                    select fk_deg, ord, Clm, Slm from ord
                    where fk_deg <= ?;
                """, (self.limit,)
            )

            for d, o, c, s in db:

                self.Clm[d, o] = c
                self.Slm[d, o] = s

        # Compute normalization parameters

        self.norm_n1 = <double*>calloc(self.limit, sizeof(double))
        if self.norm_n1 == NULL:
            raise MemoryError('Memory allocation failed for self.norm_n1')

        self.norm_n2 = <double*>calloc(self.limit, sizeof(double))
        if self.norm_n2 == NULL:
            raise MemoryError('Memory allocation failed for self.norm_n2')

        self.norm_n1_n1 = <double*>calloc(self.limit, sizeof(double))
        if self.norm_n1_n1 == NULL:
            raise MemoryError('Memory allocation failed for self.norm_n1_n1')

        self._norm_n1_m = <double*>calloc(self.limit * n, sizeof(double))
        if self._norm_n1_m == NULL:
            raise MemoryError('Memory allocation failed for self._norm_n1_m')
        self.norm_n1_m = <double[:self.limit, :n]>self._norm_n1_m

        self._norm_n2_m = <double*>calloc(self.limit * n, sizeof(double))
        if self._norm_n2_m == NULL:
            raise MemoryError('Memory allocation failed for self._norm_n2_m')
        self.norm_n2_m = <double[:self.limit, :n]>self._norm_n2_m

        for i in range(2, self.limit + 1):

            self.norm_n1[i - 1] = sqrt((2. * i + 1.) / (2. * i - 1.))
            self.norm_n2[i - 1] = sqrt((2. * i + 1.) / (2. * i - 3.))
            self.norm_n1_n1[i - 1] = (
                sqrt((2. * i + 1.) / (2. * i)) / (2. * i - 1.)
            )

            for j in range(1, i + 1):

                self.norm_n1_m[i - 1, j - 1] = sqrt(
                    (i - j) * (2. * i + 1.) / ((i + j) * (2. * i - 1.))
                )

                self.norm_n2_m[i - 1, j - 1] = sqrt(
                    (i - j) * (i - j - 1.) * (2. * i + 1.) /
                    ((i + j) * (i + j - 1.) * (2. * i - 3.))
                )

        # Initialize ds

        self.ds_array = np.zeros((6,))
        self.ds = self.ds_array

        # Load spice kernels

        furnsh_c(kernels_path)

        # Light speed

        self.c = clight_c()

    cdef void harmonics(
        self, const int max_deg, double t, double r_vec[3]
    ):
        """ Acceleration due to non-sphericity of the Moon.

        Input
        ------
        `max_deg` : const int

            Truncation degree for the spherical harmonics expansion of the
            lunar gravity field.

        `t` : double

            Epoch at which the acceleration is to be computed.

        `r_vec` : double [3]

            Satellite's position vector with respect to SCRF at given epoch.
        """

        cdef:

            int idx, n_idx, m_idx, nm1_idx, nm2_idx, mm1_idx
            double e1, r2, r, r_cos_phi, sin_phi, cos_phi, root3, root5
            double n, m, nm1, nm2, e2, e3, e4, e5

            double R_icrf2pa[3][3]
            double r_i[3]
            double cos_m_lambda[max_deg + 1]
            double sin_m_lambda[max_deg + 1]
            double R_r[max_deg + 1]
            double Pn[max_deg + 1]
            double dPn[max_deg + 1]
            double z_partials[3]
            double xy_partials[3]
            double ddr[3]

            double sec_Pnm[max_deg + 1][max_deg + 1]
            double cos_dPnm[max_deg + 1][max_deg + 1]

        # Initialize sec_Pnm and cos_dPnm

        for idx in range(max_deg + 1):
            for n_idx in range(max_deg + 1):
                sec_Pnm[idx][n_idx] = 0.
                cos_dPnm[idx][n_idx] = 0.

        # Compute position vector in PA reference frame

        pxform_c("J2000", "MOON_PA_DE440", t, R_icrf2pa)

        for idx in range(3):
            r_i[idx] = (
                R_icrf2pa[idx][0] * r_vec[0] +
                R_icrf2pa[idx][1] * r_vec[1] +
                R_icrf2pa[idx][2] * r_vec[2]
            )

        # Compute spherical coordinates: r, lat, lon

        e1 = r_i[0] * r_i[0] + r_i[1] * r_i[1]

        r2 = e1 + r_i[2] * r_i[2]

        r = sqrt(r2)

        r_cos_phi = sqrt(e1)

        sin_phi = r_i[2] / r

        cos_phi = r_cos_phi / r

        cos_m_lambda[0] = 1.
        sin_m_lambda[0] = 0.

        if r_cos_phi != 0:
            sin_m_lambda[1] = r_i[1] / r_cos_phi
            cos_m_lambda[1] = r_i[0] / r_cos_phi

        R_r[0] = 1.
        R_r[1] = moon.R / r

        # Initialize normalised associated Legendre functions (Lear algorithm)

        root3 = sqrt(3.)
        root5 = sqrt(5.)

        Pn[0] = 1.
        Pn[1] = root3 * sin_phi

        dPn[0] = 0.
        dPn[1] = root3

        sec_Pnm[1][1] = root3

        cos_dPnm[1][1] = -root3 * sin_phi

        # Normalized associated Legendre functions

        if max_deg >= 2:

            for n_idx in range(2, max_deg + 1):

                n = n_idx
                nm1 = n - 1.
                nm2 = n - 2.

                nm1_idx = n_idx - 1
                nm2_idx = n_idx - 2

                R_r[n_idx] = R_r[nm1_idx] * R_r[1]

                sin_m_lambda[n_idx] = (
                  2. * cos_m_lambda[1] * sin_m_lambda[nm1_idx]
                  - sin_m_lambda[nm2_idx]
                )
                cos_m_lambda[n_idx] = (
                  2. * cos_m_lambda[1] * cos_m_lambda[nm1_idx]
                  - cos_m_lambda[nm2_idx]
                )

                e1 = 2. * n - 1.

                Pn[n_idx] = (
                    e1 * sin_phi * self.norm_n1[nm1_idx] * Pn[nm1_idx]
                    - nm1 * self.norm_n2[nm1_idx] * Pn[nm2_idx]
                ) / n

                dPn[n_idx] = self.norm_n1[nm1_idx] * (
                    sin_phi * dPn[nm1_idx] + n * Pn[nm1_idx]
                )

                sec_Pnm[n_idx][n_idx] = (
                    e1 * cos_phi * self.norm_n1_n1[nm1_idx]
                    * sec_Pnm[nm1_idx][nm1_idx]
                )

                cos_dPnm[n_idx][n_idx] = -sin_phi * n * sec_Pnm[n_idx][n_idx]

                e1 = e1 * sin_phi
                e2 = -sin_phi * n

                for m_idx in range(1, n_idx):

                    m = m_idx
                    mm1_idx = m_idx - 1

                    e3 = (
                        self.norm_n1_m[nm1_idx, m_idx - 1]
                        * sec_Pnm[nm1_idx][m_idx]
                    )
                    e4 = n + m
                    e5 = (
                        e1 * e3
                        - (e4 - 1.) * self.norm_n2_m[nm1_idx, m_idx - 1]
                        * sec_Pnm[n_idx - 2][m_idx]
                    ) / (n - m)

                    sec_Pnm[n_idx][m_idx] = e5

                    cos_dPnm[n_idx][m_idx] = e2 * e5 + e3 * e4


        # Though Pn[0] should be 1, I prefere to separate the effect of the
        # Moon as a punctual mass from the rest of the accelerations

        Pn[0] = 0.

        # Acceleration with respect to PA reference frame

        z_partials[0] = -sin_phi * cos_m_lambda[1]
        z_partials[1] = -sin_phi * sin_m_lambda[1]
        z_partials[2] = cos_phi

        xy_partials[0] = -sin_m_lambda[1]
        xy_partials[1] = cos_m_lambda[1]
        xy_partials[2] = 0.

        for idx in range(3):

            ddr[idx] = - moon.mu * r_i[idx] / (r * r2)

        if max_deg >= 2:

            for n_idx in range(2, max_deg + 1):

                n = n_idx

                for idx in range(3):

                    ddr[idx] += (moon.mu / r2) * R_r[n_idx] * self.Clm[n_idx, 0] * (
                        (-r_i[idx] / r) * (n + 1.) * Pn[n_idx]
                        + z_partials[idx] * cos_phi * dPn[n_idx]
                    )

                for m_idx in range(1, n_idx + 1):

                    m = m_idx

                    for idx in range(3):

                        ddr[idx] += (moon.mu / r2) * R_r[n_idx] * (
                            (
                                cos_m_lambda[m_idx] * self.Clm[n_idx, m_idx]
                                + sin_m_lambda[m_idx] * self.Slm[n_idx, m_idx]
                            ) * (
                                (-r_i[idx] / r) * (n + 1.) * cos_phi
                                * sec_Pnm[n_idx][m_idx]
                                + z_partials[idx] * cos_dPnm[n_idx][m_idx]
                            ) + xy_partials[idx] * sec_Pnm[n_idx][m_idx] * m *
                            (
                                cos_m_lambda[m_idx] * self.Slm[n_idx, m_idx]
                                - sin_m_lambda[m_idx] * self.Clm[n_idx, m_idx]
                            )
                        )

        for idx in range(3):

            self.ds[3 + idx] = (
                R_icrf2pa[0][idx] * ddr[0] +
                R_icrf2pa[1][idx] * ddr[1] +
                R_icrf2pa[2][idx] * ddr[2]

            )

    cdef void third_body(
        self, double t, double r_vec[3], Body body
    ):
        """ Acceleration due to third body perturbation.

        Input
        -----
        `t` : double
            
            Epoch at which the acceleration is to be computed.

        `r_vec` : double [3]

            Satellite's position vector with respect to SCRF at given epoch.

        `id` : int

            Third body NAIF ID code. (https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html)

        `mu` : double
            
            Standard gravitational parameter of the third body [km^3/s^2]
        """
        cdef:
            double X_moon[3]
            double X_sat[3]
            double d_moon, d_sat, d_moon3, d_sat3
            int i

        spkgps_c(moon.id, t, "J2000", body.id, X_moon, &d_moon)
        
        d_moon = d_moon * self.c
        d_moon3 = d_moon * d_moon * d_moon

        for i in range(3):
            X_sat[i] = X_moon[i] + r_vec[i]

        d_sat = sqrt(
            X_sat[0] * X_sat[0] +
            X_sat[1] * X_sat[1] +
            X_sat[2] * X_sat[2]
        )

        d_sat3 = d_sat * d_sat * d_sat

        for i in range(3):
            self.ds[i + 3] += -body.mu * (
                (X_sat[i] / d_sat3) - (X_moon[i] / d_moon3)
            )

    def __dealloc__(self):
        """Perform memory deallocation and close SPICE kernels"""

        kclear_c()

        if self._Clm is not NULL:
            free(self._Clm)

        if self._Slm is not NULL:
            free(self._Slm)

        if self.norm_n1 is not NULL:
            free(self.norm_n1)

        if self.norm_n2 is not NULL:
            free(self.norm_n2)

        if self.norm_n1_n1 is not NULL:
            free(self.norm_n1_n1)

        if self._norm_n1_m is not NULL:
            free(self._norm_n1_m)

        if self._norm_n2_m is not NULL:
            free(self._norm_n2_m)

        printf("%d\n", 0)

    def __call__(self, double t, double [:] s):
        """Update velocity and acceleration values.

        Input
        -----
        `t` : double

            Epoch at which the derivative of the state is to be computed.

        `s` : double[:]

            State of the satellite at given epoch (SCRF).

        Output
        -------
        Derivative of the state of the satellite (Velocity and acceleration).
        """

        cdef double r_vec[3]
        cdef int i

        for i in range(3):
            r_vec[i] = s[i]
            self.ds[i] = s[i + 3]

        self.harmonics(self.limit, t, r_vec)

        # self.third_body(t, r_vec, earth)
        # self.third_body(t, r_vec, sun)
        # self.third_body(t, r_vec, jupiter)
        # self.third_body(t, r_vec, mars)
        # self.third_body(t, r_vec, venus)

        return self.ds_array


cdef class ENRKE:

    def __init__(self, const double ec, const double tol = 3.0e-15):
        """ENRKE class constructor

        Input
        -----
        `ec` : const double

            Eccentricity of the keplerian orbit.

        `tol` : const double, optional

            Method's accuracy.
            Default is `3.0e-15`.
        """

        self.ec = ec
        self.tol = tol

        self.PI = 3.1415926535897932385
        self.TWOPI = 6.2831853071795864779

    cdef void ENRKE_evaluate(self, double M, double * Eout):
        """Compute eccentric anomaly from mean anomaly.

        Input
        ------
        `M` : double

            Value of the mean anomaly.

        `Eout` : double *

            Pointer to the value of the eccentric anomaly.
        """

        cdef:

            double delta, Eapp, flip, f, fp, fpp, fppp, fp3, ffpfpp, f2fppp
            double Mr
            double tol2s = 2. * self.tol / (self.ec + 2.2e-16)
            double al = self.tol / 1.0e7
            double be = self.tol / 0.3

        # Fit angle in range (0, 2pi) if needed

        Mr = M % self.TWOPI

        if Mr > self.PI:

            Mr = self.TWOPI - Mr
            flip = 1.

        else:

            flip = -1.

        if (self.ec > 0.99 and Mr < 0.0045):

            fp = 2.7 * Mr
            fpp = 0.301
            f = 0.154

            while (fpp - fp > (al + be * f)):

                if (f - self.ec * sin(f) - Mr) > 0.:

                    fpp = f

                else:

                    fp = f

                f = 0.5 * (fp + fpp)

            Eout[0] = (M + flip * (Mr - f)) % self.TWOPI

        else:

            Eapp = Mr + 0.999999 * Mr * (self.PI - Mr) / (
                2. * Mr + self.ec - self.PI +
                2.4674011002723395 / (self.ec + 2.2e-16)
            )

            fpp = self.ec * sin(Eapp)
            fppp = self.ec * cos(Eapp)
            fp = 1. - fppp
            f = Eapp - fpp - Mr

            delta = - f / fp

            fp3 = fp * fp * fp
            ffpfpp = f * fp * fpp
            f2fppp = f * f * fppp

            delta = delta * (fp3 - 0.5 * ffpfpp + f2fppp / 3.) / (
                fp3 - ffpfpp + 0.5 * f2fppp
            )

            while (delta * delta > fp * tol2s):
                Eapp += delta
                fp = 1. - self.ec * cos(Eapp)
                delta = (Mr - Eapp + self.ec * sin(Eapp)) / fp

            Eapp += delta

            Eout[0] = (M + flip * (Mr - Eapp)) % self.TWOPI


cdef class Kepler(ENRKE):

    def __init__(
        self, const double a, const double e, const double Omega,
        const double i, const double omega, const double t0
    ):
        """Kepler extension type constructor.

        Input
        -----
        `a` : const double

            Semimajor axis [ km ].

        `e` : const double

            Eccentricity.

        `Omega` : const double

            Right ascension of the ascending node [ rad ].

        `i` : const double

            Inclination angle [ rad ].

        `omega` : const double

            Argument of periapsis [ rad ].

        `t0` : const double

            Initial epoch.
        """

        self.a = a
        self.e = e
        self.Omega = Omega
        self.i = i
        self.omega = omega
        self.t0 = t0

        # ----- Initialize ENRKE class ----- #

        super().__init__(self.e)

        # ----- Orbital period ----- #

        self.T = 2. * pi * sqrt(self.a * self.a * self.a / moon.mu)

        # ----- Recurrent parameters ----- #

        self.sqrt_ec = sqrt((1. + self.e) / (1. - self.e))

        self.state_array = np.zeros((6,), dtype=np.float64)
        self.state_memview = self.state_array

    cdef void time2nu(self, double t, double * nu):
        """Compute epoch from true anomaly.

        Input
        -----
        `t` : double

            Epoch in which true anomaly is to be computed.

        `nu` : double *

            Pointer to the value of the true anomaly.
        """

        cdef double M = 2. * pi * (t - self.t0) / self.T
        cdef double E

        self.ENRKE_evaluate(M, &E)

        nu[0] = 2. * atan2(self.sqrt_ec * sin(0.5 * E), cos(0.5 * E))

    cdef void _nu2state(self, double nu, double[:] state):
        """Calculate cartesian state from true anomaly.

        Input
        ------
        `nu` : double

            True anomaly [ rad ].

        `state` : double[:]

            Memory view where the value where the new state is to be stored.
        """

        cdef:
            double omega_nu, cos_omega, sin_omega, cos_omega_nu
            double sin_omega_nu, cos_i, sin_i, cos_Omega, sin_Omega, e_e, r_mod
            double u_mod, f_cos, f_sin, sin_omega_nu_cos_i, f_cos_cos_i

        omega_nu = self.omega + nu

        cos_omega = cos(self.omega)
        sin_omega = sin(self.omega)

        cos_omega_nu = cos(omega_nu)
        sin_omega_nu = sin(omega_nu)

        cos_i = cos(self.i)
        sin_i = sin(self.i)

        cos_Omega = cos(self.Omega)
        sin_Omega = sin(self.Omega)

        e_e = self.e * self.e

        r_mod = self.a * (1. - e_e) / (1. + self.e * cos(nu))

        u_mod = sqrt(moon.mu / (self.a * (1. - e_e)))

        f_cos = cos_omega_nu + self.e * cos_omega
        f_sin = sin_omega_nu + self.e * sin_omega

        sin_omega_nu_cos_i = sin_omega_nu * cos_i
        f_cos_cos_i = f_cos * cos_i

        state[0] = r_mod * (cos_omega_nu * cos_Omega
                            - sin_omega_nu_cos_i * sin_Omega)
        state[1] = r_mod * (cos_omega_nu * sin_Omega
                            + sin_omega_nu_cos_i * cos_Omega)
        state[2] = r_mod * sin_omega_nu * sin_i
        state[3] = u_mod * (-f_cos_cos_i * sin_Omega - f_sin * cos_Omega)
        state[4] = u_mod * (f_cos_cos_i * cos_Omega - f_sin * sin_Omega)
        state[5] = u_mod * f_cos * sin_i

    def nu2state(self, double nu):
        """Python interface for the _nu2state method.

        Input
        ------
        `nu` : double

            True anomaly [ rad ]
        """

        state = np.zeros((6,), dtype=np.float64)

        self._nu2state(nu, state)

        return state

    def __call__(self, double t):
        """State of the ideal satellite at given epoch.

        Input
        -----
        `t` : double

            Epoch at which the ideal state of the satellite is to be computed.

        Output
        -------
        NumPy array with the cartesian state of the ideal satellite.
        """

        self.time2nu(t, &self.NU)
        self._nu2state(self.NU, self.state_memview)

        return self.state_array



# EOF
