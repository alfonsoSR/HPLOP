# cython: language_level = 3
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False
# cython: initializedcheck = False

from hplop.solver.rkn1210_coeffs cimport A, B, Bp, Bhat, Bphat, c
from libc.math cimport ceil, floor
from libc.stdio cimport printf
from libc.float cimport DBL_EPSILON as EPS
from libc.stdlib cimport malloc, calloc, free, exit
from cpython.mem cimport PyMem_Realloc as realloc
import numpy as np


cdef class rkn1210:

    cdef:

        double [:, :] A
        double [:] B
        double [:] Bp
        double [:] Bhat
        double [:] Bphat
        double [:] c

        int rejected, accepted, index, size

        double * _t_out
        double [:] t_out
        
        double * _s_out
        double [:, ::1] s_out

        double * _est_error
        double [:, ::1] est_error

        double * step_size

        double * _ds
        double [:, ::1] ds

    def __cinit__(self):
        """rkn1210 extension type constructor"""

        # ----- Initialize coefficient's arrays ----- #

        self.A = A
        self.B = B
        self.Bp = Bp
        self.Bhat = Bhat
        self.Bphat = Bphat
        self.c = c

        # ----- Output ----- #

        self.size = <int>self.max(
            100., floor(5e5 / (24. * (6. + 1.)) - 1.)
        )

        self.index = 0

        self._t_out = <double*>calloc(self.size, sizeof(double))
        if self._t_out == NULL:
            raise MemoryError('Memory allocation failed for self._t_out')
        else:
            self.t_out = <double[:self.size]>self._t_out

        # self.t_out = <double*>calloc(self.size, sizeof(double))
        # if self.t_out == NULL:
        #     raise MemoryError('Memory allocation failed for self.t_out')

        self._s_out = <double*>calloc(self.size * 6, sizeof(double))
        if self._s_out == NULL:
            raise MemoryError('Memory allocation failed for self._s_out')
        else:
            self.s_out = <double[:self.size, :6]>self._s_out

        self._ds = <double*>calloc(self.size * 6, sizeof(double))
        if self._ds == NULL:
            raise MemoryError('Memory allocation failed for self._ds')
        else:
            self.ds = <double[:self.size, :6]>self._ds

        self.step_size = <double*>calloc(self.size, sizeof(double))
        if self.step_size == NULL:
            raise MemoryError('Memory allocation failed for self.step_size')

        self._est_error = <double*>calloc(self.size * 6, sizeof(double))
        if self._est_error == NULL:
            raise MemoryError('Memory allocation failed for self._est_error')
        else:
            self.est_error = <double[:self.size, :6]>self._est_error

        self.accepted = 0
        self.rejected = 0

    cdef void grow_arrays(self):
        """Allocate more memory for the results"""

        self.size = <int>ceil(1.2 * self.size)

        # self.t_out = <double*>realloc(self.t_out, self.size * sizeof(double))
        # if self.t_out == NULL:
        #     raise MemoryError('Memory reallocation failed for self.t_out')

        self._t_out = <double*>realloc(self._t_out, self.size * sizeof(double))
        if self._t_out == NULL:
            raise MemoryError('Memory reallocation failed for self._t_out')
        else:
            self.t_out = <double[:self.size]>self._t_out
        
        self._s_out = <double*>realloc(
            self._s_out, self.size * 6 * sizeof(double)
        )
        if self._s_out == NULL:
            raise MemoryError('Memory reallocation failed for self._s_out')
        else:
            self.s_out = <double[:self.size, :6]>self._s_out

        self._ds = <double*>realloc(self._ds, self.size * 6 * sizeof(double))
        if self._ds == NULL:
            raise MemoryError('Memory reallocation failed for self._ds')
        else:
            self.ds = <double[:self.size, :6]>self._ds

        self.step_size = <double*>realloc(
            self.step_size, self.size * sizeof(double)
        )
        if self.step_size == NULL:
            raise MemoryError('Memory reallocation failed for self.step_size')

        self._est_error = <double*>realloc(
            self._est_error, self.size * 6 * sizeof(double)
        )
        if self._est_error == NULL:
            raise MemoryError('Memory reallocation failed for self._est_error')
        else:
            self.est_error = <double[:self.size, :6]>self._est_error

    cdef double norm(self, double [:] array):
        """Infinite norm of an unidimensional array
        
        Input
        -----
        `array` : double[:]

        Output
        ------
        `norm(array)` -> double
        """

        cdef:
            
            double term, out = 0.
            short idx

        for idx in range(array.shape[0]):

            term = self.abs(array[idx])

            if term > out:

                out = term

        return out

        
    cdef double max(self, double a, double b):
        """Maximum value among two numbers
        
        Input
        -----
        `a` : double
        `b` : double

        Output
        ------
        `max(a, b)` -> double
        """

        if a > b:

            return a

        else:

            return b

    cdef double min(self, double a, double b):
        """Minimum among two values
        
        Input
        ------
        `a` : double
        `b` : double

        Output
        -------
        `min(a, b)` -> double
        """

        if a < b:

            return a

        else:

            return b

    cdef double abs(self, double a):
        """Absolute value of a number

        Input
        ------
        `a` : double

        Output
        -------
        `abs(a)` -> double
        """

        if a > 0.:

            return a

        else:

            return -a

    cdef void solve(
        self, fun, double t0, double tend,
        double [:] s0, double atol, double rtol
    ):
        """Solve equations of motion between two epochs, using
            RKN1210 method.

            Algorithm based on: https://github.com/rodyo/FEX-RKN1210
        
        Input
        ------
        `fun` : callable

            Python or Cython function representing the equations of motion.

            The function is expected to be as follows:

                `double[:] fun(double, double[:])`

            Where the output memview has shape (3,) and the one used
            as input has shape (6,)

        `t0` : double

            Initial epoch.

        `tend` : double

            Final epoch.

        `s0` : double[:]

            Initial state of the satellite (Cartesian, SCRF).

        `atol` : double

            Absolute tolerance to be used by the solver.

        `rtol` : double

            Relative tolerance to be used by the solver.

        """

        cdef:

            double h, h_new, hmin, hf, f1, f2, f3, f4, t
            double max_step, growth, growth_term
            
            int direction

            short i, j, k

            double _s[6]
            double [:] s = _s

            double _new_s[6]
            double [:] new_s = _new_s

            double _sol[6]
            double [:] sol = _sol

            double hc[17]
            double R[6]
            double f[6][17]
            double hfB[3]
            double hfBp[3]
            double hfBhat[3]
            double hfBphat[3]

            double delta[3]
            double step_tol[3]
            # bint accept

            int counter = 0

        # ----- Initialize f ----- #

        for i in range(6):

            for j in range(17):

                f[i][j] = 0.

        # ----- Step calculation parameters ----- #

        f1 = 0.92
        f2 = 11.
        f3 = 1. / 11.
        f4 = 7.

        hmin = 16 * EPS * t0

        max_step = tend - t0

        direction = 1 - 2 * (tend < t0)

        # ----- Initialize output ----- #

        # t = 0.
        # self.t_out[0] = t0

        t = t0
        self.t_out[0] = t0

        for i in range(6):

            s[i] = s0[i]
            self.s_out[0, i] = s0[i]

        sol = fun(t0, np.asarray(s0))

        for i in range(3):

            f[i][0] = sol[i + 3]

        # ----- Compute first step ----- #

        h = (atol ** f3) / self.max(self.norm(sol), 1e-4)
        h = self.min(max_step, self.max(h, hmin))
        h = direction * self.abs(h)

        # ----- Main loop ----- #

        while self.abs(t - tend) > 0.:

            hmin = 16. * t * EPS * direction

            if (direction * (t + h)) > (direction * tend):

                h = direction * self.max(self.abs(hmin), self.abs(t - tend))

            for j in range(17):

                hc[j] = h * self.c[j]

                for i in range(3):

                    R[i] = s[i] + hc[j] * s[i + 3]

                    for k in range(17):

                        R[i] += f[i][k] * h * h * self.A[k, j]

                    R[i + 3] = s[i + 3]

                sol = fun(t + hc[j], np.asarray(R))

                for i in range(3):

                    f[i][j] = sol[i + 3]

            for i in range(3):

                hf = h * f[i][0]

                hfB[i] = hf * self.B[0]
                hfBp[i] = hf * self.Bp[0]
                hfBhat[i] = hf * self.Bhat[0]
                hfBphat[i] = hf * self.Bphat[0]

                for j in range(1, 17):

                    hf = h * f[i][j]

                    hfB[i] += hf * self.B[j]
                    hfBp[i] += hf * self.Bp[j]
                    hfBhat[i] += hf * self.Bhat[j]
                    hfBphat[i] += hf * self.Bphat[j]

                new_s[i] = s[i] + h * (s[i + 3] + hfBhat[i])
                new_s[i + 3] = s[i + 3] + hfBphat[i]

                delta[i] = self.max(
                    self.abs(h * (hfBhat[i] - hfB[i])),
                    self.abs(hfBphat[i] - hfBp[i])
                )

                step_tol[i] = self.min(
                    atol, rtol * self.max(
                        self.abs(new_s[i]),
                        self.abs(new_s[i + 3])
                    )
                )
                # accept[i] = (delta[i] <= step_tol[i])

                # delta[i] = self.abs(h * (hfBhat[i] - hfB[i]))
                # delta[i + 3] = self.abs(hfBphat[i] - hfBp[i])

                # step_tol[i] = self.min(atol, rtol * self.abs(new_s[i]))
                # step_tol[i + 3] = self.min(atol, rtol * self.abs(new_s[i + 3]))

                # accept[i] = (delta[i] <= step_tol[i])
                # accept[i + 3] = (delta[i + 3] <= step_tol[i + 3])

            # print(f"Delta s: {np.asarray(s) - np.asarray(new_s)}")
            # print(f"Delta: {np.asarray(delta)}")
            # print(f"Step tol: {np.asarray(step_tol)}")

            if (
                (delta[0] <= step_tol[0]) and
                (delta[1] <= step_tol[1]) and
                (delta[2] <= step_tol[2])
            ): 

                self.index += 1

                if self.index > self.size:

                    self.grow_arrays()

                t = t + h
                
                self.t_out[self.index] = t

                for i in range(3):

                    s[i] = new_s[i]
                    s[i + 3] = new_s[i + 3]

                    self.s_out[self.index, i] = s[i]
                    self.s_out[self.index, i + 3] = s[i + 3]

                    self.ds[self.index, i] = s[i + 3]
                    self.ds[self.index, i + 3] = f[i][0]

                    self.est_error[self.index, i] = delta[i]
                    self.est_error[self.index, i + 3] = delta[i]
                    # self.est_error[self.index, i + 3] = delta[i + 3]

                self.step_size[self.index] = h

                self.accepted += 1

            else:

                self.rejected += 1

            # for i in range(6):
            # for i in range(3):

            #     accept[i] = (delta[i] <= EPS)

            if (
                (delta[0] <= EPS) and
                (delta[1] <= EPS) and
                (delta[2] <= EPS)
            ):

                # print("Calcula normal")

                h = direction * self.min(
                    self.abs(max_step),
                    f4 * self.abs(h)
                )

            else:

                # print("Calcula con EST")

                growth = 0.
                growth_term = 0.

                for i in range(3):

                    growth_term = f1 * (
                        step_tol[i] / (f2 * self.abs(h) * delta[i])
                    ) ** f3

                    if growth_term > growth:

                        growth = growth_term

                h_new = direction * self.min(
                    self.abs(max_step),
                    growth * self.abs(h)
                )

                if h_new != 0.:

                    h = h_new

                # print(f"Growth: {growth}")
                # print(f"h_new: {h_new}")

            # ----- Final check on new step ----- #

            if self.abs(h) < self.abs(hmin):

                printf(
                    "At time t = %.6f, step size fell below hmin: h = %e\n",
                    t - t0, h
                )

                counter += 1

                h = hmin

                if counter == 10:

                    exit(1)

            # print(f"Aceptados: {self.accepted}")
            # print(f"Rechazados: {self.rejected}")
            # print(f"Current t: {self.t_out[self.index]}")
            # print(f"Last h: {self.step_size[self.index]}")
            # print(f"Next h: {h}")

            # if int(input("New: ")) == 0:

            #     exit(0)

        # print(f"Last h: {h}")
        # print(f"Last t: {self.t_out[self.index]}")
        # print(f"Expected t: {tend}")

    
    def __call__(
        self, fun, double t0, double tend, double [:] s0,
        double atol=5e-14, double rtol=5e-14
    ):
        """Propagate orbit between given epochs.

        Input
        ------
        Input
        ------
        `fun` : callable

            Python or Cython function representing the equations of motion.

            The function is expected to be as follows:

                `double[:] fun(double, double[:])`

            Where the output memview has shape (3,) and the one used
            as input has shape (6,)

        `t0` : double

            Initial epoch.

        `tend` : double

            Final epoch.

        `s0` : double[:]

            Initial state of the satellite (Cartesian, SCRF).

        `atol` : double, optional

            Absolute tolerance to be used by the solver.
            Default is `5e-14`.

        `rtol` : double, optional

            Relative tolerance to be used by the solver.
            Default is `5e-14`.

        """

        self.solve(fun, t0, tend, s0, atol, rtol)

        # print(f"Total steps: {self.accepted + self.rejected}")
        # print(f"Accepted: {self.accepted}")
        # print(f"Rejected: {self.rejected}")
        # print(f"Relation: {<double>self.rejected / <double>self.accepted}")

        return self.t_out[:self.index + 1], self.s_out[:self.index + 1]

    def __dealloc__(self):
        """Perform memory deallocation"""

        if self._t_out is not NULL:
            free(self._t_out)

        if self._s_out is not NULL:
            free(self._s_out)

        if self._ds is not NULL:
            free(self._ds)

        if self.step_size is not NULL:
            free(self.step_size)

        if self._est_error is not NULL:
            free(self._est_error)

        printf("%d\n", 0)
