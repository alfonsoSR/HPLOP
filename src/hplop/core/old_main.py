import numpy as np
import numpy.typing as npt
from datetime import datetime
import spiceypy as spice
from hplop.pck.pck import moon_data
from hplop.core.equations import Kepler, motion_law
from hplop.solver.rkn1210 import rkn1210
from scipy.integrate import solve_ivp
# from hplop.core.dynamics import Dynamics

# Type alias for numpy arrays

ndarray = npt.NDArray[np.float64]

# Global

MU_MOON, R_MOON = moon_data()


class main:

    def __init__(self, state: ndarray, initial_epoch: str, nu: float = 0.,
                 cartesian: bool = False) -> None:
        """High Precision Lunar Orbit Propagator (HPLOP) Constructor

        Input
        ------
        `state` : ndarray
            Initial state of the satellite. If `cartesian` is set to `False`
            it is assumed to containg a set of classical orbital elements
            sorted as follows: `[a, e, Omega, i, omega]`.

            If `cartesian` is set to `True`, then the `state` should contain
            the cartesian components of the position and velocity vectors.
            Both the orbital elements and the vectors are assumed to have
            being defined using the SCRF system.
        `initial_epoch` : str
            UTC epoch at which the state of the satellite is known and
            from which the orbit will be propagated.

            The epoch should be provided as an ISO-formatted string:
            `YYYY-MM-DD hh:mm:ss`.

            Default ephemerides are DE440, so epochs should lie between
            1549-DEC-31 00:00:00 and 2650-JAN-25 00:00:00.
        `nu` : float, optional
            Initial value for the true anomaly in degrees.
            Default is 0.

        `cartesian` : bool, optional
            Whether the initial state of the satellite is expressed in terms
            of orbital elements or cartesian coordinates.
            Default is `False`, so orbital elements are used.
        """

        # ----- Initial epoch in TDB ----- #

        try:

            # Check if date format is correct by using a function that
            # fails if not (Should be replaced by something better)

            datetime.fromisoformat(initial_epoch)
            self.t0 = spice.str2et(initial_epoch)

        except ValueError:

            raise

        # ----- Initial state ----- #

        if cartesian:

            self.y0 = state

        else:

            # ----- Check orbital elements ----- #

            self.check_elements(state, nu)

            # ----- Initialize keplerian orbit ----- #

            kepler = Kepler(self.a, self.e,
                            self.Omega, self.i, self.omega, self.t0)

            self.y0 = kepler.nu2state(self.nu)

        self.solver = rkn1210()

        return None

    def check_elements(self, state: ndarray, nu: float) -> None:
        """Check if provided orbital elements have valid values.

        Input
        -----
        `state` : ndarray
            Set of orbital elements sorted as `[a, e, Omega, i, omega]`.

        `nu` : float
            Initial value of the true anomaly.
        """

        a, e, Omega, i, omega = state

        try:

            # Check if the orbit is not elliptical

            if 0. <= e < 1.:
                self.e = e
            else:
                raise ValueError("Eccentricity is higher or equal to 1")

            # Check if the satellite will crash against the moon

            if a * (1. - self.e) > R_MOON:
                self.a = a
            else:
                raise ValueError(
                    "The radius of the periapsis is smaller than R_MOON"
                )

            # Check if all angles have valid values

            angle_list: ndarray = np.array([Omega, i, omega, nu])

            if all(angle_list >= 0.) and all(angle_list < 360.):
                self.Omega = np.deg2rad(Omega)
                self.i = np.deg2rad(i)
                self.omega = np.deg2rad(omega)
                self.nu = np.deg2rad(nu)
            else:
                raise ValueError(
                    "Orientation angle or true anomaly is out of range"
                )

        except ValueError:
            raise

        return None

    def propagate_orbit(
            self, t_span: float, max_deg: int,
            rtol: float = 5e-14, atol: float = 5e-14,
            kpath: str = "spice/metak",
            db_name: str = "grgm1200b"
    ) -> tuple:
        """Import the equations of motion and propagate the orbit.

        Input
        ------
        `t_span` : float

            Diference, in seconds, between the initial and final epochs.

        `max_deg` : int

            Truncation degree for the spherical harmonics expansion of the
            lunar gravity field.

        `rtol` : float, optional

            Relative tolerance to be used by the solver.
            Default is `5e-14`.

        `atol` : float, optional

            Absolute tolerance to be used by the solver.
            Default is `5e-14`.

        `kpath` : str, optional

            Relative (or absolute) path to the file where spice kernels
            to be used are listed.
            Default is `spice/metak`.

        `db_name` : str, optional

            Name of the database containing the set of spherical harmonics'
            coefficients to be used. The name should belong to an SQL database
            stored at the default localhost-location. (Should add a further
            description of the DB format).
            Default is `grgm1200b`.

        Output
        ------
        The method returns a tuple of two NumPy Arrays: sol.t and sol.y.

        The first contains a set of epochs at which the state of the satellite
        is known. The second is made of a set of ndarrays with the state of
        the satellite ( cartesian coordinates in SCRF ) for each of the
        previous epochs.
        """

        natural_motion = motion_law(
            max_deg, kernels_path=bytes(kpath, 'utf-8'))

        t, s = self.solver(
            natural_motion, self.t0, self.t0 + t_span, self.y0,
            rtol=rtol, atol=atol
        )

        return np.asarray(t), np.asarray(s).swapaxes(0, 1)

    def prop_ivp(
        self, t_span: float, max_deg: int,
        method: str = "LSODA", rtol: float = 2.23e-14,
        atol: float = 2.23e-18, db_name: str = "grgm1200b",
        kpath: str = "spice/metak"
    ) -> tuple:

        natural_motion = motion_law(
            max_deg, kernels_path=bytes(kpath, 'utf-8'))

        sol = solve_ivp(
            natural_motion, (self.t0, self.t0 + t_span), self.y0,
            method=method, rtol=rtol, atol=atol
        )

        return sol.t, sol.y
