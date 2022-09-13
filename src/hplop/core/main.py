from hplop.pck.pck import moon_data
from hplop.utils.utils import Case, kernels, case_info, get_kernels, database
from datetime import datetime
import spiceypy as spice
import numpy.typing as npt
import numpy as np
from hplop.core.equations import Kepler, motion_law
from hplop.solver.rkn1210 import rkn1210
from scipy.integrate import solve_ivp

# Type alias for numpy arrays

ndarray = npt.NDArray[np.float64]

# Global

MU_MOON, R_MOON = moon_data()


class main:

    def __init__(self, case: Case) -> None:
        """High Precision Lunar Orbit Propagator (HPLOP) Constructor

        Input
        -----
        `case` : Case
            Case definition via configuration dataclass provided in `utils`.
        """

        self.case = case

        # ----- Display information about the case ----- #

        case_info(self.case)

        # ----- Check if harmonics' database is available ----- #

        database(case.db_path, case.harmonics_db, case.db_name)

        # ----- Assure that all kernels are available ----- #

        get_kernels(case.root, case.kernels)

        # ----- Transform time span to seconds if given in days ----- #

        if self.case.days:

            self.tspan = self.case.tspan * 24. * 3600.

        else:

            self.tspan = self.case.tspan

        # ----- Initial epoch in TDB ----- #

        with kernels(self.case.root):

            try:

                # Check if date format is correct by using a function that
                # fails if not (Should be replaced by something better)

                datetime.fromisoformat(self.case.initial_epoch)
                self.t0 = spice.str2et(self.case.initial_epoch)

            except ValueError:

                raise

        # ----- Initial state ----- #

        if self.case.cartesian:

            # Should check if input is reasonable

            self.y0 = self.case.initial_state

        else:

            # ----- Check orbital elements ----- #

            self.check_elements(self.case.initial_state, self.case.nu)

            # ----- Initialize keplerian orbit ----- #

            kepler = Kepler(
                self.a, self.e, self.Omega, self.i, self.omega, self.t0
            )

            self.y0 = kepler.nu2state(self.nu)

        # ----- Initialize solver ----- #

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

    def propagate_orbit(self) -> tuple:

        with kernels(self.case.root):

            natural_motion = motion_law(
                self.case.harmonics_deg,
                db_name=f"{self.case.db_path}/{self.case.db_name}.db",
                kernels_path=bytes(f"{self.case.root}/metak", "utf-8")
            )

            t, s = self.solver(
                natural_motion, self.t0, self.t0 + self.tspan, self.y0,
                rtol=self.case.rtol, atol=self.case.atol
            )

            return np.asarray(t), np.asarray(s).swapaxes(0, 1)

    def prop_ivp(self, method: str = "LSODA") -> tuple:

        with kernels(self.case.root):

            natural_motion = motion_law(
                self.case.harmonics_deg,
                db_name=f"{self.case.db_path}/{self.case.db_name}.db",
                kernels_path=bytes(f"{self.case.root}/metak", "utf-8")
            )

            sol = solve_ivp(
                natural_motion, (self.t0, self.t0 + self.tspan),
                self.y0, method=method, rtol=2.23e-14,
                atol=2.23e-18
            )  # type: ignore

        return sol.t, sol.y

    # def prop_ivp(
    #     self, t_span: float, max_deg: int,
    #     method: str = "LSODA", rtol: float = 2.23e-14,
    #     atol: float = 2.23e-18, db_name: str = "grgm1200b",
    #     kpath: str = "spice/metak"
    # ) -> tuple:

    #     natural_motion = motion_law(
    #         max_deg, kernels_path=bytes(kpath, 'utf-8'))

    #     sol = solve_ivp(
    #         natural_motion, (self.t0, self.t0 + t_span), self.y0,
    #         method=method, rtol=rtol, atol=atol
    #     )

    #     return sol.t, sol.y
