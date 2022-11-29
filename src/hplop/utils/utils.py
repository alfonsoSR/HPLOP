import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
import spiceypy as spice
import sqlite3
import os

# Type alias for NumPy arrays

ndarray = npt.NDArray[np.float64]


@dataclass
class Case:
    """Case configuration dataclass

    Attributes
    -----------
    `root` : str

        Root directory where SPICE `metak` file and kernels are to be found.

    `kernels` : list

        List of url from which unavailable kernels can be retreived.

        Kernels are retrieved via `curl`, so this step of the simulation
        should be done manually if that command is not available.

    `initial_epoch` : str

        UTC epoch at which the state of the satellite is known and
        from which the orbit will be propagated.

        The epoch should be provided as an ISO-formatted string:
        `YYYY-MM-DD hh:mm:ss`.

        Default ephemerides are DE440, so epochs should lie between
        1549-DEC-31 00:00:00 and 2650-JAN-25 00:00:00.

    `initial_state` : ndarray

        Initial state of the satellite. If `cartesian` is set to `False`
        it is assumed to containg a set of classical orbital elements
        sorted as follows: `[a, e, Omega, i, omega]`.

        If `cartesian` is set to `True`, then the `state` should contain
        the cartesian components of the position and velocity vectors.
 
        Both the orbital elements and the vectors are assumed to have
        being defined using the SCRF system.

    `tspan` : float

        Difference between the initial and final epochs of the simulation.

        The difference is assumed to be given in days if `days` is set to
        `True`, and in seconds otherwise.

    `db_name` : str

        Name of the database containing the set of spherical harmonics'
        coefficients to be used. The name should belong to an SQL database
        stored at the default localhost-location. (Should add a further
        description of the DB format).
        Default is `grgm1200b`.

    `harmonics_deg` : int

        Truncation degree for the spherical harmonics expansion of the
        lunar gravity field.

    `cartesian` : bool, optional

        Whether the initial state of the satellite is expressed in terms
        of orbital elements or cartesian coordinates.

        Default is `False`, so orbital elements are used.

    `is_grail` : bool, optional

        DEPRECATED.

        Default is `False`.

    `nu` : float, optional

        Initial value for the true anomaly in degrees.
        Default is 0.

    `days` : bool, optional

        Whether the span of the simulation is given in days or seconds.

        Default is `True`, so days are assumed.

    `rtol` : float, optional

        Relative tolerance to be used by the solver.

        Default is `5e-14`.

    `atol` : float, optional

        Relative tolerance to be used by the solver.

        Default is `5e-14`.
    """
    root: str
    kernels: list
    initial_epoch: str
    initial_state: np.ndarray
    tspan: float
    db_name: str
    harmonics_db: str
    harmonics_deg: int
    db_path: str = "databases"
    cartesian: bool = False
    is_grail: bool = False
    nu: float = 0.
    days: bool = True
    rtol: float = 5e-14
    atol: float = 5e-14


class sqlite:
    """SQLite3 context manager
    
    Input
    -----
    `file_name` : str
        
        Name of the database (without extension) to be used.
    """

    def __init__(self, file_name: str) -> None:

        self.file_name = f"{file_name}.db"
        self.connection = sqlite3.connect(self.file_name)

        return None

    def __enter__(self):

        return self.connection.cursor()

    def __exit__(self, exc_type, exc_value, traceback) -> None:

        self.connection.commit()
        self.connection.close()

        return None


class database:

    def __init__(self, root: str, database: str, file: str) -> None:

        # Check if root directory exists

        if not os.path.isdir(root):

            os.system(f"mkdir {root}")

        # Check if source file exists in root directory

        self.source = f"{root}/{os.path.basename(database)}"

        if not os.path.isfile(self.source):

            print(f"Downloading {self.source}")

            os.system(
                f"curl -# -o {self.source} {database}"
            )

        # Check if database exists and create it otherwise

        self.path = f"{root}/{file}.db"

        if not os.path.isfile(self.path):

            self.create_database()

            pass

        return None

    def create_database(self) -> None:

        with sqlite(self.path) as db:

            db.execute("""
                create table deg (deg int not null, primary key (deg));
            """)

            db.execute("""
                create table ord (
                    id integer primary key autoincrement,
                    fk_deg int not null,
                    ord int not null,
                    Clm double not null,
                    Slm double not null,
                    foreign key(fk_deg) references deg(deg)
                );
            """)

            with open(self.source) as source:

                rows = source.readlines()[1:]

                for row in rows:

                    _d, _o, _C, _S = row.split(",")[:4]

                    d = int(_d)
                    o = int(_o)
                    C = float(_C)
                    S = float(_S)

                    try:

                        db.execute(
                            """
                                insert into deg (deg) values (?);
                            """, (d,)
                        )

                    except sqlite3.IntegrityError:

                        pass

                    db.execute(
                        """
                            insert into ord (id, fk_deg, ord, Clm, Slm)
                            values (NULL, ?, ?, ?, ?);
                        """, (d, o, C, S)
                    )

        return None


class kernels:
    """Context manager for SPICE kernels"""

    def __init__(self, root: str) -> None:

        self.root = root
        spice.furnsh(f"{self.root}/metak")

    def __enter__(self) -> None:

        return None

    def __exit__(self, exc_type, exc_value, traceback):

        spice.kclear()


def get_kernels(root: str, kernels: list) -> None:

    # Check if root directory exists and create it otherwise

    if not os.path.isdir(root):

        os.system(f"mkdir {root}")

    # Check if kernels' directory exists and create it otherwise

    kpath = f"{root}/kernels"

    if not os.path.isdir(kpath):

        os.system(f"mkdir {kpath}")

    # Check if all kernels are available and download the missing ones

    with open(f"{root}/metak", "w") as metak:

        metak.write(r"\begindata" + "\nKERNELS_TO_LOAD=(\n")

        for idx, kernel in enumerate(kernels):

            file = os.path.basename(kernel)

            if not os.path.exists(f"{kpath}/{file}"):

                print(f"Downloading {file}")

                os.system(f"curl -# -o {kpath}/{file} {kernel}")

            metak.write(f"'{root}/kernels/{file}'")

            if idx != len(kernels):

                metak.write(",\n")

        metak.write(')\n'+r'\begintext'+'\n')

    return None


def case_info(case: Case) -> None:

    print("#########################################")
    print(f"Initial epoch: {case.initial_epoch}")

    if case.days:

        print(f"Time span: {case.tspan} days")

    else:

        print(f"Time span: {(case.tspan / (24. * 3600.)):0.4f} days")

    print(f"Harmonics database: {case.db_name}")
    print(f"Harmonics degree: {case.harmonics_deg}")


def state2rv(state: np.ndarray) -> tuple:
    """Get radius and velocity modulus from state vector

    Input
    ------
    `state` : ndarray
        Satellite's state in a series of epochs.

    Output
    ------
    `r` : ndarray
        Orbital radius at each epoch

    `v` : ndarray
        Orbital velocity at each epoch
    """

    r = np.sqrt(state[0]*state[0] + state[1]*state[1] + state[2]*state[2])
    v = np.sqrt(state[3]*state[3] + state[4]*state[4] + state[5]*state[5])

    return r, v


def rel_error(orbit: np.ndarray, ref: np.ndarray) -> tuple:
    """Position and velocity relative errors with respect to
    a reference orbit.

    Input
    -----
    `orbit` : ndarray
        State vectors' array for the orbit of interest.

    `ref`   : ndarray
        State vectors' array for the orbit to be used as reference

    Output
    ------
    `dr`    : ndarray
        Relative position error for each epoch.

    `dv`    : ndarray
        Relative velocity error for each epoch.
    """

    r, v = state2rv(orbit)

    r_ref, v_ref = state2rv(ref)

    dr = (r_ref - r)/r_ref
    dv = (v_ref - v)/v_ref

    return dr, dv


def grail_orbit(t: ndarray, case: Case) -> ndarray:

    with kernels(case.root):

        grail = np.swapaxes(
            spice.spkezr(
                "GRAIL-A", t, "J2000", "NONE", "moon"
            )[0], 0, 1  # type: ignore
        )

    return grail


def save_results(t: ndarray, s: ndarray, file: str) -> None:

    sol = np.concatenate((t[None, :], s))

    np.save(file, sol)

    print(f"Results saved to {file}.npy")
