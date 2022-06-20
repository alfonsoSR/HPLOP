import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from hplop.pck.pck import moon_data
import hplop.utils.utils as ut

# Type aliases

float_array = npt.NDArray[np.float64]

# Moon parameters

MU_MOON, R_MOON = moon_data()


def orbit_3D(*orbits: float_array) -> None:
    """3D representation of an orbit.

    Input
    ------
    `orbit` : ndarray((n, 6))
        Array defining the kinematic state of the satellite at different
        epochs.

        Each state consists on six cartesian components of the position
        and velocity vectors: [x, y, z, u, v, w]
    """
    fig = plt.figure(facecolor=("#292c33"))

    ax = fig.add_subplot(projection="3d", proj_type='ortho')

    ax.set_facecolor('#292c33')

    ax.axis('off')

    for orbit in orbits:
        ax.plot(orbit[0], orbit[1], orbit[2])

    ax.set_box_aspect(
        np.ptp([ax.get_xlim(), ax.get_ylim(), ax.get_zlim()], axis=1)
    )

    plt.show()

    return None


def satellite_altitude(t: float_array, orbit: float_array) -> None:
    """Graphical representation of the evolution with time of the
    satellite's altitude with respect to mean lunar radius.

    Input
    ------
    `t` : ndarray((n,))
        Epochs when the satellite's state is known.

    `orbit` : ndarray((n, 6))
        Array defining the kinematic state of the satellite at different
        epochs.

        Each state consists on six cartesian components of the position
        and velocity vectors: [x, y, z, u, v, w]
    """

    h = np.sqrt(np.sum(((orbit*orbit)[:3]), axis=0)) - R_MOON

    t = (t - t[0])/(24.*3600.)

    fig, ax = plt.subplots()

    ax.plot(t, h)

    ax.set_xlabel("Days since initial epoch")
    ax.set_ylabel("Height above lunar surface [km]")
    ax.set_title("Evolution of satellite's altitude")

    plt.show()

    return None


def position_vector(*states: tuple) -> None:
    '''Time-variation of the position vector components

    Input
    -----

    `states`: tuple, *
        Each tuple is expected to have two elements: t, and orbit.
        `t` : ndarray
            Epochs when the satellite's state is known.

        `orbit` : ndarray
            Array defining the kinematic state of the satellite at different
            epochs.

            Each state consists on six cartesian components of the position
            and velocity vectors: [x, y, z, u, v, w]'''

    fig, ax = plt.subplots(3, 1, sharex=True)

    labels = ["$x - x_{ref}$", "$y - y_{ref}$", "$z - z_{ref}$"]

    for idx, label in enumerate(labels):

        ax[idx].set_ylabel(label)  # type: ignore

    for t, orbit in states:

        t = (t - t[0])/(24.*3600.)

        for ax_i, r_i, label in zip(ax, orbit[:3], labels):  # type: ignore

            ax_i.plot(t, r_i, label=label)

    ax[-1].set_xlabel("Days since initial epoch")  # type: ignore

    plt.show()

    return None


def velocity_vector(*states: tuple) -> None:
    '''Time-variation of the velocity vector components

    Input
    -----

    `states`: tuple, *
        Each tuple is expected to have two elements: t, and orbit.
        `t` : ndarray
            Epochs when the satellite's state is known.

        `orbit` : ndarray
            Array defining the kinematic state of the satellite at different
            epochs.

            Each state consists on six cartesian components of the position
            and velocity vectors: [x, y, z, u, v, w]'''

    fig, ax = plt.subplots(3, 1, sharex=True)

    labels = ["$u - u_{ref}$", "$v - v_{ref}$", "$w - w_{ref}$"]

    for idx, label in enumerate(labels):

        ax[idx].set_ylabel(label)  # type: ignore

    for t, orbit in states:

        t = (t - t[0])/(24.*3600.)

        for ax_i, r_i, label in zip(ax, orbit[:3], labels):  # type: ignore

            ax_i.plot(t, r_i, label=label)

    ax[-1].set_xlabel("Days since initial epoch")  # type: ignore

    plt.show()

    return None


def error(t: float_array, orbit: float_array, ref: float_array) -> None:

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(11, 5))

    labels = [r"$1 - \dfrac{r}{r_{ref}}$", r"$1 - \dfrac{v}{v_{ref}}$"]

    for idx, label in enumerate(labels):

        ax[idx].set_ylabel(label)  # type: ignore

    t = (t - t[0])/(24.*3600.)

    dr, dv = ut.rel_error(orbit, ref)

    # diff = orbit - ref

    # dr = np.sqrt(diff[0]*diff[0] + diff[1]*diff[1] + diff[2]*diff[2])
    # dv = np.sqrt(diff[3]*diff[3] + diff[4]*diff[4] + diff[5]*diff[5])

    ax[0].plot(t, dr)  # type: ignore
    ax[1].plot(t, dv)  # type: ignore

    ax[-1].set_xlabel("Days since initial epoch")  # type: ignore

    plt.show()

    return None


def compute_periapsis(orbit: float_array) -> float_array:
    """Compute the radius of periapsis from array of states

    Input
    ------
    `orbit` : ndarray((n, 6))
        Array defining the kinematic state of the satellite at different
        epochs.

        Each state consists on six cartesian components of the position
        and velocity vectors: [x, y, z, u, v, w]

    Output
    ------
    `r_p` : ndarray((n,))
        Periapsis' radius of the osculating orbit fitting each state.

    """

    r = np.sqrt(np.sum(((orbit*orbit)[:3]), axis=0))
    v = np.sqrt(np.sum(((orbit*orbit)[3:]), axis=0))

    h_vec = np.array([
        orbit[1]*orbit[5] - orbit[2]*orbit[4],
        orbit[2]*orbit[3] - orbit[0]*orbit[5],
        orbit[0]*orbit[4] - orbit[1]*orbit[3]
    ])

    h = np.sqrt(np.sum(h_vec*h_vec, axis=0))

    p = h*h/MU_MOON

    e = np.sqrt(
        1. + p*(v*v/MU_MOON - 2./r)
    )

    return p/(1. + e)


def periapsis(t: float_array, *orbits: float_array, show: bool = True) -> None:
    """Graphical representation of the evolution with time of the
    periapsis' radius.

    Input
    ------
    `t` : ndarray((n,))
        Epochs when the satellite's state is known.

    `orbits` : tuple(ndarray((n, 6)))
        Tuple of arrays defining the kinematic state of each satellite at
        different epochs.

        Each state consists on six cartesian components of the position
        and velocity vectors: [x, y, z, u, v, w]

    `show` : bool, optional
        Automatically show the plot. Default is True.
    """

    r_ps = (compute_periapsis(orbit) for orbit in orbits)

    t = (t - t[0])/(24.*3600.)

    fig, ax = plt.subplots()

    for r_p in r_ps:

        ax.plot(t, r_p)

    ax.set_xlabel("Days since initial epoch")
    ax.set_ylabel("Radius of periapse [km]")
    ax.set_title("Evolution with time of the periapsis' radius")

    if show:

        plt.show()

    return None

###############################################
#
# INCLUDE AN ESTIMATION OF THE HAMILTONIAN
#
###############################################

# EOF
