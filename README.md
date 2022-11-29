# HPLOP

An open source, High Precision Lunar Orbit Propagator written in Cython, and Python.

## Introduction

HPLOP is a High Precision Lunar Orbit Propagator. It allows to perform simulations of perturbed lunar orbits using accurate models of the lunar gravitational field and JPL's planetary ephemeris. It also includes Cython implementations of several IVP solvers, from basic to state of the art.

*IMPORTANT:* This code is still in development, and is provided "as is". Use it at your own risk.

## Current scope

- Propagate the perturbed, or keplerian orbit of a single lunar satellite.
- Precise models for non-sphericity, and third body perturbations.
- Cython implementations of different IVP solvers.
- The library must be installable through `pip` and `conda`, and system-independent.
- Proper documentation (Readthedocs page).
- Proper testing.

## Expected scope

- Include a precise model for Solar Radiation Pressure.
- Allow users to write, and use their own IVP solvers.
- Configuration tools: Download SPICE kernels, create spherical harmonics database, etc.
- Utilities (maybe out of scope?).

## Out of scope

- Propagate constellations of satellites.
- Attitude determination.
- Control algorithms (GNC & AOCS).
- Graphical representation tools.
