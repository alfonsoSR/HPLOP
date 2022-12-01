:html_theme.sidebar_secondary.remove:

.. toctree::
   :maxdepth: 1
   :hidden:

   Getting Started <getting_started/index>
   User Guide <user_guide/index>
   API Reference <api_reference/index>
   Contributing <contributor_guide/index>


HPLOP documentation
===================

**Version**: |version|

HPLOP is a High Precision Lunar Orbit Propagator. It allows to perform simulations of perturbed lunar orbits using accurate models of the lunar gravitational field and JPL's planetary ephemeris. It also includes Cython implementations of several IVP solvers, from basic to state of the art.

.. grid:: 2

   .. grid-item-card:: 
      :img-top: _static/index-images/getting_started.svg
      :link: getting_started
      :link-type: ref

      Getting Started

      ^^^

      If this is your first time working with HPLOP, check out the getting started guide. You will learn how to install the package and to propagate an orbit from scratch.


   .. grid-item-card::
      :img-top: _static/index-images/user_guide.svg
      :link: user_guide
      :link-type: ref

      User Guide

      ^^^

      The user guide contains detailed information about HPLOP: from a description of the structure of the library, to detailed explanations regarding perturbation models.

   .. grid-item-card::
      :img-top:  _static/index-images/api.svg
      :link: api
      :link-type: ref

      API Reference

      ^^^

      The reference guide contains a detailed description of all the functions, modules, and objects included in HPLOP. The reference describes how the methods work and which parameters can be used.

   .. grid-item-card::
      :img-top:  _static/index-images/contributor.svg
      :link: contributor
      :link-type: ref

      Contributor's Guide

      ^^^

      In the contributor's guide you will find all the information you need to contribute to this project.
