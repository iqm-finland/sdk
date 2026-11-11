IQM EXA Core
############

:Version: |version|
:Date: |today|

EXA is a set of client libraries for composing flexible and reusable experiments
to be run on IQM quantum computers.
It allows exploring the properties of a quantum computer on a deeper level than the
traditional circuit representation.

The EXA Core library contains the framework upon which Experiments can be built.
It does not contain the Experiments themselves.

Contents
========

The user guide has two parts.
The first part covers the basic setup and usage. If you are new to EXA, start here.
The second part is an extensive guide for writing new Experiments.
The example notebook demonstrates a subset of the features.

.. toctree::
   :maxdepth: 2

   readme

.. toctree::
   :maxdepth: 1

   Basics of EXA <basic_guide/index>
   Creating EXA experiments <guide/index>
   Example Notebooks <notebooks/index>
   API
   Changelog <changelog>
   License <license>
   References <references>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
