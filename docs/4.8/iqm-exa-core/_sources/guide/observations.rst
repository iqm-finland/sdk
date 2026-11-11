.. _guide_observations:

Producing observations
======================

You should now know how to write the majority of the analysis function and add plots to the
:class:`~exa.core.analyze_result.analyze_result.AnalyzeResult` object.
The other important output of the experiment run are observations. You should make sure you understand how they work
by reading the relevant sections of the basic guide first: :doc:`../basic_guide/observation`.
Note that similarly to analysis, observations are also not entirely mature right now, and some changes might be made in near future.

Why are experiments making observations?
########################################

Just like :doc:`plots <plotting>` are a means of communication with humans, observations are a means of communication
with the software around the quantum computer. Think of an observation as a "unit of knowledge" about the system.
Most observations closely map to settings, with their different kinds explained earlier in the :doc:`sweeping <sweep>`
chapter. The name "observation" is therefore a little bit of a misnomer - you cannot really *observe* that the flux
voltage on the coupler line is -0.034, if you are the one setting this value. Yet **the way to set the value** is to save
a :class:`~exa.core.observation.observation.LocalObservation` for it. You need to get used to this nomenclature
to write good experiments.

There are also observations which do not map to settings. An important kind are the **quality metrics** observations.
Those observations are actually "observations" in the common sense, as they tell us something about the quality of the
system, e.g the :math:`T_1` time or the fidelity of a specific implementation of the CZ gate. Quality metrics observations
need to follow specific syntax to be picked up by the automated tooling which then creates visualizations of how certain
metrics change over space and time.

Finally, you can just save anything you want as an observation. Simply insert any string satisfying certain regex
rules into the :attr:`~exa.core.observation.observation.LocalObservation.dut_field` attribute. It is good practice
to nevertheless apply some logic here and do it for a good reason. Those observations are accessible from the analysis
result - maybe you want them to change the execution logic of a larger sequence of experiments.

Observation structure
#####################

The experiment analysis needs to create specifically :class:`~exa.core.observation.observation.LocalObservation`
objects, and this may be done directly, but is heavily discouraged. You should generally never need to just initialize
this class yourself. Instead, use the appropriate :class:`~exa.core.analyze_result.analyze_result.AnalyzeResult`
methods introduced later.

Let's however have a look at what are the different fields of the class.

.. code-block:: python

    from exa.core.observation import LocalObservation

    obs = LocalObservation.create(
        dut_field=dut_field, # string, a name of the observation
        value=value, # value of the observation, has to be a number, a string or a numpy array (not a list!)
        unit=unit, # string, unit of the observation. Important!
        uncertainty=uncertainty, # standard error/uncertainty in the observation value. Has to use the same unit/format
        tags=tags, # list of strings, observations are fetched from the DB based on the tags
        run_id=run_id, # UUID identifier of the source of the observation
        validation_rules=validation_rules, # any rules used to check the validity of the observation
    )

DUT fields
**********

The DUT field is the name of the observation. For setting observations, it needs to exactly map to a setting using the
dot syntax, e.g ``"controllers.QB1.drive.frequency"``. For quality metrics, it needs to start with ``"metrics"``. The DUT
field decides what the observation is. An observation with the same DUT field, which was created later, will override
any earlier observation with the same DUT field, if it is fetched together or put into the same observation set. When
fetching observations from the DB or stash, only the latest observation for a given DUT field, satisfying certain
loading rules (like specific tags) will be fetched.

The DUT field needs to have a format of a dot separated string, where each segment needs to start with a letter or
the underscore ``"_"``, and the segments cannot include dots ``"."`` or double underscores ``"__"``, unless they represent locus
strings.

Values and uncertainties
************************

A value of the observation can be anything. However, it has to be of the same type as any other observation
with the same DUT field. For setting observations, the value needs to be of the same type as the setting, with an
exception for the settings which are lists - the observation needs to be saved as a numpy array.

Units
*****

The setting units are important, because in :py:mod:`iqm.pulse`, certain logic exists which converts time-like and
frequency-like settings into dimensionless reduced quantities, and the units are used for this purpose. This only matters
for gates settings -- for everything else, just make sure the unit is the same as the previous saved observation and
makes physical sense. Always use the SI unit, and not its derived version, e.g use "s" and not "ns", "Hz" and not "MHz".

Tags
****

Tags are not to be populated by experiments, they can be set by the user later. Unless you have a very good reason, do
not attach any tags to the observations.

Run ID
******

This value is the experiment run ID, which will be populated automatically as long as you use the correct method of
adding observations to your :class:`~exa.core.analyze_result.analyze_result.AnalyzeResult`.

Validation rules
****************

You can add here an object of the class :class:`~exa.core.observation.observation_validation_rules.RangeValidationRule`,
representing a range in which the observation value should realistically be. This helps prevent saving e.g negative
:math:`T_1` times, or single Hz qubit frequencies. It is recommended to do so, but only if the range is actually strictly
limited, e.g do not add an upper limit for :math:`T_1` - they might indeed reach more than 1s one day!

Adding observations to the analysis result
##########################################

Finally, we learn how to do it correctly! This depends on the observation type.

Settings observations
*********************

For the setting observations, use :meth:`~exa.core.analyze_result.analyze_result.AnalyzeResult.add_setting_observation`,
providing the actual setting object for which to save the observation, its value and uncertainty, and optionally
validation rules. Everything else will be automatically created from the settings.

Metrics observations
********************

For the metrics observations, use :meth:`~exa.core.analyze_result.analyze_result.AnalyzeResult.add_observation_with_qon`,
where QON stands for "Qualified Observation Name" and should be first created using either the class
:class:`~iqm.station_control.client.qon.QONMetric` or one of its subclasses. For example, to save gate fidelity, you can
use:

.. code-block:: python

    from iqm.station_control.client.qon import QONGateMetric

    qon = QONGateMetric(
        method = "rb",
        gate = "prx",
        implementation = "drag_crf",
        locus_str = "QB4",
        metric = "fidelity",
        suffixes = {"par": "d2"},
    )

    analyze_result.add_observation_with_qon(qon=qon, value=0.994, unit="", uncertainty=0.002)

Misc observations
*****************

You can also specify the DUT field yourself, by using :meth:`~exa.core.analyze_result.analyze_result.AnalyzeResult.add_observation`.

.. code-block:: python

    analyze_result.add_observation(
        dut_field="misc.QB2.leakage_floor",
        value=0.02,
        unit="",
        uncertainty=0.001,
    )

Conclusions
###########

Now you should know how to add observations to your analysis result. This becomes more important when you start integrating
the experiments together. For now, the main purpose of the observations will be modifying the calibration.




