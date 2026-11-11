.. _guide_analysis:

Automated data analysis
=======================

You have your circuit and sweeps, and you have run the experiment. Now it is time to analyze the data. We have seen a
simple analysis function in the :doc:`prototyping <prototype>` chapter. Here we will explore analysis in more detail.

By the time the analysis function is called, automatic post-processing has happened and produced the
:class:`~iqm.cpc.core.run_result.RunResult` object for us to use. The post-processing can be adjusted via post-processing
passes, which belong to the :doc:`compiler <compiler>` chapter, but we will here assume the default post-processing has
run. The division between post-processing and analysis is not clear-cut, and it is possible to arbitrarily process
data during the analysis. If possible, using compiler passes is preferred, but the analysis is the most flexible
part of the framework for a reason.

In this chapter, we will learn about how to approach analysing data produced by the :class:`~exa.core.experiment.experiment.Experiment`
class, different fitting methods, some basics of operating on :py:mod:`xarray`, as well as some additional tricks to making
your analysis fast and reliable.

Purpose of analysis
###################

Automatic analysis is where your experiment finally produces the information it was designed to produce. This is done by
calculation of one or multiple **derived quantities**. It may proceed in phases, deriving further quantities from ones
derived earlier.

Remember, analysis is a communication tool! It is important to strike the balance between communicating to the experimentalist using the
experiment class, and enabling further automation. For an experimentalist, keeping all the derived quantities making
the experiment visualization easy is important. It enables flexible use of the experiment class, as a starting point
for further research. For an automated process, speed, reliability, and keeping just enough data to explain why
something went wrong, are crucial. Each raw dataset is stored in the database, but the analysis needs
to run every time it is recovered. Thus the speed requirement for analysis should be stricter than for the execution,
as it could be performed multiple times for the same raw data.

.. note::
    Think about the purpose of the experiment class. Sometimes, producing a single number at the end is not the
    goal, but instead visualizing some 2D quantity. This is commonly true for experiments where the circuit may be
    complicated, and the measured quantity is highly sensitive to small variations in the input parameters.

The analysis is allowed to create observations and plots. However, plots should never be built in the main body of the
analysis, because they can be pretty expensive, both in time and memory, even on relatively small datasets, and the
automated routines consisting of many experiments typically do not need to make them at all. Instead, make all the
actual plotting in a separate :doc:`plotting function <plotting>`.

Observations are described in a :doc:`later chapter <observations>` of this guide. Each observation is either a quantity
metric, or it should map to somewhere in the setting tree. For this reason, think about if producing observations
makes sense with arbitrary extra sweeps. Maybe it does with some sweeps, but not others?

As mentioned before, the analysis framework is not very mature, but it is very flexible. Still, organizing it internally
into multiple stages using separate functions that can be then reused is a good practice, even if not enforced by
a structure of some compiler.


Basics of analysis functions
############################

A good way of starting the analysis is to create the :class:`~exa.core.analyze_result.analyze_result.AnalyzeResult`,
and extract the dataset and settings from it.

.. code-block:: python

    result = AnalyzeResult(run_result)
    dataset = result.dataset
    settings = result.settings

Then, try to work as much as possible with the ``result`` variable, adding arrays, observations and plots to it. You
can use these methods:

.. code-block:: python

    from exa.core.analyze_result import PlotDefinition
    from iqm.station_control.client.qon import QONGateMetric

    qon_metric = QONGateMetric(*args)

    result.add_arrays(array_1, array_2, array_3)
    result.plots["plot_name"] = PlotDefinition(plotting_function, default_kwargs)
    result.add_setting_observation(setting=setting_node, value=value, uncertainty=error)
    result.add_observation_with_qon(qon=qon_metric, value=gate_fidelity)

A useful function is :func:`iqm.cpc.core.dataset.annotate`, which attaches Parameter to a :class:`xarray.DataArray`:

.. code-block:: python

    from iqm.cpc.core.dataset import annotate
    from exa.common.data import DataType, Parameter

    array_1_param = Parameter("optimal_time", "Optimal Time to do the thing", "s", data_type=DataType.FLOAT)

    annotate(array_1, array_1_param, prefix=prefix)
    result.add_arrays(array_1)

It modifies the array in place, but also returns it, which makes it easier to use it on views into some bigger array.
The parameter label is automatically used as a plot label when plotting from :py:mod:`xarray`, and the ``prefix``
argument, typically the name of the qubit or a pair, is appended to both the parameter name and label by the
:func:`~iqm.cpc.core.dataset.annotate` function.

.. note::
    **Q: What quantities should I add to the dataset? I would like to avoid from it becoming too large.**

    A: Come back to the purpose of your experiment. If it is meant to be a flexible research tool, generally any
    quantity carrying useful information should be added to the dataset. Such a flexible tool could be run with extra
    sweeps added by the user. For this reason, even the **zero-dimensional quantities**, like the final observations
    produced along with their uncertainties, should be added to the dataset, making sure any extra sweeps become a new
    dimension on this variable. Stashing information, for example fit parameters, in the attributes of the array is not
    recommended for this reason, as it makes it harder to plot them. However, some of the fit parameters are not really
    **interesting** on their own, so skipping adding them to the dataset is OK.

As a reminder, the analysis function needs to **return** the :class:`~exa.core.analyze_result.analyze_result.AnalyzeResult`
object, if it created a new one.

Arguments and data variables
############################

Similarly to the compiler passes, the analysis functions can take some extra context arguments.

1. Arguments ``chip_topology`` (:class:`~exa.common.qcm_data.chip_topology.ChipTopology` ), ``chip_properties`` and
   ``gate_definitions`` (mapping of names of :class:`~iqm.pulse.quantum_ops.QuantumOp` s) are always available.
2. Additionally, when the first argument is ``group`` and you're using the parallel execution framework, the arguments
   ``data_color`` (:class:`~xarray.Dataset`, reduced to only contain data pertaining to this group), and ``circuit_index``
   (index of the circuit that contains this color group, relevant for sequential execution of overlapping groups)
   are available.
3. In the parallel framework, ``analyze_result`` could also be given as an argument, in which case a new
   :class:`~exa.core.analyze_result.analyze_result.AnalyzeResult` needs not be created or returned. The function should
   instead modify the received object.

The dataset will contain some raw and post-processed data, depending on the
:attr:`~exa.core.experiment_options.ExperimentOptions.readout_mode` option. Each of these :class:`~xarray.DataArray` s
name is of the format ``f"{label}_{name}"``, where the ``label`` is by default a qubit name, such as ``'QB1'``. If the
measure operations were performed with custom keys (which can be added through the ``key`` argument when creating the
timebox), the labels are ``f"{qubit}__{key}"``. This helps separate data collected with different kinds of readouts.
For example, the default heralding key is ``"__HERALD"``, resulting in data variable labels like ``"QB1____HERALD"``.

The default readout variables are:

1. The variable ``readout`` contains **averaged** raw measurement results, either the averaged complex or averaged
   raw counts, depending on the readout mode - complex in ``complex`` and ``single_shot`` modes, averaged counts
   otherwise.
2. The variable ``readout_single_shot`` contains single shot **complex** data, individual in-phase and quadrature
   amplitudes of the integrated signal. This variable always has a ``repetitions`` dimension, corresponding to shots.
   It is only present if the readout mode is ``single_shot``.
3. The variable ``state_single_shot`` contains single shot thresholded data, each being either 0 or 1. It also has a
   ``repetitions`` dimension. If the readout mode is ``single_shot_threshold``, it is what is returned from the readout
   instrument. In the ``single_shot`` (complex) readout mode, the post-processing passes produce this variable by
   thresholding in software, if the readout calibration information is already available. In the ``counter`` mode,
   it is calculated from the counts.
4. The variables ``contrast`` and ``phase_optimal_quad`` are calculated in ``complex`` and ``single_shot`` modes
   by performing a principal component analysis of the averaged complex data along the first dimension. They correspond
   to the amplitude and phase of the first principal component. These are useful variables before the readout is
   calibrated.
5. The variable ``excited_state_probability`` is the **readout-error-mitigated** best estimate for the probability
   of the qubit collapsing to the excited state when measured. In ``complex`` and ``single_shot`` modes, it is
   calculated from the averaged complex data using the average complex response of the ground and excited state
   measured with a readout calibration experiment. In the other modes, it is calculated from the averaged counts by
   inverting the readout assignment matrix. This mitigation is necessary so the variable is unbiasedly the same in all
   readout modes, assuming no higher states are present.
6. The ``counter.result`` variable does not have a label, because it corresponds to all the qubits. It is the default
   variable when the readout mode is ``counter``, but also calculated in ``single_shot`` and ``single_shot_threshold``
   modes. These are the raw, unmitigated frequencies of different bitstrings (like "110", "010") measured across all
   qubits.
7. Only when using ``counter`` mode, if the ``subscribed_components`` are a mapping from names to subsets of qubits,
   there will be additional variables ``f"{group_name}_counter_readout"``, containing counts only for those subsets.
   The ordering of the qubits within this subset is inside ``run_result.additional_run_properties["readout_groups"]``,
   and will need to be extracted if the ordering of the bits inside the bitstring is important. This is the default
   behaviour when using the ``counter`` mode while ``parallel_group`` is ``CONNECTED_PAIRS``, in which case the names of
   the groups are the names of couplers connecting the qubits and the groups each have two qubits.

The general principle behind the post-processing is to always give the user all the available information, so all the
variables which are possible to be calculated, are calculated. You could always remove or add more post-processing
passes, if that is what you require!

Extra arguments
***************

Analysis functions are allowed to have extra arguments, which can be populated when calling
:meth:`~exa.core.experiment.experiment.Experiment.analyze`. Make sure that the analyze call without any
arguments works almost all, or ideally, all the time. Each experimental data are different, and a parameter default that works
on one station might not work on another. It is good practice to make the arguments dimensionless and derive as much
as possible from the chip and component properties stored in the settings.

.. note::
    For example, flux pulse amplitude modifies the qubit frequency, but this relationship is non-linear. The qubit
    frequency itself has some maximum and minimum. In an appropriate context, "max relative qubit frequency shift" might
    be a better argument than "max qubit flux pulse amplitude". Those analysis knobs can be turned instead of rewriting
    the analysis. Including arguments with defaults in your function signature is better than using magic numbers, but
    an automated routine will at least need an automated way of adjusting them.

Manipulating xarray arrays
##########################

All the data in analysis are represented as an :py:mod:`xarray` :class:`~xarray.DataArray` s. Those structures generally
behave similarly to numpy arrays, and can be manipulated directly, using already implemented "utility functions"
or "ufuncs". Those operations will generally always return a :class:`~xarray.DataArray` and maintain all the dimension
structure, including extra sweeps, and so using them should be your first priority. Let's have a look at some examples.

.. code-block:: python

    import xarray as xr

    da1 = xr.DataArray([[1, 2, 3], [4, 1, 1]], dims=('y', 'x'))
    da2 = xr.DataArray([4, 2, 1], dims='x')

    doubled = 2 * da1

    # xarray will handle the extra dimensions on both arrays. Each slice corresponding to a specific "y" is multiplied
    # separately. Then the slices are combined, and the product has dimensions of the larger array
    array_product = da1 * da2

    # bitwise XOR, useful for QEC applications
    array_xor = da1 ^ da2

    # when calculating statistical measures, specify dimensions to avoid collapsing the array to a single number!
    da1_mean_x = da1.mean("x") # dimensions: "y"
    da1_std_x = da1.std("x")

The arrays in analysis will have dimensions and coordinates corresponding to sweeps. Additionally, single shot arrays
will have a ``repetitions`` dimension and coordinate. Instead of indexing with integers, use ``sel``, ``isel`` or ``loc``
to ensure the correct slice is selected.

.. code-block:: python

    # phase is output of exa.core.util.phase_helper.get_phase_array
    cphase = (phase.sel({"conditioned": True}, drop=True) - phase.sel({"conditioned": False}, drop=True)) % (2 * np.pi)

This allows the original ``phase`` array to have any extra coordinates. Here we used ``sel`` which selects based on the
**value** of the coordinate. This is generally the correct approach, but you might also want to use ``isel``, which
selects based on the **index** -- you need to know what is the correct index, but you don't need to
know the value, which is helpful if it was generated automatically.

Handling pointwise sweeps
*************************

This is one of consequences of the analysis not being entirely mature. In the chapter about :doc:`sweeping <sweep>`
we discussed pointwise sweeps, or sweeps where multiple settings are swept in parallel. The way those are handled in
:py:mod:`xarray` is that all swept settings become coordinates of the dataset, but only the first one becomes
a dimension, and the dimensions are the same for all arrays.

For example, let's say we ran an experiment on many qubits, and swept the amplitude of a PRX gate on all of them as
a pointwise sweep. Then we want to do some analysis on it.

.. code-block:: python

    target_data = dataset[f"{qubit}_excited_state_probability"]

    prx_amplitude_setting_name = settings.get_gate_node_for_locus("prx", qubit).amplitude_i.name

    # this data array is a coordinate, and it has only one dimension
    amplitude_dimension = dataset[prx_amplitude_setting_name].dims[0] # dimension of all other arrays

    first_point = target_data.isel({amplitude_dimension: 0}, drop=True)

Using the actual setting name, not the dimension name, will sometimes work, but it is generally safer to use the
dimension name. In the above scenario, both of these names are quite convoluted, and will generally only differ by the
name of the qubit and optionally, the PRX gate implementation.

Changing coordinates and dims
*****************************

Sometimes there is a need to change the coordinate of the data. For example, the swept quantity was voltage on the
flux line of the coupler, but the analysis needs the coupler frequency to derive meaningful results. We can use
some :py:mod:`xarray` functionality to address this problem.

First, we can use :meth:`~xarray.DataArray.assign_coords` to assign new coordinates to our array.

.. code-block:: python

    import xarray as xr
    from iqm.cpc.core.dataset import annotate

    # Obtain coupler_frequency from the coupler voltages, for
    # example, say using a custom function called get_frequency_from_voltage:
    coupler_frequency: xr.DataArray = get_frequency_from_voltage(coupler_voltage, **dispersion_parameters)
    
    annotate(coupler_frequency, "coupler_frequency", prefix=coupler_name)

    # assign_coords returns a new object
    target_data = target_data.assign_coords({coupler_frequency.name: coupler_frequency})

    # to make the coordinate aligned with the coupler_voltage explicitly:
    target_data = target_data.assign_coords({coupler_frequency.name: (coupler_voltage.name, coupler_frequency)})

Now, this will add a new coordinate, but this coordinate is not a dimension of the array. Maybe this is fine, but if
it is not, you should follow it up with:

.. code-block:: python

    swapped = target_data.swap_dims({coupler_voltage.name: coupler_frequency.name})

The method :meth:`~xarray.DataArray.swap_dims` returns a **new** :class:`~xarray.DataArray` with swapped dims.

.. note::
    Be careful with swapping dims on the arrays already in the dataset. This might prevent the analysis from being
    re-runnable, as the dataset is modified in-place, and the :meth:`~xarray.DataArray.swap_dims` method will expect
    the old dimension to still be a dimension. If you invoke it twice, it will break on the second call.

Other useful methods of :class:`~xarray.DataArray` and :py:mod:`xarray` functionalities include:

1. Concatenation of multiple arrays along a new or existing coordinate: :func:`~xarray.concat`.
2. Merging multiple :class:`~xarray.Dataset` objects (you need to be careful in analysis): :meth:`~xarray.Dataset.merge` or :func:`~xarray.merge`.
3. Renaming the variables, :meth:`~xarray.DataArray.rename`.

Additionally, you can make coordinate arrays directly from :class:`~exa.common.data.parameter.Parameter`.

.. code-block:: python

    from exa.common.data import DataType, Parameter
    import numpy as np

    coord_array_1_param = Parameter("optimal_time", "Optimal Time to do the thing", "s", data_type=DataType.FLOAT)
    coord_array_1 = coord_array_1_param.build_data_array(np.linspace(0.0, 1e-7, 100))

Which populates attributes like ``standard_name``, ``long_name`` correctly. Those arrays are useful when combined with
concatenation of multiple arrays.

.. code-block:: python

    import xarray as xr

    array1 = coord_array_1_param.build_data_array(np.linspace(0.0, 1e-7, 50))
    array2 = coord_array_1_param.build_data_array(np.linspace(0.0, 1e-7, 50))
    concatenated_array = xr.concat([array1,array2], coord_array_1)

This produces a new array with a new coordinate and dimension given by ``coord_array_1``.

Using apply_ufunc
*****************

Many numpy utility functions have been ported to :py:mod:`xarray`, but not all. Some will still work on xarray objects,
but produce a numpy array. In such a scenario, first try using :func:`~xarray.apply_ufunc`. A good example is
:func:`~numpy.unwrap`, since it is commonly used in quantum experiment analysis:

.. code-block:: python

    import numpy as np
    from xarray import apply_ufunc

    unwrapped = apply_ufunc(
        np.unwrap,
        phase_array,
        input_core_dims=[[vz_param_name]],
        output_core_dims=[[vz_param_name]],
        vectorize = False # np.unwrap is already vectorized
    )

Check the documentation for :func:`~xarray.apply_ufunc` to learn more tricks. Most importantly, ``input_core_dims`` and
``output_core_dims`` are always lists of lists of dimension names, corresponding to the dimensions along which **not** to vectorize,
or in other words, the dimensions the original utility function expects the data to have. In the unwrapping example, it
is the dimension along which to unwrap. Because the unwrapping function **takes** one array, there is only one list of
dimensions. Each list corresponds to a separate array. Because the unwrapping function **returns** one array, the
argument ``output_core_dims`` also has only one element. Finally, the argument ``vectorize`` is set to ``False``, because
the unwrapping function is already vectorized.

.. note::
    **Q: What functions can I use inside apply_ufunc?**

    A: Any function which takes and returns some numpy arrays will work. If some of the arguments the original
    function needs should be fixed, or are not :class:`~xarray.DataArray` s, you might consider writing a small wrapper which fixes
    them. Otherwise, you may write your own function, which takes some arrays and returns some arrays, and then setting
    ``vectorize = True`` will make :func:`~xarray.apply_ufunc` work with xarray arrays of any dimensions, albeit
    the implementation of that is no more efficient than a for loop. Before going down this rabbit hole, please consider if
    the functions :func:`~iqm.cpc.core.dataset.apply_along_coordinate` or :func:`~iqm.cpc.core.dataset.multifit_along_coordinates`
    are not better suited to the task.


Fitting
#######

Often, you would like to specifically fit some model to the data, in order to extract some fitted parameters. These
tasks form a subset of "manipulating xarray arrays", and can in principle be done using :func:`~xarray.apply_ufunc` and
arbitrary fitting tools. However, there already exist more specific helpers for fitting, which will also produce
properly annotated arrays and sometimes even plots.

The basic tools are written on top of :func:`scipy.optimize.curve_fit`, and should be used if the problem is quite
simple, there is not much variability, and performance is important. Then, :class:`exa.core.model_selection_fitter.fitter.Fitter`
is built on top of :py:mod:`lmfit` and supports more advanced statistics and automatic selection of the best model. Both
work fully with :py:mod:`xarray` and allow arbitrary extra dimensions.

Using curve_fit
***************

First, let's have a look at :func:`~iqm.cpc.core.dataset.apply_along_coordinate`. This helper function is made mostly
for fitting, but it might be used for other purposes. Let's have a look at the syntax:

.. code-block:: python

    fit, param_0_array, param_1_array, param_1_stddev_array = multifit_along_coordinates(
        dataset[target_name], # data array to apply the function to
        coord=coordinate_name, # coordinate along which to apply it
        func=fitting_function, # fitting function of specific input/output
        returns=["param_0", "param_1", "param_1_stddev",], # additionally returned array names
        prefix=qubit, # add this to all array names other than the fit
        add_to=dataset, # add all arrays to this dataset - can be None
    )

The most important mechanics to understand here is the interplay of the outputs of ``fitting_function``, the ``returns``
argument and the returned arrays. The ``fitting_function`` must take only two 1D numpy arrays, understood to be
x-values and y-values representing some data to be fitted, and return two things: first, a dictionary mapping arbitrary
names of fit parameters to their values, and second - a numpy array representing the fit.

The names given to ``returns`` are keys in that dictionary. The function will package these values into data arrays.
The first returned array is the ``fit``, which will have the same dimensions and coordinates as the input array. All
the other arrays are fit parameters, and they will be missing the dimension corresponding to the given coordinate along
which the fit is performed.

Exa contains several typical fitting functions, and it is not difficult to write your own. The ``fitting_function`` is
allowed to do anything, as long as the output is correctly formatted. These examples are in
:py:mod:`~exa.core.util.model_functions`. Typically, you'd write a function that takes some meta-parameters, or guesses,
and that returns the fitting function proper, eg:

.. code-block:: python

    from exa.core.util.model_functions import get_oscillation_fitter
    fit, period, period_stddev = apply_along_coordinate(
        analyze_result.run_result.dataset[target_name],
        coord=amplitude_sweep,
        func=get_oscillation_fitter(**{key: value.get(qubit) for key, value in guesses.items()}),
        returns=["period", "period_stddev"],
    )

To generate these fitting functions we can use the :func:`iqm.cpc.core.dataset.compute_1d_fit` function. This function
takes the x and y data, as well as a model function and some guesses, and wraps :func:`scipy.optimize.curve_fit` to
produce output in the correct format. The keys in the returned dict are the names of the arguments of the model
function plus those same names with a ``_stddev`` suffix, which is the standard deviation calculated from the output
covariance matrix for each parameter.

.. note::
    **Q: The limitation to only 1D models is quite severe, why wouldn't I just use apply_ufunc?**

    A: Realistically this is what you need most of the time, and all you need to define is the model function and
    some guessing logic, for which there are many examples. With raw :func:`~xarray.apply_ufunc`, you need to do the same calls
    to :func:`scipy.optimize.curve_fit` every time, extract the uncertainties yourself etc, which generates a lot of
    bloat. If you need something more powerful, be sure to check the next section though.


Using Model Selection Fitter
****************************

Alternatively to using :func:`~iqm.cpc.core.dataset.apply_along_coordinate`, you may want to consider a solution built
on top of :py:mod:`lmfit` called the Model Selection Fitter (MSF). Please check the
`lmfit documentation <https://lmfit.github.io/lmfit-py/index.html>`_ to learn more about making your own models, since
it is something you will need. This section only covers the usage of MSF.

What MSF does for you:

1. Fit multiple :class:`lmfit.Model` s to the data.
2. Parallelize over extra dimensions.
3. Select the best model, separately for each sweep spot, based on the Akaike Information Criterion.
4. Wrap everything into :class:`~xarray.DataArray` s.
5. Generate informative plot definitions.

Let's look at an example. A common reason to select from multiple models arises in a :math:`T_2` Ramsey experiment,
where the signal not only oscillates, but also decays. This decay might be a simple exponential decay, a Gaussian decay,
or a combination of both. That is, we need three different models.

.. code-block:: python

    import exa.core.model_selection_fitter as msf

    fitter = msf.Fitter()
    model_exp = msf.GaussExpSineModel(name="exp")
    model_exp.set_param_hint(name="T_gauss", value=np.inf, vary=False)
    model_gauss = msf.GaussExpSineModel(name="gauss")
    model_gauss.set_param_hint(name="T_exp", value=np.inf, vary=False)
    model_both = msf.GaussExpSineModel(name="both")

    fitter.add_model(model_gauss)
    fitter.add_model(model_exp)
    fitter.add_model(model_both)
    fitter_options = {"max_nfev": 1500, "method": "leastsq", "nan_policy": "raise"}

    # set qubit name:
    # qubit = ...

    target_data = analyze_result.dataset
    target_name = f"{qubit}_excited_state_probability"

    fit_result = fitter.fit(
        target_data, # Dataset or DataArray
        datavar=target_name, # name of the DataArray within the Dataset, ignored if DataArray given
        t="pulse_delay", # name of the independent variable t as a coordinate of the DataArray
        fitter_options=fitter_options,
    )

    best_fit = annotate(fit_result["best_fit"], "fit")
    r_squared = annotate(fit_result.r2s, "R2")
    t_2 = annotate(fit_result.T_2, "T2")
    t_2_stderr = annotate(fit_result.T_2_stderr, "T2_stddev",)

    best_model_index = np.atleast_1d(fit_result.best_model_index.values).flat[0]
    plot_definition = fit_result.plot_definitions.isel(models=best_model_index).values.flat[0]
    analyze_result.plots[qubit] = plot_definition

Here we construct different models by simply taking the same model and setting some of the parameters to not vary.
As you can see, Model Selection Fitter produces many extra arrays, such as :math:`R^2` or standard errors of each
fit parameter. It also produces a plot definition, which can be immediately attached to the analyze result. More about
those in the :doc:`plotting <plotting>` chapter.

.. note::
    **Q: Why are you annotating? I thought that is done automatically...**

    A: Apply along coordinates does annotate the data arrays, but MSF doesn't, so you need to do it yourself to the
    output.

Model Selection Fitter uses the model's :meth:`lmfit.Model.guess` method, so it should always be defined. Additionally,
it is good practice to make sure each of the used models has the same fit parameters, even if they are fixed or
quantities derived from other parameters via an expression. This will ensure proper error propagation, which lmfit
will perform for you, using the full covariance matrix -- sometimes, the uncertainty in some derived quantity can be
**lower** than either of the uncertainties of the "original" parameters, if those parameters are strongly correlated.

The :py:mod:`lmfit` library supports models with an arbitrary number of independent variables, and MSF will correctly
package them into :class:`~xarray.DataArray` s. Check the collection of models in Exa and in the lmfit library proper
for examples.

Fitting 2-dimensional data
**************************

In most analysis protocols fitting only a single dimension of the data at the time is sufficient. Additionally,
collecting data along only a single dimension is much faster, and reducing the number of dimensions your experiment
needs is one of the most effective ways of making it quicker to execute. But what if your problem is fundamentally larger?

As mentioned above, the :py:mod:`lmfit` library and by extension MSF support two-dimensional models. Simply define a
model with two independent coordinates and some parameters and proceed like described in the previous section. You might
even assume the shape of one of the dimensions, e.g. always using two frequencies for a Ramsey experiment.

There is a variation of this problem which would require substantial modifications to lmfit Models to solve, and there
is a solution to it in Exa. This is where you have multiple 1-D datasets which generally follow the same kind of model,
but some parameters vary between them. For example, multiple exponential decays which are expected to have the same
decay rate, but different amplitudes. Or :math:`T_2` traces where the Gaussian part is considered the same for all
traces, but the :math:`T_1` contribution varies quickly. To fit such a dataset, we need to treat some parameters as
**global**, affecting all traces, and some as **local**, affecting just one. If all parameters are global, the problem
reduces to a 2-D fit; if all are local, it reduces to multiple 1-D fits.

There is a tool called :func:`~iqm.cpc.core.dataset.multifit_along_coordinates` which allows you to fit your data like that.
Let's go through its usage.

The function itself is mostly an :py:mod:`xarray` wrapper that works similarly to :func:`~iqm.cpc.core.dataset.apply_along_coordinate`
with slightly different arguments:

.. code-block:: python

    fit, param_0_array, param_1_array, param_1_stddev_array = multifit_along_coordinates(
        dataset[target_name], # data array to apply the function to
        main_coord=main_coordinate_name, # the coordinate describing the 1-D slice
        supp_coord=supp_coordinate_name, # the coordinate describing the second index
        func=fitting_function, # fitting function of specific input/output
        returns=["param_0", "param_1", "param_1_stddev",], # additionally returned array names
        cloned_arguments=["param_0"], # names of arguments which are cloned the number of times equal to shape of supp_coord
        prefix=qubit, # add this to all array names other than the fit
        add_to=dataset, # add all arrays to this dataset - can be None
    )

In the above example, the ``main_coord`` is the independent coordinate along which regular 1-D fitting would take place.
In the example with multiple decaying exponentials, this would be e.g. the ``time`` variable. The ``supp_coord`` is the
other coordinate. This coordinate could be an index of measurement, initial state, or something else that **modifies**
the main experiment, which is measuring the exponential decay in time.

The ``cloned_arguments`` is a list of arguments to ``fitting_function`` which will be cloned along ``supp_coord``. The
number of times it is cloned depends on the number of points in ``supp_coord``. If there are 5 points in ``supp_coord``,
there will be 5 parameters in the model, independently fitted to the data, all representing the variable ``param_0``,
but for different values of ``supp_coord``.

The returned arrays will have different shapes and dimensions depending on which arguments were cloned. The ``param_0_array``
will have a dimension ``supp_coordinate_name``, which ``param_1_array`` will not have. Of course ``fit`` is the total
2-D array which has the same dimensions as the input data array.

For this to work, the ``fitting_function`` has to be quite specific. It should still return a dict of fitted parameters
and a fit, but the values in the dict need to be themselves either numbers or 1-D numpy arrays, depending on which
arguments have been "cloned". This means this function needs to take a kwarg ``cloned_arguments`` too, to know what to do.

Thus the ``fitting_function`` has 3 positional arguments: x, y and z, corresponding to the main coord, supp coord and data,
as well as the 4th, keyword argument ``cloned_arguments``. Note that if anything is cloned, x and y are not equivalent,
as the cloning is done along y.

The simplest way to make such a function is to use :func:`~iqm.cpc.core.dataset.compute_2d_fit`:

.. code-block:: python

    def _model(x, y, param_0, param_1):
        #.. do some math
        return fit # 2d array

    def _fitting_function(x, y, z, cloned_arguments=None):

        estimates = get_estimates(x, y, z, cloned_arguments)
        fit_results = compute_2d_fit(
            x, # 1-D array corresponding to main_coord
            y, # 1-D array corresponding to supp_coord
            z, # 2-D array corresponding to data
            _model, # model function returning a 2d array
            estimates, # dict of model estimates, cloned arguments should be given as arrays
            cloned_arguments, # list of cloned arguments
            only_index_y=True # if True, replace the values of y with 0, 1, 2, ... for the model evaluation
        )
        return fit_results

Inside this function all the heavy lifting associated with upgrading the simple model to a model with some arguments
cloned is done for you. This enables changing which arguments are cloned without rewriting the fitting function.

The actual model function should be written to work with parameters ``param_0``, ``param_1`` being numbers or 1-D numpy
arrays with the same shape as y. If the model only uses simple arithmetic, numpy will generally do it automatically.
Note that y itself does not need to be used in the model and often is not - the
functional relationship between y and the data is captured through the cloned arguments. This means the actual values
of y could be anything, e.g. strings, and setting ``only_index_y=True`` is essentially saying that the value of y is not
important.

If your model does explicitly depend on y and you're not cloning any arguments, it is most likely easier to just use
the model selection fitter instead.

Conclusions
###########

You should now know how to write a good analysis function that works with :class:`~exa.core.experiment.experiment.Experiment`.
Now let's learn more about observations and plotting.
