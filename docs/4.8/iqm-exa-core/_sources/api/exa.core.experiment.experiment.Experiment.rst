Experiment
==========

.. currentmodule:: exa.core.experiment.experiment

Module: :mod:`exa.core.experiment.experiment`

.. autoclass:: Experiment
   :members:
   :private-members: _run
   :show-inheritance:

   
   
   .. rubric:: Attributes

   .. autosummary::
   
      ~Experiment.allowed_component_grouping_modes
      ~Experiment.allowed_qpu_types
      ~Experiment.chad
      ~Experiment.chip_layout
      ~Experiment.chip_topology
      ~Experiment.circuit_stages
      ~Experiment.coupler_mapping
      ~Experiment.default_label
      ~Experiment.default_options
      ~Experiment.executor
      ~Experiment.final_stages
      ~Experiment.gate_definitions
      ~Experiment.gates
      ~Experiment.input_type
      ~Experiment.last_run
      ~Experiment.name
      ~Experiment.observation_handler
      ~Experiment.parallel_group
      ~Experiment.pp_stages
      ~Experiment.pulse_stages
      ~Experiment.storage
   
   


   
   
   .. rubric:: Methods

   .. autosummary::
      :nosignatures:
   
      ~Experiment.analyze
      ~Experiment.chip_properties
      ~Experiment.create
      ~Experiment.default_analyze_function
      ~Experiment.default_circuit_generation_function
      ~Experiment.default_sweep_generation_function
      ~Experiment.default_sweep_parameters
      ~Experiment.from_dut_label
      ~Experiment.get_by_job_id
      ~Experiment.get_by_run_id
      ~Experiment.get_default_analyze_function
      ~Experiment.get_default_circuit_generation_function
      ~Experiment.get_default_circuit_settings
      ~Experiment.get_default_sweep_generation_function
      ~Experiment.get_default_sweeps
      ~Experiment.get_default_sweeps_as_dict
      ~Experiment.get_settings
      ~Experiment.has_required_gates
      ~Experiment.inspect_segment
      ~Experiment.make_final_mutations
      ~Experiment.print_default_sweeps
      ~Experiment.process_run_components
      ~Experiment.register_gate_implementation
      ~Experiment.remove_operations
      ~Experiment.run
      ~Experiment.set_calibration_data_for_gate_implementation
      ~Experiment.set_gate_implementation_as_default
      ~Experiment.setup_additional_init_properties
      ~Experiment.submit_run
      ~Experiment.update_compilation_context
      ~Experiment.update_settings
   
   



.. rubric:: Inheritance

.. inheritance-diagram:: exa.core.experiment.experiment.Experiment
   :parts: 1
