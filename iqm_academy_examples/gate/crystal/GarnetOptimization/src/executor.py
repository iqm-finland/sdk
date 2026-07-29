import numpy as np
import os.path
import json
from iqm.qiskit_iqm import IQMJob


class NotBegunException(Exception):
    "Raised when a status update is requested before circuits have been submitted."
    pass


class CircuitExecutor:
    """
    Abstract class to define CircuitExecutor objects.
    """

    def __init__(self):
        raise Exception("That's an abstract Class. Use subclasses!")


class Resonance_batched_executor(CircuitExecutor):
    """
    Batched executor for use with resonance. Takes a list of circuits and divide them in several batches, each with a maximum number of instruction given by max_batch.
    These batches are submitted to Resonance, and can be retrieved at a later time using the metadata folder defined by filename.
    """

    def __init__(
        self,
        backend,
        max_circuit=500,
        max_batch=4500,
        strip_metadata=False,
        initial_layout=None,
        max_tries=10,
        max_concurrent=10000,
    ):
        """

        Initialises a Resonance_bached_executor instance. Use this to submit large numbers of circuits to resonance.

        :param backend: IQM backend to be used (e.g. Garnet, Deneb.) from IQMProvider(server_url, token=api_token).get_backend()
        :param max_circuit: maximum number of circuits per batch. Default 500
        :param max_batch: maximum number of instructions per batch. Default 4500
        :param strip_metadata: set True to remove circuit metadata. Default False
        :param initial_layout: NOT currently implemented.  Default None
        :param max_tries : How many times a batch will be submitted to Resonance before giving up.
        :param max_concurrent: How many jobs will submitted in one submission (via .run)

        returns: a Resonance_batched_executor object.

        """
        self.backend = backend
        self.max_circuit = max_circuit
        self.max_batch = max_batch
        self.strip_metadata = strip_metadata
        self.initial_layout = initial_layout
        self.max_tries = max_tries
        self.max_concurrent = max_concurrent

    def run(
        self,
        filename: str,
        list_of_circuits,
        shots: int,
        verbose=True,
        calibration_set_id=None,
    ):
        """

        Submits batches of circuits to resonance.

        :param filename: A folder where status data, job_ids and results will be saved. Identifies project.
        :param list_of_circuits: a list of compiled circuits to be evaluated.
        :param shots: number of shots to perform for each circuit. Currently must be the same for each circuit.
        :param verbose: Provides additional information during running.  Default True
        :param calibration_set_id : NOT currently implemented.


        returns: None

        """
        # First we see if we are beginning a new job, or resuming an ongoing one.

        Custom_Calibration_Implemented = False  # Not currently available.

        if os.path.isdir(filename):
            with open(filename + "/job_id.json", "r") as f:
                job_ids = json.load(f)
            with open(filename + "/status.json", "r") as g:
                status_list = json.load(g)
        else:
            os.makedirs(filename)
            job_ids = []
            status_list = []

        # Batching
        circuit_instructions = [len(c) for c in list_of_circuits]
        batch = []
        list_of_batches = []
        instruction_counter = 0
        circuit_counter = 0

        if verbose:
            print(
                "Dealing with {} circuits, {:.3e} instructions, and {:.3e} total shots.".format(
                    len(list_of_circuits),
                    np.sum(circuit_instructions),
                    len(list_of_circuits) * shots,
                )
            )

        for c, l in zip(list_of_circuits, circuit_instructions):

            if self.strip_metadata:
                new_c = c.copy()
                new_c.metadata = None
            else:
                new_c = c

            if (
                (self.max_batch is None) or (instruction_counter + l <= self.max_batch)
            ) and (
                (self.max_circuit is None) or (circuit_counter + 1 <= self.max_circuit)
            ):  # If there is still space in the current batch
                batch.append(new_c)
                instruction_counter += l
                circuit_counter += 1

            else:  # The current batch is full: add it to the list ans start with a new one
                list_of_batches.append(batch)
                batch = [new_c]
                instruction_counter = l
                circuit_counter = 1

        list_of_batches.append(batch)
        if verbose:
            print("Created {} batches!".format(len(list_of_batches)))
            print(
                "Optimistic and approximate runtime: {} seconds".format(
                    9 * len(list_of_batches) + len(list_of_circuits) * shots / 1000
                )
            )

        # keep track of #circuits in each batch 1.
        with open(filename + "/batch_sizes.json", "w") as b:
            json.dump([len(batch) for batch in list_of_batches], b, indent=2)

        # Running
        number_of_batches = len(list_of_batches)
        self.number_of_batches = number_of_batches
        if len(status_list) > 0:
            summary, current_status = self.status(filename)
            if verbose:
                print("resuming on:", summary)
        else:
            if verbose:
                print("beginning first submission!")
            current_status = []
            summary = []

        execution_backend = self.backend

        total_running = 0
        for j, batch in enumerate(list_of_batches):
            if j < len(
                current_status
            ):  # in which case, have to check the job's status.
                job_status = current_status[j]
                if (
                    job_status == "DONE"
                    or job_status == "INITIALIZING"
                    or job_status == "QUEUED"
                    or job_status == "RUNNNING"
                    or job_status == "VALIDATING"
                ):  # move on
                    if job_status == "DONE":
                        None
                    else:
                        total_running += 1
                else:
                    if len(status_list[j]) < self.max_tries:
                        if total_running < self.max_concurrent:
                            if verbose:
                                print(
                                    "Submitting batch # {} (out of {})".format(
                                        j, len(list_of_batches)
                                    ),
                                    end="\r",
                                )

                            if Custom_Calibration_Implemented:
                                resubmitted_job = execution_backend.run(
                                    batch,
                                    calibration_set_id=calibration_set_id,
                                    shots=shots,
                                )
                            else:
                                resubmitted_job = execution_backend.run(
                                    batch, shots=shots
                                )

                            total_running += 1
                            Current_Id = resubmitted_job.job_id()

                            status_list[j].append(resubmitted_job.status().name)
                            job_ids[j].append(resubmitted_job.job_id())

                            with open(filename + "/job_id.json", "w") as f:
                                json.dump(
                                    job_ids, f, indent=2
                                )  # indent=2 is not needed but makes the file human-readable
                            with open(filename + "/status.json", "w") as g:
                                json.dump(status_list, g, indent=2)

                        else:
                            print(
                                "maximum waiting jobs has been met. Please wait and try again."
                            )
                            break
                    else:
                        print("Batch %s Has exceeded the maximum number of tries" % (j))

            else:  # this means the batch has never been submitted.
                if total_running >= self.max_concurrent:
                    print(
                        "maximum waiting jobs has been met. Please wait and try again."
                    )
                    break
                else:
                    if verbose:
                        print(
                            "Submitting batch # {} (out of {})".format(
                                j, len(list_of_batches)
                            ),
                            end="\r",
                        )
                    if Custom_Calibration_Implemented:
                        submitted_job = execution_backend.run(
                            batch, calibration_set_id=calibration_set_id, shots=shots
                        )
                    else:
                        submitted_job = execution_backend.run(batch, shots=shots)

                    total_running += 1
                    Current_Id = submitted_job.job_id()

                    status_list.append([submitted_job.status().name])
                    job_ids.append([submitted_job.job_id()])

                    with open(filename + "/job_id.json", "w") as f:
                        json.dump(job_ids, f, indent=2)
                    with open(filename + "/status.json", "w") as g:
                        json.dump(status_list, g, indent=2)

    def status(self, filename, update=True):
        """

        Provides the user with the status of all submitted batches.

        :param filename: A folder where status data, job_ids and results will be saved. Identifies project.
        :param update: if True, returns the latest info from the server. If False, uses saved info. Default True.

        returns: Summary: a dictionary giving the number of jobs with each status.
                 current_status: of list of the status of every submitted job.

        """
        try:
            if os.path.isdir(filename):

                with open(filename + "/status.json", "r") as g:
                    status_list = json.load(g)

                if len(status_list) == 0:
                    raise NotBegunException
                else:
                    if update == True:
                        with open(filename + "/job_id.json", "r") as f:
                            job_ids = json.load(f)
                        for batch_no, ids in enumerate(job_ids):
                            last_id = ids[-1]

                            current_job = IQMJob(self.backend, job_id=last_id)
                            status_list[batch_no][-1] = current_job.status().name

                        with open(filename + "/status.json", "w") as g:
                            json.dump(status_list, g, indent=2)

                    current_status = [k[-1] for k in status_list]
                    Possible_Status = [
                        "CANCELLED",
                        "DONE",
                        "ERROR",
                        "INITIALIZING",
                        "QUEUED",
                        "RUNNING",
                        "VALIDATING",
                    ]
                    Summary = [
                        (status, current_status.count(status))
                        for status in Possible_Status
                    ]
            else:
                raise NotBegunException

            return Summary, current_status
        except NotBegunException:
            print("Must have run circuits before asking for status.")

    def return_results(self, filename, save=True, update=True):
        """

        Retrieves the string outcomes of every batch, and presents the status of each incomplete job.

        :param filename: A folder where status data, job_ids and results will be saved. Identifies project.
        :param update: if True, returns the latest info from the server. If False, uses saved info. Default True.

        returns: Summary: counts. If all batches finished, a list of qiskit count dictionaries.
                                  If not all finished, a list of batches providing either count dictionaries
                                                       or status updates.

        """
        if update == True:
            try:
                if os.path.isdir(filename):

                    self.status(filename, update)  # Have to get the current updates.

                    with open(filename + "/job_id.json", "r") as f:
                        job_ids = json.load(f)

                    with open(filename + "/status.json", "r") as g:
                        status_list = json.load(g)

                    try:
                        number_of_batches = self.number_of_batches
                    except:
                        # keep track of #circuits in each batch 1.
                        with open(filename + "/batch_sizes.json", "r") as b:
                            batch_sizes = json.load(b)
                        number_of_batches = len(batch_sizes)
                        self.number_of_batches = number_of_batches

                    summary, current_status = self.status(filename)
                    summary_dict = {k[0]: k[1] for k in summary}
                    if summary_dict["DONE"] == self.number_of_batches:
                        print(
                            "All batches finished; presenting final results in original list form"
                        )
                        counts = []
                        with open(filename + "/batch_sizes.json", "r") as b:
                            batch_sizes = json.load(b)

                        for batch_number in range(self.number_of_batches):
                            current_id = job_ids[batch_number][-1]

                            current_job = IQMJob(self.backend, job_id=current_id)
                            batch_counts = current_job.result().get_counts()

                            if batch_sizes[batch_number] == 1:
                                batch_counts = [batch_counts]
                            counts += batch_counts

                    else:
                        print(
                            "Not all batches finished; presenting partial results in batch form"
                        )
                        counts = []
                        with open(filename + "/batch_sizes.json", "r") as b:
                            batch_sizes = json.load(b)
                        for batch_number in range(self.number_of_batches):

                            if batch_number >= len(job_ids):  # never been submitted
                                counts.append("Submission 0 status: never submitted. ")
                            elif status_list[batch_number][-1] == "DONE":

                                current_id = job_ids[batch_number][-1]
                                current_job = IQMJob(self.backend, job_id=current_id)
                                batch_counts = current_job.result().get_counts()
                                if batch_sizes[batch_number] == 1:
                                    batch_counts = [batch_counts]
                                counts.append(batch_counts)
                            else:
                                counts.append(
                                    "Submission %s status: %s "
                                    % (
                                        len(status_list[batch_number]),
                                        status_list[batch_number][-1],
                                    )
                                )
                    # save here
                    with open(filename + "/results.json", "w") as h:
                        json.dump(counts, h, indent=2)
                    return counts
                else:
                    raise NotBegunException

            except NotBegunException:
                print("Must have run circuits before asking for results.")
        else:
            try:
                if os.path.isdir(filename):
                    with open(filename + "/results.json", "r") as h:
                        counts = json.load(h)
                    return counts
                else:
                    raise NotBegunException
            except NotBegunException:
                print("Must have run circuits before asking for results.")
