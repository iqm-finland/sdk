.. _guide_maturity:

Is your experiment mature?
==========================

We have reached the end of this guide! Congratulations. You should now have all the information needed to design
great Exa experiments, which future generations of physicists will enjoy. Before we close off, let's talk about
**maturity** of your experiment class.

What is maturity of an experiment class?
########################################

When we are born, we require full attention of an adult, typically a parent. Then we grow up and become gradually
more independent and responsible for our choices. Our guardians begin to trust us, giving us more freedom, and
eventually we begin our own independent lives. We continue to grow, learn and make mistakes, but on our own.

Maturity of an experiment class is achieved when the experiment is able to function with no involvement from its
original creator. When you first add the experiment, or any code, to the shared codebase, you might assign yourself
as an owner of that code. You continue teaching colleagues how to use it and fixing bugs. Eventually, no one
needs you to run, fix bugs and add features to the experiment and then it becomes mature.

Signs of immaturity
###################

Let's go through some obvious signs of immaturity.

You are needed
**************

If your colleagues need you, your presence, your replies to their messages or your involvement to:

1. Be able to run the code -- for example, because the code is not part of a common codebase, but lives in some notebook,
   which you have to send out -- your experiment is still a baby.
2. Be able to understand what the code is doing -- typically, because the physics behind the experiment is complicated
   and it is not clearly explained in training materials -- your experiment is a teenager.
3. Be able to fix bugs in the code -- usually because the amount and quality of documentation is lacking, or the code
   itself is too complicated -- your experiment is not mature.

Ideally, you as the original designer, are never needed at all and you don't even know if the experiment is run.
Getting there, will however require work.

Analysis is missing cases
*************************

Experimental physics is tricky. There is no foolproof way to automatically test an experiment, because it is a piece
of code which runs physics experiments and attempts to understand the results. Sometimes, accidentally,
an experimental run will run into physics which may not be entirely new (don't get your hopes up for a Nobel Prize!),
but may have been unexpected in this context.

Often the analysis might be improved to handle these additional cases. This might only occur after you
start measuring a new device or something fundamental changes -- about the current device or the rest of the software stack.
You might not necessarily be the person making these changes, but you do hold certain implied responsibility over them,
like a parent of a kid who is misbehaving at school.

Crutches are needed for effective use
*************************************

When using a mature, independent, well-scoped experiment class, you should generally only ever need three lines of code:

.. code-block:: python

    result = experiment.run(components).analyze()
    result.render_report()
    result.observations.tag(tags).save()

If you notice that your colleagues are sharing among them "code snippets" which are **any** longer than the above,
regarding the standalone use of your experiment, it has still not graduated high school. Of course, options, settings,
analyze arguments exist for a reason - they should be changed for *non-standard* use. Anything that is shared as
a useful snippet, where the same options, settings and analyze arguments are changed in the exact same way every single
time eventually **becomes a standard** and as such, should be integrated into an existing or a new
experiment class.

Let your experiment grow up at its own pace
###########################################

Releasing a toddler into the world on their own is probably not the smartest idea, but you can and should let a high
schooler have their freedom. When merging/pulling your experiment into a common codebase, do not wait until it is
perfectly mature and ready. For sure, introducing bugs you know about is not good, but rarely does the experiment need
to cover all the experimental bases and be entirely self-contained to be useful. Keeping an experiment in a feature
branch until it is perfect will prevent your coworkers from gaining any real value from it, and will prevent you from
learning what the real issues are.

Aim for a piece of code which:

1. Does what it is designed to do in several useful cases.
2. Is well documented and tested.
3. Can be reviewed in 15-20 minutes by someone of the same level of knowledge as you.
4. Is easy to extend/improve upon.

Initially, you might need to still "parent" that code for a while. Maybe your coworkers discover that the default options
should be different, or make better plots than you - integrate them in using a follow-up MR/PR.

Other useful tips
#################

Keep experiments modular
************************

Because of the modular nature of the :class:`~exa.core.experiment.experiment.Experiment` class, where circuit generation,
sweeps and analysis are attached as static methods, and compiler passes can be freely modified, it is generally strongly
recommended to **inherit each subclass directly from Experiment**. This has very few exceptions, usually having to do
with large frameworks of several methods being added to a new base class, which takes place of the parent Experiment.
Sometimes you will need to copy a small amount of code from a different source, or define a function. This is ok.
Making a chain of 10 experiments all deriving from each other leads to much bigger problems.

Test what is important
**********************

When writing automatic tests, aim to "break" the experiment, aim for things that are actually important for it to get right.
We can generally trust that :py:mod:`numpy` will work correctly, but if your experiment involves really tricky physics,
and that physics might be misunderstood by a coworker who didn't study this area in detail and changed accidentally,
cover that part in well-designed automatic tests. The tests can take time to run, so adding multiple tests that just
check trivial things is typically a waste of compute.

Add references to physics papers where necessary
************************************************

When writing documentation, it is possible to use Sphinx ``:cite:`` to include citations to scientific articles. If
your experiment is derived from such an article, consider adding a citation to the docstring.

Profile the execution time of your experiment
*********************************************

Running physics experiments, even on a fast platform like superconducting qubits, can take time. But constructions
of run definitions and analysis should not be just a little bit faster, it should be a lot faster. As chips grow,
the number of involved qubits will grow, but the execution time may remain the same or very similar. It is important
to keep the classical part also fast. You may use tools such as ``snakeviz``, ``cProfile`` or :py:mod:`timeit` to
check how performant are different parts of the experiment. In crucial applications, accelerating the execution by
writing parts of it in a lower-level, compiled language, such as Rust or C++, could be the solution.

Be wary of introducing additional dependencies
**********************************************

Python is a language commonly used by physicists in academia to create useful frameworks. You might want to use some of
them in your experiments. When writing production-quality code, consider if a package you're introducing is well and actively supported.
This can be judged for example by the activity on the project on GitHub, number of releases last year, support of latest
Python etc. Oftentimes the person driving the package development leaves the academic group and the package becomes
effectively abandoned. Depending on abandoned work will introduce issues to your codebase down the line. In such situations,
try to find an alternative solution. Depending on the license, you might be able to simply copy the code from an
abandoned open-source project to your codebase. Remember to follow the license in such cases, sometimes simply attributing
the original contributors is sufficient.

Conclusions
###########

Now, you should have everything you need to design and take care of an Exa experiment subclass until it reaches
its full potential. Good luck and farewell!




