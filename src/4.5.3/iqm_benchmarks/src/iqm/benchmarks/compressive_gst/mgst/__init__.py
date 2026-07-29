# MIT License
#
# Copyright (c) 2021 Raphael Brieger, Ingo Roth, Martin Kliesch
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""mgst: Compressive Gate Set Tomography core package.

This implementation is a successor of the earlier
[mGST repository](https://github.com/rabrie/mGST/tree/main), with more
features added and optimized for use on IQM-systems. For a usage tutorial,
please refer to the Jupyter notebook under
`docs/examples/example_gst.ipynb`.

The repository contains the Python implementation of a flexible and
efficient gate set tomography technique based on low-rank approximations,
as described in the following research paper.

**Title**: Compressive gate set tomography

**Authors**: Raphael Brieger, Ingo Roth, and Martin Kliesch

**Links**: <https://arxiv.org/abs/2112.05176>,
<https://journals.aps.org/prxquantum/pdf/10.1103/PRXQuantum.4.010325>

## Paper Abstract

Flexible characterization techniques that identify and quantify
experimental imperfections under realistic assumptions are crucial for the
development of quantum computers. Gate set tomography is a characterization
approach that simultaneously and self-consistently extracts a tomographic
description of the implementation of an entire set of quantum gates, as
well as the initial state and measurement, from experimental data.
Obtaining such a detailed picture of the experimental implementation is
associated with high requirements on the number of sequences and their
design, making gate set tomography a challenging task even for only two
qubits.
In this work, we show that low-rank approximations of gate sets can be
obtained from significantly fewer gate sequences and that it is sufficient
to draw them randomly. Such tomographic information is needed for the
crucial task of dealing with coherent noise. To this end, we formulate the
data processing problem of gate set tomography as a rank-constrained tensor
completion problem. We provide an algorithm to solve this problem while
respecting the usual positivity and normalization constraints of quantum
mechanics by using second-order geometrical optimization methods on the
complex Stiefel manifold. Besides the reduction in sequences, we
demonstrate numerically that the algorithm does not rely on structured gate
sets or an elaborate circuit design to robustly perform gate set
tomography. Therefore, it is more flexible than traditional approaches. We
also demonstrate how coherent errors in shadow estimation protocols can be
mitigated using estimates from gate set tomography.

## Use Case

This GST implementation allows users to perform gate set tomography on
quantum devices using low-rank approximations, which reduces the number of
gate sequences and the post-processing time required. The standard sequence
design needed for compressive GST is given by random sequences of short
depth, and the python package provides functions that generate those
sequences based on high level parameters such as the number of gates and
the system size. In its standard configuration, the mGST algorithm assumes
no prior knowledge about the gate set parameters and uses local
optimization coupled with random initializations until a high quality fit
is obtained. Alternatively, an initial guess for the initialization can be
supplied, which can lead to large reductions in runtime.

For single qubit gate set tomography we recommend using 100 sequences with
at least 1000 shots each. For two qubits with model rank up to 4, we
recommend 1000 sequences and also at least 1000 shots each. The runtime
varies for different gate sets and ranks, but is expected to be on the
order of a minute for a single qubit and on the order of an hour for two
qubits. Details can be found in the linked publication.

"""
