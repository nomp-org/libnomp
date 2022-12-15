Introduction
============

nomp: Framework for Domain Specific Compilers Based on Clang and loopy
----------------------------------------------------------------------

`nomp` is a domain specific compiler framework and a runtime for C programming
language which can be used to program accelerators (GPUs, CPUs, etc.) easily
and efficiently.

`nomp` consists of two main components: `nompcc` and `libnomp`. `nompcc` is the
`clang`_ based compiler frontend which pre-process and compile C source.
`libnomp` is the `loopy`_ based runtime which can generate and tune accelerator
code (i.e., kernels) from C source at execution time and then do low level
dispatch to accelerators using other high performing programming models like
OpenCL/Cuda.

`nomp` uses a pragma based programming model to annotate C source with details
on which sections of C source must be executed on accelerators as well as when
to perform data tranfers between driver (most often a CPU) and accelerators.
`nompcc` converts these pragmas to `libnomp` runtime calls during compilation.
Then at program execution time, based on runtime calls, `libnomp` wil perform
kernel generation, transformation, execution and other tasks related to executing
kernels on accelerators by using `loopy` and other much lower level runtimes
like Cuda/OpenCL.

Instead of acting solely based on pragmas, `libnomp` can consume a user written
domain and/or kernel specific transformations script (written using `loopy` API)
at execution time thus giving users more control on kernel generation and
execution. This is an external scrpt that doesn't require any changes in
original C source and can be customized and/or changed without recompiling the
original C source.

Real power of `nomp` comes from its use of `loopy` at execution time. `loopy` is
a code generator for array based code on accelerators. In contrast to mainstream
programming models like OpenCL/CUDA etc., which force users to make
implementation choices at program logic level, `loopy` separate program logic
from implementation details. For example, all the afore-mentioned programming
models force users to decide memory location for arrays at compile time (shared
memory vs global memory vs registers) and mapping of loops to hardware axes.
`loopy` provides an API for users to experiment and tune these details at
execution time and thus providing a more portable way of writing kernels.

`nomp` architecture diagram is shown in the following figure.

.. image:: figures/nomp_diagram.png
   :alt: nomp_architecture.

`nomp` (Originally a recursive acronym: "Nomp isn't OpenMP"), combines unique
features available across several mainstream programming models for
accelerators and complement them with `loopy`. `nomp` is:

#. **Easy to use** since it uses a simple pragma based syntax like OpenMP.
#. **High performant** since it uses high perfomant programming models like
   OpenCL/Cuda for dispatching the kernel to accelerators.
#. **Portable** since it uses `loopy` to separate algorithm and its final
   schedule on hardware. This gives users (a performance enginner) a chance to
   customize/optimize based on problem input and/or target hardware it will be
   executed on.
#. **External** to source code since its pragma based and uses external
   transformation scripts. These can be turned off without making any changes
   to the source code and the C source code would work as expected on CPU.
#. **Customizable** to each domain since it lets users reuse domain specific
   programming patterns.

.. toctree::
   :maxdepth: 3
   :caption: Table of Contents

   self
   build
   usage
   user-api
   internal-api
   developer-docs

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

.. _loopy: https://documen.tician.de/loopy/
.. _clang: https://clang.llvm.org/
