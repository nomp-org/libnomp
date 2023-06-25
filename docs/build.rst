Build Instructions
==================

libnomp
-------

Clone `libnomp`_ repo and change directory to `libnomp`:

.. code-block:: bash

    git clone https://github.com/nomp-org/libnomp.git
    cd libnomp

Install dependencies using `conda`_ to build and run `libnomp`:

.. code-block:: bash

    conda env create -f environment.yml
    conda activate libnomp

You can use mamba to install the dependencies faster.

.. code-block:: bash

    mamba env create -f environment.yml
    mamba activate libnomp

If you are planning on contributing to `libnomp`, install the dev dependencies
instead:

.. code-block:: bash

    conda env create -f environment-dev.yml
    conda activate libnomp-dev

Similarly, you can install the dev dependencies with mamba as well:

.. code-block:: bash

    mamba env create -f environment-dev.yml
    mamba activate libnomp-dev

Use `lncfg` to configure CMake build and `lninstall` to install `libnomp`. To
see the available options, check out `lncfg -h` or `lncfg ---help`. `libnomp`
currently supports the following backends:

#. OpenCL
#. CUDA
#. HIP
#. DPC++
#. ISPC

For example, to build `libnomp` with OpenCL backend enabled, use the following
commands:

.. code-block:: bash

    cd libnomp
    ./lncfg --enable-opencl --install-dir ${HOME}/.nomp
    ./lninstall

If the installation is successful, `lninstall` will prompt you to update the
shell configuration file (e.g., `.bashrc`) to set `NOMP_INSTALL_DIR` and append
`PATH` variable with `NOMP_INSTALL_DIR/bin`. `NOMP_INSTALL_DIR` environment
variable must be set in order to use `libnomp`. This will also enable you to use
`lnrun` script without using its full path to open documentation, run tests,
debug tests, etc. You can avoid this prompt by either providing `--yes` or
`--no` to the `lninstall` script. You have to open a new shell in order for
changes to take affect.

.. code-block:: bash

    cd libnomp
    ./lncfg --enable-opencl --install-dir ${HOME}/.nomp
    ./lninstall --yes

You can manually set the `NOMP_INSTALL_DIR` variable as well.

.. code-block:: bash

    export NOMP_INSTALL_DIR=${HOME}/.nomp

You might additionally want to specify OpenCL library path as below if CMake
can't find OpenCL:

.. code-block:: bash

    ./lncfg --enable-opencl --opencl-lib /lib/x86_64-linux-gnu/libOpenCL.so

If you used `conda` to install OpenCL (for example `pocl`_), do the following:

.. code-block:: bash

    ./lncfg --enable-opencl --opencl-lib ${CONDA_PREFIX}/lib/libOpenCL.so --opencl-inc ${CONDA_PREFIX}/include

nompcc
------

Clone `nompcc`_ repo first and change directory to `llvm-project`:

.. code-block:: bash

    git clone https://github.com/nomp-org/llvm-project.git
    cd llvm-project

If you are using Linux, build the llvm-project as follows:

.. code-block:: bash

    nprocs=$(grep -c ^processor /proc/cpuinfo)
    mkdir build; cd build
    cmake -G "Unix Makefiles" ../llvm                    \
                       -DLLVM_ENABLE_PROJECTS="clang"    \
                       -DLLVM_TARGETS_TO_BUILD="X86"     \
                       -DLLVM_OPTIMIZED_TABLEGEN=ON      \
                       -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                       -DCMAKE_C_COMPILER=`which gcc`    \
                       -DCMAKE_CXX_COMPILER=`which g++`  \
                       -DBUILD_SHARED_LIBS=on
    make -j${nprocs}

If you are using OSX with Apple silicon, build the llvm-project as follows:

.. code-block:: bash

    nprocs=$(sysctl -n hw.ncpu)
    mkdir build; cd build
    cmake -G "Unix Makefiles" ../llvm                                \
                        -DLLVM_ENABLE_PROJECTS="clang"               \
                        -DLLVM_TARGETS_TO_BUILD="AArch64"            \
                        -DLLVM_OPTIMIZED_TABLEGEN=ON                 \
                        -DCMAKE_BUILD_TYPE=RelWithDebInfo            \
                        -DCMAKE_C_COMPILER=`which clang`             \
                        -DCMAKE_CXX_COMPILER=`which clang++`         \
                        -DCMAKE_OSX_ARCHITECTURES='arm64'            \
                        -DDEFAULT_SYSROOT="$(xcrun --show-sdk-path)" \
                        -DBUILD_SHARED_LIBS=on
    make -j${nprocs}

This will build the clang compiler in `bin/clang`. Set environment variable
`NOMP_CLANG_DIR` to point to this clang binary directory:

.. code-block:: bash

    export NOMP_CLANG_DIR=`pwd`/bin


Documentation
-------------

We use `Doxygen` for in source documentations and render those with `Sphinx`
and `Breathe`. These packages must be available if you install the dev
dependencies using `conda`. You can enable docs by passing either `-docs` or
`--enable-docs` option to `lncfg` script.

.. code-block:: bash

    ./lncfg --enable-docs
    ./lninstall

Use `lnrun` to open the user documentation locally. You can specify the browser
with option `-B`. For example, to open the documentation in firefox:

.. code-block:: bash

    lnrun docs -B firefox

If you do not specify the browser, it opens the documentation in chrome by
default.


.. _libnomp: https://github.com/nomp-org/libnomp/
.. _nompcc: https://github.com/nomp-org/llvm-project/
.. _pocl: https://github.com/pocl/pocl/
.. _conda: https://docs.conda.io/en/latest/miniconda.html
