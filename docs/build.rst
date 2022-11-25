Build Instructions
==================

libnomp
-------

Clone the repo first

.. code-block:: bash

    git clone https://github.com/nomp-org/libnomp.git

Install dependencies using conda to build and run `libnomp`

.. code-block:: bash

    conda env create -f environment.yml
    conda activate libnomp

You can use mamba to install the dependencies faster

.. code-block:: bash

    mamba env create -f environment.yml
    mamba activate libnomp

If you are planning on contributing to `libnomp`, install the dev dependencies
instead

.. code-block:: bash

    conda env create -f environment-dev.yml
    conda activate libnomp-dev

Similarly, you can install the dev dependencies with mamba as well

.. code-block:: bash

    mamba env create -f environment-dev.yml
    mamba activate libnomp-dev

Use `lncfg` to configure cmake for libnomp and `lninstall` to install libnomp.
For the available options, you can check `lncfg -h`.

.. code-block:: bash

    cd libnomp
    ./lncfg
    ./lninstall

You might additionally want to specify OpenCL libray path like below if CMake
can't find OpenCL

.. code-block:: bash

    ./lncfg -o /lib/x86_64-linux-gnu/libOpenCL.so.1

Clang frontend
--------------

Clone the llvm-project repo first

.. code-block:: bash

    git clone https://github.com/nomp-org/llvm-project.git

If you are using Linux, build the llvm-project as follows

.. code-block:: bash

    nprocs=$(grep -c ^processor /proc/cpuinfo)
    cd llvm-project
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

If you are using OSX with Apple silicon, build the llvm-project as follows

.. code-block:: bash

    nprocs=$(sysctl -n hw.ncpu)
    cd llvm-project
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

This will build clang compiler in `bin/clang`. Set `NOMP_CLANG_DIR` to point to
this clang binary directory

.. code-block:: bash

    export NOMP_CLANG_DIR=`pwd`/bin
