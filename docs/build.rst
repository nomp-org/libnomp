Build instructions
==================

libnomp
-------

Clone the libnomp repo first::

    git clone https://github.com/nomp-org/libnomp.git

Install dependencies using conda to build and run `libnomp`::

    conda env create -f environment.yml
    conda activate libnomp

If you are planning on contributing to `libnomp`, install the dev dependencies
instead::

    conda env create -f environment-dev.yml
    conda activate libnomp-dev

Use `cmake` to build the repo after installing the dependencies::

    cd libnomp
    mkdir build; cd build
    export NOMP_INSTALL_DIR=${HOME}/.nomp
    cmake .. -DCMAKE_INSTALL_PREFIX=${NOMP_INSTALL_DIR}
    make install
    cd ..

You might additionally want to specify OpenCL libray path like below if CMake
can't find OpenCL::

    cmake .. -DCMAKE_INSTALL_PREFIX=${HOME}/.nomp -DOpenCL_LIBRARY=/lib/x86_64-linux-gnu/libOpenCL.so.1

Linking with clang frontend
---------------------------

Clone the llvm-project repo first::

    git clone https://github.com/nomp-org/llvm-project.git

If you are using Linux, build the llvm-project as follows::

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

If you are using OSX with Apple silicon, build the llvm-project as follows::

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
this clang binary directory::

    export NOMP_CLANG_DIR=`pwd`/bin

Compile instructions
----------------------

Create new folder(`nomp-test`) and add following files in the folder.

`foo.c`::

    #include <stdio.h>

    void foo(double *a) {
    #pragma nomp update(to : a[0, 10])
    #pragma nomp for transform("transforms:foo")
        for(int i = 0; i<10; i++)
            a[i] = i;
    #pragma nomp update(from : a[0, 10])
    #pragma nomp update(free : a[0,10])
    }

    int main(int argc, char *argv[]){
    #pragma nomp init("opencl", 0, 0)
        double a[10] = {0};
        for (int i=0; i<10; i++)
            printf("a[%d] = %f \n", i, a[i]);
        foo(a);
        for (int i=0; i<10; i++)
            printf("a[%d] = %f \n", i, a[i]);
    #pragma nomp finalize
        return 0;
    }

`transforms.py`::

    import loopy as lp

    LOOPY_LANG_VERSION = (2018, 2)

    def foo(knl):
        (g,) = knl.default_entrypoint.all_inames()
        knl = lp.tag_inames(knl, [(g, "g.0")])
        return knl

`nompcc`::

    #!/bin/bash

    if [ -z "${NOMP_INSTALL_DIR}" ]; then
    echo "Error: NOMP_INSTALL_DIR is not defined !"
    exit 1
    fi
    if [ -z "${NOMP_CLANG_DIR}" ]; then
    echo "Error: NOMP_CLANG_DIR is not defined !"
    exit 1
    fi

    NOMP_LIB_DIR=${NOMP_INSTALL_DIR}/lib
    NOMP_INC_DIR=${NOMP_INSTALL_DIR}/include

    ${NOMP_CLANG_DIR}/clang -fnomp -include nomp.h -I${NOMP_INC_DIR} "$@" -Wl,-rpath,${NOMP_LIB_DIR} -L${NOMP_LIB_DIR} -lnomp

Now you can link libnomp installation to the clang compiler using `nompcc` script. Make sure both `NOMP_INSTALL_DIR` and `NOMP_CLANG_DIR` are set before using it.

Run instructions
------------------

To compile any file containing `nomp` pragmas as below::

    ./nompcc foo.c
