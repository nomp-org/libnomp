How to use nomp
===============

Make sure both `NOMP_INSTALL_DIR` and `NOMP_CLANG_DIR` are set and follow
the :doc:`Build instructions <build>` correctly. 

Create new folder(`nomp-test`) and add following files in the folder.

The `foo.c` file contains the example with nomp pragmas. 

..  code-block:: c
    :caption: foo.c

    #include <stdio.h>

    void foo(double *a) {
    #pragma nomp update(to : a[0, 10])
    #pragma nomp for transform("transforms", "foo")
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

The `transforms.py` file contains the `foo` function that creates the loopy kernel. 

..  code-block:: python
    :caption: transforms.py

    import loopy as lp

    LOOPY_LANG_VERSION = (2018, 2)

    def foo(knl):
        (g,) = knl.default_entrypoint.all_inames()
        knl = lp.tag_inames(knl, [(g, "g.0")])
        return knl

`nompcc` contains the script that link libnomp installation to the clang compiler. 

..  code-block:: bash
    :caption: nompcc

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

To compile any file containing `nomp` pragmas, use `nompcc` as below

..  code-block:: bash

    ./nompcc foo.c
