Nomp Examples
=============

Make sure both `libnomp` and `nompcc` is built and `NOMP_INSTALL_DIR` and
`NOMP_CLANG_DIR` are set before trying this section. Follow the instructions
in :doc:`Build Instructions <build>` to build and set these variables correctly.

This example requires creating three files in your working directory. First,
let's create `foo.c` file which contains the C program annotated with `nomp`
pragmas.

.. code-block:: c
    :caption: foo.c

    #include <stdio.h>

    void foo(double *a) {
    #pragma nomp for transform("transforms", "foo")
      for(int i = 0; i < 100; i++)
        a[i] = i;
    }

    int main(int argc, const char *argv[]) {
    #pragma nomp init(argc, argv)
      double a[100] = {0};
      for (int i = 0; i < 100; i++)
        printf("a[%d] = %f \n", i, a[i]);

    #pragma nomp update(to : a[0, 100])
      foo(a);
    #pragma nomp update(from : a[0, 100])

      for (int i=0; i<100; i++)
        printf("a[%d] = %f \n", i, a[i]);

    #pragma nomp update(free : a[0, 100])
    #pragma nomp finalize

        return 0;
    }

Then let's create the `transforms.py` file which contains the `foo` Python
function that is called by `libnomp` during the creation of accelerator kernel.
This file has the loopy transformations that are applied to the kernel.

.. code-block:: python
    :caption: transforms.py

    import loopy as lp

    LOOPY_LANG_VERSION = (2018, 2)

    def foo(knl, context):
        (iname,) = knl.default_entrypoint.all_inames()
        i_inner, i_outer = f"{iname}_inner", f"{iname}_outer"
        knl = lp.split_iname(
            knl, iname, 32, inner_iname=i_inner, outer_iname=i_outer
        )
        knl = lp.tag_inames(knl, {i_outer: "g.0", i_inner: "l.0"})
        return knl

Finally, let's create `nompcc` which is a helper script that links libnomp
installation to the clang compiler during compilation.

.. code-block:: bash
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

Now, compile `foo.c` containing `nomp` pragmas using `nompcc` script as follows:

.. code-block:: bash

    chmod +x nompcc
    ./nompcc foo.c -o foo

You can now run the compiled executable by simply providing commnad line
arguments to initialize and use backends, devices, etc.

.. code-block:: bash

    ./foo --nomp-backend opencl --nomp-device 0

Read more about arguments accepted by nomp_init() under
:doc:`User API <user-api>`.
