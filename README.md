# libnomp

[![Unit tests](https://github.com/nomp-org/libnomp/actions/workflows/ci.yml/badge.svg)](https://github.com/nomp-org/libnomp/actions/workflows/ci.yml)

`libnomp` is a [loopy](https://mathema.tician.de/software/loopy/) based runtime
for C programming language to create domain specific compilers for building HPC
applications on accelerators (e.g. GPUs). Read more about how to download, build
and use libnomp on [nomp-org website](https://nomp-org.github.io/libnomp/).

While `libnomp` can be used as a standalone C library, it is intended to be used
with the clang based compiler front-end [nompcc](https://github.com/nomp-org/llvm-project)
to easily build HPC applications using a pragma based programming model without
having to directly use `libnomp` C API. `libnomp` and `nompcc` together form the
`nomp` compiler framework.
