# libnomp

[![Unit tests](https://github.com/nomp-org/libnomp/actions/workflows/ci.yml/badge.svg)](https://github.com/nomp-org/libnomp/actions/workflows/ci.yml)

`libnomp` is a `loopy` based runtime for C for programming accelerators.

## Build instructions

Clone the repo and change directory to `libnomp`:

```bash
git clone https://github.com/nomp-org/libnomp.git
cd libnomp
```

Install dependencies using conda to build and run `libnomp`:

```bash
conda env create -f environment.yml
conda activate libnomp
```

If you are planning on contributing to `libnomp`, install the dev dependencies
instead:

```bash
conda env create -f environment-dev.yml
conda activate libnomp-dev
```

Use `lncfg` to configure CMake for `libnomp` and `lninstall` to install `libnomp`.
For the available options, you can check `lncfg -h` or `lncfg --help`. To build
`libnomp` with OpenCL backend enabled, use the following commands:

```bash
./lncfg --enable-opencl
./lninstall
```

`lninstall` prompts you to update `.bashrc` to append `PATH` variable with
the `libnomp` install directory. It also updates `.bashrc` to set
`NOMP_INSTALL_DIR` environment variable which must be set in order to use
`libnomp`. This will also enable you to use `lnrun` script without using the
full path to open documentation, run tests, debug tests, etc.

You might additionally want to specify OpenCL libray path as below if CMake
can't find OpenCL:

```bash
./lncfg --enable-opencl --opencl-lib /lib/x86_64-linux-gnu/libOpenCL.so
```

or if you are using `conda` to install OpenCL:
```bash
./lncfg --enable-opencl --opencl-lib ${CONDA_PREFIX}/lib/libOpenCL.so --opencl-inc ${CONDA_PREFIX}/include
```

### Build documentation

We use `Doxygen` for in source documentations and render those with `Sphinx`
and `Breathe`. These packages should be available if you install the dev
dependencies using conda. You can enable docs by passing either `-docs` or
`--enable-docs` option to `lncfg` script.

```bash
./lncfg --enable-docs
./lninstall
```

Use `lnrun` to open the user documentation locally. You can specify the
browser by providing it after `-B`. For example, to open the documentation
in firefox,

```bash
lnrun docs -B firefox
```

If you do not specify the browser, it opens the documentation in chrome by
default.

## Run tests

After building, run the tests to see if everything is working.

```bash
lnrun test
```
