# libnomp

[![Unit tests](https://github.com/nomp-org/libnomp/actions/workflows/ci.yml/badge.svg)](https://github.com/nomp-org/libnomp/actions/workflows/ci.yml)

`libnomp` is a `loopy` based runtime for C for programming accelerators.

## Build instructions

Clone the repo first:

```bash
git clone https://github.com/nomp-org/libnomp.git
```

Install dependencies using conda to build and run `libnomp`:

```bash
conda env create -f environment.yml
conda activate libnomp
```

You can use mamba to install the dependencies faster:

```bash
mamba env create -f environment.yml
mamba activate libnomp
```

If you are planning on contributing to `libnomp`, install the dev dependencies
instead:

```bash
conda env create -f environment-dev.yml
conda activate libnomp-dev
```

Similarly, you can install the dev dependencies with mamba as well:

```bash
mamba env create -f environment-dev.yml
mamba activate libnomp-dev
```

Use `lncfg` to configure cmake for libnomp and `lninstall` to install libnomp. 
For the available options, you can check `lncfg -h`.

```bash
cd libnomp
./lncfg
./lninstall 
```

`lninstall` prompts you to update your `.bashrc` to set the `PATH` and 
`NOMP_INSTALL_DIR` variables. This will allow you to use `lnrun` command to 
open the documentation, run the tests, and debug the provided test.

You might additionally want to specify OpenCL libray path like below if CMake
can't find OpenCL:

```bash
./lncfg -ol /lib/x86_64-linux-gnu/libOpenCL.so
```

or if you are using `conda` to install OpenCL:
```bash
./lncfg -ol ${CONDA_PREFIX}/lib/libOpenCL.so -oi ${CONDA_PREFIX}/include/
```

### Build documentation

We use `Doxygen` for in source documentations and render those with `Sphinx` and
`Breathe`. These packages should be available if you install the dev dependencies
using conda. You can enable docs by passing either `-d` or `--enable-docs` option
to the `lncfg` script.

```bash
./lncfg -d
./lninstall
```

Use `lnrun` to open the user documentation locally. You can specify the browser by
providing it after `-B`. For example, to open the documentation in firefox, 
    
```bash
lnrun docs -B firefox
```

If you do not specify the browser, it opens the documentation in chrome by default.

## Run tests

After building, you can run all the tests in `tests/` directory.

```bash
lnrun test
```
