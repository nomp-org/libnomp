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

```bash
./lncfg -o /lib/x86_64-linux-gnu/libOpenCL.so.1
```

### Build documentation

We use `Doxygen` for in source documentations and render those with `Sphinx` and
`Breathe`. These packages should be available if you install the dev dependencies
using conda.

```bash
./lncfg -d
./lninstall
```

Use `lnrun` to open the user documentation locally. To open the documentation in chrome, 

```bash
lnrun docs
```

## Run tests

After building, you can run tests in `tests/` directory.

```bash
lnrun test
```
