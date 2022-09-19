# libnomp

Runtime for [nomp](https://github.com/nomp-org/nomp)

## Build instructions

Clone the repo first:
```bash
git clone --recurse-submodules https://github.com/nomp-org/libnomp.git
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

Use `cmake` to build the repo after installing the dependencies:
```
cd libnomp
mkdir build; cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${HOME}/.nomp
make install
cd ..
```

You might additionally want to specify OpenCL libray path like below if CMake
can't find OpenCL:
```bash
cmake .. -DCMAKE_INSTALL_PREFIX=${HOME}/.nomp -DOpenCL_LIBRARY=/lib/x86_64-linux-gnu/libOpenCL.so.1
```

## Run tests

After building, you can run tests in `tests/` directory.
```bash
export NOMP_INSTALL_DIR=${HOME}/.nomp
cd tests
./run-tests.sh
cd -
```

## Build documentation

We use `Doxygen` for in source documentations and render those with `Sphinx` and
`Breathe`. These packages should be available if you install the dev dependencies
using conda (See build instructions).

Build docs
```bash
cmake .. -DCMAKE_INSTALL_PREFIX=${HOME}/.nomp -DENABLE_DOCS=ON
make install
```

Open `build/docs/sphinx/index.html` on the browser to view the documentation
locally.

## Developer documentation

Please run `clang-format` before comitting any changes you make on the source
files. `clang-format` will be available if you install the dev dependencies with
conda. Below are some examples on how to use `clang-format`:
```
clang-format -i src/*.[ch]
clang-format -i tests/*.[ch]
clang-format -i src/nomp.c
```
