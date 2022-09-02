# libnomp

Runtime for [nomp](https://github.com/nomp-org/nomp)

## Build instructions

Clone the repo first:
```bash
git clone --recurse-submodules https://github.com/nomp-org/libnomp.git
```

Install dependencies:
```bash
pip3 install git+https://github.com/inducer/loopy
pip3 install pycparser pyopencl
```

Use `cmake` to build the repo after installing the dependencies:
```
cd libnomp
mkdir build; cd build
cmake .. -DCMAKE_INSTALL_PREFIX=${HOME}/.nomp
make install
cd ..
```

You might additionally want to specify OpenCL libray path if CMake can't
find OpenCL:
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

We use `Doxygen` for in source documentations and render those with `Sphinx`
and `breath`. First, install required dependencies:
```bash
conda install -c conda-forge doxygen breathe
conda install sphinx sphinx_rtd_theme
```

Build docs
```bash
cmake .. -DCMAKE_INSTALL_PREFIX=${HOME}/.nomp -DENABLE_DOCS=ON
make install
```

Open `build/docs/sphinx/index.html` on the browser to view the documentation
locally.

## Developer documentation

Please run `clang-format src/*.[ch]` before comitting any changes you make on
the source files. `clang-format` can also be installed from `conda`.
```bash
conda install -c conda-forge clang-format
```
