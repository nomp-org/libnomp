# libnomp

Runtime for [nomp](https://github.com/nomp-org/nomp)

## Build instructions

Clone the repo first:
```bash
git clone --recurse-submodules https://github.com/nomp-org/libnomp.git
```

Install dependencies:
```bash
conda env create -f environment.yml
conda activate libnomp
```

Install Dev Dependencies:
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
