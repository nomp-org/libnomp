# libnomp

Runtime for [nomp](https://github.com/nomp-org/nomp)

## Build instructions

Clone the repo first:
```bash
git clone https://github.com/nomp-org/libnomp.git
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
cmake .. -DCMAKE_INSTALL_DIR=${HOME}/.nomp
make install
cd ..
```

## Run tests

After building, you can run tests in `tests/` directory.
```bash
export NOMP_INSTALL_DIR=${HOME}/.nomp
cd tests
./run-tests.sh -g nomp-api
```
