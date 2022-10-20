# libnomp

Runtime for [nomp](https://github.com/nomp-org/nomp)

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

### Variable names

We use `snake_case` for all variables. In addition, for preprocessor constants and
enum members we use capital letters.

### Braces

We omit the braces for a block if it has only a single statement. For example, `if`
with a single statement in the body, or `for` with a single statement in the body.
We keep braces in the following `for` loop even if they can be omitted without any
change to program logic.

```C
for (unsigned i = 0; i < 10; i++) {
  if (i < 5)
    printf("i < 5");
}
```

Also, in an `if-then-else` block, if at least one branch has more than one statement
in the body, we will use braces for all the branches. So we have the following code:

```C
if (a < 5) {
  printf("a < 5");
} else if (a == 5) {
  printf("a == 5");
} else {
  b = a;
  printf("a > 5");
}
```

instead of:

```C
if (a < 5)
  printf("a < 5");
else if (a == 5)
  printf("a == 5");
else {
  b = a;
  printf("a > 5");
}
```

### Formatting files before committing

Please run `clang-format` before committing any changes you make on the source
files. `clang-format` will be available if you install the dev dependencies with
conda. Below are some examples on how to use `clang-format`:
```bash
clang-format -i src/*.[ch]
clang-format -i tests/*.[ch]
clang-format -i src/nomp.c
```

If you change any python files, please use `black` and `isort` to format the python
code before committing. Also check any issues in the code with `flake8`. `black`,
`isort` and `flake8`  will be available if you install the dev dependencies with
conda. Below are some examples on how to use `black`, `isort` and `flake8`:
```bash
black -l 80 **/*.py; isort **/*.py
flake8
```
