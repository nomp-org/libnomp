#!/bin/bash

# script variables
SOURCE_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR="$SOURCE_DIR/build"
CMAKE_OPTS="-DCMAKE_INSTALL_PREFIX=${HOME}/.nomp  -DOpenCL_LIBRARY=/lib/x86_64-linux-gnu/libOpenCL.so.1"
RUN_TESTS=false
TEST_GROUPS="*"
BACKEND="opencl"
OUTPUT=""
BUILD_SUCCESS=false
VERBOSE=false
DEBUG_TEST=false
PORT=5005
DISPLAY_DOCS=false
BROWSER="google-chrome"

# colors
red=$(tput setaf 1)
green=$(tput setaf 2)
cyan=$(tput setaf 6)
reset=$(tput sgr0)

function print_help() {
  echo -e "usage: ./install.sh [-h|--help] [-v|--verbose] [-t|--test] [-g|--groups <test-group>]\n" \
    "\t[-d|--debug] [-p|--port <port-number>] [-D|--docs] [-B|--browser <web-browser>]\n" \
    "\t[-T|--testdir <build test directory>] [-b|--backend <backend>]\n\n" \
    "${cyan}-h/--help:${reset} Print this help and exit.\n" \
    "${cyan}-v/--verbose:${reset} Display all the build output.\n" \
    "${cyan}-t/--test:${reset} Run all the tests after the build.\n" \
    "${cyan}-g/--group:${reset} Pattern to filter the tests to be run (Default: ${TEST_GROUPS})\n" \
    "${cyan}-d/--debug:${reset} Run the specified test with gdbserver.\n" \
    "${cyan}-p/--port:${reset} Port for which gdbserver host the test case (Default: ${PORT})\n" \
    "${cyan}-D/--docs:${reset} Displays the user documentation after the build.\n" \
    "${cyan}-B/--browser:${reset} Specifies the web browser to display the documentation (Default: ${BROWSER})\n" \
    "${cyan}-T/--testdir:${reset} Location of test binaries, case sensitive (Default: ${BUILD_DIR}).\n" \
    "${cyan}-b/--backend:${reset} Backend to run the tests, case insensitive (Default: ${BACKEND}).\n"
}

function run_tests() {
  TESTS=${BUILD_DIR}/nomp-api-${TEST_GROUPS}
  if ! compgen -G "${TESTS}" >/dev/null; then
    echo -e "\n${red}No tests found for: ${TESTS}${reset}"
    exit 1
  fi

  echo -e "\nRunning tests..."
  ERR=0
  for t in $(ls ${TESTS}); do
    OUTPUT=$($t ${BACKEND})
    if [ $? -eq 0 ]; then
      echo "$t: ${green}Passed${reset}"
    else
      echo "$t: ${red}Failed${reset}"
      echo -e "\t$OUTPUT\n"
      ERR=$((ERR + 1))
    fi
  done
  [ $ERR -gt 0 ] && echo -e "\n${red}There are test failures${reset}"
  exit ${ERR}
}

function debug_test() {
  TEST_NAME="nomp-api-${DEBUG_TEST}"
  BUILD_TEST="${BUILD_DIR}/${TEST_NAME}"
  if [[ -f "$BUILD_TEST" ]]; then
    echo -e "\nDebugging session started for: ${cyan}${TEST_NAME}${reset}"
    gdbserver "localhost:${PORT}" "${BUILD_TEST}"
    echo "Debugging session ended"
  else
    echo -e "\n${red}Test not found: nomp-api-${DEBUG_TEST}$reset"
    exit 1
  fi
}

# create build directory if not exists
[[ ! -d "${BUILD_DIR}" ]] && mkdir build

# set env variables
[[ -n "${NOMP_INSTALL_DIR}" ]] && export "NOMP_INSTALL_DIR=${HOME}/.nomp"

# parse the options
while [ $# -gt 0 ]; do
  case $1 in
  -h | --help)
    print_help
    exit 0
    ;;
  -t | --test) RUN_TESTS=true ;;
  -v | --verbose) VERBOSE=true ;;
  -p | --port) shift && PORT="${1}" ;;
  -d | --debug) CMAKE_COMMAND+=" -DCMAKE_BUILD_TYPE=DEBUG" && shift && DEBUG_TEST="${1}" ;;
  -g | --group) shift && TEST_GROUPS="${1}" ;;
  -D | --docs) CMAKE_COMMAND+=" -DENABLE_DOCS=ON" && DISPLAY_DOCS=true ;;
  -B | --browser) shift && BROWSER="${1}" ;;
  -b | --backend) shift && BACKEND="${1}" ;;
  -T | --testdir) shift && BUILD_DIR="${1}" ;;
  *) echo "${red}Invalid argument: ${1}${reset}" &&
    echo "See ${cyan}./install.sh -h${reset} or ${cyan}./install.sh --help${reset} for the accepted commands" &&
    exit 1 ;;
  esac
  shift
done

# build libnomp
echo "Building libnomp..."
cd "${BUILD_DIR}" &&
  OUTPUT+=$(cmake .. "$CMAKE_OPTS") &&
  OUTPUT+=$(make install) &&
  BUILD_SUCCESS=true

# running subsequent tasks
if [ "${BUILD_SUCCESS}" = true ]; then
  [ "${VERBOSE}" = true ] && echo "${OUTPUT}"
  echo "${green}Successfully built libnomp${reset}"
  cd "${SOURCE_DIR}/tests" &&
    if [ "${RUN_TESTS}" = true ]; then
      run_tests
    elif [ ! "${DEBUG_TEST}" = false ]; then
      debug_test
    fi
  [[ "${DISPLAY_DOCS}" = true ]] &&
    echo -e "Starting web browser..." &&
    if command -v "${BROWSER}" &>/dev/null; then
      nohup "$BROWSER" "${BUILD_DIR}/docs/sphinx/index.html" 1>/dev/null 2>/dev/null &
      exit 0
    else
      echo "${red}Browser not found: ${BROWSER}${reset}"
      exit 1
    fi
else
  echo "${OUTPUT}"
  echo "${red}Failed to build libnomp${reset}"
  exit 1
fi
