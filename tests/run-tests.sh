#!/bin/bash

########################## Don't touch what follows ############################
TESTDIR=`cd ..; pwd`/build
BACKEND=opencl

export NOMP_ANNOTATE_SCRIPT=sem
export NOMP_ANNOTATE_FUNCTION=annotate

function print_help() {
  echo "./run-tests.sh [-h|--help] [-g|--group <list of test groups>] [-b|--backend <backend>]"
  echo "-h/--help: Print this help and exit."
  echo "-t/--testdir: Location of test binaries, case sensitive (Default: ${TESTDIR})."
  echo "-b/--backend: Backend to run the tests, case insensitive (Default: ${BACKEND})."
}

while [[ $# -gt 0 ]]; do
  key="$1"

  case ${key} in
    -h|--help)
      shift
      print_help
      exit 0
      ;;
    -t|--testdir)
      shift
      TESTDIR=$1
      shift
      ;;
    -b|--backend)
      shift
      BACKEND=$1
      shift
      ;;
  esac
done

err=0
for t in `ls ${TESTDIR}/nomp-api-*`; do
  $t ${BACKEND}
  if [ $? -eq 0 ]; then
    echo "$t: Pass";
  else
    echo "$t: Fail";
    err=$((err + 1))
  fi
done
exit ${err}
