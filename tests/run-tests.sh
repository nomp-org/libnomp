#!/bin/bash

TESTDIR=`cd ..; pwd`/build

########################## Don't touch what follows ############################
if [ ! -d ${TESTDIR} ]; then
  echo "Test directory '${TESTDIR}' not found."
  exit 1
fi

TESTGROUPS=nomp-api, #vec-init,vec-add
BACKEND=opencl

function print_help() {
  echo "./run-tests.sh [-h|--help] [-g|--group <list of test groups>] [-b|--backend <backend>]"
  echo "--help: Print this help and exit."
  echo "--group: Comm separated list of test groups, case sensitive (Default: ${TESTGROUPS})."
  echo "--backend: Backend to run the tests, case insensitive (Default: ${BACKEND})."
}

while [[ $# -gt 0 ]]; do
  key="$1"

  case ${key} in
    -h|--help)
      shift
      print_help
      exit 0
      ;;
    -g|--group)
      shift
      TESTGROUPS=$1
      shift
      ;;
    -b|--backend)
      shift
      BACKEND=$1
      shift
      ;;
  esac
done

if [[ -z ${TESTGROUPS} ]] || [[ -z ${BACKEND} ]]; then
  echo "Error parsing command line arguments. Run './run-tests.sh --help' for help."
  exit 1
fi

IFS=','; group_array=(${TESTGROUPS}); unset IFS;
err=0
for g in ${group_array[@]}; do
  for t in `ls ${TESTDIR}/${g}*`; do
    $t
    if [ $? -eq 0 ]; then
      echo "$t: Pass";
    else
      echo "$t: Fail";
      err=$((err+1))
    fi
  done
done
exit ${err}
