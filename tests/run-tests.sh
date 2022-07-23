#!/bin/bash

########################## Don't touch what follows ############################
TESTDIR=`cd ..; pwd`/build
TESTGROUPS=nomp-api,vec-init,vec-add
BACKEND=opencl

function print_help() {
  echo "./run-tests.sh [-h|--help] [-g|--group <list of test groups>] [-b|--backend <backend>]"
  echo "-h/--help: Print this help and exit."
  echo "-t/--testdir: Location of test binaries, case sensitive (Default: ${TESTDIR})."
  echo "-g/--group: Comma separated list of test groups, case sensitive (Default: ${TESTGROUPS})."
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

IFS=','; group_array=(${TESTGROUPS}); unset IFS;
err=0
for g in ${group_array[@]}; do
  for t in `ls ${TESTDIR}/${g}*`; do
    $t
    if [ $? -eq 0 ]; then
      echo "$t: Pass";
    else
      echo "$t: Fail";
      err=$((err + 1))
    fi
  done
done
exit ${err}
