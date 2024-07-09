#!/bin/bash

USAGE="
$0 upcxx-utils_install_prefix

A simple script to spawn the tests within a slurm environment

"

install=$1
if [ -z "$install" -o ! -d "$install/test/bin" ]
then
  echo "$USAGE"
  echo "Could not find the install directory"
  exit 1
fi

set -e

nodes=${SLURM_JOB_NUM_NODES:=1}
threads=$(lscpu -p | grep -v '#' | awk -F, '{print $2}' | sort | uniq | wc -l)
threads=$((threads*nodes))
for test in $install/test/bin/test_combined 
do
  echo "Running $test with $threads threads over $nodes nodes at $(date) on $(uname -n)"
  echo "Running $test with $threads threads over $nodes nodes at $(date) on $(uname -n)" >> tests.log
  upcxx-run -n $threads -N $nodes $test >> tests.log 2>&1
  echo "Completed at $(date)" >> tests.log
  echo "Completed at $(date)"
done
