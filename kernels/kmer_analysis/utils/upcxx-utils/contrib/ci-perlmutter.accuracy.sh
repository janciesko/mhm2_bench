#!/bin/bash

# CI Accuracy script for perlmutter

set -e

uname -a
pwd
date

if [ -z "${UPCXX_UTILS_SOURCE}" ] || [ -z "${CI_SCRATCH}" ]
then
  echo "please set the UPCXX_UTILS_SOURCE and CI_SCRATCH environmental variables"
  exit 1
fi

export INSTALL_PREFIX=${CI_SCRATCH}/install
export BUILD_PREFIX=${CI_SCRATCH}/build
export RUN_PREFIX=${CI_SCRATCH}/runs
export GASNET_BACKTRACE=1

module load PrgEnv-gnu
module load cmake
module load cpe-cuda
module load cudatoolkit
module swap gcc/11.2.0

module use /global/common/software/m2878/perlmutter/modulefiles
module rm upcxx
# module load upcxx
module load upcxx/nightly ; export GASNET_OFI_RECEIVE_BUFF_SIZE=recv ; export GASNET_OFI_NUM_RECEIVE_BUFFS=400 ; export FI_CXI_RX_MATCH_MODE=software ; export FI_MR_CACHE_MONITOR=memhooks

module list
which cc
which CC
which nvcc
which upcxx

CC --version
upcxx --version
nvcc --version

BUILD_NAME=gnu
#  UPCXX_UTILS_BUILD=${BUILD_PREFIX}-${BUILD_NAME}-${t}
#  UPCXX_UTILS_INSTALL=${INSTALL_PREFIX}-${BUILD_NAME}-${t}
cd ${CI_SCRATCH}

echo "Testing 3 node job RelWithDebug and Release test_combined"
nodes=3
job1=$(sbatch --parsable --job-name="CIuuv-${CI_COMMIT_SHORT_SHA}" --account=m2865 -C gpu --nodes=$nodes --qos=debug --time=30:00 --wrap="set -e; set -x; module list; cd ${BUILD_PREFIX}-${BUILD_NAME}-RelWithDebug && upcxx-run -N $nodes -n $((128*nodes)) ./test/test_combined && cd - && cd ${BUILD_PREFIX}-${BUILD_NAME}-Release && upcxx-run -N $nodes -n $((128*nodes)) ./test/test_combined && echo Good")

echo "Waiting for multi-node combined tests: jobs $job1"
date
for job in $job1
do
 while /bin/true
 do
  sleep 60
  date
  sacct=$(sacct -j $job -o state -X -n 2>/dev/null || true)
  if [ -n "${sacct}" -a -z "$(echo "${sacct}" | grep ING)" ] ; then break ; fi
 done

 echo "sacct $sacct"
 sacct=$(sacct -j $job -X -n)
 echo "sacct $sacct"
 cat slurm-${job}.out
 wasgood=$(echo "${sacct}" | grep -v '0:0' || true)
 if [ -z "$wasgood" ] ; then  true ; else  echo "job ${job} failed somehow - ${wasgood}"; false ; fi
done

echo "Completed at $(date) in ${SECONDS} s"
