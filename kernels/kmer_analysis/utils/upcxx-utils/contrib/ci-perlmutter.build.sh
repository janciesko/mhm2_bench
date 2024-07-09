#!/bin/bash

# CI Build script for perlmutter

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
for t in Debug RelWithDebInfo RelWithDebug Release
do
  UPCXX_UTILS_BUILD=${BUILD_PREFIX}-${BUILD_NAME}-${t}
  mkdir -p ${UPCXX_UTILS_BUILD}
  cd $UPCXX_UTILS_BUILD
  echo "Building ${t} version of ${BUILD_NAME}"
  cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}-${BUILD_NAME}-${t} -DCMAKE_BUILD_TYPE=${t} ${UPCXX_UTILS_SOURCE}
  make -j 32 install
done
echo "Done building"
