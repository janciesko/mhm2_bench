#!/bin/bash

# set the version to install
UPCXXVER=${UPCXXVER:=2023.3.0}
CONFIGURE_OPTS="${CONFIGURE_OPTS:=}"

USAGE="

$0 /path/to/install [ Optional extra options to upcxx ./configure script (like: --enable-cuda --enable-valgrind --with-default-network=smp --disable-ibv --with-cxx=mpicxx ...) ]
   See https://bitbucket.org/berkeleylab/upcxx/src/master/INSTALL.md#markdown-header-configuration-linux
   (to build smp/ibv version on cori exvivo: module rm craype-network-aries)

"

installdir=${1}
shift

if [ -z "${installdir}" ]
then
  echo "$USAGE"
  exit 1
fi

getcores()
{
  if lscpu
  then
     :
  fi 2>/dev/null | awk '/^CPU\(s\):/ {print $2}'
  if sysctl -a
  then
     :
  fi 2>/dev/null | awk '/^machdep.cpu.core_count/ {print $2}'
}
BUILD_THREADS=${BUILD_THREADS:=$(getcores)}

builddir=${TMPDIR:=/dev/shm}
[ -d "$builddir" ] && [ -w "$builddir" ] || builddir=/tmp

codedir=$HOME

CC=${CC:=gcc}
CXX=${CXX:=g++}

# do not have any MPI dependency in SMP build
if [ -x mpicc ]
then
  CONFIGURE_OPTS="--with-cxx=mpicxx ${CONFIGURE_OPTS}"
else
  # do not have any mpi dependencies
  unset MPICC
  unset MPICXX
fi

UPCXX=${UPCXX:=upcxx}

oops()
{
  echo "uh oh, something bad happened!"
  exit 1
}

echo "Building upcxx, installdir=$installdir builddir=$builddir codedir=$codedir"

trap oops 0

set -e
set -x


cd $codedir
builddir=$builddir/upcxx-$USER-mhm2-builds
rm -rf ${builddir}
mkdir -p ${builddir}

if [ ! -x $installdir/bin/upcxx ]
then
  echo "Building UPC++"
  # build upcxx
  cd $codedir

  UPCXXDIR=upcxx-${UPCXXVER}
  UPCXXTAR=${UPCXXDIR}.tar.gz
  UPCXXURLBASE=https://bitbucket.org/berkeleylab/upcxx/downloads
  if [ "${UPCXXVER%-snapshot}" != "${UPCXXVER}" ]
  then
    UPCXXTAR=${UPCXXDIR%-snapshot}.tar.gz
    UPCXXURLBASE=https://upcxx-bugs.lbl.gov/snapshot
  fi

  rm -f $UPCXXTAR
  [ -f $UPCXXTAR ] || curl -LO $UPCXXURLBASE/${UPCXXTAR}
  cd $builddir
  [ -d ${UPCXXDIR} ] || tar -xvzf $codedir/${UPCXXTAR}
  [ -d ${UPCXXDIR} ] || mv upcxx-*/ ${UPCXXDIR}
  cd ${UPCXXDIR}

  CC=$CC CXX=$CXX ./configure --prefix=$installdir $@
  make -j ${BUILD_THREADS} || make
  make install
else
  echo "upcxx is already installed $installdir/bin/upcxx"
fi

$installdir/bin/upcxx --version

set -x
trap "" 0

