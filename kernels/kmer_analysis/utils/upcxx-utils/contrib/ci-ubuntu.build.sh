#!/bin/bash

set -e

USAGE="$0 base_dir
Optionally set UPCXX_VER to download and install that version of UPCXX
Optionally set CI_UPCXX_CONFIGURE_OPTS to add extra options when building upcxx
Optionally set CI_CMAKE_OPTS to change the build options

"


BASE=$1
if [ -z "$BASE" ]
then
	echo $USAGE
	exit 1
fi
UPCXX_VER=${UPCXX_VER:=2023.3.0}
echo "Using upcxx version $UPCXX_VER"

CI_CMAKE_OPTS=${CI_CMAKE_OPTS}
echo "Using CI_CMAKE_OPTS=${CI_CMAKE_OPTS}"

CI_INSTALL=$BASE/ci-install-${CI_PROJECT_NAME}-upcxx-${UPCXX_VER}
export CI_SCRATCH=${BASE}/scratch/${CI_PROJECT_NAME}-${CI_COMMIT_SHORT_SHA}-${CI_COMMIT_REF_NAME}-${CI_COMMIT_TAG}
export INSTALL_PREFIX=${CI_SCRATCH}
export GASNET_BACKTRACE=1

echo "Establishing all tests under BASE=$BASE and CI_SCRATCH=$CI_SCRATCH"
set -x
mkdir -p ${CI_SCRATCH}
chmod a+rx ${CI_SCRATCH}
chmod g+s ${CI_SCRATCH}

export UPCXX_UTILS_SOURCE=${CI_SCRATCH}/repo
mkdir -p ${UPCXX_UTILS_SOURCE}
rsync -av $(pwd)/ ${UPCXX_UTILS_SOURCE}/
uname -a
uptime
pwd
find * -type d -ls -maxdepth 3 || /bin/true
date

echo "Purging any old tests"
find ${BASE}/scratch/ -maxdepth 1  -name "${CI_PROJECT_NAME}-*-*-*"  -mtime +7 -type d -exec rm -rf '{}' ';' || /bin/true
df -h

echo "PATH=$PATH"
FAILED=""
echo "Checking for cmake, Berkeley UPC and UPC++"
which cmake && cmake --version || FAILED="${FAILED} cmake not found"
echo "FAILED=${FAILED}" && [ -z "$FAILED" ]

echo "Checking or building upcxx"
which upcxx || UPCXXVER=${UPCXX_VER} ./contrib/install_upcxx.sh $CI_INSTALL --enable-gasnet-verbose ${CI_UPCXX_CONFIGURE_OPTS} || FAILED="${FAILED} could not install upcxx"
echo "FAILED=${FAILED}" && [ -z "$FAILED" ]

which upcxx
upcxx --version || FAILED="${FAILED} no upcxx was found"
echo "FAILED=${FAILED}" && [ -z "$FAILED" ]

echo "Building all flavors with '${CI_CMAKE_OPTS}'"

export GASNET_BACKTRACE=1

echo "Building debug"
mkdir -p $CI_SCRATCH/build-dbg $CI_SCRATCH/build-rel $CI_SCRATCH/build-rwdi $CI_SCRATCH/build-rwd-nothreads

cd $CI_SCRATCH/build-dbg
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}/${CI_PROJECT_NAME}-dbg -DCMAKE_BUILD_TYPE=Debug ${UPCXX_UTILS_SOURCE} ${CI_CMAKE_OPTS} || FAILED="${FAILED} could not configure debug"
echo "FAILED=${FAILED}" && [ -z "$FAILED" ]

make -j 16 all install || FAILED="${FAILED} could not build debug"
echo "FAILED=${FAILED}" && [ -z "$FAILED" ]

cd $CI_SCRATCH/build-rel
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}/${CI_PROJECT_NAME}-rel -DCMAKE_BUILD_TYPE=Release ${UPCXX_UTILS_SOURCE} ${CI_CMAKE_OPTS} || FAILED="${FAILED} could not configure rel"
echo "FAILED=${FAILED}" && [ -z "$FAILED" ]

make -j 16 all install || FAILED="${FAILED} could not build rel"
echo "FAILED=${FAILED}" && [ -z "$FAILED" ]

cd $CI_SCRATCH/build-rwdi
export UPCXX_CODEMODE=debug
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}/${CI_PROJECT_NAME}-rwdi -DCMAKE_BUILD_TYPE=RelWithDebInfo ${UPCXX_UTILS_SOURCE} ${CI_CMAKE_OPTS} || FAILED="${FAILED} could not configure rwdi"
echo "FAILED=${FAILED}" && [ -z "$FAILED" ]

make -j 16 all install || FAILED="${FAILED} could not build rwdi"
echo "FAILED=${FAILED}" && [ -z "$FAILED" ]

cd $CI_SCRATCH/build-rwd-nothreads
cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX}/${CI_PROJECT_NAME}-rwd-nothreads -DCMAKE_BUILD_TYPE=RelWithDebug -DUPCXX_UTILS_NO_THREADS=ON ${UPCXX_UTILS_SOURCE} ${CI_CMAKE_OPTS} || FAILED="${FAILED} could not configure rwd-nothreads"
echo "FAILED=${FAILED}" && [ -z "$FAILED" ]

make -j 16 all install || FAILED="${FAILED} could not build rwd-nothreads"
echo "FAILED=${FAILED}" && [ -z "$FAILED" ]

echo "Done with builds"
echo "Completed Successfully: '$0 $@' at $(date) on $(uname) over ${SECONDS} s"


