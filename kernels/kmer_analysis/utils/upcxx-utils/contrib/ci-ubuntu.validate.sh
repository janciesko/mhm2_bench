#!/bin/bash

set -e

USAGE="$0 base_dir
Optionally set UPCXX_VER to use a different that version of UPCXX (installed by ci-ubuntu.build.sh)
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

echo "Running make test for the versions built by ci-ubuntu.build.sh"

set -x

echo "Testing dbg"
cd $CI_SCRATCH/build-dbg
timeout -k 1m -s INT -v 20m make test || FAILED="${FAILED} could not test dbg"
echo "FAILED=${FAILED}" && [ -z "$FAILED" ]

echo "Testing rwd-nothreads"
cd $CI_SCRATCH/build-rwd-nothreads
timeout -k 1m -s INT -v 20m make test || FAILED="${FAILED} could not test rwd-nothreads"
echo "FAILED=${FAILED}" && [ -z "$FAILED" ]

echo "Testing rwdi"
cd $CI_SCRATCH/build-rwdi
timeout -k 1m -s INT -v 20m make test || FAILED="${FAILED} could not test rwdi"
echo "FAILED=${FAILED}" && [ -z "$FAILED" ]

echo "Testing rel"
cd $CI_SCRATCH/build-rel
timeout -k 1m -s INT -v 20m make test || FAILED="${FAILED} could not test rel"
echo "FAILED=${FAILED}" && [ -z "$FAILED" ]

echo "Done with unit tests"
echo "Completed Successfully: '$0 $@' at $(date) on $(uname) over ${SECONDS} s"
