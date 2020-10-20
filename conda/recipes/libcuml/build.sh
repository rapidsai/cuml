
if [ -n "$MACOSX_DEPLOYMENT_TARGET" ]; then
    # C++11 requires 10.9
    # but cudatoolkit 8 is build for 10.11
    export MACOSX_DEPLOYMENT_TARGET=10.11
fi

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    ./build.sh clean libcuml -v --allgpuarch
else
    ./build.sh clean libcuml prims -v --allgpuarch
fi
