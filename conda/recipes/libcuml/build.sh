
if [ -n "$MACOSX_DEPLOYMENT_TARGET" ]; then
    # C++11 requires 10.9
    # but cudatoolkit 8 is build for 10.11
    export MACOSX_DEPLOYMENT_TARGET=10.11
fi

if [[ -z "$PROJECT_FLASH" || "$PROJECT_FLASH" == "0" ]]; then
    ./build.sh clean libcuml -v --allgpuarch
else
    if [[ -z "$SIMPLE_BUILD" || "$SIMPLE_BUILD" == "0" ]]; then
        ./build.sh clean libcuml prims -v --allgpuarch
    else
        if [[ -z "$SINGLE_ARCH" || "$SINGLE_ARCH" == "0" ]]; then
            ./build.sh clean libcuml -v --nolibcumltest --singlegpu
        else
            ./build.sh clean libcuml -v --nolibcumltest
        fi
    fi
fi
