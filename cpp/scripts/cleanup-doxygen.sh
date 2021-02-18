#!/bin/bash

# Run using output from build. i.e.
# `./build.sh cppdocs > cppdocs.log && ./cpp/scripts/cleanup-doxygen.sh --log-file ./cppdocs.log`
# 
# Or pipe the output directly to the script
# `./build.sh cppdocs | ./cpp/scripts/cleanup-doxygen.sh`

DOC_LIST=()
DO_PROCESS_LOG=0


# Long arguments
LONG_ARGUMENT_LIST=(
    "file:"
    "log-file:"
    "all"
)

# Short arguments
ARGUMENT_LIST=(
    "f:"
)

# read arguments
opts=$(getopt \
    --longoptions "$(printf "%s," "${LONG_ARGUMENT_LIST[@]}")" \
    --name "$(basename "$0")" \
    --options "$(printf "%s" "${ARGUMENT_LIST[@]}")" \
    -- "$@"
)

if [ $? != 0 ] ; then echo "Terminating..." >&2 ; exit 1 ; fi

eval set -- "$opts"

while true; do
    case "$1" in
        h)
            show_help
            exit 0
            ;;
        -f | --file )
            shift
            echo "Adding file $1"
            DOC_LIST=("${DOC_LIST[@]}" "$1")
            ;;
        --log-file )
            shift
            echo "Adding log file $1"
            # DOC_LIST=("${DOC_LIST[@]}" $(sed -nr 's/^.*?(\/home\/mdemoret\/Repos\/rapids\/cuml-dev2\/cpp\/.+?\.(cuh?|hp?p?|cpp)).*$/\1/p' $1  | awk '!seen[$0]++'))
            INPUT_LOG_FILE="$1"
            ;;
        --all )
            echo "Adding all"
            DOC_LIST=("${DOC_LIST[@]}" $(find . -iregex '\./cpp/.*\.\(h\|hpp\|cpp\|cuh\|cu\)$'))
            ;;
        --)
            shift
            break
            ;;
    esac
    shift
done

# Check to see if a pipe exists on stdin.
if [ -p /dev/stdin ]; then
    echo "Log was piped to this script!"

    # STDIN_DATA=$(cat)

    # echo "Input: $STDIN_DATA"

    # STDIN_DOC_LIST=$(sed -nr 's/^.*?(\/home\/mdemoret\/Repos\/rapids\/cuml-dev2\/cpp\/.+?\.(cuh?|hp?p?|cpp)).*$/\1/p' | awk '!seen[$0]++')

    # DOC_LIST=("${DOC_LIST[@]}" $(sed -nr 's/^.*?(\/home\/mdemoret\/Repos\/rapids\/cuml-dev2\/cpp\/.+?\.(cuh?|hp?p?|cpp)).*$/\1/p' | awk '!seen[$0]++'))
    
    # Just ensure INPUT_LOG_FILE is set to null
    INPUT_LOG_FILE=
fi

if [[ -n "${INPUT_LOG_FILE+1}" ]]; then
    echo "INPUT_LOG_FILE defined: '${INPUT_LOG_FILE}', Appending"
    DOC_LIST=("${DOC_LIST[@]}" $(cat ${INPUT_LOG_FILE} | sed -nr 's/^.*?(\/home\/mdemoret\/Repos\/rapids\/cuml-dev2\/cpp\/.+?\.(cuh?|hp?p?|cpp)).*$/\1/p' | awk '!seen[$0]++'))
fi

# Populate the doclist

# Process just the files that generate warnings
# DOC_LIST=($(sed -nr 's/^(\/home\/mdemoret\/Repos\/rapids\/cuml-dev2\/cpp\/.+?\.(cuh?|hp?p?|cpp)).*$/\1/p' $1  | awk '!seen[$0]++'))

# DOC_LIST=($(sed -nr 's/^.*?(\/home\/mdemoret\/Repos\/rapids\/cuml-dev2\/cpp\/.+?\.(cuh?|hp?p?|cpp)).*$/\1/p' $1  | awk '!seen[$0]++'))


# Find all files in all folders
# DOC_LIST=($(find . -iregex '\./cpp/.*\.\(h\|hpp\|cpp\|cuh\|cu\)$'))

# Manual specification
# DOC_LIST=(
#    "/home/mdemoret/Repos/rapids/cuml-dev2/cpp/src/tsne/tsne_runner.cuh"
# )

echo "DOC_LIST: $DOC_LIST, Count: ${#DOC_LIST[@]}"

# exit

# set -x

i=0

for f in "${DOC_LIST[@]}"; do

    FULL_PATH=$(realpath $f)

    echo "Fixing File: $f"

    # subl -b $f
    # read -n 1 -s
    # subl -b --command "set_file_type {\"syntax\": \"Packages/C++/C++.sublime-syntax\"}"
    # read -n 1 -s
    # # subl -b --command "doxy_update_comments { \"reparse\": true }"
    # subl -b --command "doxy_select_comments {\"kind\":\"doxy\"}"
    # read -n 1 -s
    # subl -b --command "doxy_update_comments {\"reparse\":\"False\", \"new_style\":\"preferred\"}"
    # read -n 1 -s
    # subl -b --command "save {}"
    # read -n 1 -s
    # subl -b --command "close_file {}"
    # read -n 1 -s

    subl -w -b --command \
        "doxy_chain_commands {\"commands\": [ \
            [\"window.doxy_open_file\", { \"file\":\"$FULL_PATH\" }], \
            [\"view.set_file_type\", { \"syntax\": \"Packages/C++/C++.sublime-syntax\" }], \
            [\"view.doxy_select_comments\", { \"kind\":\"doxy\"}], \
            [\"view.doxy_update_comments\", { \"reparse\": true }], \
            [\"window.save\", {}], \
            [\"window.close_file\", {}] \
         ] }"
   # read -n 1 -s

#    i=$((i + 1))

#    if [[ $i -gt 10 ]]; then
#       sleep 2
#       i=0
#    fi
done

# @input/output param ([a-zA-Z_][a-zA-Z0-9_]*):? 
# @param[inout] $1 