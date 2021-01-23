#!/bin/bash

echo -e "\033[1mUSAGE: ./run.sh <version>\033[0m" 
echo "e.g:   ./run.sh v0"
echo "Available targets: v0 v1 v2"

echo "Optional arguments: --debug"
echo "e.g:   ./run.sh v0 --debug"

if [[ $# == 0 || ($1 != "v0" &&  $1 != "v1" && $1 != "v2") ]]; then
    exit;
elif [ $# == 2 ] && [ $2 != "--debug" ]; then
    echo -e "\n\x1B[31mError\x1B[0m: Invalid passed optional argument\nDid you mean \033[1m--debug\033[0m ?"
    exit;
elif [ $# == 3 ]; then
    echo -e "\n\x1B[31mError\x1B[0m: Invalid number of arguments"
    exit;
fi

echo -e "\n\033[1mCompiling...\033[0m"
make $1

echo -e "\n\033[1mOctave...\033[0m"
octave src/octave/pipeline.m $1 $2
