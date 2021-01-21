#!/bin/bash

echo -e "\033[1mUSAGE: ./run.sh <version>\033[0m" 
echo "e.g:   ./run.sh v0"
echo "Available targets: v0 v1 v2"

if [[ $# -lt 1 || ($1 != "v0" &&  $1 != "v1" && $1 != "v2") ]]; then
    exit;
fi

echo -e "\n\033[1mCompiling...\033[0m"
make $1

echo -e "\n\033[1mOctave...\033[0m"
octave src/octave/nlm.m $1 
