#!/bin/bash

set -e
set -u
set -o pipefail


# Install python and pip. Don't modify this to add Python package dependencies,
wget -O Miniforge3.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3.sh -b -p $1

$1/bin/conda create -n $2 python=3.12
