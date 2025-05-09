#!/bin/bash

input_file=$1
output_file=$2
mask_file=$3

export FREESURFER_HOME=/usr/local/freesurfer-7.4.1
source $FREESURFER_HOME/SetUpFreeSurfer.sh

# synth strip function using freesurfer
$FREESURFER_HOME/bin/mri_synthstrip -i $input_file -o $output_file -m $mask_file
