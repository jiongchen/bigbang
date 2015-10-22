#!/bin/bash

BIN=../build/bin/test_simple
INPUT_MESH=../dat/beam.tet
INPUT_CONS=../dat/beam_fixed.fv
OUTPUT_FOLDER=../build/bin/simple

$BIN -i $INPUT_MESH -c $INPUT_CONS -o $OUTPUT_FOLDER -t 0.1 --wg 2.0
