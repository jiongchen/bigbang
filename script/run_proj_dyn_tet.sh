#!/bin/bash

exe=../build/bin/test_proj_dyn_tet
mesh=../dat/beam.tet
cons=../dat/beam_fixed.fv
outfolder=../build/bin/proj_dyn/arap

if [ ! -e "$exe" ]; then
  echo executable binary not exists!
  exit 1
elif [ ! -e "$mesh" ]; then
  echo mesh not exists!
  exit 1
fi

time $exe -i $mesh -c $cons -o $outfolder -t 0.033 --method 0 -m 25000 -n 30
