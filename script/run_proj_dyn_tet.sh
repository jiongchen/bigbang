#!/bin/bash

exe=../build/bin/test_proj_dyn_tet
mesh=../dat/beam_dense.tet
cons=../dat/beam_dense_fixed.fv
driv=../dat/beam_dense_handle.fv
outfolder=../build/bin/proj_dyn/arap/dense

if [ ! -e "$exe" ]; then
  echo executable binary not exists!
  exit 1
elif [ ! -e "$mesh" ]; then
  echo mesh not exists!
  exit 1
fi

time $exe -i $mesh -c $cons -f $driv -o $outfolder --method $2 -m 25000 -n $1 --ws 2000 --wg 0.0
