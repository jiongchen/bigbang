#!/bin/bash

exe=../build/bin/test_proj_dyn_tet
mesh=../dat/beam_dense.tet
cons=../dat/beam_dense_fixed.fv
driv=../dat/beam_dense_handle.fv
outfolder=../build/bin/proj_dyn/arap/dense
twist_sr=0.80
fall_down_sr=0.87

if [ ! -e "$exe" ]; then
  echo executable binary not exists!
  exit 1
elif [ ! -e "$mesh" ]; then
  echo mesh not exists!
  exit 1
fi

# twist the beam without gravity exerted
time $exe -i $mesh -c $cons -f $driv -o $outfolder --method $1 -m 25000 -n 20 --ws 2000 --wg 0.0 --spectral_radius $twist_sr
