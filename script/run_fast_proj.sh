#!/bin/bash

exe=../build/bin/test_fast_proj
mesh=../dat/cloth_25.obj
cons=../dat/plane_fixed.fv
outfolder=../build/bin/FASTPROJ/plane25

if [ ! -e "$exe" ]; then
  echo executable binary not exists!
  exit 1
elif [ ! -e "$mesh" ]; then
  echo mesh not exists!
  exit 1
fi

time $exe -i $mesh -c $cons -o $outfolder --proj 1 -t 0.01 --wb 0.1 -n 300