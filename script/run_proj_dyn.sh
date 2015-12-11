#!/bin/bash

exe=../build/bin/test_proj_dyn
mesh=../dat/plane.obj
cons=../dat/plane_fixed.fv
outfolder=../build/bin/proj_dyn

if [ ! -e "$exe" ]; then
  echo executable binary not exists!
  exit 1
elif [ ! -e "$mesh" ]; then
  echo mesh not exists!
  exit 1
fi

time $exe -i $mesh -c $cons -o $outfolder --wb 0.000 --method 1 -m 20000 -n 2