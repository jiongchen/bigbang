#!/bin/bash

exe=../build/bin/test_proj_dyn
mesh=../dat/plane.obj
cons=../dat/plane_fixed_2.fv
outfolder=../build/bin/proj_dyn/JTS

if [ ! -e "$exe" ]; then
  echo executable binary not exists!
  exit 1
elif [ ! -e "$mesh" ]; then
  echo mesh not exists!
  exit 1
fi

if [ ! -d "$outfolder" ]; then
  mkdir -p $outfolder
fi
time $exe -i $mesh -c $cons -o $outfolder -t 0.033 --wb 0.01 --method $1 -m 25000 -n 100 --spectral_radius 0.9995