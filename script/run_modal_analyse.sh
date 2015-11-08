#!/bin/bash

folder=../build/bin/basis
exe=../build/bin/test_modal_basis
mesh=../dat/beam.tet
cons=../dat/beam_fixed.fv

if [[ -d "$folder" ]]; then
  echo -e "\033[31mDelete basis dir\033[33m"
  rm -rf $folder
fi

time $exe $mesh $cons $folder