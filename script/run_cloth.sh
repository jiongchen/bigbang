#!/bin/bash

exe=../build/bin/test_cloth
mesh=../dat/plane.obj
cons=../dat/plane_fixed.fv
outfolder=../build/bin/cloth

time $exe -i $mesh -c $cons -o $outfolder --bw98 0