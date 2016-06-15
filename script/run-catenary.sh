#!/bin/bash

EXE=../build/bin/main_catenary_dyn
LENGTH=1.0
VERT_NUM=16
DENS=1.0
TIMESTEP=0.01
OUT_DIR=../result/catenary
FRAMES=2
WS=1e2
WB=1e-2
WG=1.0
WP=1e3

if [ ! -d "$OUT_DIR" ]; then
    mkdir -p $OUT_DIR
fi

$EXE --length=$LENGTH --vert_num=$VERT_NUM --density=$DENS --timestep=$TIMESTEP --out_dir=$OUT_DIR --ws=$WS --wb=$WB --wg=$WG --wp=$WP
