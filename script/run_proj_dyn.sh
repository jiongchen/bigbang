#!/bin/bash

EXE=../build/bin/test_proj_dyn
MESH=../dat/plane.obj
CONS=../dat/plane_fixed_2.fv
FRAMES=300
METHOD=0
TIMESTEP=0.033
WS=1e3
WB=1e-2
MAXITER=5000
OUTDIR=../result/projective/ms-method$METHOD-t$TIMESTEP-ws$WS-wb$WB

if [ ! -d "$OUTDIR" ]; then
  mkdir -p $OUTDIR
fi

$EXE -i $MESH -c $CONS -o $OUTDIR -t $TIMESTEP --ws $WS --wb $WB --method $METHOD -n $FRAMES -m $MAXITER | tee $OUTDIR/log.txt
