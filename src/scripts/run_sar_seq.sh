#!/bin/bash
#
# Usage: 
# run_sar_seq.sh pattern imdir enl significance 
#

alpha="${@: -1}"
enl=("${@: -2}")
imdir=("${@: -3}")

fns=$(ls -l $imdir | grep $1 | \
     grep -v 'sarseq' | grep -v 'enl' | \
     grep -v 'mmse' | grep -v 'gamma' | \
     grep -v 'warp' | grep -v 'sub' |  awk '{print $9}')
     
python scripts/sar_seq.py  -s $alpha  \
                     ${fns//$1/$imdir$1} sarseq.tif $enl 