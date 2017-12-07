#!/bin/sh
# Usage:
#
#   ./normalize warpbandnumber spectral_subset referencefile targetfile [spatial_subset]
#
#   Only the spatial subset is optional.
#
#   Spectral and spatial subsets must be lists, e.g., for Landsat images:
#
#   ./ normalize 4 [1,2,3,4,5,7] reference.tif target.tif [500,500,2000,2000]
#
#
echo 
echo "--------------------------------------------"
echo "Automatic relative radiometric normalization"
echo "--------------------------------------------"
echo Reference image $3
echo Target image $4
echo Warp band $1
echo Spectral subset $2
echo Spatial subset ${5:-full}

if [ $# -eq 4 ]
then
	echo "----------------------------"
	echo "Image-image registration ..."
	warp=$(python /crc/register.py -b $1 $3 $4 | tee /dev/tty \
	   | grep written \
	   | awk '{print $5}')
	echo "----------------------------"   
	echo "IR-MAD ..."
	mad=$(python /crc/iMad.py -n -p $2 $3 $warp | tee /dev/tty \
	   | grep written \
	   | awk '{print $4}' )
	echo "----------------------------"
	echo "Radiometric normalization ..."
	norm=$(python /crc/radcal.py -n -p $2 $mad | tee /dev/tty \
	   | grep written \
	   | awk '{print $4}' )
	echo "----------------------------"   
elif [ $# -eq 5 ]
then
	echo "----------------------------"
	echo "Image-image registration ..."
	warp=$(python /crc/register.py -b $1 -d $5 $3 $4 | tee /dev/tty \
	   | grep written \
	   | awk '{print $5}')
	echo "----------------------------"   
	echo "IR-MAD ..."
	mad=$(python /crc/iMad.py -n -d $5 -p $2 $3 $warp | tee /dev/tty \
	   | grep written \
	   | awk '{print $4}' )
	echo "----------------------------"
	echo "Radiometric normalization ..."
	norm=$(python /crc/radcal.py -n -d $5 -p $2 $mad $4 | tee /dev/tty \
	   | grep full \
	   | awk '{print $5}' )
	echo "----------------------------"   
else
   echo "Incorrect number of arguments"
   return 1
fi   	
return 0