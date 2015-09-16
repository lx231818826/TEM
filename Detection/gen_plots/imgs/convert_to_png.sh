#!/bin/bash


for i in *.svg;
do
    filename=`ls $i | gawk -F '.' '{print $1;}'`
    echo $filename
    echo 'inkscape -z '$filename'.svg --export-png='$filename'.png'
    `inkscape -z $i --export-png=$filename.png`
    `inkscape -z $i --export-pdf=$filename.pdf`
done
