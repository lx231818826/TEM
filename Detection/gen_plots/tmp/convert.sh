#!/bin/bash

ls ../imgs/*.svg | sed 's/.svg//' | awk  '{
print "echo inkscape -z --file="$1".svg --export-pdf="$1".pdf --export-width=640"
print "inkscape -z --file="$1".svg --export-pdf="$1".pdf --export-width=640"
print "echo inkscape -z --file="$1".svg --export-png="$1".png --export-width=640"
print "inkscape -z --file="$1".svg --export-png="$1".png --export-width=640"
}' | sh


mv -v ../imgs/*.{png,pdf} .
