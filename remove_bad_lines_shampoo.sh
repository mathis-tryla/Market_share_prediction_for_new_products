#!/bin/bash

indexes=$(awk -F '[\t;]' 'NF-1!=12 {print NR}' $1)
read -a indexesArr <<< $indexes

for index in "${indexesArr[@]}"
do
    sed -i -e ''"${index}d"'' $1
done
echo "-- remove bad lines from dataset DONE"
