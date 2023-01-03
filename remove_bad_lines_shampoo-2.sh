#!/bin/bash

indexes=$(awk -F '[\t;]' 'NF-1!=12 {print NR}' $1)
echo "indexes done"
#cat $indexes
#indexesArr=($indexes)
read -a indexesArr <<< $indexes
echo "indexesArr done"

for index in "${indexesArr[@]}"
do
    sed -i -e ''"${index}d"'' $1
done
