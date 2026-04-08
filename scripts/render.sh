#!/bin/bash
set -o errexit # Exit on error
set -o nounset # Trigger error when expanding unset variables

root_dir=$1
exp_name=$2
echo "Dataset root dir: ${root_dir}"
list=$(basename -a $root_dir/*/)
echo "TFs list: " $list

for i in $list; do
#* inference
python render.py -m ./output/${exp_name}/${i}/2dgs  \
                 -s ${root_dir}/${i} \
                 --skip_mesh

done

