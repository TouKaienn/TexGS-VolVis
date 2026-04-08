#!/bin/bash
set -o errexit # Exit on error
set -o nounset # Trigger error when expanding unset variables

root_dir=$1
exp_name=$2
text_prompt_TFs=$3
edit_name=$4
echo "Dataset root dir: ${root_dir}"
list=($(basename -a ${root_dir}/*/))
echo "TFs list: " ${list[@]}


# for i in $list; do

i="TF02"

# #* optimize texture
CUDA_LAUNCH_BLOCKING=1 python paint3d.py --eval \
                -t 'stylize' \
                -s ${root_dir}/${i} \
                -m ./output/${exp_name}/${i}/${edit_name} \
                -init ./output/${exp_name}/${i}/2dgs/point_cloud/iteration_30000/point_cloud.ply \
                --iteration 4000 \
                --text_prompt "${text_prompt_TFs}" \
                --edit_name ${edit_name} \

# done