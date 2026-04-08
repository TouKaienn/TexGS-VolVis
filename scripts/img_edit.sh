#!/bin/bash
set -o errexit # Exit on error
set -o nounset # Trigger error when expanding unset variables

root_dir=$1
exp_name=$2
edit_name=$3
shift 3
style_img_path=($@)
echo "Dataset root dir: ${root_dir}"
list=($(basename -a ${root_dir}/*/))
echo "TFs list: " ${list[@]}


for i in $(seq 0 $((${#list[@]} - 1))); do
tic=$(date +%s)

# #* optimize texture
CUDA_LAUNCH_BLOCKING=1 python imgEdit.py --eval \
                -t 'stylize' \
                -s ${root_dir}/${list[i]} \
                -m ./output/${exp_name}/${list[i]}/${edit_name}_Img \
                -init ./output/${exp_name}/${list[i]}/2dgs/point_cloud/iteration_30000/point_cloud.ply \
                --iteration 3000 \
                --texture_lr 0.0025 \
                --style_img_path "${style_img_path[i]}" \
                --edit_name ${edit_name} \

toc=$(date +%s)
echo "Processing ${list[i]}'s ${style_img_path[i]} editing took $((toc - tic)) seconds" >> output/${exp_name}/${list[i]}/time.txt
done