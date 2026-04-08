#!/bin/bash
set -o errexit # Exit on error
set -o nounset # Trigger error when expanding unset variables

root_dir=$1
exp_name=$2
echo "Dataset root dir: ${root_dir}"
list=$(basename -a $root_dir/*/)
echo "TFs list: " $list


for i in $list; do
# i="TF01"
# #* optimize 2DGS
mkdir -p ./output/${exp_name}/${i}/2dgs
mkdir -p ./output/${exp_name}/${i}/texgs
tic=$(date +%s)
python train.py --eval \
                -s ${root_dir}/${i} \
                -m ./output/${exp_name}/${i}/2dgs >> ./output/${exp_name}/${i}/2dgs/training_log.txt
toc=$(date +%s)
echo "2DGS training time for ${i}: $((toc - tic)) seconds" >> ./output/${exp_name}/${i}/2dgs/2dgs_training_time.txt

#* optimize recolorable texGS
tic=$(date +%s)
python train.py --eval \
                -t 'TexGS' \
                -s ${root_dir}/${i} \
                -m ./output/${exp_name}/${i}/texgs \
                -init ./output/${exp_name}/${i}/2dgs/point_cloud/iteration_30000/point_cloud.ply \
                --iteration 3000 >> ./output/${exp_name}/${i}/texgs/texgs_training_log.txt
toc=$(date +%s)
echo "TexGS training time for ${i}: $((toc - tic)) seconds" >> ./output/${exp_name}/${i}/texgs/texgs_training_time.txt


done

#* inference
# python render.py -m ./output/chameleon/TF01/texgs  \
#                  -s /home/dullpigeon/SSD/TFImgData/chameleonRGBa/TF01 \
#                  --skip_mesh
