#!/bin/sh
# 分别下载https://thor.robots.ox.ac.uk/flash3d/test.pickle.gz等到data/RealEstate10K目录下
DATA_PATH=data/RealEstate10K
file_list=(
    # test.pickle.gz
    # train.pickle.gz
    # pcl.test.tar
    pcl.train.tar
    # valid_seq_ids.train.pickle.gz
    SHA512SUMS
)
ROOT_URL=https://thor.robots.ox.ac.uk/flash3d
cd $DATA_PATH 
for item in "${file_list[@]}"; do
    curl -C - -O $ROOT_URL/$item
done
sha512sum -c SHA512SUMS

