#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This script is modified from the original version by Jianfei Guo:
# https://github.com/PJLab-ADG/neuralsim/blob/main/dataio/autonomous_driving/waymo/download_waymo.sh

# NOTE: Before proceeding, you need to fill out the Waymo terms of use and complete `gcloud auth login`.

dest=$1 # destination directory for waymo dataset
limit=$2 # optional limit for the number of files to download

down_dest=$dest/downloads
proc_dest=$dest/input

mkdir -p $down_dest

wget -nc -P $dest https://raw.githubusercontent.com/nv-tlabs/Cosmos-Drive-Dreams/refs/heads/main/cosmos-drive-dreams-toolkits/assets/waymo_all.json
wget -nc -P $dest https://raw.githubusercontent.com/nv-tlabs/Cosmos-Drive-Dreams/refs/heads/main/cosmos-drive-dreams-toolkits/assets/waymo_caption.csv
jfile=$dest/waymo_all.json

# read json file. the json file is a list of clip ids
lst=$(jq -r '.[]' $jfile)
total_files=$(jq '. | length' $jfile)

counter=0

# loop through the clip ids in lst as filename
for filename in $lst; do
    # filename_full is 'segment-' + $filename + '_with_camera_labels'
    filename_full="segment-${filename}_with_camera_labels.tfrecord"

    if [ -f "${down_dest}/${filename_full}" ]; then
        echo "[${counter}/${total_files}] Skipping ${filename_full}, already exists."
        continue
    fi

    if [ -d "${proc_dest}/${filename}" ]; then
        echo "[${counter}/${total_files}] Skipping ${filename_full}, already has been processed."
        continue
    fi

    if [[ -n "$limit" && "$limit" -gt 0 && "$counter" -eq "$limit" ]]; then
        echo "Downloaded all $limit file(s). Exiting."
        break
    fi

    # can be in training
    train_source=gs://waymo_open_dataset_v_1_4_2/individual_files/training
    gsutil cp -n ${train_source}/${filename_full} ${down_dest} >/dev/null 2>&1
    found_in_train=$?

    if [ "$found_in_train" -eq 0 ]; then
        echo "[$((counter + 1))/${total_files}] Dowloaded $filename_full from training"
    else
        # or can be in validation
        val_source=gs://waymo_open_dataset_v_1_4_2/individual_files/validation
        gsutil cp -n ${val_source}/${filename_full} ${down_dest} >/dev/null 2>&1
        found_in_val=$?
        echo "[$((counter + 1))/${total_files}] Dowloaded $filename_full from validation"
    fi

    if [[ "$found_in_train" -eq 0 || "$found_in_val" -eq 0 ]]; then
        counter=$((counter + 1))
    else
        echo "[${counter}/${total_files}] Skipping $filename_full, not found in training or validation sets."
    fi

done
