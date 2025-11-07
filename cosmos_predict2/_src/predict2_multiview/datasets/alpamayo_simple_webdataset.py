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

import gzip
import os
import pickle

from cosmos_predict2._src.imaginaire import config
from cosmos_predict2._src.imaginaire.datasets.webdataset.config.schema import DatasetInfo, TarSample
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.predict2_multiview.datasets.alpamayo_raw_webdataset import AlpamayoRawWebdataset
from cosmos_predict2._src.predict2_multiview.datasets.cache_utils import get_cam_t5_cache_dir
from cosmos_predict2._src.predict2_multiview.utils.object_store import ObjectStore


class AlpamayoSimpleWebdataset(AlpamayoRawWebdataset):
    def parse_dataset_info(
        self,
        dataset_info: list[DatasetInfo],
        use_multithread: bool = False,
    ):
        r"""Parse metadata about the list of tar files.

        Args:
            dataset_info (list): List of dictionaries containing paths to metadata files.
        """
        if use_multithread:
            log.warning("use_multithread not supported for AlpamayoSimpleWebdataset. Ignoring it.")
        t5_object_store_initialized = False
        t5_prefix = None
        # we have to build two clients if caption is stored in team-sil-videogen
        caption_json_dset_id = "caption_json_dset: 0"
        cam_t5_cache_dir = get_cam_t5_cache_dir()

        for dset_num, dset_info in enumerate(dataset_info):
            # For each dataset, we parse the file paths and store them as a list of TarSample.
            # TarSample will then be used by each worker to load the data.
            use_object_store = dset_info.object_store_config.enabled
            self.use_object_store = use_object_store
            dset_id = "dset: {}".format(dset_num)
            sampling_data = (dset_info.opts["view_indices_options"], dset_info.opts["view_id_to_camera_key"])
            download_t5_tar = dset_info.opts["download_t5_tar"]
            assert download_t5_tar is False, "download_t5_tar must be False for simple webdataset"

            overfit_firstn = dset_info.opts["overfit_firstn"]
            if use_object_store:
                data_object_store_reader = ObjectStore(config_object_storage=dset_info.object_store_config)
                # Create PBSS config if data is loaded from PBSS
                bucket_dset = dset_info.object_store_config.bucket
                s3_client_dset = data_object_store_reader.client
                self.s3_client[dset_id] = s3_client_dset
                self.bucket[dset_id] = bucket_dset
                if not t5_object_store_initialized:
                    dset_info.opts["t5_store_config"] = {
                        "object_store": config.ObjectStoreConfig(
                            credentials="credentials/team-sil-videogen.secret",
                            bucket="alpamayo_v2.2",
                        ),
                        "prefix": "reformat_caption_av2.2_7views_2_10_second_align_format",
                        "video_prefix": "predict2_multiview/mvd/AV-V2.2/videos",
                    }
                    t5_store_config = dset_info.opts["t5_store_config"]

                    t5_object_store_reader = ObjectStore(config_object_storage=t5_store_config["object_store"])
                    self.s3_client[caption_json_dset_id] = t5_object_store_reader.client
                    self.bucket[caption_json_dset_id] = t5_store_config["object_store"].bucket
                    video_prefix = t5_store_config["video_prefix"]
                    t5_prefix = t5_store_config["prefix"]

                    t5_object_store_initialized = True

            # Read all wdinfo files and obtain the DataSample list
            for train_chunk_id_clip_id_pkl_gz in dset_info.wdinfo:
                with gzip.open(os.path.join(cam_t5_cache_dir, train_chunk_id_clip_id_pkl_gz), "rb") as f:
                    train_chunk_id_clip_id_list = pickle.load(f)

                data_list = []
                for chunk_id_clip_id in train_chunk_id_clip_id_list:
                    chunk_id, clip_id = chunk_id_clip_id.split("/")
                    t5_tar_path = f"{t5_prefix}/{chunk_id}/{clip_id}.tar"
                    video_tar_url_no_ext = f"{video_prefix}/{chunk_id}/{clip_id}"
                    data_list.append(
                        (video_tar_url_no_ext, (t5_tar_path, caption_json_dset_id, sampling_data, download_t5_tar))
                    )

                if overfit_firstn > 0:
                    data_list = sorted(data_list, key=lambda x: x[0])[:overfit_firstn]
                sample_keys_full_list_per_tar = [None] * len(data_list)
                tar_files_list = data_list
                tar_files = [
                    TarSample(
                        path=tar_file,
                        root=bucket_dset,
                        keys=(
                            dset_info.per_dataset_keys if dset_info.per_dataset_keys else self.data_keys
                        ),  # use per dataset keys if available
                        meta=dset_info,
                        dset_id=dset_id,
                        sample_keys_full_list=sample_keys_full_list,
                    )
                    for tar_file, sample_keys_full_list in zip(
                        tar_files_list, sample_keys_full_list_per_tar, strict=True
                    )
                ]

                # Update the master winfo
                self.wdinfo.tar_files.extend(tar_files)
                self.wdinfo.total_key_count += len(data_list)
                self.wdinfo.chunk_size = 1
        log.info(f"Total number of samples: {self.wdinfo.total_key_count}")
