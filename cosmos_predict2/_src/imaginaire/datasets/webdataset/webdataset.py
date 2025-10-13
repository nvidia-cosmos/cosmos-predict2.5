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

import json
import os
import time
import warnings
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
from typing import Callable

import omegaconf
import webdataset as wds
from webdataset.handlers import reraise_exception

from cosmos_predict2._src.imaginaire.datasets.webdataset.config.schema import (
    AugmentorConfig,
    DatasetConfig,
    DatasetInfo,
    TarSample,
    Wdinfo,
)
from cosmos_predict2._src.imaginaire.datasets.webdataset.utils.iterators import WebDataset
from cosmos_predict2._src.imaginaire.datasets.webdataset.utils.misc import (
    remove_extensions_from_keys,
    skip_keys,
    update_url,
)
from cosmos_predict2._src.imaginaire.lazy_config import instantiate
from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.utils.distributed import get_world_size
from cosmos_predict2._src.imaginaire.utils.object_store import ObjectStore


def wrap_augmentor_func_as_generator(func: Callable, data: Iterable):
    for data_dict in data:
        data_dict_out = func(data_dict)
        if data_dict_out is None:
            # Skip "unhealthy" samples
            continue
        yield data_dict_out


class Dataset:
    def __init__(
        self,
        config: DatasetConfig,
        handler: Callable = reraise_exception,
    ):
        r"""Webdataloader class

        Args:
            config: Dataset config
            world_size: Total number of GPUs
        """
        super().__init__()

        self.config = config

        self.world_size = get_world_size()

        dataset_info = config.dataset_info
        self.streaming_download = config.streaming_download

        self.s3_client = dict()
        self.bucket = dict()
        self.data_keys = config.keys

        # Parse the metadata
        self.wdinfo = Wdinfo([], 0, 0)
        self.parse_dataset_info(dataset_info=dataset_info, use_multithread=True)
        self.handler = handler
        self.augmentors = dict()

    def parse_dataset_info(self, dataset_info: list[DatasetInfo], use_multithread: bool = True):
        r"""Parse metadata about the list of tar files.

        Args:
            dataset_info (list): List of dictionaries containing paths to metadata files.
            use_multithread (bool): Whether to use multi-threaded parsing across datasets. Default: True.
        """
        log.info(f"Start parsing dataset info with {len(dataset_info)} entries, use multithread = {use_multithread}")
        tic = time.time()

        def process_single_dataset(dset_num: int, dset_info: DatasetInfo):
            # For each dataset, we parse the file paths and store them as a list of TarSample.
            # TarSample will then be used by each worker to load the data.
            use_object_store = dset_info.object_store_config.enabled
            self.use_object_store = use_object_store
            dset_id = "dset: {}".format(dset_num)
            if use_object_store:
                object_store_reader = ObjectStore(config_object_storage=dset_info.object_store_config)

                # Create PBSS config if data is loaded from PBSS
                bucket_dset = dset_info.object_store_config.bucket
                s3_client_dset = object_store_reader.client
            else:
                object_store_reader = None
                s3_client_dset = None
                bucket_dset = None

            tar_samples = []
            total_key_count = 0
            chunk_sizes = []

            # Read all wdinfo files and obtain the DataSample list
            for wdinfo_path in dset_info.wdinfo:
                if use_object_store:
                    if not object_store_reader.object_exists(wdinfo_path):
                        raise FileNotFoundError(f"{wdinfo_path} not found")
                    cur_dset_info = object_store_reader.load_object(key=wdinfo_path, type="json")  # type: ignore
                else:
                    with open(wdinfo_path, "r") as fp:
                        cur_dset_info = json.load(fp)

                data_root = cur_dset_info["root"]
                tar_files_list = cur_dset_info["data_list"]
                local_tar_samples = [
                    TarSample(
                        path=tar_file,
                        root=data_root,
                        keys=(
                            dset_info.per_dataset_keys if dset_info.per_dataset_keys else self.data_keys
                        ),  # use per dataset keys if available
                        meta=dset_info,
                        dset_id=dset_id,
                        sample_keys_full_list=None,
                    )
                    for tar_file in tar_files_list
                ]
                tar_samples.extend(local_tar_samples)
                total_key_count += cur_dset_info["total_key_count"]
                chunk_sizes.append(cur_dset_info["chunk_size"])

            return {
                "dset_id": dset_id,
                "tar_samples": tar_samples,
                "total_key_count": total_key_count,
                "chunk_sizes": chunk_sizes,
                "s3_client": s3_client_dset,
                "bucket": bucket_dset,
            }

        dataset_results = []

        if use_multithread:
            num_workers = os.cpu_count()
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = []
                for i, dset_info in enumerate(dataset_info):
                    if len(dset_info.wdinfo) == 0:
                        log.warning(f"No wdinfo found for dataset {i}, skipping...")
                        continue
                    log.info(f"Adding: {dset_info.wdinfo}")
                    futures.append(executor.submit(process_single_dataset, i, dset_info))
                for future in as_completed(futures):
                    dataset_results.append(future.result())
        else:
            for i, dset_info in enumerate(dataset_info):
                log.info(f"Adding: {dset_info.wdinfo}")
                dataset_results.append(process_single_dataset(i, dset_info))

        # Merge results
        for result in dataset_results:
            dset_id = result["dset_id"]
            self.wdinfo.tar_files.extend(result["tar_samples"])
            self.wdinfo.total_key_count += result["total_key_count"]
            if len(set(result["chunk_sizes"])) > 1:
                warnings.warn(
                    f"Multiple chunk_size values found in {dset_id}: {result['chunk_sizes']}. Using the first one."
                )
            self.wdinfo.chunk_size = result["chunk_sizes"][0]
            if result["s3_client"]:
                self.s3_client[dset_id] = result["s3_client"]
            if result["bucket"]:
                self.bucket[dset_id] = result["bucket"]
        toc = time.time()
        log.info(
            f"Parsed dataset info with {len(dataset_info)} wdinfos (num_keys = {self.wdinfo.total_key_count}, num_tars = {len(self.wdinfo.tar_files)}) and multithread = {use_multithread}, took {(toc - tic):.2f} seconds"
        )

    @staticmethod
    # This is the function that calls each augmentor in sequence.
    def augmentor_fn(data, augmentations):
        # Build augmentor chain
        for aug_fn in augmentations:
            # Use generator function as augmentor
            # (recommended, allows skipping or replicating samples inside the augmentor)
            if getattr(aug_fn, "is_generator", False):
                data = aug_fn(data)
            else:  # Use regular function as augmentor (backward compatibility)
                data = wrap_augmentor_func_as_generator(aug_fn, data)
        yield from data

    def build_data_augmentor(self, augmentor_cfg: dict[str, AugmentorConfig]) -> Callable:
        r"""Function for building data augmentors from augmentor config."""
        augmentations = []
        for aug in augmentor_cfg.keys():
            augmentations.append(instantiate(augmentor_cfg[aug]))

        # This is the function that calls each augmentor in sequence.
        return partial(Dataset.augmentor_fn, augmentations=augmentations)

    def build_dataset(self, **kwargs) -> WebDataset:
        tar_list = self.wdinfo.tar_files
        num_tars = len(tar_list)
        assert num_tars > 0, "Did not find any data."

        shuffle_buffer_size = getattr(self.config, "buffer_size", self.wdinfo.chunk_size)

        # update distributor urls and chunk size
        distributor_fn = self.config.distributor

        distributor_fn.set_urls(tar_list)
        distributor_fn.set_chunk_size(self.wdinfo.chunk_size)

        dataset = WebDataset(
            distributor_fn,
            load_from_object_store=self.use_object_store,
            s3_client=self.s3_client,
            s3_bucket_name=self.bucket,
            streaming_download=self.streaming_download,
            handler=self.handler,
        )

        # Creating a shuffle buffer
        if shuffle_buffer_size > 0:
            dataset.append(wds.shuffle(shuffle_buffer_size))

        # Adding decoders
        # Decoders are functions that decode the input IO stream
        decoder_list = getattr(self.config, "decoders", [])
        decoder_functions = []
        for decoder in decoder_list:
            # If the specified decoder is a string, use the webdataset decoder
            # If its a callable function, use the defined function to decode data
            assert isinstance(decoder, str) or callable(decoder), "Decoder should either be callable or a str"
            decoder_functions.append(decoder)
        dataset.append(wds.decode(*decoder_functions))

        # After the decoders are added, remove extension from the keys
        # Extensions in the data keys are needed for auto-detection of decoders in webdataset.
        if self.config.remove_extension_from_keys:
            dataset.append(remove_extensions_from_keys)

        # Function to skip keys
        dataset.append(skip_keys)
        # Building augmentors
        augmentor_cfg = getattr(self.config, "augmentation", None)
        assert isinstance(augmentor_cfg, (dict, omegaconf.dictconfig.DictConfig)), (
            f"getting type: {type(augmentor_cfg)}"
        )
        augmentation_fn = self.build_data_augmentor(augmentor_cfg)
        dataset.append(augmentation_fn)

        # Updates URL names so that the collate function can handle
        dataset.append(update_url)

        dataset.total_images = self.wdinfo.total_key_count  # type: ignore
        log.info("Total number of training shards: %d" % num_tars)
        log.info("Total training key count: %d" % dataset.total_images)  # type: ignore

        return dataset
