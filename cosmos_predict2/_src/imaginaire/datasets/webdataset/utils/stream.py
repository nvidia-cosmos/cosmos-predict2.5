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


# PBSS
import time
from typing import Optional

from botocore.exceptions import EndpointConnectionError
from multistorageclient.types import RetryableError
from urllib3.exceptions import ProtocolError as URLLib3ProtocolError
from urllib3.exceptions import ReadTimeoutError as URLLib3ReadTimeoutError
from urllib3.exceptions import SSLError as URLLib3SSLError

from cosmos_predict2._src.imaginaire.utils import log
from cosmos_predict2._src.imaginaire.utils.easy_io.backends import BaseStorageBackend


class RetryingStream:
    def __init__(self, easy_io_backend: BaseStorageBackend, bucket: str, key: str, retries: int = 10):  # type: ignore
        r"""Class for loading data in a streaming fashion from an object store.
        Args:
            easy_io_backend (BaseStorageBackend): easy_io backend, must support 's3://' URLs
            bucket (str): Bucket where data is stored
            key (str): Key to read
            retries (int): Number of retries
        """
        self.easy_io_backend = easy_io_backend
        self.filepath = f"s3://{bucket}/{key}"
        self.retries = retries
        self.content_size = self.easy_io_backend.size(filepath=self.filepath)
        self._amount_read = 0

        self.name = f"{bucket}/{key}"

    def read(self, amt: Optional[int] = None) -> bytes:
        r"""Read function for reading the data stream.
        Args:
            amt (int, optional): Amount of data to read
        Returns:
            chunk (bytes): Bytes read
        """

        chunk = b""
        for cur_retry_idx in range(self.retries):
            try:
                chunk = self.easy_io_backend.get(
                    filepath=self.filepath,
                    offset=self._amount_read,
                    size=amt or (self.content_size - self._amount_read),
                )
                if len(chunk) == 0 and self._amount_read != self.content_size:
                    raise IOError
                break
            except URLLib3ReadTimeoutError as e:
                log.warning(
                    f"URLLib3ReadTimeoutError: {e} {self.name} retry: {cur_retry_idx} / {self.retries}",
                    rank0_only=False,
                )
            except URLLib3ProtocolError as e:
                log.warning(
                    f"URLLib3ProtocolError: {e} {self.name} retry: {cur_retry_idx} / {self.retries}",
                    rank0_only=False,
                )
            except URLLib3SSLError as e:
                log.warning(
                    f"URLLib3SSLError: {e} {self.name} retry: {cur_retry_idx} / {self.retries}", rank0_only=False
                )
            except IOError as e:
                log.warning(
                    f"Premature end of stream. IOError {e}. Retrying...  {self.name} retry: {cur_retry_idx} / {self.retries}",
                    rank0_only=False,
                )
            except RetryableError as e:
                log.warning(
                    f"RetryableError: {e} {self.name} retry: {cur_retry_idx} / {self.retries}",
                    rank0_only=False,
                )
            except RuntimeError as e:
                log.warning(
                    f"RuntimeError: {e} {self.name} retry: {cur_retry_idx} / {self.retries}",
                    rank0_only=False,
                )
            except EndpointConnectionError as e:
                log.error(
                    f"EndpointConnectionError: {e} {self.name} retry: {cur_retry_idx} / {self.retries}",
                    rank0_only=False,
                )
            time.sleep(1)

        if len(chunk) == 0 and self._amount_read != self.content_size:
            log.warning(
                f"After {self.retries} retries, chunk is empty and self._amount_read != self.content_size {self._amount_read} != {self.content_size} {self.name}",
                rank0_only=False,
            )
            raise IOError

        self._amount_read += len(chunk)
        return chunk
