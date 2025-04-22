# SPDX-License-Identifier: Apache-2.0
from vllm.my_utils import decorate_all_methods, profile_function # added by auto-decorator-script

from dataclasses import dataclass
from typing import Tuple


@dataclass
@decorate_all_methods(profile_function) # added by auto-decorator-script
class AdapterMapping:
    # Per every token in input_ids:
    index_mapping: Tuple[int, ...]
    # Per sampled token:
    prompt_mapping: Tuple[int, ...]

    def __post_init__(self):
        self.index_mapping = tuple(self.index_mapping)
        self.prompt_mapping = tuple(self.prompt_mapping)