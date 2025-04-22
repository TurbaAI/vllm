# SPDX-License-Identifier: Apache-2.0
from vllm.my_utils import decorate_all_methods, profile_function # added by auto-decorator-script

# Adapted from llama.py
"""Inference-only Phi3 model code inherit from Llama.py"""

from vllm.model_executor.models.llama import LlamaForCausalLM


@decorate_all_methods(profile_function) # added by auto-decorator-script
class Phi3ForCausalLM(LlamaForCausalLM):

    packed_modules_mapping = {
        "qkv_proj": [
            "qkv_proj",
        ],
        "gate_up_proj": [
            "gate_up_proj",
        ],
    }
