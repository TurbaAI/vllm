# SPDX-License-Identifier: Apache-2.0
from vllm.my_utils import decorate_all_methods, profile_function # added by auto-decorator-script

# Adapted from
# https://huggingface.co/h2oai/h2ovl-mississippi-2b/blob/main/configuration_h2ovl_chat.py
# --------------------------------------------------------
# H2OVL-Mississippi
# Copyright (c) 2024 H2O.AI
# Licensed under Apache 2.0 License [see LICENSE for details]
# --------------------------------------------------------

from .internvl import InternVLChatConfig


@decorate_all_methods(profile_function) # added by auto-decorator-script
class H2OVLChatConfig(InternVLChatConfig):
    model_type = "h2ovl_chat"
