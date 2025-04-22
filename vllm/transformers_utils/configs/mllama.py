# SPDX-License-Identifier: Apache-2.0
from vllm.my_utils import decorate_all_methods, profile_function # added by auto-decorator-script

from transformers.models.mllama import configuration_mllama as mllama_hf_config


@decorate_all_methods(profile_function) # added by auto-decorator-script
class MllamaTextConfig(mllama_hf_config.MllamaTextConfig):
    '''
    Use this class to override is_encoder_decoder:
    - transformers regards mllama as is_encoder_decoder=False
    - vllm needs is_encoder_decoder=True to enable cross-attention
    '''

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.is_encoder_decoder = True


@decorate_all_methods(profile_function) # added by auto-decorator-script
class MllamaConfig(mllama_hf_config.MllamaConfig):

    def __init__(
        self,
        text_config=None,
        **kwargs,
    ):
        if isinstance(text_config, dict):
            text_config = MllamaTextConfig(**text_config)
        super().__init__(text_config=text_config, **kwargs)
