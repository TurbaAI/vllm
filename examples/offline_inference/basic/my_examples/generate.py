# SPDX-License-Identifier: Apache-2.0

import os

import torch
import tqdm
from dotenv import load_dotenv
from huggingface_hub import login
from torch.profiler import profile

from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser

load_dotenv()
hf = os.getenv("TOKEN")

login(token=hf)


model_name = "meta-llama/Llama-3.2-1B-Instruct"
parts = model_name.split("/")
str_name = f"(llama 1gpu) {parts[0]}-{parts[1]}_"


def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    prof.export_chrome_trace(
        f"examples/offline_inference/basic/my_examples/final/Transformer-like LLM/{str_name}"
        + str(prof.step_num)
        + ".json"
    )


def main(args: dict):
    # Pop arguments not used by LLM
    max_tokens = args.pop("max_tokens")
    temperature = args.pop("temperature")
    top_p = args.pop("top_p")
    top_k = args.pop("top_k")

    args["enforce_eager"] = True
    args["disable_async_output_proc"] = True
    args["trust_remote_code"] = True
    args["tensor_parallel_size"] = 1  # 8

    # Create an LLM
    llm = LLM(**args)

    # Create a sampling params object
    max_tokens = 16
    sampling_params = llm.get_default_sampling_params()
    if max_tokens is not None:
        sampling_params.max_tokens = max_tokens
    if temperature is not None:
        sampling_params.temperature = temperature
    if top_p is not None:
        sampling_params.top_p = top_p
    if top_k is not None:
        sampling_params.top_k = top_k

    # Generate texts from the prompts. The output is a list of RequestOutput
    # objects that contain the prompt, generated text, and other information.
    prompts = ["Hello, my name is"]

    # Generate embedding. The output is a list of EmbeddingRequestOutputs.
    with profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=9, active=10),
        with_stack=False,
        record_shapes=False,
        on_trace_ready=trace_handler,
    ) as prof:
        with torch.inference_mode():
            for i in tqdm.tqdm(range(10), postfix="generate"):
                outputs = llm.generate(prompts, sampling_params)
                prof.step()

    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    # Add engine args
    engine_group = parser.add_argument_group("Engine arguments")
    EngineArgs.add_cli_args(engine_group)
    # model_name = "meta-llama/Llama-3.2-1B-Instruct"

    engine_group.set_defaults(model=model_name)
    # Add sampling params
    sampling_group = parser.add_argument_group("Sampling parameters")
    sampling_group.add_argument("--max-tokens", type=int)
    sampling_group.add_argument("--temperature", type=float)
    sampling_group.add_argument("--top-p", type=float)
    sampling_group.add_argument("--top-k", type=int)
    args: dict = vars(parser.parse_args())
    main(args)
