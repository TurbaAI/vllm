# SPDX-License-Identifier: Apache-2.0

import os
from argparse import Namespace

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


model_name = "sentence-transformers/all-roberta-large-v1"
parts = model_name.split("/")
str_name = f"(RoBERTa-based) {parts[0]}-{parts[1]}_"


def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    prof.export_chrome_trace(
        f"examples/offline_inference/basic/my_examples/final/Embedding Models/8GPU/{str_name}"
        + str(prof.step_num)
        + ".json"
    )


def main(args: Namespace):
    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]

    # Create an LLM.
    # You should pass task="embed" for embedding models
    model = LLM(**vars(args))

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
        for i in tqdm.tqdm(range(10), postfix="generate classify"):
            outputs = model.embed(prompts)
            prof.step()

    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for prompt, output in zip(prompts, outputs):
        embeds = output.outputs.embedding
        embeds_trimmed = (
            (str(embeds[:16])[:-1] + ", ...]") if len(embeds) > 16 else embeds
        )
        print(
            f"Prompt: {prompt!r} \n"
            f"Embeddings: {embeds_trimmed} (size={len(embeds)})"
        )
        print("-" * 60)


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    # Set example specific arguments
    parser.set_defaults(
        model=model_name,
        task="embed",
        enforce_eager=True,
        disable_async_output_proc=True,
        trust_remote_code=True,
        tensor_parallel_size=8
    )
    args = parser.parse_args()
    main(args)
