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


model_name = "BAAI/bge-reranker-v2-m3"
parts = model_name.split("/")
str_name = f"(bge) {parts[0]}-{parts[1]}_"


def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    prof.export_chrome_trace(
        f"examples/offline_inference/basic/my_examples/{str_name}"
        + str(prof.step_num)
        + ".json"
    )


def main(args: Namespace):
    # Sample prompts.
    text_1 = "What is the capital of France?"
    texts_2 = [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.",
    ]

    # Create an LLM.
    # You should pass task="score" for cross-encoder models
    model = LLM(**vars(args))

    # Generate scores. The output is a list of ScoringRequestOutputs.
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
            outputs = model.score(text_1, texts_2)
            prof.step()

    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for text_2, output in zip(texts_2, outputs):
        score = output.outputs.score
        print(f"Pair: {[text_1, text_2]!r} \nScore: {score}")
        print("-" * 60)


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    # Set example specific arguments
    parser.set_defaults(
        model=model_name,
        task="score",
        enforce_eager=True,
        disable_async_output_proc=True,
    )
    args = parser.parse_args()
    main(args)
