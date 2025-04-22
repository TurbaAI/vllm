# SPDX-License-Identifier: Apache-2.0

import torch
import tqdm
from torch.profiler import profile

from vllm import LLM, SamplingParams


def trace_handler(prof):
    print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1))
    prof.export_chrome_trace(
        "examples/offline_inference/basic/my_examples/opt-125m_"
        + str(prof.step_num)
        + ".json"
    )


# Sample prompts.
prompts = [
    "Hello, my name is",
    "The president of the United States is",
    "The capital of France is",
    "The future of AI is",
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

# Create an LLM.
llm = LLM(model="facebook/opt-125m", enforce_eager=True, disable_async_output_proc=True)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.

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
    for i in tqdm.tqdm(range(10), postfix="generate"):
        outputs = llm.generate(prompts, sampling_params)
        prof.step()

# Print the outputs.
print("\nGenerated Outputs:\n" + "-" * 60)
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt:    {prompt!r}")
    print(f"Output:    {generated_text!r}")
    print("-" * 60)
