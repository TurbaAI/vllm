# SPDX-License-Identifier: Apache-2.0
import argparse
from vllm.my_utils import decorate_all_methods, profile_function # added by auto-decorator-script

from vllm.benchmarks.serve import add_cli_args, main
from vllm.entrypoints.cli.benchmark.base import BenchmarkSubcommandBase
from vllm.entrypoints.cli.types import CLISubcommand


@decorate_all_methods(profile_function) # added by auto-decorator-script
class BenchmarkServingSubcommand(BenchmarkSubcommandBase):
    """ The `serve` subcommand for vllm bench. """

    def __init__(self):
        self.name = "serve"
        super().__init__()

    @property
    def help(self) -> str:
        return "Benchmark the online serving throughput."

    def add_cli_args(self, parser: argparse.ArgumentParser) -> None:
        add_cli_args(parser)

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        main(args)


def cmd_init() -> list[CLISubcommand]:
    return [BenchmarkServingSubcommand()]
