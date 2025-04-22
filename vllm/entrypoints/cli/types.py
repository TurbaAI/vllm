# SPDX-License-Identifier: Apache-2.0
from vllm.my_utils import decorate_all_methods, profile_function # added by auto-decorator-script

import argparse

from vllm.utils import FlexibleArgumentParser


@decorate_all_methods(profile_function) # added by auto-decorator-script
class CLISubcommand:
    """Base class for CLI argument handlers."""

    name: str

    @staticmethod
    def cmd(args: argparse.Namespace) -> None:
        raise NotImplementedError("Subclasses should implement this method")

    def validate(self, args: argparse.Namespace) -> None:
        # No validation by default
        pass

    def subparser_init(
            self,
            subparsers: argparse._SubParsersAction) -> FlexibleArgumentParser:
        raise NotImplementedError("Subclasses should implement this method")
