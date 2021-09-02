#!/usr/bin/env python3 -u
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from fairseq_cli.train import cli_main
import argparse

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description="")
    parser.add_argument('--data', type=str, default='/git/PaDiM-master/kangqiang_result')

    cli_main()
