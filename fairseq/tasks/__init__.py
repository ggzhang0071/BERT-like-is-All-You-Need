# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib
import os

from .fairseq_task import FairseqTask

TASK_REGISTRY = {}
TASK_CLASS_NAMES = set()


def setup_task(args, **kwargs):
    return TASK_REGISTRY[args.task].setup_task(args, **kwargs)


def register_task(name):
    """
    New tasks can be added to fairseq with the
    :func:`~fairseq.tasks.register_task` function decorator.

    For example::

        @register_task('classification')
        class ClassificationTask(FairseqTask):
            (...)

    .. note::

        All Tasks must implement the :class:`~fairseq.tasks.FairseqTask`
        interface.

    Please see the

    Args:
        name (str): the name of the task
    """

    def register_task_cls(cls):
        if name in TASK_REGISTRY:
            pass
            #raise ValueError('Cannot register duplicate task ({})'.format(name))
        if not issubclass(cls, FairseqTask):
            pass
            #raise ValueError('Task ({}: {}) must extend FairseqTask'.format(name, cls.__name__))
        if cls.__name__ in TASK_CLASS_NAMES:
            pass
            #raise ValueError('Cannot register task with duplicate class name ({})'.format(cls.__name__))
        else:
            TASK_REGISTRY[name] = cls
            TASK_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_task_cls


def get_task(name):
    return TASK_REGISTRY[name]


# automatically import any Python files in the tasks/ directory
tasks_dir = os.path.dirname(__file__)
for file in os.listdir(tasks_dir):
    path = os.path.join(tasks_dir, file)
    if (
        not file.startswith('_')
        and not file.startswith('.')
        and (file.endswith('.py') or os.path.isdir(path))
    ):
        task_name = file[:file.find('.py')] if file.endswith('.py') else file
        importlib.import_module('fairseq.tasks.' + task_name)

        # expose `task_parser` for sphinx
        if task_name in TASK_REGISTRY:
            parser = argparse.ArgumentParser(add_help=False)
            group_task = parser.add_argument_group('Task name')
            # fmt: off
            group_task.add_argument('--task', metavar=task_name,
                                    help='Enable this task with: ``--task=' + task_name + '``')
            # fmt: on
            group_args = parser.add_argument_group('Additional command-line arguments')
            TASK_REGISTRY[task_name].add_args(group_args)
            globals()[task_name + '_parser'] = parser
