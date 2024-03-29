#!/usr/bin/env python
# encoding: utf-8

"""BERT finetuning runner."""

import random

import os
import processors
import train_common

import tensorflow as tf


def run(flags):

    class AllFinClassProcessor(processors.DataProcessor):

        def __init__(self, data_dir, train_ratio=0.9, seed=processors._SEED):
            fiqa_processor = processors.FiQACombineProcessor(data_dir, train_ratio=train_ratio, seed=seed)
            self._fiqa_class_processor = processors.FiQAClassProcessor(fiqa_processor, class_separation=[-0.075, 0.075])
            self._finPB_processor = processors.FinPBProcessor(data_dir, train_ratio=train_ratio, seed=seed)
            self._seed = seed

        def get_train_examples(self, data_dir):
            examples = self._fiqa_class_processor.get_train_examples(data_dir)
            examples.extend(self._finPB_processor.get_train_examples(data_dir))
            random.seed(self._seed)
            random.shuffle(examples)
            return examples

        def get_dev_examples(self, data_dir):
            examples = self._fiqa_class_processor.get_dev_examples(data_dir)
            examples.extend(self._finPB_processor.get_dev_examples(data_dir))
            random.seed(self._seed)
            random.shuffle(examples)
            return examples

        def get_labels(self):
            return ["0", "1", "2"]

    processor = AllFinClassProcessor(data_dir=flags["data_dir"])
    return train_common.run_classifier(flags, processor)


def save_improvement(output_dir, save_output_dir, res):
    train_common.compare_eval_save_model(
        res, output_dir, lambda x, y: x["eval_accuracy"] > y["eval_accuracy"], "eval_accuracy", save_output_dir)


def main(_):
    flags = tf.app.flags.FLAGS.flag_values_dict()
    res = run(flags)
    save_improvement(flags["output_dir"], os.path.join(flags["output_dir"], 'save'), res)


if __name__ == "__main__":
    tf.flags.mark_flag_as_required("data_dir")
    # tf.flags.mark_flag_as_required("vocab_file")
    # tf.flags.mark_flag_as_required("bert_config_file")
    # tf.flags.mark_flag_as_required("output_dir")
    tf.app.run()
