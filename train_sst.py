#!/usr/bin/env python
# encoding: utf-8

"""BERT finetuning runner."""

import tensorflow as tf

import processors
import train_common


def run(flags):
    processor = processors.SSTProcessor()
    train_common.run_classifier(flags, processor)


def main(_):
    run(tf.app.flags.FLAGS.flag_values_dict())


if __name__ == "__main__":
    tf.flags.mark_flag_as_required("data_dir")
    # tf.flags.mark_flag_as_required("vocab_file")
    # tf.flags.mark_flag_as_required("bert_config_file")
    # tf.flags.mark_flag_as_required("output_dir")
    tf.app.run()
