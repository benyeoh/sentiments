#!/usr/bin/env python
# encoding: utf-8

"""BERT finetuning runner."""

import processors
import train_common

import tensorflow as tf


tf.flags.DEFINE_bool("use_headlines", True, "Headlines data.")

tf.flags.DEFINE_bool("use_posts", True, "Posts data.")


def run(flags):
    if flags["use_headlines"] and flags["use_posts"]:
        fiqa_processor = processors.FiQACombineProcessor(flags["data_dir"])
    elif flags["use_headlines"]:
        fiqa_processor = processors.FiQAHeadlinesProcessor(flags["data_dir"])
    else:
        fiqa_processor = processors.FiQAPostsProcessor(flags["data_dir"])
        
    processor = processors.FiQAClassProcessor(fiqa_processor, class_separation=[-0.1, 0.1])
    train_common.run_classifier(flags, processor)


def main(_):
    run(tf.app.flags.FLAGS.flag_values_dict())


if __name__ == "__main__":
    tf.flags.mark_flag_as_required("data_dir")
    # tf.flags.mark_flag_as_required("vocab_file")
    # tf.flags.mark_flag_as_required("bert_config_file")
    # tf.flags.mark_flag_as_required("output_dir")
    tf.app.run()
