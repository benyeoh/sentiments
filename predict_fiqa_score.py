#!/usr/bin/env python
# encoding: utf-8

"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import bert.modeling as modeling
import bert.tokenization as tokenization
import tensorflow as tf

import processors
import predict_score


tf.flags.DEFINE_bool("use_headlines", True, "Headlines data.")

tf.flags.DEFINE_bool("use_posts", True, "Posts data.")


def run(flags):
    def _get_predict_examples(predict_flags):
        if predict_flags.use_headlines and predict_flags.use_posts:
            processor = processors.FiQACombineProcessor(predict_flags.data_dir)
        elif predict_flags.use_headlines:
            processor = processors.FiQAHeadlinesProcessor(predict_flags.data_dir)
        else:
            processor = processors.FiQAPostsProcessor(predict_flags.data_dir)
        return processor.get_eval_examples()        
    return predict_score.predict_score(flags, _get_predict_examples)


def main(_):
    res = run(tf.app.flags.FLAGS.flag_values_dict())
    tf.logging.info(res)


if __name__ == "__main__":
    tf.flags.mark_flag_as_required("data_dir")
    tf.flags.mark_flag_as_required("init_checkpoint")
    # flags.mark_flag_as_required("vocab_file")
    # flags.mark_flag_as_required("bert_config_file")
    # flags.mark_flag_as_required("output_dir")
    tf.app.run()
