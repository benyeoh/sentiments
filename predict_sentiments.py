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


def run(flags):
    def _get_predict_examples(predict_flags):
        predict_path = os.path.join(predict_flags.data_dir, "predict.txt")
        with tf.gfile.GFile(predict_path, "r") as reader:
            examples = []
            for i, line in enumerate(reader):
                sentence = tokenization.convert_to_unicode(line)
                # Remove URLS in sentence
                sentence = re.sub(r'http\S+', '', sentence)
                examples.append(
                    processors.InputExample(guid=i, text_a=sentence, text_b=None, label=0.0))
            return examples    
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
