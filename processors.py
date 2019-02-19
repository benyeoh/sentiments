#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import collections
import json
import csv

import bert.modeling as modeling
import bert.optimization as optimization
import bert.tokenization as tokenization

import tensorflow as tf


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
          guid: Unique id for the example.
          text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
          text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
          label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class FiQAProcessor(object):

    def __init__(self, data_dir, train_ratio=0.9):
        self._data_dict = self._read_json_file(os.path.join(data_dir, "FiQA_ABSA", "task1_headline_ABSA_train.json"))    
        num_train = int(round(len(self._data_dict) * train_ratio))
        all_data = list(self._data_dict.items())
        self._train_data_dict = dict(all_data[:num_train])
        self._eval_data_dict = dict(all_data[num_train:])

    def _read_json_file(self, input_file):
        with tf.gfile.Open(input_file, "r") as f:
            data_dict = json.load(f)
            return data_dict
        return None

    def _create_examples(self, dict_data):
        examples = []
        for k, v in dict_data.items():
            guid = k
            sentence = tokenization.convert_to_unicode(v["sentence"])
            infos = v["info"]

            # We sum all the scores if we have more than 1 target
            total_sentiment_score = 0.0
            for info in infos:
                total_sentiment_score += float(info["sentiment_score"])
            sentiment_score = total_sentiment_score / float(len(infos))
            examples.append(
                InputExample(guid=guid, text_a=sentence, text_b=None, label=sentiment_score))
        return examples

    def get_train_examples(self):
        return self._create_examples(self._train_data_dict)

    def get_eval_examples(self):
        return self._create_examples(self._eval_data_dict)

