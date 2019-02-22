#!/usr/bin/env python
# encoding: utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os
import collections
import json
import csv
import re
import random

import bert.modeling as modeling
import bert.optimization as optimization
import bert.tokenization as tokenization

import tensorflow as tf
from numpy.ma.core import negative
from random import seed


_SEED = 3509843095


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


class FiQAPostsProcessor(object):

    def __init__(self, data_dir, train_ratio=0.9, seed=_SEED):
        self._data_dict = self._read_json_file(os.path.join(data_dir, "FiQA_ABSA", "task1_post_ABSA_train.json"))
        num_train = int(round(len(self._data_dict) * train_ratio))
        all_data = list(self._data_dict.items())
        random.seed(seed)
        random.shuffle(all_data)
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
            # Remove URLS in sentence
            sentence = re.sub(r'http\S+', '', sentence)
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


class FiQAHeadlinesProcessor(object):

    def __init__(self, data_dir, train_ratio=0, seed=_SEED):
        self._data_dict = self._read_json_file(os.path.join(data_dir, "FiQA_ABSA", "task1_headline_ABSA_train.json"))
        num_train = int(round(len(self._data_dict) * train_ratio))
        all_data = list(self._data_dict.items())
        random.seed(_SEED)
        random.shuffle(all_data)
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


class FiQACombineProcessor(object):

    def __init__(self, data_dir, train_ratio=0.9, seed=_SEED):
        self._posts = FiQAPostsProcessor(data_dir, train_ratio, seed=seed)
        self._headlines = FiQAHeadlinesProcessor(data_dir, train_ratio, seed=seed)
        self._seed = seed

    def get_train_examples(self):
        examples = self._posts.get_train_examples()
        examples.extend(self._headlines.get_train_examples())
        random.seed(self._seed)
        random.shuffle(examples)
        return examples

    def get_eval_examples(self):
        examples = self._posts.get_eval_examples()
        examples.extend(self._headlines.get_eval_examples())
        random.seed(self._seed)
        random.shuffle(examples)
        return examples


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class FiQAClassProcessor(DataProcessor):

    def __init__(self, fiqa_processor, class_separation=[0.0]):
        self._processor = fiqa_processor
        self._class = class_separation

    def _get_class(self, val):
        for i, x in enumerate(self._class):
            if val < x:
                return i
        return len(self._class)

    def get_train_examples(self, data_dir):
        train_examples = self._processor.get_train_examples()
        for example in train_examples:
            example.label = self.get_labels()[self._get_class(example.label)]
        return train_examples

    def get_dev_examples(self, data_dir):
        eval_examples = self._processor.get_eval_examples()
        for example in eval_examples:
            example.label = self.get_labels()[self._get_class(example.label)]
        return eval_examples

    # def get_test_examples(self, data_dir):
    #    return self._get_dev_examples(data_dir)

    def get_labels(self):
        return [str(i) for i in range(len(self._class) + 1)]


class FinPBProcessor(DataProcessor):

    def __init__(self, data_dir, train_ratio=0.9, seed=_SEED):
        #_100 = self._parse_txt(os.path.join(data_dir, 'financial_phrasebank', 'Sentences_AllAgree.txt'))
        _75 = self._parse_txt(os.path.join(data_dir, 'financial_phrasebank', 'Sentences_75Agree.txt'))
        #_66 = self._parse_txt(os.path.join(data_dir, 'financial_phrasebank', 'Sentences_66Agree.txt'))
        #sentences = self._parse_txt(os.path.join(data_dir, 'financial_phrasebank', 'Sentences_50Agree.txt'))
        sentences = _75
        random.seed(seed)
        random.shuffle(sentences)
        num_train = int(round(len(sentences) * train_ratio))
        self._train = sentences[:num_train]
        self._eval = sentences[num_train:]

    def _parse_txt(self, filepath):
        with tf.gfile.Open(filepath, "r") as f:
            sentences = []
            for line in f:
                sentence = None
                label_id = "0"
                if '@positive' in line:
                    label_id = "2"
                    sentence = line[:line.index('@positive')]
                elif '@neutral' in line:
                    label_id = "1"
                    sentence = line[:line.index('@neutral')]
                else:
                    assert '@negative' in line
                    sentence = line[:line.index('@negative')]
                sentences.append((sentence, label_id))
            return sentences
        return None

    def get_train_examples(self, data_dir):
        return self._create_examples(self._train, "train")

    def get_dev_examples(self, data_dir):
        return self._create_examples(self._eval, "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]

    def _create_examples(self, sentences, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, (sentence, label_id) in enumerate(sentences):
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(sentence)
            text_b = None
            if set_type == "test":
                label = "0"
            else:
                label = label_id
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class SSTProcessor(DataProcessor):

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")),
            "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0])
            text_b = None
            if set_type == "test":
                label = "0"
            else:
                label = tokenization.convert_to_unicode(line[1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class XnliProcessor(DataProcessor):
    """Processor for the XNLI data set."""

    def __init__(self):
        self.language = "zh"

    def get_train_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(
            os.path.join(data_dir, "multinli",
                         "multinli.train.%s.tsv" % self.language))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "train-%d" % (i)
            text_a = tokenization.convert_to_unicode(line[0])
            text_b = tokenization.convert_to_unicode(line[1])
            label = tokenization.convert_to_unicode(line[2])
            if label == tokenization.convert_to_unicode("contradictory"):
                label = tokenization.convert_to_unicode("contradiction")
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_dev_examples(self, data_dir):
        """See base class."""
        lines = self._read_tsv(os.path.join(data_dir, "xnli.dev.tsv"))
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "dev-%d" % (i)
            language = tokenization.convert_to_unicode(line[0])
            if language != tokenization.convert_to_unicode(self.language):
                continue
            text_a = tokenization.convert_to_unicode(line[6])
            text_b = tokenization.convert_to_unicode(line[7])
            label = tokenization.convert_to_unicode(line[1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
            text_a = tokenization.convert_to_unicode(line[8])
            text_b = tokenization.convert_to_unicode(line[9])
            if set_type == "test":
                label = "contradiction"
            else:
                label = tokenization.convert_to_unicode(line[-1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[3])
            text_b = tokenization.convert_to_unicode(line[4])
            if set_type == "test":
                label = "0"
            else:
                label = tokenization.convert_to_unicode(line[0])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # Only the test set has a header
            if set_type == "test" and i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            if set_type == "test":
                text_a = tokenization.convert_to_unicode(line[1])
                label = "0"
            else:
                text_a = tokenization.convert_to_unicode(line[3])
                label = tokenization.convert_to_unicode(line[1])
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples
