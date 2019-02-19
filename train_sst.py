#!/usr/bin/env python
# encoding: utf-8

"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import bert.modeling as modeling
import bert.tokenization as tokenization
import tensorflow as tf

import model_classifier
import processors
import common


def run(flags):
    # I'm lazy
    class AttrDict(dict):
        def __init__(self, *args, **kwargs):
            super(AttrDict, self).__init__(*args, **kwargs)
            self.__dict__ = self
            
    FLAGS = AttrDict()
    FLAGS.update(flags)    

    tf.logging.set_verbosity(tf.logging.INFO)

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if not FLAGS.use_dropout:
        tf.logging.info("Disabling dropout")
        bert_config.attention_probs_dropout_prob = 0.0
        bert_config.hidden_dropout_prob = 0.0

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    processor = processors.SSTProcessor()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None

    train_examples = processor.get_train_examples(FLAGS.data_dir)
    #train_examples = train_examples[:1]

    num_train_steps = int(
        len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_classifier.model_fn_builder(
        bert_config=bert_config,
        num_labels=len(label_list),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps,
        use_tpu=FLAGS.use_tpu,
        use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_size=FLAGS.eval_batch_size,
        predict_batch_size=FLAGS.predict_batch_size)

    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    model_classifier.file_based_convert_examples_to_features(
        train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)

    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = model_classifier.file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)
    if FLAGS.use_tpu:
        # TPU requires a fixed batch size for all batches, therefore the number
        # of examples must be a multiple of the batch size, or else examples
        # will get dropped. So we pad with fake examples which are ignored
        # later on. These do NOT count towards the metric (all tf.metrics
        # support a per-instance weight, and these get a weight of 0.0).
        while len(eval_examples) % FLAGS.eval_batch_size != 0:
            eval_examples.append(model_classifier.PaddingInputExample())

    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    model_classifier.file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
        assert len(eval_examples) % FLAGS.eval_batch_size == 0
        eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = model_classifier.file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)

    result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
        tf.logging.info("***** Eval results *****")
        for key in sorted(result.keys()):
            tf.logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))


def main(_):
    run(tf.app.flags.FLAGS.flag_values_dict())


if __name__ == "__main__":
    tf.flags.mark_flag_as_required("data_dir")
    # tf.flags.mark_flag_as_required("vocab_file")
    # tf.flags.mark_flag_as_required("bert_config_file")
    # tf.flags.mark_flag_as_required("output_dir")
    tf.app.run()
