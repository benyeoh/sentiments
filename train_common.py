from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import bert.modeling as modeling
import bert.tokenization as tokenization
import tensorflow as tf

import model_classifier
import processors
import new


# Required parameters
tf.flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

tf.flags.DEFINE_string(
    "bert_config_file", "bert_models/uncased_L-12_H-768_A-12/bert_config.json",
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

tf.flags.DEFINE_string("vocab_file", "bert_models/uncased_L-12_H-768_A-12/vocab.txt",
                       "The vocabulary file that the BERT model was trained on.")

tf.flags.DEFINE_string(
    "output_dir", "tmp/sentiments/",
    "The output directory where the model checkpoints will be written.")

# Other parameters

tf.flags.DEFINE_string(
    "init_checkpoint", "bert_models/uncased_L-12_H-768_A-12/bert_model.ckpt",
    "Initial checkpoint (usually from a pre-trained BERT model).")

tf.flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

tf.flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

tf.flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

tf.flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

tf.flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

tf.flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

tf.flags.DEFINE_float("num_train_epochs", 3.0,
                      "Total number of training epochs to perform.")

tf.flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

tf.flags.DEFINE_integer("save_checkpoints_steps", 1000,
                        "How often to save the model checkpoint.")

tf.flags.DEFINE_integer("iterations_per_loop", 1000,
                        "How many steps to make in each estimator call.")

tf.flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

tf.flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

tf.flags.DEFINE_bool("use_dropout", True, "Whether to use dropout.")


def write_eval_results(result, output_dir):
    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
        tf.logging.info("***** Eval results *****")
        for key in sorted(result.keys()):
            tf.logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))


def read_eval_results(output_dir):
    output_eval_file = os.path.join(output_dir, "eval_results.txt")
    if tf.gfile.Exists(output_eval_file):
        with tf.gfile.GFile(output_eval_file, "r") as reader:
            res = {}
            for line in reader:
                key_val = line.split(' = ')
                key = key_val[0]
                val = key_val[1]
                try:
                    val = int(val)
                except ValueError:
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                res[key] = val
            return res
    return None


def compare_eval_save_model(new, output_dir, compare_fn, compare_key, save_output_dir):

    prev = read_eval_results(save_output_dir)
    if prev is None or (compare_fn(new, prev)):
        new_save_output_dir = os.path.join(save_output_dir, '%s_%f' % (compare_key, new[compare_key]))
        tf.gfile.MakeDirs(new_save_output_dir)

        tf.logging.info("We found a better result. New: %f" % (new[compare_key]))
        tf.logging.info("Saving checkpoint files to %s" % (new_save_output_dir))

        num_global_steps = new["global_step"]
        model_name = 'model.ckpt-%d' % num_global_steps
        filenames = [model_name + '.data-00000-of-00001',
                     model_name + '.index',
                     model_name + '.meta']

        for filename in filenames:
            from_path = os.path.join(output_dir, filename)
            to_path = os.path.join(new_save_output_dir, filename)
            tf.gfile.Copy(from_path, to_path)

        write_eval_results(new, save_output_dir)


def run_classifier(flags, processor):
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
    write_eval_results(result, FLAGS.output_dir)
    return result
