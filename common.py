import tensorflow as tf


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

