#!/usr/bin/env python
# encoding: utf-8

"""BERT finetuning runner."""

import os

import processors
import train_common
import train_all_fin_class
import train_finpb
import train_fiqa_score

import tensorflow as tf


tf.flags.DEFINE_bool("num_pretrain_epochs", True, "Num epochs to pretrain.")
tf.flags.DEFINE_bool("pretrain_finpb_only", False, "Only pretrain with financial_phrasebank data.")


def save_improvement(output_dir, save_output_dir, res):
    train_common.compare_eval_save_model(
        res, output_dir, lambda x, y: x["eval_loss"] < y["eval_loss"], "eval_loss", save_output_dir)


def run(flags):
    num_train_epochs = flags["num_train_epochs"]
    output_dir = flags["output_dir"]
    
    if flags["num_pretrain_epochs"] > 0.0:
        flags["num_train_epochs"] = flags["num_pretrain_epochs"]
        flags["output_dir"] = os.path.join(output_dir, "pretrain")

        tf.logging.info("------------------ Running pretraining --------------------------")
        if flags["pretrain_finpb_only"]:
            results = train_finpb.run(flags)
        else:
            results = train_all_fin_class.run(flags)
            
        flags["num_train_epochs"] = num_train_epochs
        checkpoint_name = "model.ckpt-%d" % results["global_step"]
        flags["init_checkpoint"] = os.path.join(flags["output_dir"], checkpoint_name)

    tf.logging.info("------------------ Running scoring --------------------------")
    flags["output_dir"] = output_dir
    results = train_fiqa_score.run(flags)
    return results


def main(_):
    run(tf.app.flags.FLAGS.flag_values_dict())


if __name__ == "__main__":
    tf.flags.mark_flag_as_required("data_dir")
    # tf.flags.mark_flag_as_required("vocab_file")
    # tf.flags.mark_flag_as_required("bert_config_file")
    # tf.flags.mark_flag_as_required("output_dir")
    tf.app.run()
