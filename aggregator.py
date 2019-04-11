#!/usr/bin/env python
# encoding: utf-8

import os
import json
import datetime
from dateutil import parser
from dateutil import tz

import tensorflow as tf

import logging
sh = logging.StreamHandler()
logger = logging.getLogger()
logger.addHandler(sh)
logger.setLevel(level=logging.INFO)

import spacy

import newscrawler
import db_manager
import predict_sentiments


def fetch_recent_headlines(temp_dir, db):
    logger.info(">>>>>> Fetching headlines <<<<<<<\n")

    # Fetch and store headlines for processing
    temp_newscrawler_paths, source_ids = newscrawler.run_all(temp_dir)

    all_recent_headlines = {}
    for i in range(len(source_ids)):
        path = temp_newscrawler_paths[i]
        res = db.fetch(source=source_ids[i], max_rows=1)
        if len(res) > 0:
            logger.info(res[0])
            latest_date = res[0]['date']
        else:
            latest_date = None
        recent_headlines = []

        with open(path, 'r') as fd:
            headlines = json.load(fd)
            for headline in headlines:
                headline_date = datetime.datetime.strptime(headline['date'], '%Y-%m-%dT%H:%M:%SZ')
                if latest_date is None or (headline_date > latest_date):
                    recent_headlines.append({'headline': headline['headline'], 'date': headline_date})
        all_recent_headlines[source_ids[i]] = recent_headlines
    return all_recent_headlines


def insert_recent_headlines(db, recent_headlines):
    logger.info(">>>>>>> Inserting headlines <<<<<<<<\n")
    to_insert = []
    for source_id, headlines in recent_headlines.iteritems():
        for headline in headlines:
            to_insert.append(
                {'text': headline['headline'], 'score': headline['score'], 'date': headline['date'], 'targets': headline['targets'], 'source': source_id})
    logger.info("Inserting %d entries" % len(to_insert))
    if len(to_insert) > 0:
        for row in to_insert:
            logger.info("Date: %s\nText: %s\nScore: %f\nEntities: %s\nSource: %s\n" % 
                        (row['date'], row['text'], row['score'], row['targets'], row['source']))
        db.insert(to_insert)


def tag_entities_recent_headlines(recent_headlines):
    def _tag(text):
        all_ents = []
        doc = nlp(text)
        if len(doc.ents) == 1:
            label = doc.ents[0].label_
            if label in [u'NORP', u'ORG', u'GPE']:
                if doc.ents[0].text == text:
                    all_ents.append(doc.ents[0].text)
                else:
                    all_ents.extend(_tag(doc.ents[0].text))
        elif len(doc.ents) > 1:
            for ent in doc.ents:
                label = ent.label_
                if label in [u'NORP', u'ORG', u'GPE']:
                    all_ents.extend(_tag(ent.text))
        return all_ents

    logger.info(">>>>>>>> Tagging entities <<<<<<<<<<\n")
    nlp = spacy.load('en_core_web_lg')
    for source_id, headlines in recent_headlines.iteritems():
        for headline in headlines:
            all_ents = _tag(headline['headline'])
            headline['targets'] = str(all_ents)


def score_recent_headlines(data_dir,
                           output_dir,
                           checkpoint_path,
                           recent_headlines):
    logger.info(">>>>>>>>>>> Scoring headlines <<<<<<<<<\n")
    temp_score_headlines = os.path.join(data_dir, 'predict.txt')
    with open(temp_score_headlines, 'w') as fd:
        for source_id, headlines in recent_headlines.iteritems():
            for headline in headlines:
                fd.write((headline['headline'] + '\n').encode('utf-8'))

    params = tf.app.flags.FLAGS.flag_values_dict().copy()
    params["data_dir"] = data_dir
    #params["debug_examples"] = True
    params["init_checkpoint"] = checkpoint_path
    #params["do_lower_case"] = BERT_MODEL.startswith('uncased')
    params["output_dir"] = output_dir

    res = predict_sentiments.run(params)
    index = 0
    for source_id, headlines in recent_headlines.iteritems():
        for headline in headlines:
            headline['score'] = res[index][1][0]
            index += 1

            
def run(db_path, data_dir, output_dir, checkpoint_path, temp_dir='/tmp'):
    db = db_manager.DBManager(db_path)
    recent_headlines = fetch_recent_headlines(temp_dir, db)
    score_recent_headlines(data_dir, output_dir, checkpoint_path, recent_headlines)
    tag_entities_recent_headlines(recent_headlines)
    insert_recent_headlines(db, recent_headlines)


if __name__ == "__main__":
    data_dir = 'data'
    output_dir = 'tmp/sentiments/predict_sentiments'
    checkpoint_path = 'models/FiQA/model.ckpt-1347' #'models/sentiments/model.ckpt-1531'

    run('tmp/test_db', data_dir, output_dir, checkpoint_path)
