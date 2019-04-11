#!/usr/bin/env python
# encoding: utf-8

import peewee
import sqlite3
import datetime


class DBManager(object):

    def __init__(self, path):
        self._db = peewee.Proxy()

        class Sentiments(peewee.Model):

            date = peewee.DateTimeField()
            text = peewee.TextField()
            score = peewee.DoubleField()
            targets = peewee.TextField()
            source = peewee.TextField()

            class Meta:
                database = self._db
                db_table = 'sentiments'

        self._model = Sentiments
        self._initialize(path)

    def _initialize(self, db_path, backend='sqlite'):
        db = None
        if backend == 'sqlite':
            db = peewee.SqliteDatabase(db_path)
        self._db.initialize(db)
        self._db.connect()
        self._model.create_table(safe=True)

    def fetch(self, date_from=None, date_to=None, source=None, max_rows=1):
        query = self._model.select()
        cond = self._model.text != ''
        cond = (cond & (self._model.date >= date_from)) if date_from is not None else cond
        cond = (cond & (self._model.date <= date_to)) if date_to is not None else cond
        cond = (cond & (self._model.source == source)) if source is not None else cond
        query = query.where(cond)
        query = query.order_by(self._model.date.desc())
        query = query.limit(max_rows)
        return query.dicts()

    def insert(self, rows):
        self._model.insert_many(rows).execute()


if __name__ == "__main__":
    db_mgr = DBManager('/tmp/testy.sqlite')
    db_mgr.insert([{'text': 'test1', 'score': 0.1, 'date': datetime.datetime.now(), 'targets': '["aaa"]', 'source': 'testy_source1'},
                   {'text': 'test2', 'score': -0.1, 'date': datetime.datetime.now(), 'targets': '["bbb"]', 'source': 'testy_source1'},
                   {'text': 'test3', 'score': 0.0, 'date': datetime.datetime.now(), 'targets': '["ccc"]', 'source': 'testy_source2'}])
    res = db_mgr.fetch(source='testy_source1', max_rows=100)
    print(len(res))
    print(res[0])


