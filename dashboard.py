#!/usr/bin/env python

import os
from flask import Flask, request, session, g, redirect, url_for, abort, render_template, flash

import db_manager


app = Flask(__name__,
            static_folder="dashboard/static",
            template_folder="dashboard/templates")
app.config["DB_PATH"] = 'tmp/test_db'

db = db_manager.DBManager(app.config["DB_PATH"])


@app.route('/', methods=['GET', 'POST'])
def main():
    num_rows = request.values.get('num_rows')
    if num_rows is None:
        num_rows = 50

    col_names = ['date',
                 'text',
                 'score',
                 'targets',
                 'source']

    rows = []
    results = db.fetch(max_rows=num_rows)
    if len(results) > 0:
        # Assume all rows have same schema
        #col_names = results[0].keys()
        for row in results:
            values = []
            for col in col_names:
                values.append(row[col])
            rows.append(values)

    return render_template('main.html',
                           col_names=col_names,
                           rows=rows)
