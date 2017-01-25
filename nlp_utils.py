# -*- coding: UTF-8 -*-
from __future__ import unicode_literals, print_function
from flask import Flask
from flask import request
from spacy.en import English
from io import BytesIO
import datetime
#import html
import requests
import re
import json
import os
import sys
import PyPDF2
from bs4 import BeautifulSoup
import Iscore

nlp = English()

app = Flask(__name__)

@app.route('/get_text')
def get_text():
   url = request.args.get('target')
   page = requests.get(url)
   page = BeautifulSoup(page.content, 'html.parser')
   for script in page.find_all("script"):
    script.decompose()
    text = tokenize_texts(page.get_text())
   tt = []
   s = 'menu'
   for t in text:
       if t.endswith('.') and s not in t:
           t = t.replace('\n', ' ')
           t = t.replace(':', ' ')
           t = re.sub( '\s+', ' ', t ).strip()
           #t.sub(r'\.([a-zA-Z])', r'. \1', t)
           tt.append(t)
   return json.dumps(tt)

@app.route('/generate_kb')
def generate_kb():
   url = request.args.get('target')
   kb_name = request.args.get('kb_name')
   page = requests.get(url)
   page = BeautifulSoup(page.content, 'html.parser')
   for script in page.find_all("script"):
    script.decompose()
    text = tokenize_texts(page.get_text())
   tt = []
   s = 'menu'
   for t in text:
       if t.endswith('.') and s not in t:
           t = t.replace('\n', ' ')
           t = t.replace(':', ' ')
           t = t.replace('.', '')
           t = t.replace(',', '')
           t = t.replace('(', '')
           t = t.replace(')', '')
           t = t.replace('-', ' ')
           t = re.sub( '\s+', ' ', t ).strip()
           t = re.sub('[^A-Za-z0-9]+', ' ', t)
           #t.sub(r'\.([a-zA-Z])', r'. \1', t)
           if len(t) > 1:
               tt.append(t)
       thefile = open("%s.txt" % kb_name, 'w' )
       for item in tt:
           thefile.write(item.encode('utf-8').strip())
           thefile.write('\n')
       thefile.close
   return json.dumps(tt)

@app.route('/get_text_from_pdf')
def get_from_pdf():
    url = request.args.get('target')
    pdf = requests.get(url)
    memoryFile = BytesIO(pdf.content)
    pdfReader = PyPDF2.PdfFileReader(memoryFile)
    if pdfReader.isEncrypted:
        pdfReader.decrypt('')
    num_pages = pdfReader.getNumPages()
    tt = ""
    for i in range(num_pages):
        pageObj = pdfReader.getPage(i)
        t = pageObj.extractText()
        t = (clean_string(t))
        tt = tt + t
    tt = tokenize_texts(tt)
    return json.dumps(tt)

@app.route('/tokenize')
def tokenize_texts(texts):
    raw_text = texts
    doc = nlp(raw_text)
    sentences = [sent.string.strip() for sent in doc.sents]
    return sentences

@app.route('/answers')
def get_answers():
    question = request.args.get('question')
    file = request.args.get('file')
    answers = Iscore.dynamic_score(file, question)
    return json.dumps(answers)


def clean_string(t):
    t = t.replace('\n', ' ')
    t = t.replace('¥!', ' ')
    t = t.replace('¥', '')
    t = " ".join(t.split())
    return t

if __name__ == '__main__':
    app.run(host= '0.0.0.0',port=8080, threaded=True)
