# -*- coding: utf-8 -*-

from __future__ import unicode_literals, print_function
from flask import Flask
from flask import request
from spacy.en import English
from io import BytesIO
from dragnet import content_extractor, content_comments_extractor
import requests
import json
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from cStringIO import StringIO
from bs4 import BeautifulSoup
import re
import Iscore

nlp = English()

app = Flask(__name__)

@app.route('/get_text')
def get_text():
   url = request.args.get('target')
   page = requests.get(url)
   page = content_extractor.analyze(page.content)
   text = tokenize_texts(page.decode('utf-8'))
   return json.dumps(text)

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
    tt = convert_pdf_to_txt(memoryFile)
    tt = clean_string(tt)
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
    t = t.decode("utf-8")
    t = t.replace('\n', ' ')
    t = " ".join(t.split())
    return t

def convert_pdf_to_txt(m_file):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()

    for page in PDFPage.get_pages(m_file, pagenos, maxpages=maxpages, password=password,caching=caching, check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    device.close()
    retstr.close()
    return text

if __name__ == '__main__':
    app.run(host= '0.0.0.0',port=8080, threaded=True)
