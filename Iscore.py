from CNNQAClassifier import load_word2vec, get_wang_model_input, get_wang_conv_model_input, load_model, parse_entry
import json
import requests

batch_size = 1
word_vec_file = 'data/GoogleNews-vectors-negative300.bin'
best_wang_model_file = 'checkpoint-21-0.83.hdf5'
word_vecs = load_word2vec(word_vec_file)
best_wang_model = load_model(best_wang_model_file)
import datetime
max_qs_l = 23
max_ans_l = 40

def load_sentences_from_file(f):
    with open(f, 'r') as myfile:
        data = myfile.readlines()
    data = [x.strip() for x in data]
    return data

def build_kb(file_name, url):
    payload = {'target': url}
    raw_base = json.loads(requests.get('http://localhost:5000/get_text', params=payload).text)
    thefile = open(file_name, 'w')
    for sentence in raw_base:
        thefile.write("%s\n" % sentence.encode('ascii', 'ignore').decode('ascii'))

def run_test():
    raw_base = load_sentences_from_file('rb.txt')
    q = 'what are mortgage points'
    kb = raw_base
    qset = []
    for sentence in raw_base:
        qset.append([q,sentence, '0'])
    test_samples = parse_entry(qset)
    max_qs_l = 23
    max_ans_l = 40
    test_samples_sent_matrix, test_labels = get_wang_model_input(test_samples, max_qs_l, max_ans_l)
    test_q_tensor, test_a_tensor = get_wang_conv_model_input(test_samples_sent_matrix, max_qs_l, max_ans_l)
    test_probs = best_wang_model.predict([test_q_tensor, test_a_tensor], batch_size=batch_size)
    zipped = zip(test_probs, qset)
    s = sorted(zipped, key=lambda ranked: ranked[0], reverse=True)
    print s[0][0][0], s[0][1][1]

def dynamic_score(file_name,question):
    print(datetime.datetime.now())
    raw_base = load_sentences_from_file(file_name)
    q = question
    kb = raw_base
    qset = []
    for sentence in raw_base:
        qset.append([q,sentence, '0'])
    test_samples = parse_entry(qset)
    max_qs_l = 23
    max_ans_l = 40
    test_samples_sent_matrix, test_labels = get_wang_model_input(test_samples, max_qs_l, max_ans_l)
    test_q_tensor, test_a_tensor = get_wang_conv_model_input(test_samples_sent_matrix, max_qs_l, max_ans_l)
    print(datetime.datetime.now())
    test_probs = best_wang_model.predict([test_q_tensor, test_a_tensor], batch_size=batch_size)
    print(datetime.datetime.now())
    zipped = zip(test_probs, qset)
    s = sorted(zipped, key=lambda ranked: ranked[0], reverse=True)
    sorted_answers = []
    top = 5
    #print s[0][0][0], s[0][1][1]
    for i in range(0, top):
        print s[i][0][0], s[i][1][1]
        sorted_answers.append([str(s[i][0][0]), s[i][1][1]])
    return sorted_answers
