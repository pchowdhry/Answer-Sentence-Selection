import numpy as np
import json
import requests
import os
from collections import defaultdict
from collections import namedtuple
from keras.models import load_model
from scipy.spatial.distance import cdist
import time
os.environ["THEANO_FLAGS"] = "device=gpu0,floatX=float32,dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic"
#os.environ["THEANO_FLAGS"] = "floatX=float32,dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic,blas.ldflags=-L/usr/lib/ -lblas"
#os.environ["THEANO_FLAGS"] = "floatX=float32,dnn.conv.algo_bwd_filter=deterministic,dnn.conv.algo_bwd_data=deterministic"

batch_size = 100
word_vec_file = 'data/GoogleNews-vectors-negative300.bin'
best_wang_model_file = 'checkpoint-23-0.94.hdf5'

best_wang_model = load_model(best_wang_model_file)
import datetime
#max_qs_l = 23
#max_ans_l = 40
vec_dim=300
sent_vec_dim=300
ans_len_cut_off=40
word_vecs = {}
stop_words=[]
idf = defaultdict(float)
data_folder=""
qtype_map, qtype_invmap = {}, {}
QASample = namedtuple("QASample", "Qs Ans QsWords AnsWords Label")

def load_word2vec(fname):
    """
    Loads 300x1 word vecs from Google (Mikolov) word2vec
    """
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
    word_vecs["<unk>"] = np.random.uniform(-0.25, 0.25, vec_dim)
    return word_vecs

word_vecs = load_word2vec(word_vec_file)

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

def get_sent_matrix(words):
    """
    Given a sentence, gets the input in the required format.
    """
    vecs = []
    vec = np.zeros(vec_dim, dtype='float32')
    for word in words:
        if word_vecs.has_key(word):
            vec = word_vecs[word]
        else:
            print(word)
            vec = word_vecs["<unk>"]
        vecs.append(np.array(vec))
    return np.array(vecs)

def parse_entry(qset):
    samples = []
    for e in qset:
        parts = e
        qs = parts[0]
        qs=qs.replace('\n','')
        ans = parts[1]
        ans=ans.replace('\n','')
        qwords = qs.split()
        answords = ans.split()
        label = int(parts[2].replace('\n', ''))
        sample = QASample(Qs=qs, Ans=ans, QsWords=qwords, AnsWords=answords, Label=label)
        samples.append(sample)
    return samples

def f_decompose(matrix, hatmatrix):
    # finding magnitude of parallel vector
    mag = np.sum(hatmatrix * matrix, axis=1) / np.sum(hatmatrix * hatmatrix, axis=1)
    # multiplying magnitude with hatmatrix vector
    plus = mag.reshape(-1, 1) * hatmatrix
    minus = matrix - plus
    return plus, minus

def compose_decompose(qmatrix, amatrix):
    qhatmatrix, ahatmatrix = f_match(qmatrix, amatrix, window_size=3)
    qplus, qminus = f_decompose(qmatrix, qhatmatrix)
    aplus, aminus = f_decompose(amatrix, ahatmatrix)
    return qplus, qminus, aplus, aminus

def f_match(qmatrix, amatrix, window_size=3):
    A = 1 - cdist(qmatrix, amatrix, metric='cosine')  # Similarity matrix
    Atranspose = np.transpose(A)
    qa_max_indices = np.argmax(A,
                               axis=1)  # 1-d array: for each question word, the index of the answer word which is most similar
    # Selecting answer word vectors in a window surrounding the most closest answer word
    qa_window = [range(max(0, max_idx - window_size), min(amatrix.shape[0], max_idx + window_size + 1)) for max_idx in
                 qa_max_indices]
    # Selecting question word vectors in a window surrounding the most closest answer word
    # Finding weights and its sum (for normalization) to find f_match for question for the corresponding window of answer words
    qa_weights = [(np.sum(A[qword_idx][aword_indices]), A[qword_idx][aword_indices]) for qword_idx, aword_indices in
                  enumerate(qa_window)]
    # Then multiply each vector in the window with the weights, sum up the vectors and normalize it with the sum of weights
    # This will give the local-w vecotrs for the Question sentence words and Answer sentence words.
    qhatmatrix = np.array([np.sum(weights.reshape(-1, 1) * amatrix[aword_indices], axis=0) / weight_sum for
                           ((qword_idx, aword_indices), (weight_sum, weights)) in
                           zip(enumerate(qa_window), qa_weights)])

    # Doing similar stuff for answer words
    aq_max_indices = np.argmax(A,
                               axis=0)  # 1-d array: for each   answer word, the index of the question word which is most similar
    aq_window = [range(max(0, max_idx - window_size), min(qmatrix.shape[0], max_idx + window_size + 1)) for max_idx in
                 aq_max_indices]
    aq_weights = [(np.sum(Atranspose[aword_idx][qword_indices]), Atranspose[aword_idx][qword_indices]) for
                  aword_idx, qword_indices in enumerate(aq_window)]
    ahatmatrix = np.array([np.sum(weights.reshape(-1, 1) * qmatrix[qword_indices], axis=0) / weight_sum for
                           ((aword_idx, qword_indices), (weight_sum, weights)) in
                           zip(enumerate(aq_window), aq_weights)])
    return qhatmatrix, ahatmatrix

def get_wang_model_input(samples, max_qs_l, max_ans_l):
    """
    Returns the training samples and labels as numpy array
    """
    s1samples_list = []
    s2samples_list = []
    labels_list = []

    for sample in samples:
        q_len=len(sample.QsWords)
        if q_len > max_qs_l:
            q_len=max_qs_l
        a_len=len(sample.AnsWords)
        if a_len > max_ans_l:
            a_len=max_ans_l
        s1samples_list.append(get_sent_matrix(sample.QsWords[0:q_len]))
        s2samples_list.append(get_sent_matrix(sample.AnsWords[0:a_len]))
        labels_list.append(sample.Label)

    samples_sent_matrix = [s1samples_list, s2samples_list]
    labels = labels_list
    return samples_sent_matrix, labels

def get_wang_conv_model_input(samples_sent_matrix, max_qs_l, max_ans_l):
    token = np.zeros((2, vec_dim), dtype='float')
    qsamples = samples_sent_matrix[0]
    asamples = samples_sent_matrix[1]
    q_list = []
    a_list = []
    for qmatrix, amatrix in zip(qsamples, asamples):
        qplus, qminus, aplus, aminus = compose_decompose(qmatrix, amatrix)
        # Padding questions
        qpad_width = ((0, max_qs_l - qplus.shape[0]), (0, 0))
        qplus_pad = np.pad(qplus, pad_width=qpad_width, mode='constant', constant_values=0.0)
        qminus_pad = np.pad(qminus, pad_width=qpad_width, mode='constant', constant_values=0.0)
        # Padding answers
        apad_width = ((0, max_ans_l - aplus.shape[0]), (0, 0))
        aplus_pad = np.pad(aplus, pad_width=apad_width, mode='constant', constant_values=0.0)
        aminus_pad = np.pad(aminus, pad_width=apad_width, mode='constant', constant_values=0.0)
        qplusminus=np.concatenate((qplus_pad, token, qminus_pad))
        aplusminus = np.concatenate((aplus_pad, token, aminus_pad))
        # Adding these padded matrices to list
        q_list.append(qplusminus)
        a_list.append(aplusminus)
    q_tensor = np.array(q_list)
    a_tensor = np.array(a_list)
    return q_tensor, a_tensor

class Timer:
    def __enter__(self):
        self.start = time.clock()
        return self

    def __exit__(self, *args):
        self.end = time.clock()
        self.interval = self.end - self.start

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
    #print s[0][0][0], s[0][1][1]

def dynamic_score(file_name,question):

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
    with Timer() as t:
        test_probs = best_wang_model.predict([test_q_tensor, test_a_tensor], batch_size=batch_size)
    print('Request took %.03f sec.' % t.interval)
    zipped = zip(test_probs, qset)
    s = sorted(zipped, key=lambda ranked: ranked[0], reverse=True)
    sorted_answers = []
    top = 5
    #print s[0][0][0], s[0][1][1]
    for i in range(0, top):
        print s[i][0][0], s[i][1][1]
        sorted_answers.append([str(s[i][0][0]), s[i][1][1]])
    return sorted_answers

#end